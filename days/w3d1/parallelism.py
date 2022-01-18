import os
import random
import torch.distributed as dist
import torch.multiprocessing as mp
from torchtyping import tensor_type
import gin
import sys
from days.utils import import_object_from_qualified_name
import torch as t
import numpy as np
from days.utils import *
import os
import signal
import transformers
from einops import *
import json
from tqdm import tqdm
import torchtext

DEVICE = "cpu"
DEVICES = [4, 5, 6, 7]
MAX_LEN = 1024


class FirstElement(t.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x):
        return self.module(x)[0]
    
class LastElement(t.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x):
        return self.module(x)[:,-1,:]

    
def extract_batches(tokenizer, dataset, micro_per_mini = 8, microbatch_size = 4,  max_seq_length=128, vocab_size = 50400):
    random.seed(2)
    # dataset.sort(key=lambda x: len(x[1]))
    minibatch_size = micro_per_mini * microbatch_size
    
    number_of_batches = math.ceil(len(dataset) / minibatch_size)
    
    print("number of batches", number_of_batches)
    print(type(dataset))
    dataset = np.array_split(dataset, number_of_batches)

    batches = []

    for batch in dataset:
        scores, reviews = zip(*batch)
        tokens = tokenizer(list(reviews))

        # print(len(tokens['input_ids'][0]))
        tokens = tokens["input_ids"]

        tokens = [review_token[:max_seq_length] for review_token in tokens]
        
        
        tokens = [review_token if len(review_token) == max_seq_length else ([random.randint(0, vocab_size) for _ in range(max_seq_length - len(review_token))] + review_token) for review_token in tokens]
        
        
        X = t.tensor(tokens)
        y = t.tensor([1 if s == "pos" else 0 for s in scores])
        
        X = t.split(X, microbatch_size, dim = 0)
        y = t.split(y, microbatch_size, dim = 0)

        batches.append((X,y))

    random.shuffle(batches)
    return batches



##################################
# print(len(extract_batches(dataset_train)))


def load_data():
    print("loading data")
    tensor_path = "/home/ubuntu/lw.pt"
    if os.path.exists(tensor_path):
        tokens = t.load(tensor_path)
        print("tokens shape", tokens.shape)
    else:
        lw_json = json.load(open("/home/ubuntu/lw_corpus.json"))
        print("have json")
        texts = [x["text"] for x in lw_json]
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "gpt2", TOKENIZERS_PARALLELISM=True
        )
        eot_id = 50256
        eot_id = tokenizer("<|endoftext|>")["input_ids"][0]
        tokens = tokenizer(texts)["input_ids"]
        for seq in tokens:
            seq.append(eot_id)
        print("tokenized")
        tokens = t.LongTensor(list(itertools.chain(*tokens)))

        t.save(tokens, tensor_path)
    return rearrange(
        tokens[: tokens.shape[0] - (tokens.shape[0] % MAX_LEN)],
        "(b s) -> b s",
        s=MAX_LEN,
    )


def init_model():
    t.random.manual_seed(0)
    # storing model locally because huggingface throttles checking
    if os.path.exists("/home/ubuntu/gpt2_copy.pt"):
        return t.load("/home/ubuntu/gpt2_copy.pt")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    return model


@gin.configurable
class DistributedDataLoader:
    def __init__(
        self,
        rank,
        size,
        data_size=(1024,),
        data_fn="days.w2d5.dataparallel.load_data",
        mini_batch_size=1,
        random_seed=0,
    ):
        self.rank = rank
        self.size = size
        self.data_size = data_size
        self.mini_batch_size = mini_batch_size
        if rank == 0:
            self.data_tensors = import_object_from_qualified_name(data_fn)()
            print("data tensors size", self.data_tensors.shape)
            t.manual_seed(random_seed)
            perm = t.randperm(self.data_tensors.shape[0])
            self.data_tensors = self.data_tensors[perm]
            n_batches = self.data_tensors.shape[0] // (mini_batch_size * size)
            self.batches = self.data_tensors[
                : self.data_tensors.shape[0]
                - (self.data_tensors.shape[0] % (mini_batch_size * size))
            ]
            self.batches = self.batches.reshape(-1, size, mini_batch_size, *data_size)
            print("self batches shape", self.batches.shape)
            self.len = len(self.batches)
        else:
            self.len = -1
            self.batches = None
        btsr = t.Tensor([self.len]).to(DEVICE)
        dist.broadcast(btsr, src=0)
        self.len = int(btsr.cpu().item())

    def __len__(self):
        return self.len

    def __iter__(self):
        if self.batches is not None:
            for mini_batches in self.batches:
                mini_batches = mini_batches.to(DEVICE)
                dist.broadcast(
                    mini_batches, src=0
                )  # all processes must do this, else all wait forever
                my_batch = mini_batches[self.rank]
                yield my_batch
        else:
            for _ in range(self.len):
                mini_batches = t.zeros(
                    self.size,
                    self.mini_batch_size,
                    *self.data_size,
                    dtype=t.int64,
                    device=DEVICE,
                )
                dist.broadcast(mini_batches, src=0)
                my_batch = mini_batches[self.rank]
                yield my_batch


def alladd_grad(model):

    reduce_ops = [
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        for param in model.parameters()
    ]
    for op in reduce_ops:
        op.wait()


def add_grad(buckets, rank):
    reduce_ops = []
    for i, bucket in enumerate(buckets):
        for param in bucket:
            reduce_ops.append(dist.reduce(param.grad, i, async_op=True))
    for op in reduce_ops:
        op.wait()


def broadcast_updated_params(buckets, rank):
    reduce_ops = []
    for i, bucket in enumerate(buckets):
        for param in bucket:
            reduce_ops.append(dist.broadcast(param, i, async_op=True))
    for op in reduce_ops:
        op.wait()


def killgroup():
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)



    
@gin.configurable()
def run(
    rank,
    size,
    micro_per_mini,
    microbatch_size,
    embedding_dim = 4096,
    max_seq_length = 128,
):
    print("i'm rank", rank)
    # device = "cuda:" + str(rank)
        
    model = t.load("gptj_{}.pt".format(rank + 1))
    
    model.train()
    model.to(DEVICES[rank])

    # If rank 0, loads data, splits things, keeps a minibatch
    # else, listen for a minibatch from rank 1
    
    current_device = DEVICES[rank]
    if rank == 0:
        tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        
        data_train = torchtext.datasets.IMDB(root='.data',split='train')
        data_test = torchtext.datasets.IMDB(root='.data',split='test')

        dataset_train = []
        for d in data_train:
            dataset_train.append(d)

        dataset_test = []
        for d in data_test:
            dataset_test.append(d)

        all_batches = extract_batches(tokenizer=tokenizer, dataset=dataset_train, micro_per_mini = micro_per_mini, microbatch_size = microbatch_size,  max_seq_length=max_seq_length, vocab_size = 50400)
    
    num_of_microbatches = t.tensor([0]).to(current_device)
    
    if rank == 0:
        num_of_microbatches = t.tensor([len(all_batches)]).to(current_device)
        
    dist.broadcast(num_of_microbatches, 0)
    
    print("before loop", rank, num_of_microbatches)
    
    x = t.zeros([microbatch_size, max_seq_length, embedding_dim]).to(current_device)
    
    groups = [[dist.new_group([i,j]) for j in range(size)] for i in range(size)]
    
    # microbatch_size, seq_len, embedding_dim
    y = t.zeros([microbatch_size],dtype=t.int64).to(current_device)
    for j in range(int(num_of_microbatches.item())):
        if rank == 0:
            mini_batch = all_batches[j]
        outputs = []
        print("{} is beginning the inner loop".format(rank))
        for i in range(microbatch_size):
            print("microbatch:", i)
            if rank == 0:
                # get next microbatch, send
                print("rank 0 conditional started")
                microbatch = mini_batch[0][i].to(current_device)
                x = model(microbatch)
                y = mini_batch[1][i].to(current_device)
                print("dtype of first y", y.dtype)
                print("shape of first y", y.shape)
                print("value of first y", y)


                #dist.broadcast(x, DEVICES[0], [DEVICES[0],DEVICES[1]])
                #dist.broadcast(y, DEVICES[0], [DEVICES[0], DEVICES[size - 1]])
                print("starting x")
                
                dist.broadcast(x, 0, groups[0][1])
                
                # t.cuda.synchronize(DEVICES[-1])
                
                print("starting y")
                print("first device at s -1", t.cuda.get_device_name(size - 1))
                dist.broadcast(y, 0, groups[0][size - 1])
                print("rank 0 conditional ended")
            elif rank != size - 1:
                print("rank {} conditional started".format(rank))
                # wait to receive from #device rank-1, send
                dist.broadcast(x, rank - 1, groups[rank - 1][rank])
                x = model(x)
                dist.broadcast(x, rank, groups[rank][rank + 1])
                print("rank {} conditional ended".format(rank))
            else:
                print("Last conditional started")

                print("dtype of last y", y.dtype)
                print("shape of last y", y.shape)
                print("value of last y", y)
                # wait to receive from device size - 1
                # x = dist.broadcast(x, DEVICES[rank - 1], [DEVICES[rank - 1], DEVICES[rank]])
                print("last device at s -1", t.cuda.get_device_name(size - 1))
                #t.cuda.synchronize(DEVICES[0])

                dist.broadcast(y, 0, groups[0][size - 1])

                print("finished last broadcast of y")
                dist.broadcast(x, rank - 1, groups[rank-1][rank])
                output = model(x)
                print("finished last broadcast of x")
                outputs.append((output, y))
                # save x as final value
                print("Last conditional ended")
                print("Done!")
        print("outputs:", outputs)
        print(i)
        break
                
#             # do forward

#             if rank == size - 1:
#                 # do something with final output (put this in array)
#             else:
#                 # pass on to rank + 1

#             # save activations to tensor for backward


#         for i in range(micro_batches):
#             if rank == size - 1:
#                 # get correct microbatch activation
#             else:
#                 # get activation for microbatch i from device rank + 1

    
    
#     dataloader = DistributedDataLoader(rank=rank, size=size)
#     dist.barrier()
#     pbar = tqdm(enumerate(dataloader))
#     for batch_num, batch in pbar:
#         out = model(batch.to(DEVICE)).logits
#         loss = t.nn.CrossEntropyLoss()(
#             rearrange(out[:-1], "a b c -> (a b) c"),
#             rearrange(batch[1:], "a b -> (a b)"),
#         )
#         loss.backward()
#         if sharded_optimizer:
#             add_grad(param_buckets, rank)
#         else:
#             alladd_grad(model)
#         optimizer.step()
#         optimizer.zero_grad()
#         if sharded_optimizer:
#             broadcast_updated_params(param_buckets, rank)
#         # print(rank, "loss", loss.cpu().detach().numpy())
#         print(rank, batch_num)
#         pbar.set_description(f"loss {loss.cpu().item()}")
#     print(rank, "done training")
#     dist.barrier()

#     if rank == 0:
#         killgroup()


@gin.configurable
def init_process(
    rank, size, micro_per_mini, microbatch_size, run, device, backend="gloo"
):  # gloo is algo for sharing gradients. nccl better?
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29504"  # make the master available for mutual contact
    if device == "cuda":
        global DEVICE
        DEVICE = "cuda:" + str(DEVICES[rank])
    dist.init_process_group(backend, rank=rank, world_size=size)
    print("inited process group", rank)

    run(rank, size, micro_per_mini, microbatch_size)



@gin.configurable
def create_processes(
    local_parallelism=2,
    micro_per_mini = 8,
    microbatch_size = 4,
    device="cpu",
):
    # raise AssertionError(":)")
    processes = []
    mp.set_start_method("spawn")
    for rank in range(local_parallelism):  # process index = rank
        p = mp.Process(target=init_process, args=(rank, local_parallelism, micro_per_mini, microbatch_size, run, device))
        p.start()
        processes.append(p)
    # pytorch join requires you to join in order of completion!???


if __name__ == "__main__":
    # gin.parse_config_file(sys.argv[1])
    
    micro_per_mini = 1
    microbatch_size = 1
    max_seq_length = 2
    size = len(DEVICES)
    
    create_processes(local_parallelism = size, micro_per_mini = micro_per_mini, microbatch_size = microbatch_size)
    
#     tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

#     data_train = torchtext.datasets.IMDB(root='.data',split='train')

#     dataset_train = []
#     for d in data_train:
#         dataset_train.append(d)

    
    
#     all_batches = extract_batches(tokenizer=tokenizer, dataset=dataset_train, micro_per_mini = micro_per_mini, microbatch_size = microbatch_size,  max_seq_length=max_seq_length, vocab_size = 50400)
#     first_batch_X, _ = all_batches[0]
    
#     model = transformers.AutoModelForSequenceClassification.from_pretrained("EleutherAI/gpt-j-6B")
    
    
#     for i, microbatch in enumerate(first_batch_X):
#         #print(microbatch)
#         print(microbatch.shape)
#         print(type(microbatch))
#         print("true_outputs", i, model(microbatch).logits)
        