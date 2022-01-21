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

DEVICE = "cuda"
DEVICES = [0, 1, 2]
MAX_LEN = 1024


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
        mini_batch_size=4,
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
                #print("my batch ", my_batch.shape)
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
                #print("my batch2 ", my_batch.shape)
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
    model_init_fn_name="days.w2d5.dataparallel.init_model",
    sharded_optimizer=False,
):
    print("i'm rank", rank, " sharded is ", sharded_optimizer)
    # device = "cuda:" + str(rank)
    model = import_object_from_qualified_name(model_init_fn_name)()
    model.train()
    model.to(DEVICE)
    if sharded_optimizer:
        # use naive random param split
        all_params = list(model.parameters())
        random.seed(0)
        random.shuffle(all_params)
        per_bucket = len(all_params) // size
        param_buckets = [
            all_params[i * per_bucket : (i + 1) * per_bucket] for i in range(rank)
        ]
        params = param_buckets[rank]
    else:
        params = list(model.parameters())
    optimizer = t.optim.Adam(params, lr=1e-4)

    # If rank 0, loads data, splits things, keeps a minibatch
    # else, listen for a minibatch from rank 1
    dataloader = DistributedDataLoader(rank=rank, size=size)
    dist.barrier()
    #print("thru the barrier, i'm rank", rank, " sharded is ", sharded_optimizer)
    pbar = tqdm(dataloader)
    for batch in pbar:
        #print("ere ", rank)
        #print("batch1 ", batch)
        out = model(batch.to(DEVICE)).logits
        #print("fore ", rank)
        loss = t.nn.CrossEntropyLoss()(
            rearrange(out[:, :-1], "a b c -> (a b) c"),
            rearrange(batch[:, 1:], "a b -> (a b)"),
        )
        #print("out; ", out[:-1], "batch ", batch[1:])
        loss.backward()
        #print("here ", rank)
        if sharded_optimizer:
            add_grad(param_buckets, rank)
        else:
            alladd_grad(model)
        #print("there", rank)
        optimizer.step()
        optimizer.zero_grad()
        #print("everywhere", rank)
        if sharded_optimizer:
            broadcast_updated_params(param_buckets, rank)
        # print(rank, "loss", loss.cpu().detach().numpy())
        #print(rank, batch_num)
        pbar.set_description(f"loss {loss}")
    #print(rank, "done training")
    dist.barrier()

    if rank == 0:
        killgroup()


@gin.configurable
def init_process(
    rank, size, run, device, backend="nccl"
):  # gloo is algo for sharing gradients. nccl better?
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"  # make the master available for mutual contact
    if device == "cuda":
        global DEVICE
        DEVICE = "cuda:" + str(DEVICES[rank])
    dist.init_process_group(backend, rank=rank, world_size=size)
    print("inited process group", rank)

    run(rank, size)


@gin.configurable
def create_processes(
    local_parallelism=2,
    device="cuda",
):
    # raise AssertionError(":)")
    processes = []
    mp.set_start_method("spawn")
    for rank in range(local_parallelism):  # process index = rank
        p = mp.Process(target=init_process, args=(rank, local_parallelism, run, device))
        p.start()
        processes.append(p)
    # pytorch join requires you to join in order of completion!???


if __name__ == "__main__":
    # gin.parse_config_file(sys.argv[1])
    create_processes()
