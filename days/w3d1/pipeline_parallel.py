from torch.distributed.distributed_c10d import broadcast
import transformers
import torch as t
import torchtext
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from days.utils import import_object_from_qualified_name
from days.w2d5.dataparallel import killgroup
from einops import rearrange

DEVICES = ["cuda:4", "cuda:5", "cuda:6", "cuda:7"]
MAX_LEN = 128
BATCH_SIZE = 4

def load_data(batch_size=BATCH_SIZE, random_seed=0):
    print("loading data")
    tensor_path_tokens = "/home/ubuntu/dm_and_nina/days/w3d1/imdb_tokens.pt"
    tensor_path_labels = "/home/ubuntu/dm_and_nina/days/w3d1/imdb_labels.pt"
    if os.path.exists(tensor_path_tokens) and os.path.exists(tensor_path_labels):
        batches = t.load(tensor_path_tokens)
        batched_labels = t.load(tensor_path_labels)
        print("batches shapes", batches.shape, batched_labels.shape)
    else:
        data_train, _ = torchtext.datasets.IMDB(root='.data', split=('train', 'test'))
        data_train = list(data_train)
        tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        tokenizer.pad_token = tokenizer.eos_token
        tokens = tokenizer([_t[1] for _t in data_train], padding="longest", max_length=MAX_LEN, truncation=True)["input_ids"]
        tokens = t.tensor(tokens).long()
        labels = t.tensor([0 if _t[0] == "neg" else 1 for _t in data_train])
        print(tokens.shape, labels.shape)

        t.manual_seed(random_seed)
        perm = t.randperm(tokens.shape[0])
        tokens = tokens[perm]
        labels = labels[perm]
        leftover = tokens.shape[0] % batch_size
        batches = rearrange(tokens[leftover: ], "(k b) l -> k b l", b = batch_size)
        batched_labels = rearrange(labels[leftover: ], "(k b) -> k b", b = batch_size)

        t.save(batches, tensor_path_tokens)
        t.save(batched_labels, tensor_path_labels)
    return batches, batched_labels

class GPTJPart(t.nn.Module):
    def __init__(self, model, part_num):
        super(GPTJPart, self).__init__()
        num_blocks = len(model.transformer.h)

        if part_num == 0:

            # self.layers = t.nn.Linear(MAX_LEN, 4096)

            self.layers = t.nn.Sequential(
                model.transformer.wte,
                model.transformer.drop,
                *model.transformer.h[0:num_blocks//4]
            )

        elif part_num == 1:

            # self.layers = t.nn.Linear(4096, 4096)

            self.layers = t.nn.Sequential(
                *model.transformer.h[num_blocks//4:2*num_blocks//4]
            )

        elif part_num == 2:

            # self.layers = t.nn.Linear(4096, 4096)
            
            self.layers = t.nn.Sequential(
                *model.transformer.h[2*num_blocks//4:3*num_blocks//4]
            )

        elif part_num == 3:

            # self.layers = t.nn.Linear(4096, 2)
            
            self.layers = t.nn.Sequential(
                *model.transformer.h[3*num_blocks//4:],
                model.transformer.ln_f,
                model.score
            )

    def forward(self, x):
        return self.layers(x)

def run(
    rank,
    size
):
    print("i'm rank", rank)
    # model = GPTJPart(full_model, rank)
    model = t.load(f"/home/ubuntu/dm_and_nina/days/w3d1/part_{rank}.pt")
    model.train()
    model.to(DEVICES[rank])
    
    # optimizer = t.optim.SGD(model.parameters(), lr=1e-4)

    # If rank 0, loads data, splits things, keeps a minibatch
    # else, listen for a minibatch from rank 1
    # dataloader = DistributedDataLoader(rank=rank, size=size)
    # batch_size = dataloader.batch_size

    num_batches = t.tensor(0).to(DEVICES[rank])

    if rank == 0:
        tokens, labels = load_data(batch_size=BATCH_SIZE)
        # num_batches = t.tensor(tokens.shape[0]).to(DEVICES[rank])
        # num_batches = t.tensor(5).to(DEVICES[rank])
        num_batches = t.tensor(tokens.size(dim=0)).to(DEVICES[rank])

    group_all = dist.new_group([0, 1, 2, 3])
    group_0_3 = dist.new_group([0, 3])
    group_0_1 = dist.new_group([0, 1])
    group_1_2 = dist.new_group([1, 2])
    group_2_3 = dist.new_group([2, 3])

    dist.broadcast(num_batches, src=0, group=group_all)

    dist.barrier()

    print('Got to for loop')

    for i in range(num_batches.item()):

        print('Got to batch')

        dist.barrier()

        print('After barrier')

        label = t.zeros(BATCH_SIZE).to(DEVICES[rank])
        if rank == 0:
            label = labels[i].to(DEVICES[rank])
        dist.broadcast(label, src = 0, group=group_0_3)

        if rank == 0:
            print('Started rank 0')
            x = model(tokens[i].to(DEVICES[rank])).logits
            dist.broadcast(x, src=0, group=group_0_1)
            
            print('Finished rank 0')
        if rank == 1:
            x = t.zeros(BATCH_SIZE, 128, 4096).to(DEVICES[rank])
            dist.broadcast(x, src=0, group=group_0_1)
            x = model(x)
            dist.broadcast(x, src=1, group=group_1_2)
        if rank == 2:
            x = t.zeros(BATCH_SIZE, 128, 4096).to(DEVICES[rank])
            dist.broadcast(x, src=1, group=group_1_2)
            x = model(x)
            dist.broadcast(x, src=2, group=group_2_3)
        if rank == 3:
            x = t.zeros(BATCH_SIZE, 128, 4096).to(DEVICES[rank])
            dist.broadcast(x, src=2, group=group_2_3)
            x = model(x).logits
            loss = t.nn.functional.cross_entropy(x, labels[i].to(DEVICES[rank]))
            print(loss.item())
            # loss.backward()
            # optimizer.step()
        dist.barrier()

    dist.barrier()
    if rank == 0:
        killgroup()

def init_process(
    rank, size, run, backend="nccl"
):  # gloo is algo for sharing gradients. nccl better?
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29504"  # make the master available for mutual contact
    dist.init_process_group(backend, rank=rank, world_size=size)
    print("united process group", rank)

    run(rank, size)


def create_processes(
    local_parallelism=4,
):
    processes = []
    mp.set_start_method("spawn")
    for rank in range(local_parallelism):  # process index = rank
        p = mp.Process(target=init_process, args=(rank, local_parallelism, run))
        p.start()
        processes.append(p)
    # pytorch join requires you to join in order of completion!???


def split_model():
    if all([os.path.exists(f"/home/ubuntu/dm_and_nina/days/w3d1/part_{i}.pt") for i in range(len(DEVICES))]):
        print("Parts already saved")
        return
    model = transformers.AutoModelForSequenceClassification.from_pretrained("EleutherAI/gpt-j-6B")
    parts = [GPTJPart(model, i) for i in range(len(DEVICES))]
    for i, part in enumerate(parts):
        print(f"Saving part {i}")
        t.save(part, f"/home/ubuntu/dm_and_nina/days/w3d1/part_{i}.pt")


if __name__ == "__main__":
    split_model()
    create_processes(local_parallelism=len(DEVICES))
