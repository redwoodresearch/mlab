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

DEVICE = "cpu"
MAX_LEN = 512


def load_data():
    tensor_path = "/home/ubuntu/lw.pt"
    if os.path.exists(tensor_path):
        return t.load(tensor_path)
    texts = open("/home/ubuntu/lw.txt").readlines()
    largetext = "<|endoftext|>".join(texts)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer(largetext, return_tensors="pt")["input_ids"]
    t.save(tokens, tensor_path)
    return rearrange(tokens[: -tokens.shape[0] % MAX_LEN], "(b s) -> b s", s=512)


def init_model():
    t.random.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    return model


@gin.configurable
class DistributedDataLoader:
    def __init__(
        self,
        rank,
        size,
        data_fn="days.dataparallel.load_data",
        mini_batch_size=4,
        random_seed=0,
    ):
        self.rank = rank
        self.size = size
        if rank == 0:
            self.data_tensors = import_object_from_qualified_name(data_fn)()
            t.manual_seed(random_seed)
            perm = t.randperm(self.data_tensors[0].shape[0])
            self.data_tensors = self.data_tensors[perm]

            self.batches = [
                to_batches(batch, mini_batch_size, trim=True)
                for batch in to_batches((self.data_tensors,), mini_batch_size * size)
            ]
            self.len = len(self.batches)
        else:
            self.len = -1
            self.batches = None
        blst = [self.len]
        print("broadcast length from", self.rank)
        dist.broadcast_object_list(blst, src=0)
        self.len = blst[0]

    def __len__(self):
        return self.len

    def __iter__(self):
        if self.batches is not None:
            for mini_batches in self.batches:
                dist.broadcast_object_list(
                    mini_batches, src=0
                )  # all processes must do this, else all wait forever
                my_batch = mini_batches[self.rank]
                yield my_batch
        else:
            for _ in range(self.len):
                mini_batches = [None for _ in range(self.size)]
                dist.broadcast_object_list(mini_batches, src=0)
                my_batch = mini_batches[self.rank]
                yield my_batch


def alladd_grad(model):

    reduce_ops = [
        dist.all_reduce(prop.grad, op=dist.ReduceOp.SUM, async_op=True)
        for param in model.parameters()
    ]
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
    model_init_fn_name="days.dataparallel.init_model",
    sharded_optimizer=False,
):
    print("i'm rank", rank)
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
        params = model.parameters()
    optimizer = t.optim.Adam(params, lr=1e-4)

    # If rank 0, loads data, splits things, keeps a minibatch
    # else, listen for a minibatch from rank 1
    dataloader = DistributedDataLoader(rank=rank, size=size)
    for batch_num, batch in enumerate(dataloader):
        # print("batch", batch)
        out = model(batch[0].to(DEVICE))
        loss = t.nn.CrossEntropyLoss()(out[:-1], batch[0][1:])
        loss.backward()
        alladd_grad(model, "grad")
        optimizer.step()
        optimizer.zero_grad()
        if sharded_optimizer:
            broadcast_updated_params(param_buckets, rank)
        # print(rank, "loss", loss.cpu().detach().numpy())
        print(rank, batch_num)
    print(rank, "done training")
    dist.all_reduce(t.zeros(1), op=dist.ReduceOp.SUM)

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
        DEVICE = "cuda:" + str(rank)
    dist.init_process_group(backend, rank=rank, world_size=size)
    print("inited process group", rank)

    run(rank, size)


@gin.configurable
def create_processes(
    local_parallelism=2,
    device="cpu",
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
    local_parallelism = (
        2 if len(sys.argv) < 3 else int(sys.argv[2])
    )  # number of processes in parallel
    device = "cpu" if sys.argv[3] == "cpu" else "cuda"
    if sys.argv[1] == "master":
        # gin.parse_config_file(sys.argv[2])
        create_processes(local_parallelism, device)
    else:
        tmpfilename = ".ginny_weasly"
        with open(tmpfilename, "w") as f:
            f.write(sys.argv[1])
        # gin.parse_config_file(tmpfilename)
        init_process()
