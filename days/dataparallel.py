import os
import torch
from torch import random
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import gin
import sys
from utils import import_object_from_qualified_name
import torch as t


@gin.configurable
class DistributedDataLoader:
    def __init__(self, data_fn, mini_batch_size, rank, size, random_seed=0):
        self.rank = rank
        self.size = size
        if rank == 0:
            self.data_list = list(import_object_from_qualified_name(data_fn)())
            random.seed(random_seed)
            random.shuffle(self.data_list)
            self.batches = [[self.data_list[x : x + mini_batch_size] for x in range(rank)] for y in range(1)]  # wrong
            self.len = len(self.batches)
        else:
            self.len = -1
        blst = [self.len]
        dist.broadcast_object_list(blst, src=0)
        self.len = blst[0]

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        if self.batches:
            for mini_batches in self.batches:
                dist.broadcast_object_list(mini_batches, src=0)
                my_batch = mini_batches[self.rank]
                yield my_batch
        else:
            for _ in range(self.len):
                mini_batches = [None for _ in range(self.size)]
                dist.broadcast_object_list(mini_batches, src=0, tag=7474)
                my_batch = mini_batches[self.rank]
                yield my_batch


def alladd_grad(model):
    reduce_ops = [dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True) for param in model.parameters()]
    for op in reduce_ops:
        op.wait()


@gin.configurable(denyList=["rank", "size"])
def run(
    rank,
    size,
    model_init_fn_name=gin.REQUIRED,
):
    device = "cuda:" + str(rank)
    model = import_object_from_qualified_name(model_init_fn_name)
    model.train()
    model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-4)
    dataloader = DistributedDataLoader()
    for batch in dataloader:
        out = model(batch[0])
        loss = out - batch[1]
        loss.backward()
        alladd_grad(model)
        optimizer.step()


@gin.configurable
def init_process(rank, size, backend, tensors_per_batch):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    run(rank, size)


@gin.configurable
def create_processes(
    local_parallelism=gin.REQUIRED,
    local_batch_size=gin.REQUIRED,
    backend="mpi",
    dataset_fn_name=gin.REQUIRED,
    model_fn_name=gin.REQUIRED,
    lr=1e-4,
):

    processes = []
    mp.set_start_method("spawn")
    for rank in range(1, local_parallelism + 1):
        p = mp.Process(target=init_process, args=(rank, local_parallelism, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    if sys.argv[1] == "master":
        gin.parse_config_file(sys.argv[1])
        local_parallelism = sys.argv[2]
        create_processes()
    else:
        tmpfilename = ".ginny_weasly"
        with open(tmpfilename, "w") as f:
            f.write(sys.argv[1])
        gin.parse_config_file(tmpfilename)
        init_process()
