import test_all
import os
import random
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import gin
import sys
from utils import import_object_from_qualified_name
import torch as t
import numpy as np


def load_data():
    randns = t.rand(1000, 2, 10)
    return [randns[i] for i in range(1000)]


def init_model():
    t.random.manual_seed(0)
    model = t.nn.Linear(10, 10)
    return model


@gin.configurable
class DistributedDataLoader:
    def __init__(self, rank, size, data_fn="days.dataparallel.load_data", mini_batch_size=8, random_seed=0):
        self.rank = rank
        self.size = size
        if rank == 0:
            self.data_list = list(import_object_from_qualified_name(data_fn)())
            random.seed(random_seed)
            random.shuffle(self.data_list)
            self.data_list = self.data_list[: -(len(self.data_list) % (mini_batch_size * size))]
            print("overflow", len(self.data_list) % (mini_batch_size * size))
            self.batches = [np.array_split(x, size) for x in np.array_split(self.data_list, mini_batch_size * size)]
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
                dist.broadcast_object_list(mini_batches, src=0)
                my_batch = mini_batches[self.rank]
                yield my_batch
        else:
            for _ in range(self.len):
                mini_batches = [None for _ in range(self.size)]
                dist.broadcast_object_list(mini_batches, src=0)
                my_batch = mini_batches[self.rank]
                yield my_batch


def alladd_grad(model):
    reduce_ops = [dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True) for param in model.parameters()]
    for op in reduce_ops:
        op.wait()


@gin.configurable()
def run(
    rank,
    size,
    model_init_fn_name="days.dataparallel.init_model",
):
    print("i'm rank", rank)
    # device = "cuda:" + str(rank)
    device = "cpu"
    model = import_object_from_qualified_name(model_init_fn_name)()
    model.train()
    model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-4)
    dataloader = DistributedDataLoader(rank=rank, size=size)
    for batch in dataloader:
        # print("batch", batch)
        out = model(batch[0])
        loss = t.sum(out - batch[1])
        loss.backward()
        alladd_grad(model)
        optimizer.step()
        optimizer.zero_grad()
        print("loss", loss.cpu().detach().numpy())
        raise AssertionError("I want to error!")


@gin.configurable
def init_process(rank, size, run, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    print("inited process group", rank)
    import test_all

    run(rank, size)


@gin.configurable
def create_processes(
    local_parallelism=2,
):
    # raise AssertionError(":)")
    processes = []
    mp.set_start_method("spawn")
    for rank in range(local_parallelism):
        p = mp.Process(target=init_process, args=(rank, local_parallelism, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    if sys.argv[1] == "master":
        # gin.parse_config_file(sys.argv[2])
        create_processes()
    else:
        tmpfilename = ".ginny_weasly"
        with open(tmpfilename, "w") as f:
            f.write(sys.argv[1])
        # gin.parse_config_file(tmpfilename)
        init_process()
