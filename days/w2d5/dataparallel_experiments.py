import os
import tempfile
import time
from typing import Callable

import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
from days.w2d5.lesswrong_data import LesswrongDistributedDataLoader

GPU_DEVICES = ["cuda:0", "cuda:1"]


def run():
    """Distributed function to be implemented later."""
    rank = dist.get_rank()
    device = GPU_DEVICES[rank]
    t.cuda.set_device(device)

    dl = LesswrongDistributedDataLoader(
        rank=rank,
        world_size=dist.get_world_size(),
        device=device,
        mini_batch_size=4,
        seq_len=10,
    )

    for x in dl:
        print(f"rank {rank} batch: {x}")
        break

    while True:
        time.sleep(1)
    # if rank != 0:
    #     x = t.zeros((2, 2), device=gpu_id)
    # else:
    #     x = t.rand((2, 2), device=gpu_id)
    # print(f"Process {rank} before broadcast: {x=}, {x.device=}")

    # dist.broadcast(x, 0)
    # print(f"Process {rank} after broadcast: {x=}, {x.device=}")

    # r = t.rand(5, device=gpu_id)
    # print(f"Process {rank} before all_reduce: {r=}, {r.device=}")
    # dist.all_reduce(r)
    # print(f"Process {rank} after all_reduce: {r=}, {r.device=}")


def init_process(
    rank: int,
    size: int,
    fn: Callable[[], None],
    shared_tmp_file_name: str,
    backend: str = "nccl",
):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29499"
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=size,
        # init_method=f"file://{shared_tmp_file_name}",
    )
    fn()


if __name__ == "__main__":
    shared_tmp_file_name = tempfile.NamedTemporaryFile().name

    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(
            target=init_process, args=(rank, size, run, shared_tmp_file_name)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
