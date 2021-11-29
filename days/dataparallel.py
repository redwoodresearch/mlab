import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import gin
import sys


@gin.configurable(denyList=["rank", "size"])
def run(
    rank,
    size,
    dataloader_fn,
    model_init_fn,
):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print("Rank 0 started sending")
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print("Rank 1 started receiving")
    req.wait()
    print("Rank ", rank, " has data ", tensor[0])


@gin.configurable
def init_process(rank, size, backend="mpi"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    run(rank, size)


@gin.configurable
def create_processes(local_parallelism=gin.REQUIRED):
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
        gin.parse_config_file(sys.argv[1])
        local_parallelism = sys.argv[2]
    else:
        tmpfilename = ".ginny_weasly"
        with open(tmpfilename, "w") as f:
            f.write(sys.argv[1])
        gin.parse_config_file(tmpfilename)
