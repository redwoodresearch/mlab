import json
import os
from re import L
from time import ctime, time
from typing import Optional

import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers
from comet_ml import Experiment
from days.utils import to_batches

DEVICES = [f"cuda:{i}" for i in range(4)]
TRAINING_DATA_X_PATH = "imdb_train_xs_1024.pt"
TRAINING_DATA_Y_PATH = "imdb_train_ys.pt"
TRAINING_DATA_SIZE = 25_000
SEQUENCE_LENGTH = 1024
BATCH_SIZE = 2
COMPONENT_PATHS = [f"component{i}.pt" for i in range(4)]

"""
use t.mp to start up N processes and t.dist to create a process group. You can (and should) copy over your code. In each process, t.load one of your sub-models that you saved earlier onto the appropriate device and initialize a plain SGD optimizer (to save memory) on your model’s params. Also, have your rank-0 process load and batch the IMDB sentiment classification data.

In pipeline parallelism, we often send data between two nodes. Pyt Nccl doesn't support dist.send though (I think raw Nccl does), so we're going to have to create a new process group with dist.new_group for every pair of processes that ever need to communicate, and then broadcast in those groups.
"""


def get_device() -> str:
    return DEVICES[dist.get_rank()]


def send_tensor_to(my_tensor: Optional[t.Tensor], src: int, dst: int):
    rank = dist.get_rank()
    print(f"{rank}  Entered send_tensor_to({src} -> {dst})")

    group = dist.new_group([src, dst])
    if rank not in [src, dst]:
        return None

    # Broadcast my_tensor metadata from src to dst
    print(f"{rank} attempting to broadcast metadata... ({src} -> {dst})")
    metadata = [None, None] if rank == dst else [my_tensor.shape, my_tensor.dtype]
    dist.broadcast_object_list(metadata, src=src, group=group)
    print(f"{rank} broadcasted metadata {metadata} ({src} -> {dst})")

    shape, dtype = metadata
    if rank == dst:
        my_tensor = t.empty(shape, dtype=dtype, device=get_device())

    dist.broadcast(my_tensor, src, group)
    return my_tensor


def run(rank, size):
    device = f"cuda:{rank}"
    print(f"Beginning loop for rank {rank}")

    if rank == 0:
        """
        Load and microbatch (?) the training data
        """

        training_data_xs = t.load(TRAINING_DATA_X_PATH).to(device)  # TODO shuffle data
        # training_data_ys = t.load(TRAINING_DATA_Y_PATH).to(device)
        assert training_data_xs.shape == t.Size(
            [TRAINING_DATA_SIZE, SEQUENCE_LENGTH]
        ), f"{training_data_xs.shape}"

        assert TRAINING_DATA_SIZE % BATCH_SIZE == 0
        num_batches = TRAINING_DATA_SIZE // BATCH_SIZE

        training_data = training_data_xs.reshape(
            num_batches, BATCH_SIZE, SEQUENCE_LENGTH
        )
        print("Finished loaded training data")

    component = t.load(COMPONENT_PATHS[rank]).to(device)
    optimizer = t.optim.SGD(component.parameters(), lr=1e-3)

    # Forward pass
    prv_activations = training_data[0] if rank == 0 else None
    if rank == 0:
        prv_activations = prv_activations[:, :10]
        print(prv_activations.shape)
    for i in range(size - 1):
        activations = component(prv_activations) if rank == i else None
        prv_activations = send_tensor_to(activations, src=i, dst=i + 1)


def init_process(rank, size, fn, backend="gloo"):  # TODO NCCL
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29499"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
