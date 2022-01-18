from importlib.metadata import requires
import os
from typing import Optional

import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
from gptj_parallel import *
import torch.nn.functional as F

DEVICES = [f"cuda:{i}" for i in range(4, 8)]
TRAINING_DATA_X_PATH = "imdb_train_xs_1024.pt"
TRAINING_DATA_Y_PATH = "imdb_train_ys.pt"
TRAINING_DATA_SIZE = 25_000
SEQUENCE_LENGTH = 1024
BATCH_SIZE = 1
COMPONENT_PATHS = [f"component{i}.pt" for i in range(4)]
PROCESS_GROUPS = {}

"""
use t.mp to start up N processes and t.dist to create a process group. You can (and should) copy over your code. In each process, t.load one of your sub-models that you saved earlier onto the appropriate device and initialize a plain SGD optimizer (to save memory) on your model’s params. Also, have your rank-0 process load and batch the IMDB sentiment classification data.

In pipeline parallelism, we often send data between two nodes. Pyt Nccl doesn't support dist.send though (I think raw Nccl does), so we're going to have to create a new process group with dist.new_group for every pair of processes that ever need to communicate, and then broadcast in those groups.
"""


def get_device() -> str:
    return DEVICES[dist.get_rank()]


def get_process_group(*ranks):
    return PROCESS_GROUPS[tuple(sorted(ranks))]


def add_process_group(*ranks: int):
    ranks = tuple(sorted(ranks))
    if ranks not in PROCESS_GROUPS:
        PROCESS_GROUPS[ranks] = dist.new_group(ranks)


def _send_receive_tensor(my_tensor: Optional[t.Tensor], src: int, dst: int) -> t.Tensor:
    """Should only be called by the src and dst processes."""

    rank = dist.get_rank()
    # print(f"{rank}  Entered send_tensor({src} -> {dst})")
    assert rank in (src, dst)

    # Broadcast my_tensor metadata from src to dst
    # print(f"{rank} attempting to broadcast metadata... ({src} -> {dst})")
    metadata = [None, None] if rank == dst else [my_tensor.shape, my_tensor.dtype]
    dist.broadcast_object_list(metadata, src=src, group=get_process_group(src, dst))
    # print(f"{rank} broadcasted metadata {metadata} ({src} -> {dst})")

    shape, dtype = metadata
    if rank == dst:
        my_tensor = t.empty(shape, dtype=dtype, device=get_device())

    dist.broadcast(my_tensor, src, get_process_group(src, dst))
    return my_tensor


def send_tensor(x: t.Tensor, src: int, dst: int):
    _send_receive_tensor(x, src, dst)


def receive_tensor(src: int, dst: int) -> t.Tensor:
    return _send_receive_tensor(None, src, dst)


def run(rank, size):
    device = get_device()
    t.cuda.set_device(device)
    print(f"Beginning loop for rank {rank}")

    if rank == 0:
        """
        Load and microbatch (?) the training data
        """

        training_data_xs = t.load(TRAINING_DATA_X_PATH).to(device)  # TODO shuffle data
        training_data_ys = t.load(TRAINING_DATA_Y_PATH).to(device)

        # training_data_ys = t.load(TRAINING_DATA_Y_PATH).to(device)

        assert training_data_xs.shape == t.Size(
            [TRAINING_DATA_SIZE, SEQUENCE_LENGTH]
        ), f"{training_data_xs.shape}"
        assert training_data_ys.shape == t.Size([TRAINING_DATA_SIZE])

        assert TRAINING_DATA_SIZE % BATCH_SIZE == 0
        num_batches = TRAINING_DATA_SIZE // BATCH_SIZE

        training_data_xs = training_data_xs.reshape(
            num_batches, BATCH_SIZE, SEQUENCE_LENGTH
        )
        training_data_ys = training_data_ys.reshape(
            num_batches, BATCH_SIZE,
        )
        print("Finished loaded training data")

    component: GPTJComponent = t.load(COMPONENT_PATHS[rank]).to(device)
    optimizer = t.optim.SGD(component.parameters(), lr=1e-3)

    # Forward pass
    # print(training_data_xs.shape, "the current shape")
    prv_activations = training_data_xs[0] if rank == 0 else None
    if rank == 0:
        print(training_data_xs.shape, "the current shape")
        print(prv_activations.shape)
        prv_activations = prv_activations[:, :10]
        # t.save(prv_activations, "input_test.pt")
        pass


    else:
        prv_activations = t.autograd.Variable(
            receive_tensor(rank - 1, rank), requires_grad=True
        )

    activations = component(prv_activations)

    if rank < size - 1:
        send_tensor(activations, rank, rank + 1)

    # print(f"Finished forward pass rank {rank}")

    if rank == size - 1:
        activations = component(prv_activations)
        # print(activations.shape)
        loss = t.linalg.norm(activations[1])
        loss.backward()

    if rank < size - 1:
        activations_grad = receive_tensor(src=rank + 1, dst=rank)
        # print(f"Begin backward call at rank {rank}, {activations_grad.shape}")
        # activations = t.autograd.Variable(activations, requires_grad=True)
        # activations.backward(inputs=activations_grad)

        # TODO: Make this not so hacky...
        (activations.flatten() @ activations_grad.flatten()).backward()
        # print(f"Finished backward call at rank {rank}")

    if rank > 0:
        send_tensor(prv_activations.grad, rank, rank - 1)

    # print(f"Start optimizer step at {rank}")
    optimizer.step()
    # print(f"End optimizer at step {rank}")


def init_process(rank, size, fn, backend="gloo"):  # TODO NCCL
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29499"
    dist.init_process_group(backend, rank=rank, world_size=size)
    for i in range(size - 1):
        add_process_group(i, i + 1)

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