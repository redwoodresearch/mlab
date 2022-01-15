from comet_ml import Experiment

# So that organize imports does not move comet_ml below torch
if True:
    pass

import os
import sys
from typing import Callable

import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import transformers
from days.w2d5.lesswrong_data import LesswrongDistributedDataLoader
from torch import optim
from tqdm import tqdm

GPU_DEVICES = ["cuda:0", "cuda:1"]
API_KEY = "UALDNRUcAdPMtRTrKFbliwZ9y"
PROJECT_NAME = "beth-tony-gpt2-lesswrong"


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def average_gradients_sharded(model, partition_size):
    size = float(dist.get_world_size())
    for pos, param in enumerate(model.parameters()):
        owner = get_owner(pos, partition_size)
        dist.reduce(param.grad.data, owner, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def get_owner(param_position, partition_size):
    owner_rank = param_position // partition_size
    return owner_rank


def broadcast_parameters(model, partition_size):
    for pos, param in enumerate(model.parameters()):
        owner = get_owner(pos, partition_size)
        dist.broadcast(param.data, owner)


def run():
    t.manual_seed(1234)

    rank = dist.get_rank()
    print(f"{rank} Running...")
    if rank == 0:
        experiment = Experiment(
            api_key=API_KEY,
            project_name=PROJECT_NAME,
            auto_metric_logging=False,
            auto_output_logging=False,
        )
        experiment.log_parameter("rank", rank)

    device = GPU_DEVICES[rank]
    t.cuda.set_device(device)

    print(f"{rank} Loading data...")
    dl = LesswrongDistributedDataLoader(
        rank=rank,
        world_size=dist.get_world_size(),
        device=device,
        mini_batch_size=16,
    )
    print(f"{rank} Loaded data, now loading model...")
    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    flattened_params = list(model.parameters())
    partition_size = len(flattened_params) // dist.get_world_size() + 1
    # each process has the rankth half of the parameters
    params = flattened_params[rank * partition_size : (rank + 1) * partition_size]

    optimizer = optim.SGD(params, lr=0.01, momentum=0.5)

    num_steps = 0
    print(f"{rank} Starting training loop...")
    for epoch in range(10):
        epoch_loss = 0
        batches = 0

        it = tqdm(dl, total=dl.n_batches) if rank == 0 else dl
        for data in it:
            optimizer.zero_grad()

            # Shape: (mini_batch_size, seq_len, vocab_size)
            output = model(data).logits

            vocab_size = output.shape[-1]
            loss = F.cross_entropy(output.reshape(-1, vocab_size), data.flatten())
            loss.backward()
            average_gradients_sharded(model, partition_size)
            optimizer.step()
            broadcast_parameters(model, partition_size)
            loss = loss.detach()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= dist.get_world_size()
            if rank == 0:
                # experiment.log_metric("loss", loss.item(), step=num_steps, epoch=epoch)
                # print(f"Epoch {epoch} Step {num_steps} Loss {loss.item()}")
                it.set_description(f"Loss {loss.item():.3f}")
                # it.refresh()
                # sys.stdout.flush()

            batches += 1
            epoch_loss += loss.item()
            num_steps += 1


def init_process(
    rank: int,
    size: int,
    fn: Callable[[], None],
    backend: str = "nccl",
):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29499"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn()


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
