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
from torch import optim
import transformers
from tqdm import tqdm

from days.w3d1.gptj_parallel import GPTJComponent


DEVICES = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
API_KEY = "UALDNRUcAdPMtRTrKFbliwZ9y"
PROJECT_NAME = "beth-tony-gpt2-lesswrong"


def run():
    rank = dist.get_rank()
    device = DEVICES[rank]
    print(f"{rank} Running on {device=}")
    assert rank in (0, 1, 2, 3)

    if rank == 0:
        # experiment = Experiment(
        #     api_key=API_KEY,
        #     project_name=PROJECT_NAME,
        #     auto_metric_logging=False,
        #     auto_output_logging=False,
        # )
        # experiment.log_parameter("rank", rank)
        pass

    original_gptj = transformers.AutoModelForSequenceClassification.from_pretrained(
        "EleutherAI/gpt-j-6B"
    )
    model_component = GPTJComponent(idx=rank, gptj=original_gptj)


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
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
