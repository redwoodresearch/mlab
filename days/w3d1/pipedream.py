from comet_ml import Experiment
import os
import random
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchtyping import tensor_type
from typing import Tuple, List
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
import json
import torchtext
from tqdm import tqdm
import numpy as np

DEVICE = "cpu"
DEVICES = [2, 3]
MAX_LEN = 1024
ZERO = 0


class SaneBlock(t.nn.Module):
    def __init__(self, h_block: t.nn.Module):
        super().__init__()
        self.h_block = h_block

    def forward(self, x):
        return self.h_block(x)[0]

def start_comet(hyperparams):
    experiment = Experiment(
        api_key="oGcm04SiEeJM89dRyU0vcOFzd",
        project_name="mlab_pipedream",
        workspace="luuf",
        auto_output_logging=False,
    )
    experiment.log_parameters(hyperparams)

def killgroup():
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


def get_imdb_collate_fn(
    max_seq_length: int,
    tokenizer: transformers.AutoTokenizer,
    device: str,
):

    myrandom = np.random.default_rng(182)
    padding = list(myrandom.integers(0, tokenizer.vocab_size, max_seq_length))

    def fn(raw_xs: List[Tuple[str, str]]) -> Tuple[t.Tensor, t.Tensor]:
        labels: Tuple[str, ...]
        texts: Tuple[str, ...]
        labels, texts = zip(*raw_xs)

        xs = tokenizer(
                list(texts),
                # padding="longest",
                max_length=max_seq_length,
                truncation=True,
            )['input_ids']
        
        xs = [padding[len(x):] + x for x in xs]
        xs = t.tensor(xs, dtype=t.long, device=device)

        ys = t.tensor([int(l == "pos") for l in labels], dtype=t.long, device=device)

        return xs, ys

    return fn



@gin.configurable()
def run(
    rank,
    size,
    model_prefix = 'mlp',
    seed = 0,
    batch_size = 1,
    num_microbatches=1,
    epochs=1,
    lr=1e-4
):
    random.seed(seed)
    t.random.manual_seed(seed)
    print("i'm rank", rank)
    # device = "cuda:" + str(rank)
    model = t.load(model_prefix + str(rank) + '.pt')
    model.train()
    model.to(DEVICE)
    optimizer = t.optim.SGD(model.parameters(), lr=lr)

    if rank != ZERO:
        start_comet({'batch_size': batch_size, 'num_microbatches': num_microbatches, "seed": seed, "lr": lr})

    ub_size = batch_size // num_microbatches
    microbatch_shape = (ub_size, 128, 4096)
    labels_shape = (batch_size,)
    # Agree on num batches
    if rank == ZERO:
        tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

        data_train, data_test = torchtext.datasets.IMDB(root='.data', split=('train', 'test'))
        data_train = list(data_train)
        data_test = list(data_test)
        
        collate_fn = get_imdb_collate_fn(128, tokenizer, DEVICE)
        dataloader = DataLoader(
            data_train,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            # num_workers=2,
            drop_last = True,
            # pin_memory=True,
        )
        
        # dl_test = DataLoader(
        #     data_test,
        #     batch_size=batch_size*2,
        #     collate_fn=collate_fn,
        #     shuffle=True,
        #     num_workers=2,
        #     pin_memory=True,
        # )

        num_batches = len(dataloader)
    else:
        num_batches = 0
    num_batches_tensor = t.tensor([num_batches], device=DEVICE);
    dist.broadcast(num_batches_tensor, src=0)
    num_batches = num_batches_tensor.item()

    loss_fn = t.nn.CrossEntropyLoss()

    for e in range(epochs):
        if rank == 0:
            data_iterator = iter(dataloader)
        for i in range(num_batches):
            optimizer.zero_grad()

            # Let last layer know labels so it can compute loss
            if rank == ZERO:
                batch, labels = next(data_iterator)
            else:
                labels = t.zeros(labels_shape, dtype=t.long, device=DEVICE)
            dist.broadcast(labels, src=0)

            intermediates = []

            # Forward
            for j in range(num_microbatches+1):
                # Run batch through first stage and broadcast result to second stage
                if j < num_microbatches:
                    if rank == ZERO:
                        intermediate = model(batch[j*ub_size:(j+1)*ub_size])
                    else:
                        intermediate = t.zeros(microbatch_shape, device=DEVICE, requires_grad=True)
                        
                    # Don't broadcast yet! Let second stage do work before we synchronize

                # Run through second stage and compute loss
                if j != 0:
                    if rank != ZERO:
                        out = model(intermediates[j-1])

                        # loss = t.sqrt(t.sum((out - labels)**2))
                        loss = loss_fn(input=out[:,-1], target=labels[(j-1)*ub_size:j*ub_size])
                        del out
                        if i % 4 == 0:
                            print(f"batch {i} loss {loss}")
                        loss.backward()

                        intermediate_grad = intermediates[j-1].grad
                        intermediates[j-1] = None
                    else:
                        intermediate_grad = t.zeros_like(intermediates[j-1])

                    # Broadcast grads to first stage
                    dist.broadcast(intermediate_grad, src=1)

                # Ok, now we're synchronizing, send the next intermediate forward
                if j < num_microbatches:
                    dist.broadcast(intermediate, src=0)
                    intermediates.append(intermediate)

                # Finish backward pass
                if j != 0:
                    if rank == ZERO:
                        intermediates[j-1].backward(gradient = intermediate_grad)
                        intermediates[j-1] = None
                        #print(f"zero received grad updates {intermediate_grad}")

                    del intermediate_grad

            # Free stored stuff
            del intermediates

            optimizer.step()

            if i % 300 == 299:
                t.save(model, f"pipedream_model_r{rank}_e{e}_b{i}.pt")

    print(rank, "done training")
    dist.barrier()

    if rank == 0:
        killgroup()


@gin.configurable
def init_process(
    rank, size, run, device, backend="nccl"
):  # gloo is algo for sharing gradients. nccl better?
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29502"  # make the master available for mutual contact
    if device == "cuda":
        global DEVICE
        DEVICE = "cuda:" + str(DEVICES[rank])
    dist.init_process_group(backend, rank=rank, world_size=size)
    print("inited process group", rank)

    run(rank, size, model_prefix='stage_', batch_size=16, num_microbatches=4, lr=3e-6)


@gin.configurable
def create_processes(
    local_parallelism=2,
    device="cuda",
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
    # gin.parse_config_file(sys.argv[1])
    create_processes()
