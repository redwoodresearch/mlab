from typing import List

from torch import nn
import os
import random
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import gin
import sys
from utils import import_object_from_qualified_name
import torch as t

# I think the gradient accumulation will just work out?
@gin.configurable()
def pprun(
    rank,
    size,
    model_file_name,
    dataloader_fn_name: str,
    minibatch_size: int,
    num_batches,
    y_size,
    x_size,
    checkpoint_every=10,
):
    import test_all

    print("i'm rank", rank)
    # device = "cuda:" + str(rank)
    device = "cpu"
    model: nn.Module = t.load(model_file_name)
    model.train()
    model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-4)
    if rank == 0:
        dataloader = import_object_from_qualified_name(dataloader_fn_name)(
            batch_size=minibatch_size * size
        )
    for batch_num in range(num_batches):
        if rank == 0:
            minibatches = [next(dataloader) for _ in range(size)]
            forward_sends = []
            out_tensors = []
            for batch in minibatches:
                dist.isend(batch["y"], size - 1)
                out = model(batch["x"])
                out_tensors.append(out)
                forward_sends.append(dist.isend(out, rank + 1))
            for forward_pass in forward_sends:
                forward_pass.wait()
            grad_buffer = t.zeros_like(out)
            for out_tensor, _ in enumerate(out_tensors):
                dist.recv(grad_buffer, rank + 1)
                loss = t.sum(grad_buffer * out_tensor)  # whaat is this how this works?
                loss.backward()

        elif rank != size - 1:
            forward_sends = []
            out_tensors = []
            xs = []
            for minibatch_num in range(size):
                x_buffer = t.zeros(minibatch_size, *x_size)
                dist.recv(x_buffer, rank - 1)
                x_buffer.requires_grad = True
                out = model(x_buffer)
                xs.append(x_buffer)
                out_tensors.append(out)
                forward_sends.append(dist.isend(out, rank + 1))
            for forward_pass in forward_sends:
                forward_pass.wait()
            grad_buffer = t.zeros_like(out)
            backward_sends = []
            for out_tensor, x in zip(out_tensors, xs):
                dist.recv(grad_buffer, rank + 1)
                loss = t.sum(grad_buffer * out_tensor)  # whaat is this how this works?
                loss.backward()
                xgrad = x.grad
                backward_sends.append(dist.isend(xgrad, rank - 1))
            for backward_send in backward_sends:
                backward_send.wait()

        elif rank == size - 1:
            ys = t.zeros(minibatch_size, *y_size)
            dist.recv(ys, 0)
            xs = []
            losses = []
            for minibatch_num in range(size):
                x_buffer = t.zeros(minibatch_size, *x_size)
                dist.recv(x_buffer, rank - 1)
                x_buffer.requires_grad = True
                out = model(x_buffer)
                loss = t.binary_cross_entropy_with_logits(out, ys[minibatch_num])
                losses.append(loss)
                xs.append(x_buffer)
            backward_sends = []
            for x in xs:
                loss.backward()
                xgrad = x.grad
                backward_sends.append(dist.isend(xgrad, rank - 1))
            for backward_send in backward_sends:
                backward_send.wait()
        optimizer.step()
        optimizer.zero_grad()
        if batch_num % checkpoint_every == checkpoint_every - 1:
            t.save(model, f"checkpoint_rank{rank}")


@gin.configurable
def init_process(rank, size, run, *args, **kwargs):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend="gloo", rank=rank, world_size=size)
    print("inited process group", rank)

    run(rank, size, *args, **kwargs)


@gin.configurable
def start_pipeline_cluster(
    model_paths: List[str],
):
    # raise AssertionError(":)")
    processes = []
    mp.set_start_method("spawn")
    pipe_stages = len(model_paths)
    size = pipe_stages + 1
    for rank, model_part_str in enumerate(model_paths):
        print("spawning", rank)
        p = mp.Process(
            target=init_process,
            args=(rank, size, pprun, model_part_str),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    gin.parse_config_file(sys.argv[1])
    start_pipeline_cluster()
