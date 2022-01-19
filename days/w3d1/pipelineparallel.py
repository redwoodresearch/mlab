from test_all import allclose
from typing import *

from torch import nn
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import gin
import sys
from days.utils import import_object_from_qualified_name
import torch as t
from days.utils import *
import transformers
import torchtext
from time import time

TAGS = {
    "Y": 1000,
    "X": 2000,
    "ACTIVATION": 3000,
    "GRAD": 4000,
    "SYNC": 5000,
}  # why thousands/not single digits?

# HuggingFace models return tuples in the middle (things like activation patterns), thus the [0]
class HFBlockSequence(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)  # ModuleList

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x


# call once
def make_gptj_and_save_pieces():
    model_lm = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    model = model_lm.transformer
    with open("gptj_arch.txt", "w") as f:
        f.write(str(model))
    print(model_lm)

    num_layers = 28
    chunks = [6, 8, 8, 6]  # less at ends due to embeddings/unembed
    assert sum(chunks) == num_layers
    chunk_cumsum = t.cumsum(t.tensor(chunks), dim=0).tolist()
    print("cumsum", chunk_cumsum)
    models = [
        HFBlockSequence(*model.h[start - size : start])
        for start, size in zip(chunk_cumsum, chunks)
    ]
    models[0] = nn.Sequential(model.wte, model.drop, models[0])
    models[-1] = nn.Sequential(models[-1], model.ln_f, model_lm.lm_head)
    for i, model_part in enumerate(models):
        t.save(model_part, "gpt-j-6b_part" + str(i))
    return models


def make_dataset_imdb():
    if os.path.exists("imdb_data.pt"):
        print("reading cached data tensors")
        return t.load("imdb_data.pt")
    train_data = list(torchtext.datasets.IMDB(split="train"))
    import random

    sent_to_num = {"neg": 0, "pos": 1}
    random.shuffle(train_data)
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    print("starting tokenization")
    # save as float32 so you don't have to pass dtype around, convert to long right before use
    data = [
        t.Tensor(
            [
                x[-128:]
                for x in tokenizer(
                    [
                        # gptj wasn't trained with pad token, so doing this
                        """However, when I quit and come back and try the top command again, it is refreshing the processes information again for every 3 seconds and not with the interval that I've set earlier.

I was looking for a way to configure this interval permanently. I've looked at some articles where in they mentioned to use the toprc file in /etc directory for this configuration.

But it doesn't seem like I have any such file in /etc or my home directory.

How do I set the refresh interval for top command? When generating samples interactively sometimes an "end of text" marker is included on the text. Is this related to the sanitization of the training data?However, when I quit and come back and try the top command again, it is refreshing the processes information again for every 3 seconds and not with the interval that I've set earlier.

I was looking for a way to configure this interval permanently. I've looked at some articles where in they mentioned to use the toprc file in /etc directory for this configuration.

But it doesn't seem like I have any such file in /etc or my home directory.

How do I set the refresh interval for top command? When generating samples interactively sometimes an "end of text" marker is included on the text. Is this related to the sanitization of the training data?<|endoftext|>"""
                        + x
                        for _, x in train_data
                    ],
                    padding=False,
                    truncation=False,
                )["input_ids"]
            ]
        ),
        t.Tensor([sent_to_num[x] for x, _ in train_data]),
    ]
    t.save(data, "imdb_data.pt")
    print("finished tokenization")
    return data


# 'bad address' error means you tried to use operations that weren't supported on cuda
# I think the gradient accumulation will just work out?
@gin.configurable()
def pprun(
    rank,
    size,
    model_file_name,
    model_in_shape,
    minibatch_size: int,
    num_batches,
    y_size,
    pipe_width,
    x_size=None,
    dataset_fn_name: str = "days.pipelineparallel.make_dataset",
    checkpoint_every=100,
    use_cpu=True,
    use_autocast=False,
):
    autocast_type = t.bfloat16 if use_cpu else t.float16
    device = "cpu" if use_cpu else "cuda:" + str(rank)
    model: nn.Module = t.load(model_file_name)
    model.train()
    model.to(device)
    optimizer = t.optim.SGD(model.parameters(), lr=1e-4)
    print("model loaded", rank)
    if rank == 0:
        dataset = import_object_from_qualified_name(
            dataset_fn_name
        )()  # tokenizer. embedding in the model (don't worry)
        batches = (
            to_batches(batch, minibatch_size, trim=True)
            for batch in to_batches(dataset, minibatch_size * pipe_width, trim=True)
        )
    total_examples = num_batches * minibatch_size * pipe_width
    start = time()
    batch_start = time()
    for batch_num in range(num_batches):
        dist.all_reduce(t.zeros(1))
        if rank == 0:
            minibatches = next(batches)
            forward_sends = []
            out_tensors = []
            ysends = [  # send the ys all the way to the end. This is not being used for some reason?
                dist.isend(
                    batch[1], size - 1, tag=TAGS["Y"] + i
                )  # tag on the send forces it to only be received by same tag
                for i, batch in enumerate(minibatches)
            ]

            for i, batch in enumerate(minibatches):
                with t.autocast(
                    dtype=autocast_type, device_type=device[:4], enabled=use_autocast
                ):
                    out = model(batch[0].long().to(device)).cpu()  # all the gpu action
                out_tensors.append(out)
                forward_sends.append(
                    dist.isend(out, rank + 1, tag=TAGS["ACTIVATION"] + i)
                )
            grad_buffers = [t.zeros_like(out) for _ in range(pipe_width)]
            grad_recvs = [
                dist.irecv(x, rank + 1, tag=TAGS["GRAD"] + i)
                for i, x in enumerate(grad_buffers)
            ]
            grad_buffer = t.zeros_like(out)
            for i, out_tensor in enumerate(out_tensors):
                grad_recvs[i].wait()
                grad_buffer = grad_buffers[i]
                out_tensor.backward(grad_buffer)
                out_tensors[i] = None  # why is this necessary? saves memory?
        elif rank != size - 1:
            forward_sends = []
            out_tensors = []
            xs = []
            backward_sends = []
            xs = [t.zeros(minibatch_size, *model_in_shape) for _ in range(pipe_width)]
            xjobs = [
                dist.irecv(x, rank - 1, tag=TAGS["ACTIVATION"] + i)
                for i, x in enumerate(xs)
            ]

            for minibatch_num in range(pipe_width):
                xjobs[minibatch_num].wait()
                x_buffer = xs[minibatch_num]
                x_buffer.requires_grad = True
                with t.autocast(
                    dtype=autocast_type, device_type=device[:4], enabled=use_autocast
                ):
                    out = model(x_buffer.to(device)).cpu()
                xs.append(x_buffer)
                out_tensors.append(out)
                forward_sends.append(
                    dist.isend(out, rank + 1, tag=TAGS["ACTIVATION"] + minibatch_num)
                )
            grad_buffers = [t.zeros_like(out) for _ in range(pipe_width)]
            grad_recvs = [
                dist.irecv(x, rank + 1, tag=TAGS["GRAD"] + i)
                for i, x in enumerate(grad_buffers)
            ]
            for i, (out_tensor, x) in enumerate(zip(out_tensors, xs)):
                grad_recvs[i].wait()
                grad_buffer = grad_buffers[i]
                out_tensor.backward(grad_buffer)
                xgrad = x.grad
                backward_sends.append(dist.isend(xgrad, rank - 1, tag=TAGS["GRAD"] + i))
                xs[i] = None
                out_tensors[i] = None

        elif rank == size - 1:
            xs = []
            losses = []
            backward_sends = []
            ys = [t.zeros(minibatch_size, *y_size) for _ in range(pipe_width)]
            yjobs = [dist.irecv(y, 0, tag=TAGS["Y"] + i) for i, y in enumerate(ys)]
            xs = [t.zeros(minibatch_size, *model_in_shape) for _ in range(pipe_width)]
            xjobs = [
                dist.irecv(x, rank - 1, tag=TAGS["ACTIVATION"] + i)
                for i, x in enumerate(xs)
            ]
            for minibatch_num in range(pipe_width):
                xjobs[minibatch_num].wait()
                x_buffer = xs[minibatch_num]
                x_buffer.requires_grad = True
                with t.autocast(
                    dtype=autocast_type, device_type=device[:4], enabled=use_autocast
                ):  # save memory by computing with less precision
                    out = model(x_buffer.to(device)).cpu()
                out = out[
                    :, -1, -2:
                ]  # use the last 2 tokens of LM head as classification head
                yjobs[minibatch_num].wait()
                cur_loss = nn.CrossEntropyLoss()(out.float(), ys[minibatch_num].long())
                # print(cur_loss.cpu().item())
                losses.append(cur_loss)
                xs.append(x_buffer)
            print(
                "whole batch loss",
                batch_num,
                sum([x.cpu().item() for x in losses]) / len(losses),
                "took",
                time() - batch_start,
            )
            batch_start = time()
            for i, (loss, x) in enumerate(zip(losses, xs)):
                loss.backward()
                xgrad = x.grad
                backward_sends.append(dist.isend(xgrad, rank - 1, tag=TAGS["GRAD"] + i))
                losses[i] = None
                xs[i] = None
        optimizer.step()
        optimizer.zero_grad()
        if batch_num % checkpoint_every == checkpoint_every - 1:
            if rank == 0:
                print("saving")
            filename = f"./.checkpoints/gptj_imdb_{batch_num}_rank{rank}"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:  # added this
                t.save(model, f)
    end = time()
    print(
        f"Total time: {start - end}, per batch: {(start - end)/num_batches}, per example {(start - end)/total_examples}, rank {rank}"
    )
    if rank == 0:
        tpeek("pipe", next(model.parameters()))


@gin.configurable
def init_process(rank, size, run, *args, **kwargs):
    gin.parse_config_file(sys.argv[1])
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    print("will init process group", rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=size)
    print("inited process group", rank)
    run(rank, size, *args, **kwargs)


@gin.configurable
def start_pipeline_cluster(
    model_paths: List[str], model_in_shapes: List[tuple]
):  
    processes = []
    mp.set_start_method("spawn")
    pipe_stages = len(model_paths)
    size = pipe_stages
    for rank, model_part_str in enumerate(model_paths):
        p = mp.Process(
            target=init_process,
            args=(rank, size, pprun, model_part_str, model_in_shapes[rank]),
        )
        p.start()
        processes.append(p)  # why are we doing this? and why aren't we joining?


# 2,8 produces 0.18 time units and final loss of 0.10 (noisy)
# 2,10 breaks (too much mem)
# 4,4 produced something like 0.17 time-units/example, final loss of 0.11, half-way 0.38
# 8,2 produced something like 0.10 time-units/example, final loss of 0.37, half-way 0.53

# with 4 minibatches of 12, took 3.1s
# with 1 minibatch of 48, took 4.6
#

if __name__ == "__main__":
    # make_gptj_and_save_pieces()
    gin.parse_config_file(sys.argv[1])

    start_pipeline_cluster()
