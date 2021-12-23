import test_all
from test_all import allclose
from typing import List

from torch import nn
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import gin
import sys
from utils import import_object_from_qualified_name
import torch as t
from utils import *
import torchvision
import transformers
import torchtext


def make_and_save_resnet_pieces():
    t.manual_seed(0)
    resnet = torchvision.models.resnet50()
    print(resnet)
    models = [
        nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        ),
        nn.Sequential(
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
            resnet.fc,
        ),
    ]
    t.save(models[0], ".resnet50_part0")
    randimg = t.randn(1, 3, 64, 64)
    print("intermediate shape", models[0](randimg).shape)
    t.save(models[1], ".resnet50_part1")
    seqran = models[1](models[0](randimg))
    allclose(
        seqran,
        resnet(randimg),
        "splitresnet",
    )
    return models


def make_t5_and_save_pieces():
    t5 = transformers.T5ForSequenceClassification.from_pretrained("t5-11b")
    print(t5)
    models = []


def make_dataset():
    t.manual_seed(0)
    pairs = torchvision.datasets.CIFAR10(root=".data", download=True)
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((64, 64))]
    )
    return t.stack([transforms(p[0]) for p in pairs]), t.Tensor([p[1] for p in pairs])


def make_dataset_imdb():
    train_data = list(torchtext.datasets.IMDB(split="train"))
    import random

    sent_to_num = {"neg": 0, "pos": 1}
    random.shuffle(train_data)
    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-11b")
    print(train_data[0])
    data = [
        t.stack(
            [
                tokenizer(x, return_tensors="pt", padding=512)["input_ids"]
                for x, _ in train_data
            ]
        ),
        t.tensor([sent_to_num[x] for _, x in train_data]),
    ]
    return data


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
    x_size,
    dataset_fn_name: str = "days.pipelineparallel.make_dataset",
    checkpoint_every=10,
    use_cpu=True,
):

    device = "cpu" if use_cpu else "cuda:" + str(rank)
    model: nn.Module = t.load(model_file_name)
    model.train()
    model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-4)
    if rank == 0:
        dataset = import_object_from_qualified_name(dataset_fn_name)()
        print(dataset[1].shape)
        batches = (
            to_batches(batch, minibatch_size, trim=True)
            for batch in to_batches(dataset, minibatch_size * size, trim=True)
        )
    for batch_num in range(num_batches):
        if rank == 0:
            minibatches = next(batches)
            forward_sends = []
            out_tensors = []
            for send in [dist.isend(batch[1], size - 1) for batch in minibatches]:
                send.wait()

            for batch in minibatches:
                out = model(batch[0].to(device))
                out_tensors.append(out)
                forward_sends.append(dist.isend(out, rank + 1))
            for forward_pass in forward_sends:
                forward_pass.wait()
            grad_buffer = t.zeros_like(out)
            for _, out_tensor in enumerate(out_tensors):
                dist.recv(grad_buffer, rank + 1)
                out_tensor.backward(grad_buffer)

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
                out_tensor.backward(grad_buffer)
                xgrad = x.grad
                backward_sends.append(dist.send(xgrad, rank - 1))
            for backward_send in backward_sends:
                backward_send.wait()

        elif rank == size - 1:
            ys = [t.zeros(minibatch_size, *y_size).to(device) for _ in range(size)]
            for recv in [dist.irecv(y, 0) for y in ys]:
                recv.wait()
            xs = []
            losses = []
            for minibatch_num in range(size):
                x_buffer = t.zeros(minibatch_size, *model_in_shape).to(device)
                dist.recv(x_buffer, rank - 1)
                x_buffer.requires_grad = True
                out = model(x_buffer)
                print("out shape", out.shape)
                print("ys shape", ys[minibatch_num].shape)
                cur_loss = nn.CrossEntropyLoss()(out, ys[minibatch_num].long())
                # print(cur_loss.cpu().item())
                losses.append(cur_loss)
                xs.append(x_buffer)
            backward_sends = []
            for loss, x in zip(losses, xs):
                loss.backward()
                xgrad = x.grad
                backward_sends.append(dist.isend(xgrad, rank - 1))
            for backward_send in backward_sends:
                backward_send.wait()
        optimizer.step()
        optimizer.zero_grad()
        if batch_num % checkpoint_every == checkpoint_every - 1:
            t.save(model, f"checkpoint_rank{rank}")
    if rank == 0:
        tpeek("pipe", next(model.parameters()))


def reference_training(use_cpu=True):
    num_batches = 100
    device = "cpu" if use_cpu else "cuda"
    model: nn.Module = t.nn.Sequential(
        t.load(".pipe_par_test_model_0"), t.load(".pipe_par_test_model_1")
    )
    model.train()
    model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-4)
    dataset = import_object_from_qualified_name("days.pipelineparallel.make_dataset")()
    batches = to_batches(dataset, 12 * 2, trim=True)
    for batch in batches[:num_batches]:
        optimizer.zero_grad()
        out = model(batch[0].to(device))
        cur_loss = t.binary_cross_entropy_with_logits(out, batch[1]).mean()
        cur_loss.backward()
        optimizer.step()
    tpeek("reference", next(model.parameters()))
    return list(model.parameters())


@gin.configurable
def init_process(rank, size, run, *args, **kwargs):
    gin.parse_config_file(sys.argv[1])
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    print("will init process group", rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=size)
    print("inited process group", rank)

    run(rank, size, *args, **kwargs)


@gin.configurable
def start_pipeline_cluster(model_paths: List[str], model_in_shapes: List[tuple]):
    # raise AssertionError(":)")
    processes = []
    mp.set_start_method("spawn")
    pipe_stages = len(model_paths)
    size = pipe_stages
    for rank, model_part_str in enumerate(model_paths):
        print("spawning", rank)
        p = mp.Process(
            target=init_process,
            args=(rank, size, pprun, model_part_str, model_in_shapes[rank]),
        )
        p.start()
        processes.append(p)


if __name__ == "__main__":
    # make_and_save_resnet_pieces()
    gin.parse_config_file(sys.argv[1])
    start_pipeline_cluster()
    # reference_training()
