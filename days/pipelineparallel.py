from test_all import allclose
from typing import *

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

TAGS = {"Y": 1000, "X": 2000, "ACTIVATION": 3000, "GRAD": 4000, "SYNC": 5000}


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


class HFBlockSequence(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x


def make_gptj_and_save_pieces():
    model_lm = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    model = model_lm.transformer
    with open("gptj_arch.txt", "w") as f:
        f.write(str(model))
    print(model)
    num_layers = 28
    chunks = [2, 4, 4, 4, 4, 4, 4, 2]
    chunk_cumsum = t.cumsum(t.tensor(chunks), dim=0).tolist()
    print("cumsum", chunk_cumsum)
    models = [
        HFBlockSequence(*model.h[start - size : start])
        for start, size in zip(chunk_cumsum, chunks)
    ]
    models[0] = nn.Sequential(model.wte, model.drop, models[0])
    models[7] = nn.Sequential(models[7], model.ln_f, model_lm.lm_head)
    for i, model_part in enumerate(models):
        t.save(model_part, "gpt-j-6b_part" + str(i))
    return models


def make_dataset():
    t.manual_seed(0)
    pairs = torchvision.datasets.CIFAR10(root=".data", download=True)
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((64, 64))]
    )
    return t.stack([transforms(p[0]) for p in pairs]), t.Tensor([p[1] for p in pairs])


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
            # gptj wasn't trained with pad token, so doing this
            [
                x[:256]
                for x in tokenizer(
                    [
                        x
                        + " _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _-- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _- _-"
                        for _, x in train_data
                    ],
                    padding=False,
                    truncation=True,
                    max_length=256,
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
    x_size=None,
    dataset_fn_name: str = "days.pipelineparallel.make_dataset",
    checkpoint_every=10,
    use_cpu=True,
):

    device = "cpu" if use_cpu else "cuda:" + str(rank)
    model: nn.Module = t.load(model_file_name)
    # model.half()
    model.train()
    model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-4)
    print("model loaded", rank)
    if rank == 0:
        dataset = import_object_from_qualified_name(dataset_fn_name)()
        batches = (
            to_batches(batch, minibatch_size, trim=True)
            for batch in to_batches(dataset, minibatch_size * size, trim=True)
        )
    for batch_num in range(num_batches):
        dist.all_reduce(t.zeros(1))
        if rank == 0:
            print("pre_batch_sync")
        if rank == 0:
            minibatches = next(batches)
            forward_sends = []
            out_tensors = []
            for send in [
                dist.send(batch[1], size - 1, tag=TAGS["Y"] + i)
                for i, batch in enumerate(minibatches)
            ]:
                ...

            for i, batch in enumerate(minibatches):
                with t.autocast(dtype=t.float16, device_type=device[:4]):
                    out = model(batch[0].long().to(device)).cpu()
                out_tensors.append(out)
                forward_sends.append(
                    dist.send(out, rank + 1, tag=TAGS["ACTIVATION"] + i)
                )
            grad_buffer = t.zeros_like(out)
            for i, out_tensor in enumerate(out_tensors):
                dist.recv(grad_buffer, rank + 1, tag=TAGS["GRAD"] + i)
                out_tensor.backward(grad_buffer)

        elif rank != size - 1:
            forward_sends = []
            out_tensors = []
            xs = []
            backward_sends = []
            for minibatch_num in range(size):
                x_buffer = t.zeros(minibatch_size, *model_in_shape)
                dist.recv(x_buffer, rank - 1, tag=TAGS["ACTIVATION"] + minibatch_num)
                x_buffer.requires_grad = True
                with t.autocast(dtype=t.float16, device_type=device[:4]):
                    out = model(x_buffer.to(device)).cpu()
                xs.append(x_buffer)
                out_tensors.append(out)
                forward_sends.append(
                    dist.send(out, rank + 1, tag=TAGS["ACTIVATION"] + minibatch_num)
                )
            grad_buffer = t.zeros_like(out)
            for i, (out_tensor, x) in enumerate(zip(out_tensors, xs)):
                dist.recv(grad_buffer, rank + 1, tag=TAGS["GRAD"] + i)
                out_tensor.backward(grad_buffer)
                xgrad = x.grad
                backward_sends.append(dist.send(xgrad, rank - 1, tag=TAGS["GRAD"] + i))

        elif rank == size - 1:
            xs = []
            losses = []
            backward_sends = []
            ys = [t.zeros(minibatch_size, *y_size) for _ in range(size)]
            for recv in [dist.recv(y, 0, tag=TAGS["Y"] + i) for i, y in enumerate(ys)]:
                pass
            for minibatch_num in range(size):
                x_buffer = t.zeros(minibatch_size, *model_in_shape)
                dist.recv(x_buffer, rank - 1, tag=TAGS["ACTIVATION"] + minibatch_num)
                x_buffer.requires_grad = True
                with t.autocast(dtype=t.float16, device_type=device[:4]):
                    out = model(x_buffer.to(device)).cpu()
                out = out[
                    :, -1, -2:
                ]  # use the last 2 tokens of LM head as classification head
                print("out shape", out.shape)
                print("ys shape", ys[minibatch_num].shape)
                cur_loss = nn.CrossEntropyLoss()(out, ys[minibatch_num].long())
                # print(cur_loss.cpu().item())
                losses.append(cur_loss)
                xs.append(x_buffer)
            for i, (loss, x) in enumerate(zip(losses, xs)):
                loss.backward()
                xgrad = x.grad
                backward_sends.append(dist.send(xgrad, rank - 1, tag=TAGS["GRAD"] + i))
        dist.all_reduce(t.zeros(1))
        if rank == 0:
            print("post_batch_sync")
        optimizer.step()
        optimizer.zero_grad()
        if batch_num % checkpoint_every == checkpoint_every - 1:
            t.save(model, f"checkpoint{batch_num}_rank{rank}")
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
    dataset = import_object_from_qualified_name("os")()
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
        p = mp.Process(
            target=init_process,
            args=(rank, size, pprun, model_part_str, model_in_shapes[rank]),
        )
        p.start()
        processes.append(p)


if __name__ == "__main__":
    # make_and_save_resnet_pieces()
    # make_gptj_and_save_pieces()
    gin.parse_config_file(sys.argv[1])
    start_pipeline_cluster()
    # reference_training()
