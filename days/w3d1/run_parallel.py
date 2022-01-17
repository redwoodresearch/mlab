import os
import random
import torch.distributed as dist
import torch.multiprocessing as mp
from torchtyping import tensor_type
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
from tqdm import tqdm

DEVICE = "cpu"
DEVICES = [2, 3]
MAX_LEN = 1024


def load_data():
    print("loading data")
    tensor_path = "/home/ubuntu/lw.pt"
    if os.path.exists(tensor_path):
        tokens = t.load(tensor_path)
        print("tokens shape", tokens.shape)
    else:
        lw_json = json.load(open("/home/ubuntu/lw_corpus.json"))
        print("have json")
        texts = [x["text"] for x in lw_json]
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "gpt2", TOKENIZERS_PARALLELISM=True
        )
        eot_id = 50256
        eot_id = tokenizer("<|endoftext|>")["input_ids"][0]
        tokens = tokenizer(texts)["input_ids"]
        for seq in tokens:
            seq.append(eot_id)
        print("tokenized")
        tokens = t.LongTensor(list(itertools.chain(*tokens)))

        t.save(tokens, tensor_path)
    return rearrange(
        tokens[: tokens.shape[0] - (tokens.shape[0] % MAX_LEN)],
        "(b s) -> b s",
        s=MAX_LEN,
    )


def init_model():
    t.random.manual_seed(0)
    # storing model locally because huggingface throttles checking
    if os.path.exists("/home/ubuntu/gpt2_copy.pt"):
        return t.load("/home/ubuntu/gpt2_copy.pt")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    return model


def alladd_grad(model):

    reduce_ops = [
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        for param in model.parameters()
    ]
    for op in reduce_ops:
        op.wait()


def add_grad(buckets, rank):
    reduce_ops = []
    for i, bucket in enumerate(buckets):
        for param in bucket:
            reduce_ops.append(dist.reduce(param.grad, i, async_op=True))
    for op in reduce_ops:
        op.wait()


def broadcast_updated_params(buckets, rank):
    reduce_ops = []
    for i, bucket in enumerate(buckets):
        for param in bucket:
            reduce_ops.append(dist.broadcast(param, i, async_op=True))
    for op in reduce_ops:
        op.wait()


def killgroup():
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)



@gin.configurable()
def run(
    rank,
    size,
    model_init_fn_name="days.w2d5.dataparallel.init_model",
    sharded_optimizer=False,
    model_prefix = 'mlp',
):
    print("i'm rank", rank)
    # device = "cuda:" + str(rank)
    model = t.load(model_prefix + str(rank) + '.pt')
    model.train()
    model.to(DEVICE)
    optimizer = t.optim.SGD(model.parameters(), lr=1e-4)

    # If rank 0, loads data, splits things, keeps a minibatch
    # else, listen for a minibatch from rank 1
    if rank == 0:
        mlp_data = t.randn((7,128)) # TODO: broadcast this thing
        dist.barrier()
        pbar = tqdm(enumerate(dataloader))
        # for batch_num, batch in pbar:
        out = model(batch.to(DEVICE)) # necessary??

        dist.broadcast(out, src=0)


        dist.broadcast(out.grad, src=1)

    elif rank == 1:
        inp = t.zeros((7,128)).to(DEVICE)
        dist.broadcast(inp, src=0)
        out = model(inp)


        loss = t.sqrt(t.sum((out - batch)**2))
        # # ss = t.nn.CrossEntropyLoss()(
        # #     rearrange(out[:-1], "a b c -> (a b) c"),
        # #     rearrange(batch[1:], "a b -> (a b)"),
        # # )
        loss.backward()
        dist.broad(inp.grad, src=1)
        # optimizer.step()
        # optimizer.zero_grad()
        # if sharded_optimizer:
        #     broadcast_updated_params(param_buckets, rank)
        # # print(rank, "loss", loss.cpu().detach().numpy())
        # print(rank, batch_num)
        # pbar.set_description(f"loss {loss.cpu().item()}")
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

    run(rank, size)


@gin.configurable
def create_processes(
    local_parallelism=2,
    device="cpu",
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
