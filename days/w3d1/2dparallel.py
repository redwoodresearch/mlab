import web_pdb
import sys
import os

if False and len(sys.argv) > 2:
    myport = 5555 + int(sys.argv[4])
    os.system(f"fuser -k {myport}/tcp")
    web_pdb.set_trace(port=myport)
from dataclasses import dataclass

from torch import nn
import torch.distributed as dist
import torch as t
from time import time
import json
import subprocess
import itertools


# Pipeline parallel and data parallel at once

# to fix hostname lan issues add this to /etc/hosts
# 104.171.200.117 104-171-200-117
# 104.171.200.214 104-171-200-214
# 104.171.200.196 104-171-200-196
@dataclass
class Config:
    stage_ips = [f"ubuntu@104.171.200.{x}" for x in [117, 117, 214, 214, 196, 196]]
    stage_dp_cuda_ids = [[0, 1], [2, 3], [0, 1], [2, 3], [0, 1], [2, 3]]
    model_in_shapes = [(1024,), (1024, 4096), (1024, 4096), (1024, 4096)]

    microbatch_size = 1
    seq_len = 1024
    master_addr = "104.171.200.117"
    master_port = "29500"
    dp_size = 2
    mp_size = 6
    model_file_prefix = "gpt-j-6b"
    data_file_prefix = "lw_tensor"
    y_shape = (1024,)
    dataset_fn_name = "days.pipelineparallel.make_dataset_imdb"
    dist_backend = "nccl"
    use_autocast = True
    pipe_width = 4
    checkpoint_every_m = 10
    use_cpu = False

    total_size = None
    stage_dp_sizes_cum = None

    def __init__(self):
        self.stage_dp_sizes_cum = t.IntTensor([0] + [len(x) for x in self.stage_dp_cuda_ids]).cumsum(0).tolist()
        self.total_size = int(self.stage_dp_sizes_cum[-1])
        self.device_type = "cpu" if self.use_cpu else "cuda"


C = Config()

# HuggingFace models return tuples in the middle (things like activation patterns), thus the [0]
class HFBlockSequence(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)  # treat like a list

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x


# call once
def make_gptj_and_save_pieces():
    import transformers

    model_lm = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    model = model_lm.transformer

    num_layers = len(model)
    assert num_layers == 28
    chunks = [4, 5, 5, 5, 5, 4]  # less at ends due to embeddings/unembed
    assert sum(chunks) == num_layers
    chunk_cumsum = t.cumsum(t.tensor(chunks), dim=0).tolist()
    print("cumsum", chunk_cumsum)
    models = [HFBlockSequence(*model.h[start - size : start]) for start, size in zip(chunk_cumsum, chunks)]
    models[0] = nn.Sequential(model.wte, model.drop, models[0])
    models[-1] = nn.Sequential(models[-1], model.ln_f, model_lm.lm_head)
    for i, model_part in enumerate(models):
        t.save(model_part, f"gpt-j-6b_part{i}.pt")
    return models


def load_data():
    import transformers

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
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", TOKENIZERS_PARALLELISM=True)
        # eot_id = tokenizer("<|endoftext|>")["input_ids"][0]
        eot_id = 50256
        tokens = tokenizer(texts)["input_ids"]
        for seq in tokens:
            seq.append(eot_id)
        print("tokenized")
        tokens = t.LongTensor(list(itertools.chain(*tokens)))

        t.save(tokens, tensor_path)
    return tokens


def get_total_rank(mp_rank, dp_rank):
    return C.stage_dp_sizes_cum[mp_rank] + dp_rank


# 'bad address' error means you tried to use operations that weren't supported on cuda
# Multiple groups cannot operate simultaneously in NCCL, so we have to await all of one group before we start the next group.
def pprun(
    mp_rank,
    dp_rank,
    total_rank,
):
    autocast_type = t.bfloat16 if C.use_cpu else t.float16
    device = "cpu" if C.use_cpu else "cuda:" + str(C.stage_dp_cuda_ids[mp_rank][dp_rank])

    # start all our fucking process groups!
    os.environ["MASTER_ADDR"] = C.master_addr
    os.environ["MASTER_PORT"] = C.master_port
    print("will init process group", total_rank)
    dist.init_process_group(backend=C.dist_backend, rank=total_rank, world_size=C.total_size)
    print("inited process group", total_rank)
    process_groups = {
        "stage": [None for _ in range(C.mp_size)],
        "pipe": [None for _ in range(C.dp_size)],
        "stage_links": [[None for _ in range(C.mp_size)] for _ in range(C.dp_size)],
    }
    print("initing subgroups", mp_rank, dp_rank)
    for g_mp_rank in range(C.mp_size):
        process_groups["stage"][g_mp_rank] = dist.new_group(
            ranks=[get_total_rank(g_mp_rank, i) for i in range(C.dp_size)],
            backend="nccl",
        )
    for g_dp_rank in range(C.dp_size):
        process_groups["pipe"][g_dp_rank] = dist.new_group(
            ranks=[get_total_rank(i, g_dp_rank) for i in range(C.mp_size)],
            backend="nccl",
        )
    for g_dp_rank in range(C.dp_size):
        for g_mp_rank in range(mp_rank):
            process_groups["stage_links"][g_dp_rank][g_mp_rank] = dist.new_group(
                ranks=[
                    get_total_rank(g_mp_rank, g_dp_rank),
                    get_total_rank((g_mp_rank + 1) % C.mp_size, g_dp_rank),
                ],
                backend="nccl",
            )
    pipe_group = process_groups["pipe"][dp_rank]
    stage_group = process_groups["stage"][mp_rank]
    fwd_group = process_groups["stage_links"][dp_rank][mp_rank]
    bwd_group = process_groups["stage_links"][dp_rank][(C.mp_size + mp_rank - 1) % C.mp_size]
    print("initiated subgroups", mp_rank, dp_rank)

    model_part_fname = f"[{C.model_file_prefix}_part{mp_rank}.pt"
    if not os.path.exists(model_part_fname):
        if dp_rank == 0:
            make_gptj_and_save_pieces()
        dist.barrier(group=stage_group)
    model: nn.Module = t.load(model_part_fname)
    model.train()
    model.to(device)
    print("loaded model", mp_rank, dp_rank)

    optimizer = t.optim.SGD(model.parameters(), lr=1e-4)  # TODO switch to sharded optimizer adam

    print("model loaded", mp_rank, dp_rank)
    num_batches = t.IntTensor([0])
    if mp_rank == 0:
        dataset = load_data()
        # each loads all data then takes its dp slice
        batches = dataset[: -(dataset.shape[0] % (C.dp_size * C.pipe_width * C.microbatch_size * C.seq_len))].reshape(
            C.dp_size, -1, C.pipe_width, C.microbatch_size, C.seq_len
        )
        batches = batches[dp_rank]
        total_examples = batches.shape[0] * batches.shape[1] * batches.shape[2]
        num_batches[0] = batches.shape[0]

    dist.broadcast(num_batches, src=get_total_rank(0, 0))
    num_batches = num_batches.item()

    start = time()
    batch_start = time()
    last_checkpoint_time = time()
    for batch_num in range(num_batches):
        dist.barrier()
        print("crossed barrier", mp_rank, dp_rank)
        if mp_rank == 0:
            pipe_batches = batches[batch_num].to(device)
            out_tensors = []

            # send batch Ys to the end so it can calculate loss
            dist.broadcast(pipe_batches, src=total_rank, group=bwd_group)

            for i, microbatch in enumerate(pipe_batches):
                with t.autocast(
                    dtype=autocast_type,
                    device_type=C.device_type,
                    enabled=C.use_autocast,
                ):
                    out = model(microbatch.long().to(device))  # all the gpu action
                out_tensors.append(out)
                dist.broadcast(out, src=total_rank, group=fwd_group)
            grad_buffers = [t.zeros_like(out) for _ in range(C.pipe_width)]
            grad_recvs = [
                dist.broadcast(
                    x,
                    src=get_total_rank(mp_rank + 1, dp_rank),
                    group=bwd_group,
                    async_op=True,
                )
                for i, x in enumerate(grad_buffers)
            ]
            grad_buffer = t.zeros_like(out)
            for i, out_tensor in enumerate(out_tensors):
                grad_recvs[i].wait()
                grad_buffer = grad_buffers[i]
                out_tensor.backward(grad_buffer)
                # release out tensor memory after each backward minibatch
                out_tensors[i] = None
        elif mp_rank != C.mp_size - 1:
            forward_sends = []
            out_tensors = []
            xs = []
            backward_sends = []
            xs = [t.zeros(C.microbatch_size, *C.model_in_shape, device=device) for _ in range(C.pipe_width)]
            xjobs = [
                dist.broadcast(
                    x,
                    src=get_total_rank(mp_rank - 1, dp_rank),
                    group=bwd_group,
                    async_op=True,
                )
                for i, x in enumerate(xs)
            ]

            for microbatch_num in range(C.pipe_width):
                xjobs[microbatch_num].wait()
                x_buffer = xs[microbatch_num]
                x_buffer.requires_grad = True
                with t.autocast(
                    dtype=autocast_type,
                    device_type=C.device_type,
                    enabled=C.use_autocast,
                ):
                    out = model(x_buffer.to(device))
                xs.append(x_buffer)
                out_tensors.append(out)
                dist.broadcast(out, src=total_rank, group=fwd_group)
            grad_buffers = [t.zeros_like(out) for _ in range(C.pipe_width)]
            grad_recvs = [
                dist.broadcast(
                    x,
                    src=get_total_rank(mp_rank + 1, dp_rank),
                    group=fwd_group,
                )
                for i, x in enumerate(grad_buffers)
            ]
            for i, (out_tensor, x) in enumerate(zip(out_tensors, xs)):
                grad_buffer = grad_buffers[i]
                out_tensor.backward(grad_buffer)
                xgrad = x.grad
                dist.broadcast(
                    xgrad,
                    src=total_rank,
                    group=bwd_group,
                )
                xs[i] = None
                out_tensors[i] = None

        elif mp_rank == C.mp_size - 1:
            xs = []
            losses = []
            backward_sends = []
            ys = [t.zeros(C.microbatch_size, C.seq_len, device=device) for _ in range(C.pipe_width)]
            yjobs = [dist.broadcast(y, get_total_rank(0, dp_rank), group=fwd_group) for i, y in enumerate(ys)]
            xs = [t.zeros(C.microbatch_size, *C.model_in_shape[mp_rank], device=device) for _ in range(C.pipe_width)]
            xjobs = [
                dist.broadcast(
                    x,
                    get_total_rank(mp_rank - 1, dp_rank),
                    group=bwd_group,
                    async_op=True,
                )
                for i, x in enumerate(xs)
            ]
            for microbatch_num in range(C.pipe_width):
                xjobs[microbatch_num].wait()
                x_buffer = xs[microbatch_num]
                x_buffer.requires_grad = True
                with t.autocast(
                    dtype=autocast_type,
                    device_type=C.device_type,
                    enabled=C.use_autocast,
                ):  # save memory by computing with less precision
                    out = model(x_buffer.to(device))
                out = out[:, -1, -2:]  # use the last 2 tokens of LM head as classification head
                cur_loss = nn.CrossEntropyLoss()(out.float(), ys[microbatch_num].long())
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
                backward_sends.append(dist.broadcast(xgrad, src=total_rank, group=bwd_group))
                losses[i] = None
                xs[i] = None

        # average grad
        reduce_ops = [
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=stage_group, async_op=True)
            for param in model.parameters()
        ]
        for op in reduce_ops:
            op.wait()

        optimizer.step()
        optimizer.zero_grad()
        if time() - last_checkpoint_time > 60 * C.checkpoint_every_m:
            last_checkpoint_time = time()
            if mp_rank == 0:
                print("saving")
            filename = f"./.checkpoints/gptj_imdb_{batch_num}_rank{mp_rank}"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:  # added this
                t.save(model, f)
    end = time()
    print(
        f"Total time: {start - end}, per batch: {(start - end)/num_batches}, per example {(start - end)/total_examples}, rank {mp_rank}"
    )


from threading import Thread
from queue import Queue, Empty


def start_cluster():  # does gin add the arguments here? crazy
    remote_procs = []
    os.system(f'ssh -i ~/mlab_ssh ubuntu@{C.master_addr} "fuser -k {C.master_port}/tcp"')
    unique_name = str(int(time() * 10))
    for ip in set(C.stage_ips):
        os.system(
            f'ssh -o StrictHostKeyChecking=no -i ~/mlab_ssh {ip} "cd mlab; git fetch -q; git reset -q --hard  origin/2dp;"',
        )
    q = Queue()

    def enqueue(out, rank):
        for line in iter(out.readline, b""):
            q.put(f"{rank}: {line}")

    for mp_rank, ip in enumerate(C.stage_ips):
        for dp_rank in range(C.dp_size):
            total_rank = C.stage_dp_sizes_cum[mp_rank] + dp_rank
            cmd = f'ssh -o StrictHostKeyChecking=no -i ~/mlab_ssh {ip} "cd ~/mlab; python days/w3d1/2dparallel.py process {mp_rank} {dp_rank} {total_rank} 1>&2 4>&2"'
            proc = subprocess.Popen(
                cmd, shell=True, bufsize=1, stdout=subprocess.STDOUT, stderr=subprocess.STDOUT, text=True
            )
            remote_procs.append(proc)
            t = Thread(target=enqueue, args=(proc.stdout, mp_rank))
            t = Thread(target=enqueue, args=(proc.stderr, mp_rank))
            t.daemon = True
            t.start()
            print("started process", mp_rank, dp_rank)

    while 1:
        try:
            line = q.get(timeout=0.5)
            print(line)
        except Empty:
            pass


# 2,8 produces 0.18 time units and final loss of 0.10 (noisy)
# 2,10 breaks (too much mem)
# 4,4 produced something like 0.17 time-units/example, final loss of 0.11, half-way 0.38
# 8,2 produced something like 0.10 time-units/example, final loss of 0.37, half-way 0.53

# with 4 minibatches of 12, took 3.1s
# with 1 minibatch of 48, took 4.6
#


if __name__ == "__main__":
    print("hi from 2dparallel")
    # import hashlib

    # os.system("touch ~/touchfile")
    # thisfile = __file__
    # tfh = hashlib.md5(open(thisfile, "rb").read()).hexdigest()
    # print("file hash", tfh)
    if sys.argv[1] == "orchestrate":
        print(
            f"""STARTING 2DP RUN___________________________
        
        
        
        
        
        """
        )
        start_cluster()
    elif sys.argv[1] == "process":
        raise AssertionError("er")
        pprun(mp_rank=int(sys.argv[2]), dp_rank=int(sys.argv[3]), total_rank=int(sys.argv[4]))
    else:
        print("ERRORIE")
