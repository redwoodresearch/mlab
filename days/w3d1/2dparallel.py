from comet_ml import Experiment
import web_pdb
import sys
import os

# os.environ["NCCL_BLOCKING_WAIT"] = "1"
# os.environ["NCCL_DEBUG"] = "INFO"

from dataclasses import dataclass

from torch import nn
import torch.distributed as dist
import torch as t
import time
import json
import subprocess
import itertools
from einops import rearrange
import random

# Pipeline parallel and data parallel at once

# to fix hostname lan issues add this to /etc/hosts
# 104.171.200.117 104-171-200-117
# 104.171.200.214 104-171-200-214
# 104.171.200.196 104-171-200-196
@dataclass
class Config:
    stage_ips = [f"ubuntu@104.171.200.{x}" for x in [117, 117, 214, 214, 196, 196]]
    stage_dp_cuda_ids = [[0, 1], [2, 3], [0, 1], [2, 3], [0, 1], [2, 3]]
    model_in_shapes = [(1024,), (1024, 4096), (1024, 4096), (1024, 4096), (1024, 4096), (1024, 4096)]

    microbatch_size = 2
    seq_len = 1024
    master_addr = "104.171.200.117"
    master_port = "29500"
    dp_size = 2
    mp_size = 6
    model_file_prefix = "gpt-j-6b"
    y_shape = (1024,)
    dataset_fn_name = "days.pipelineparallel.make_dataset_imdb"
    dist_backend = "nccl"
    use_autocast = True
    pipe_width = 4
    checkpoint_every_m = 5
    use_cpu = False
    sharded_optimizer = True

    lr = 1e-7

    total_size = None
    stage_dp_sizes_cum = None

    def __init__(self):
        self.stage_dp_sizes_cum = t.IntTensor([0] + [len(x) for x in self.stage_dp_cuda_ids]).cumsum(0).tolist()
        self.total_size = int(self.stage_dp_sizes_cum[-1])
        self.device_type = "cpu" if self.use_cpu else "cuda"


C = Config()


class HFBlockSequence(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)  # treat like a list

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x


def load_data():
    import transformers

    print("loading data")
    tensor_path = "/home/ubuntu/lw.pt"
    if os.path.exists(tensor_path):
        tokens = t.load(tensor_path)
    else:
        lw_json = json.load(open("/home/ubuntu/lw_corpus.json"))
        texts = [f'{x["karma"]}\n{x["text"]}' for x in lw_json]
        random.shuffle(texts)
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


def params_to_buckets(params, n_buckets):
    params = list(params)
    target_params_per = sum([x.numel() for x in params]) // n_buckets
    buckets = [[]]
    bucket_size = 0
    for param in params:
        if len(buckets) < n_buckets and param.numel() + bucket_size > target_params_per:
            buckets.append([])
            buckets[-1].append(param)
            bucket_size = 0
        else:
            buckets[-1].append(param)
            bucket_size += param.numel()
    return buckets


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

    def sinc():
        if not C.use_cpu:
            t.cuda.synchronize(device)
            return

    # start all our fucking process groups!
    os.environ["MASTER_ADDR"] = C.master_addr
    os.environ["MASTER_PORT"] = C.master_port
    from datetime import timedelta

    dist.init_process_group(
        backend=C.dist_backend, rank=total_rank, world_size=C.total_size, timeout=timedelta(seconds=90)
    )
    process_groups = {
        "stage": [None for _ in range(C.mp_size)],
        "pipe": [None for _ in range(C.dp_size)],
        "stage_links": [[None for _ in range(C.mp_size)] for _ in range(C.dp_size)],
    }
    for g_mp_rank in range(C.mp_size):
        process_groups["stage"][g_mp_rank] = dist.new_group(
            ranks=[get_total_rank(g_mp_rank, i) for i in range(C.dp_size)],
        )
    for g_dp_rank in range(C.dp_size):
        process_groups["pipe"][g_dp_rank] = dist.new_group(
            ranks=[get_total_rank(i, g_dp_rank) for i in range(C.mp_size)],
        )
    for g_dp_rank in range(C.dp_size):
        for g_mp_rank in range(C.mp_size):
            process_groups["stage_links"][g_dp_rank][g_mp_rank] = dist.new_group(
                ranks=[
                    get_total_rank(g_mp_rank, g_dp_rank),
                    get_total_rank((g_mp_rank + 1) % C.mp_size, g_dp_rank),
                ],
            )
    pipe_group = process_groups["pipe"][dp_rank]
    stage_group = process_groups["stage"][mp_rank]
    fwd_group = process_groups["stage_links"][dp_rank][mp_rank]
    bwd_group = process_groups["stage_links"][dp_rank][(C.mp_size + mp_rank - 1) % C.mp_size]

    model_part_fname = f"{C.model_file_prefix}_part{mp_rank}.pt"
    model: nn.Module = t.load(model_part_fname)
    model.train()
    model.to(device)
    if C.sharded_optimizer:
        param_buckets = params_to_buckets(model.parameters(), C.dp_size)
        params = param_buckets[dp_rank]
    else:
        # some hyperparameters copied from gpt3 paper
        # use 10% LR because that's what they use at end of training?
        # gpt3 uses 1.2e-4 at 2m batch size, 1.2e-5 at end of training, I'm using 24k tokens so I want 1e-7 lr?
        params = model.parameters()
    optimizer = t.optim.Adam(params, lr=C.lr / C.dp_size, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)

    print("model loaded", mp_rank, dp_rank)
    num_batches = t.IntTensor([0]).to(device)
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
    if total_rank == 11:
        experiment = Experiment(
            api_key="vABV7zo6pqS7lfzZBhyabU2Xe",
            project_name="jan15-2dp",
        )
    start = time.time()
    batch_start = time.time()
    last_checkpoint_time = time.time()
    print("num_batches", num_batches, mp_rank, dp_rank)
    for batch_num in range(num_batches):
        dist.barrier()
        sinc()  # done using global group
        if mp_rank == 0:
            pipe_batches = batches[batch_num].long().to(device)
            out_tensors = []
            # send batch Ys to the end so it can calculate loss
            dist.broadcast(pipe_batches, src=total_rank, group=bwd_group)
            sinc()  # done using bwd group

            for i, microbatch in enumerate(pipe_batches):
                with t.autocast(
                    dtype=autocast_type,
                    device_type=C.device_type,
                    enabled=C.use_autocast,
                ):
                    out = model(microbatch.to(device)).float()  # all the gpu action
                out_tensors.append(out)
                dist.broadcast(out, src=total_rank, group=fwd_group)

            grad_buffers = [t.zeros_like(out) for _ in range(C.pipe_width)]
            for i, out_tensor in enumerate(out_tensors):
                dist.broadcast(
                    grad_buffers[i],
                    src=get_total_rank(mp_rank + 1, dp_rank),
                    group=fwd_group,
                )
                grad_buffer = grad_buffers[i]
                out_tensor.backward(grad_buffer)
                # release out tensor memory after each backward minibatch
                out_tensors[i] = None
            sinc()  # done using fwd group
        elif mp_rank != C.mp_size - 1:
            out_tensors = []
            xs = []
            backward_sends = []
            xs = [t.zeros(C.microbatch_size, *C.model_in_shapes[mp_rank], device=device) for _ in range(C.pipe_width)]

            for microbatch_num in range(C.pipe_width):
                dist.broadcast(
                    xs[microbatch_num],
                    src=get_total_rank(mp_rank - 1, dp_rank),
                    group=bwd_group,
                )
                sinc()  # done using bwd group

                x_buffer = xs[microbatch_num]
                x_buffer.requires_grad = True
                with t.autocast(
                    dtype=autocast_type,
                    device_type=C.device_type,
                    enabled=C.use_autocast,
                ):
                    out = model(x_buffer.to(device)).float()
                xs.append(x_buffer)
                out_tensors.append(out)
                dist.broadcast(out, src=total_rank, group=fwd_group)
                sinc()  # done using fwd group
            grad_buffers = [t.zeros_like(out) for _ in range(C.pipe_width)]
            for i, (out_tensor, x) in enumerate(zip(out_tensors, xs)):
                dist.broadcast(
                    grad_buffers[i],
                    src=get_total_rank(mp_rank + 1, dp_rank),
                    group=fwd_group,
                )
                sinc()  # done using fwd group
                grad_buffer = grad_buffers[i]
                out_tensor.backward(grad_buffer)
                xgrad = x.grad
                dist.broadcast(
                    xgrad,
                    src=total_rank,
                    group=bwd_group,
                )
                sinc()  # done using bwd group
                xs[i] = None
                out_tensors[i] = None

        elif mp_rank == C.mp_size - 1:
            xs = []
            losses = []
            backward_sends = []
            ys = t.zeros((C.pipe_width, C.microbatch_size, C.seq_len), dtype=t.int64, device=device)
            dist.broadcast(ys, get_total_rank(0, dp_rank), group=fwd_group)
            sinc()  # done using fwd group

            xs = [t.zeros(C.microbatch_size, *C.model_in_shapes[mp_rank], device=device) for _ in range(C.pipe_width)]
            for microbatch_num in range(C.pipe_width):
                dist.broadcast(
                    xs[microbatch_num],
                    get_total_rank(mp_rank - 1, dp_rank),
                    group=bwd_group,
                )
                x_buffer = xs[microbatch_num]
                x_buffer.requires_grad = True
                with t.autocast(
                    dtype=autocast_type,
                    device_type=C.device_type,
                    enabled=C.use_autocast,
                ):  # save memory by computing with less precision
                    out = model(x_buffer.to(device)).float()
                cur_loss = nn.CrossEntropyLoss()(
                    rearrange(out[:, :-1], "a b c -> (a b) c"),
                    rearrange(ys[microbatch_num][:, 1:], "a b -> (a b)"),
                )
                # print(cur_loss.cpu().item())
                losses.append(cur_loss)
                xs.append(x_buffer)
            batch_loss = sum([x.cpu().item() for x in losses]) / len(losses)
            batch_time = time.time() - batch_start
            tokens_per_second = (C.dp_size * C.pipe_width * C.microbatch_size * C.seq_len) // batch_time
            print("whole batch loss", batch_loss, "took", batch_time, "tokens per second", tokens_per_second)
            if total_rank == 11:
                experiment.log_metric("batch_loss", batch_loss)
                experiment.log_metric("batch_time", batch_time)
                experiment.log_metric("tokens_per_second", tokens_per_second)
            batch_start = time.time()
            for i, (loss, x) in enumerate(zip(losses, xs)):
                loss.backward()
                xgrad = x.grad
                backward_sends.append(dist.broadcast(xgrad, src=total_rank, group=bwd_group))
                losses[i] = None
                xs[i] = None
        sinc()
        if C.sharded_optimizer:
            reduce_ops = []
            for i, bucket in enumerate(param_buckets):
                for param in bucket:
                    reduce_ops.append(
                        dist.reduce(param.grad, get_total_rank(mp_rank, i), group=stage_group, async_op=True)
                    )
            for op in reduce_ops:
                op.wait()

            optimizer.step()

            reduce_ops = []
            for i, bucket in enumerate(param_buckets):
                for param in bucket:
                    reduce_ops.append(
                        dist.broadcast(param, get_total_rank(mp_rank, i), group=stage_group, async_op=True)
                    )
            for op in reduce_ops:
                op.wait()
        else:
            reduce_ops = [
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=stage_group, async_op=True)
                for param in model.parameters()
            ]
            for op in reduce_ops:
                op.wait()
            optimizer.step()
        sinc()  # done using stage group
        optimizer.zero_grad()
        if time.time() - last_checkpoint_time > 60 * C.checkpoint_every_m:
            last_checkpoint_time = time.time()
            if mp_rank == 0:
                print("saving")
            filename = f"./.checkpoints/gptj_imdb_{batch_num}_rank{mp_rank}"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:  # added this
                t.save(model, f)
    end = time.time()
    print(
        f"Total time: {start - end}, per batch: {(start - end)/num_batches}, per example {(start - end)/total_examples}, rank {mp_rank}"
    )


from threading import Thread
from queue import Queue, Empty


def git_pull(ip):
    os.system(
        f'ssh -o StrictHostKeyChecking=no -i ~/mlab_ssh {ip} "pkill python; cd mlab; git fetch -q; git reset -q --hard  origin/2dp;"',
    )


def start_cluster():  # does gin add the arguments here? crazy
    remote_procs = []
    unique_name = str(int(time.time() * 10))
    for ip in set(C.stage_ips):
        git_pull(ip)

    q = Queue()

    def enqueue(out, rank):
        for line in iter(out.readline, b""):
            q.put(f"{rank}: {line}")

    for mp_rank, ip in enumerate(C.stage_ips):
        for dp_rank in range(C.dp_size):
            total_rank = C.stage_dp_sizes_cum[mp_rank] + dp_rank
            cmd = f'ssh -o StrictHostKeyChecking=no -i ~/mlab_ssh {ip} "cd ~/mlab; python days/w3d1/2dparallel.py process {mp_rank} {dp_rank} {total_rank} 1>&2 4>&2"'
            proc = subprocess.Popen(cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, bufsize=1, text=True)
            remote_procs.append(proc)
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
    if sys.argv[1] == "save_model":
        procs = []
        for mp_rank, ip in enumerate(set(C.stage_ips)):  # only do each ip once
            git_pull(ip)
            cmd = f'ssh -o StrictHostKeyChecking=no -i ~/mlab_ssh {ip} "cd ~/mlab; python days/w3d1/save_model.py"'
            proc = subprocess.Popen(cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, bufsize=1, text=True)
            procs.append(proc)
        for proc in procs:
            proc.wait()

    if sys.argv[1] == "orchestrate":
        print(
            f"""STARTING 2DP RUN___________________________
        
        
        
        
        
        """
        )
        start_cluster()
    elif sys.argv[1] == "process":
        pprun(mp_rank=int(sys.argv[2]), dp_rank=int(sys.argv[3]), total_rank=int(sys.argv[4]))
    else:
        print("ERRORIE")
