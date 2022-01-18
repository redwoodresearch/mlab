#!/usr/bin/env python
import json
import os
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torchtext
import transformers
import time

from days.utils import to_batches

DEVICES=["cuda:0","cuda:1"]
LEADER_RANK = 0

EMBEDDING_SIZE = 4096
SEQ_LEN = 128
BATCH_SIZE = 16
NUM_MICROBATCHES = 1
MICROBATCH_SIZE = BATCH_SIZE // NUM_MICROBATCHES

MODEL_DIR = "/home/ubuntu/david_tom_mlab/days/w3d1/model/"
MODEL_FILENAMES = ["gptj-0.pt", "gptj-1.pt"]

# classes needed for the model to load properly
class GiveMeFirst(nn.Module):
    def forward(self, x):
        return x[0]
class GiveMeLastSeq(nn.Module):
    def forward(self, x):
        return x[:, -1, :]

# fetch targets and tokens out of each batch of dataloader
def wrap_loader(dataloader, tokenizer, device, max_seq_len):
    for targets, inputs in dataloader:
        targets_tensor = torch.tensor(
                [rating == 'pos' for rating in targets],
                dtype=torch.long,
                device=device,
        )
        tokens = tokenizer(list(inputs), padding='longest', max_length=max_seq_len, truncation=True).input_ids
        tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        yield targets_tensor, tokens_tensor

def load_data(tokenizer, device, batch_size, max_seq_len=SEQ_LEN):
    data_train, data_test = torchtext.datasets.IMDB(root='.data', split=('train', 'test'))
    train_dataloader = DataLoader(
            list(data_train),
            batch_size=batch_size,
            shuffle=True,
            # don't want to think about how to broadcast the last batch if
            # things don't divide evenly
            drop_last=True,
    )
    test_dataloader = DataLoader(data_test, batch_size=batch_size)
    wrap = lambda data: wrap_loader(data, tokenizer=tokenizer, device=device, max_seq_len=max_seq_len)
    return wrap(train_dataloader), len(train_dataloader), wrap(test_dataloader), len(test_dataloader)

def init_process(rank, num_processes, backend='nccl'):
    import transformers
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=num_processes)
    device = DEVICES[rank]

    if rank == LEADER_RANK:
        tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        tokenizer.pad_token = tokenizer.eos_token
        training_data, train_len, test_data, test_len = load_data(
                tokenizer=tokenizer,
                device=device,
                batch_size=BATCH_SIZE
        )

    model = torch.load(MODEL_DIR + MODEL_FILENAMES[rank], map_location=device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    epoch_length = torch.empty(1, dtype=torch.long, device=device)
    if rank == LEADER_RANK:
        epoch_length[0] = train_len
    dist.broadcast(epoch_length, LEADER_RANK)

    prev_time = None
    for t in range(epoch_length.item()):
        if rank == LEADER_RANK and t % 10 == 0:
            current_time = time.time()
            if prev_time != None:
                print("10 iters time:", current_time - prev_time)
                # Without pipelining: ~18.5 secs
                # pipelining, batch size 16 and 8 microbatches: ~19.5
                # pipelining, batch size 16 and 4 microbatches: ~16.5
                # pipelining, batch size 16 and 2 microbatches: ~15.8
                # pipelining, batch size 16 and 1 microbatches: ~18.5
            prev_time = current_time

        if rank == LEADER_RANK:
            targets, inputs = next(training_data)
            micro_inputs = to_batches([inputs], batch_size=MICROBATCH_SIZE)
        else:
            targets = torch.empty(BATCH_SIZE, device=device, dtype=torch.long)
        dist.broadcast(targets, LEADER_RANK)

        # forward pass
        hidden_results = []
        losses = []
        for microbatch_idx in range(NUM_MICROBATCHES):
            #print(f"{rank=} {t=} {microbatch_idx=} forward")
            if rank == LEADER_RANK:
                micro_input = micro_inputs[microbatch_idx][0]
                hidden = model(micro_input)
                hidden_results.append(hidden)
            else:
                hidden = torch.empty(MICROBATCH_SIZE, SEQ_LEN, EMBEDDING_SIZE, device=device, dtype=torch.float32)
                hidden_results.append(hidden)
            # maybe: broadcast async? and then sync after for loop
            dist.broadcast(hidden, LEADER_RANK)
            if rank != LEADER_RANK:
                hidden.requires_grad = True
                output = model(hidden)
                target_start = microbatch_idx * MICROBATCH_SIZE
                target_end = (microbatch_idx + 1) * MICROBATCH_SIZE
                # we want to average loss across the whole batch, so divide
                loss = loss_fn(output, targets[target_start:target_end]) / NUM_MICROBATCHES
                losses.append(loss)

        # backward pass
        for microbatch_idx in range(NUM_MICROBATCHES):
            # print(f"{rank=} {t=} {microbatch_idx=} backward")
            optimizer.zero_grad()
            if rank != LEADER_RANK:
                losses[microbatch_idx].backward()
                hidden_grad = hidden_results[microbatch_idx].grad
            else:
                hidden_grad = torch.empty_like(hidden_results[microbatch_idx], device=device, dtype=torch.float32)
            # maybe: broadcast async? and then sync after for loop
            dist.broadcast(hidden_grad, 1)
            if rank == LEADER_RANK:
                hidden = hidden_results[microbatch_idx]
                hidden.backward(gradient=hidden_grad)
        optimizer.step()

if __name__ == "__main__":
    num_processes = len(DEVICES)
    processes = []
    mp.set_start_method("spawn")
    for rank in range(num_processes):
        p = mp.Process(target=init_process, args=(rank, num_processes))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
