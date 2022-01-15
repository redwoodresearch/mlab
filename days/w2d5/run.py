"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import json
import transformers

DEVICES = ["cuda:0","cuda:1","cuda:2"]
EPOCHS = 1
SEED = 11
LR = 1e-4

def run_broadcast(rank, size):
    if rank == 0:
        tensor = torch.rand(5).to(DEVICES[rank])
    else:
        tensor = torch.zeros(5).to(DEVICES[rank])
    print("Rank ", rank, "ID ", id(tensor), "data ", tensor)
    dist.broadcast(tensor=tensor, src=0)
    print("Rank ", rank, "ID ", id(tensor), "data ", tensor)

def run_all_reduce(rank, size):
    tensor = torch.rand(5, device=DEVICES[rank])
    print("Rank ", rank, "ID ", id(tensor), "data ", tensor)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print("Rank ", rank, "ID ", id(tensor), "data ", tensor)

def average_gradients(model, size):
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def train(rank, size):
    torch.manual_seed(SEED)
    loader = DistributedDataLoader(rank, size)
    model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(DEVICES[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(3):
        epoch_loss = 0.
        for i, mini_batch in enumerate(loader):
            optimizer.zero_grad()
            output = model(mini_batch, labels=mini_batch)
            epoch_loss += output.loss.detach().cpu()
            output.loss.backward()
            average_gradients(model, size)
            optimizer.step()

        print('Rank ', rank, 'epoch ', epoch, 'loss ', epoch_loss / i+1)

def init_process(rank, device, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29576'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

class DistributedDataLoader:
    def __init__(self, rank, world_size, seq_len=1024, mini_batch_size=2, leader=0, random_seed=0):
        self.rank = rank
        self.world_size = world_size
        self.mini_batch_size = mini_batch_size
        self.random_seed = random_seed
        self.leader = leader
        self.seq_len = seq_len
        if rank == leader:
            with open("lw_corpus_tokens.json") as f:
                data = torch.tensor(json.load(f), device=DEVICES[rank])
            
            num_batches = torch.tensor([data.shape[0]], device=DEVICES[rank])
            self.dataloader = DataLoader(data, 
                                    batch_size = world_size * mini_batch_size,
                                    shuffle=True)
        else:
            num_batches = torch.tensor([-1], device=DEVICES[rank])
        dist.broadcast(num_batches, src=leader)
        self.num_batches = num_batches.detach().item()

    def __iter__(self):
        if self.rank == self.leader:
            for batch in self.dataloader:
                dist.broadcast(tensor=batch, src=self.leader)
                split_tensor = torch.tensor_split(batch, self.world_size, 0)[self.rank]
                yield split_tensor.to(DEVICES[self.rank])
        else:
            print(self.num_batches)
            for _ in range(self.num_batches):
                tensor = torch.zeros(
                    self.mini_batch_size * self.world_size,
                    self.seq_len,
                    dtype=torch.int64,
                    device=DEVICES[self.rank]
                )
                dist.broadcast(tensor=tensor, src=self.leader)
                split_tensor = torch.tensor_split(tensor, self.world_size, 0)[self.rank]
                yield split_tensor


if __name__ == "__main__":

    processes = []
    mp.set_start_method("spawn")
    for index, device in enumerate(DEVICES):
        p = mp.Process(target=init_process, args=(index, device, len(DEVICES), train))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()



