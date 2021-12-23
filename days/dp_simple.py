import os
import random
import torch.distributed as dist
import torch.multiprocessing as mp
#import gin
import sys
import torch as t
import numpy as np
from sklearn.datasets import make_moons
from utils import *
import os
import signal

# class DistributedDataLoaderSlow:
#     def __init__(
#         self,
#         rank,
#         size,
#         mini_batch_size=4,
#         random_seed=0,
#     ):
#         self.rank = rank
#         self.size = size
#         if rank == 0:
#             # load and shuffle x and y
#             X, y = make_moons(n_samples=4 * 4 * 5, noise=0.1, random_state=354) 
#             self.data_tensors = [t.Tensor(X).float(), t.Tensor(y).float()]
#             t.manual_seed(random_seed)
#             perm = t.randperm(X.shape[0])
#             for i, ten in enumerate(self.data_tensors): # shuffle
#                 self.data_tensors[i] = ten[perm]

#             # batches are of size mbs * size, as mbs is the batch size for one process
#             # minibatches are of size mbs. There are "size" of them in a list, forming a batch 
#             # so self.batches is a list of batches, which are each a list of minibatches, which are each
#             # a list of parameters (like x and y), which are each a tensor of elements
#             self.batches = [to_batches(batch, mini_batch_size, trim=True)
#                            for batch in to_batches(self.data_tensors, mini_batch_size * size)]
#             self.len = t.tensor(len(self.batches))
#             print("broadcast length from", self.rank)
#             dist.broadcast(self.len, src=0) #everyone gets 0's len
#         else:
#             self.len = t.tensor(-1)
#             print("broadcast length from", self.rank)
#             dist.broadcast(self.len, src=0) #everyone gets 0's len
#             self.batches = [[] for _ in range(self.len)]

#         dist.broadcast_object_list(self.batches, src=0) #everyone gets 0's len
#         self.mini_batches = map(lambda x : x[rank], self.batches)

#     def __iter__(self):
#         return self.mini_batches

class DistributedDataLoader:
    def __init__(
        self,
        rank,
        size,
        mini_batch_size=4,
        random_seed=0,
    ):
        self.rank = rank
        self.size = size
        if rank == 0:
            # load and shuffle x and y
            X, y = make_moons(n_samples=4 * 4 * 5, noise=0.1, random_state=354) 
            self.data_tensors = [t.Tensor(X).float(), t.Tensor(y).float()]
            t.manual_seed(random_seed)
            perm = t.randperm(X.shape[0])
            for i, ten in enumerate(self.data_tensors): # shuffle
                self.data_tensors[i] = ten[perm]

            # batches are of size mbs * size, as mbs is the batch size for one process
            # minibatches are of size mbs. There are "size" of them in a list, forming a batch 
            # so self.batches is a list of batches, which are each a list of minibatches, which are each
            # a list of parameters (like x and y), which are each a tensor of elements
            self.batches = [to_batches(batch, mini_batch_size, trim=True) # chops the last batch if necessary
                           for batch in to_batches(self.data_tensors, mini_batch_size * size)]
            self.len = t.tensor(len(self.batches))
            self.batches = iter(self.batches)
        else:
            self.len = t.tensor(-1)
            self.batches = None

        print("broadcast length from", self.rank)
        dist.broadcast(self.len, src=0) #everyone gets 0's len

    # Reason to do it this way: put as much data distribution as possibe as late as possible
    # because we want to do as much training compute as possible 
    def __iter__(self):
        for _ in range(self.len):
            if self.batches is not None:
                mini_batches = next(self.batches)
            else:
                mini_batches = [None for _ in range(self.size)]
            dist.broadcast_object_list(mini_batches, src=0)
            my_batch = mini_batches[self.rank]
            yield my_batch


def alladd_grad(model): 
    
    # if you do non async, does these operations sequentially
    # Async starts them all whenever, then waits for them all to finish before continuing
    reduce_ops = [
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        for param in model.parameters()
    ]
    for op in reduce_ops:
        op.wait()
    

def init_process(rank, size, device, backend="gloo"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=rank, world_size=size)

    print("test:", os.environ["test"])

    print("inited process group", rank)

    # init model, optim, data
    t.random.manual_seed(0)
    model = t.nn.Sequential(t.nn.Linear(2, 20), t.nn.Linear(20, 1))
    model.train()
    model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-4)
    dataloader = DistributedDataLoader(rank=rank, size=size)

    # train
    for batch_num, (x, y) in enumerate(dataloader):
        # print("batch", batch)
        loss = t.sum((model(x.to(device)) - y.to(device)) ** 2)
        #print(f"Loss before: {loss}, pid: {rank}")
        optimizer.zero_grad()
        loss.backward()
        alladd_grad(model) # broadcast gradients
        optimizer.step()
        
        # print(rank, "loss", loss.cpu().detach().numpy())
        #print(rank, batch_num)
    # print(rank, "done training")
    
    # total_loss = 0
    # for x, y in dataloader:
    #     total_loss += t.sum((model(x.to(device)) - y.to(device)) ** 2)
    # print(f"Final Loss: {total_loss} and pid {rank}")

    dist.all_reduce(t.zeros(2), op=dist.ReduceOp.SUM) # syncs processes, look into?

    # ps -eg |  test.txt

    if rank == 0:
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)

if __name__ == "__main__":
    #print(t.cuda.get_device_capability("cuda:7"))    

    local_parallelism = 2 if len(sys.argv) < 3 else int(sys.argv[2])
    device = "cpu" if sys.argv[3] == "cpu" else "cuda"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["test"] = "hello"

    mp.set_start_method("spawn") #breaks if removed
    for rank in range(local_parallelism): # for each process index
        p = mp.Process(target=init_process, args=(rank, local_parallelism, device))
        p.start()


