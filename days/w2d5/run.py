#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DEVICES = [
    "cuda:0", "cuda:1", "cuda:2"
]

def run(rank, size):
    """ Distributed function to be implemented later. """
    # Broadcasting
    # x = torch.zeros([2,2]).to(DEVICES[rank])
    # if rank == 1:
    #     x = torch.randn([2,2]).to(DEVICES[rank])
    # dist.broadcast(x, src=1)
    # print(x)
    
    shape = [2,2]
    device = DEVICES[rank]
    
    # Scatter
    # x = torch.zeros(shape).to(device)
    # if rank == 1:
    #     y = [torch.zeros(shape).to(device) + 1 + 2 * i for i in range(size)]
    # else:
    #     y = None
    # dist.scatter(x, y, src=1)
    # print(x)
    
    # All-Reduce
    x = torch.zeros(shape).to(device) + rank
    dist.all_reduce(x, dist.ReduceOp.SUM)
    print(x)

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29503'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    device = DEVICES[rank]


if __name__ == "__main__":
    size = len(DEVICES)
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run, "gloo"))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()