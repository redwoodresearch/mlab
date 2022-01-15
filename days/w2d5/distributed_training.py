import os
import torch.distributed as dist
import torch.multiprocessing as mp
import time

DEVICES = [
    "cuda:0", "cuda:1", "cuda:2"
]


import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import torchtext
from typing import Optional
import transformers

from distributed_dataloader import DistributedDataLoader


SHARD_OPTIMIZER_STATE=False
"""
mini_batch_size 350: both pass
mini_batch_size 375: sharding passes
mini_batch_size 400: both fail
"""

def run(rank, size):
    """ Distributed function to be implemented later. """

    if rank == 0:
        start_time = time.time()
    
    device = DEVICES[rank] 

    model = transformers.GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    
    if SHARD_OPTIMIZER_STATE:
        params_to_optimize = []
        for i, param in enumerate(model.parameters()):
            if i % size == 0:
                params_to_optimize.append(param)
        optimizer = torch.optim.Adam(params_to_optimize, lr=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Iterate over minibatches
    model.train()
    for epoch in range(4):
        print(epoch)
        ddl = DistributedDataLoader(rank, len(DEVICES), 375, random_seed = epoch)
        for minibatch_data in ddl:
            optimizer.zero_grad()
            # Normal training loop
            # FIXME
            minibatch_data = {'input_ids': minibatch_data,
                            'attention_mask': torch.ones_like(minibatch_data, dtype=torch.long)}
            outputs = model(**minibatch_data, labels=minibatch_data['input_ids']) 
            loss = outputs.loss
            loss.backward()
            print(loss.detach())
            # All-reduce to share gradients, for each parameter
            for param in model.parameters():
                old_grad = param.grad.detach().clone()
                # Taking the mean over the gradients
                dist.all_reduce(param.grad, dist.ReduceOp.SUM)
                # assert not torch.allclose(old_grad, param.grad.detach())
                param.grad = param.grad / size
            # Does it take real long? Maybe time optimizer.step() and dist.broadcast, and compare them
            optimizer.step()
            if SHARD_OPTIMIZER_STATE:
                for i, param in enumerate(model.parameters()):
                    dist.broadcast(param.data, src=i % 3)
    
    print('Training completed')
    loss = 0.

    ddl = DistributedDataLoader(rank, len(DEVICES), 32, random_seed = epoch)

    if rank == 0:
        model.eval()
        test_data = ddl.test_dataloader
        c = 0
        for test_datum in test_data:
            test_datum = ddl.tokenize(test_datum)
            test_datum = {'input_ids': test_datum,
                        'attention_mask': torch.ones_like(test_datum, dtype=torch.long)}
            outputs = model(**test_datum, labels=test_datum['input_ids']) 
            loss += outputs.loss.detach()
            c +=1
            if c % 10 == 0:
                print(loss)
            if c > 100:
                break    
        print("eval loss: ", loss / len(test_data) / c)  

        print("time: ", time.time() - start_time)          
    

    # After 4 epochs evaluate on test set

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