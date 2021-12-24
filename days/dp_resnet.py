import os
import random
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
#import gin
import sys
import torch as t
import numpy as np
from sklearn.datasets import make_moons
from utils import *
import signal
from tqdm import tqdm


def load_data(train, bsz):
    data = torchvision.datasets.CIFAR10("~/mlab/datasets/cifar10_" + ("train" if train else "test"),
                            transform= torchvision.transforms.Compose([
                                torchvision.transforms.PILToTensor(),
                                torchvision.transforms.ConvertImageDtype(t.float),
                                torchvision.transforms.Resize((64, 64))]),
                            download=False, 
                            train=train)
    data = [t.stack([p[0] for p in data]), t.tensor([p[1] for p in data])]
    return to_batches(data, bsz, trim=True)

def load_model():
    t.random.manual_seed(0)
    return torchvision.models.resnet18() 

class DistributedDataLoader:
    def __init__(
        self,
        rank,
        size,
        mini_batch_size=10000,
        random_seed=0,
    ):
        self.rank = rank
        self.size = size
        self.mini_batch_size = mini_batch_size
        if rank == 0:
            self.batches = load_data(train=True, bsz=mini_batch_size * size)
            self.len = t.tensor(len(self.batches))
        else:
            self.len = t.tensor(-1)
            self.batches = None

        print("broadcast length from", self.rank)
        dist.broadcast(self.len, src=0) #everyone gets 0's len

    # Reason to do it this way: put as much data distribution as possibe as late as possible
    # because we want to do as much training compute as possible 
    def __iter__(self):
        for i in range(self.len):
            x_mb = t.zeros((self.mini_batch_size, 3, 64, 64), dtype=t.float32)
            y_mb = t.zeros((self.mini_batch_size), dtype=t.int64)
            tensors = [x_mb, y_mb]
            #scatter_lists = to_batches(self.batches[i], self.mini_batch_size) if self.batches is not None else [None, None]
            if self.batches is not None:
                scatter_lists = [list(rearrange(tensor, "(s m) ... -> s m ...", s=self.size)) for tensor in self.batches[i]]
            else:
                scatter_lists = [None, None]

            #dist.scatter_object_list(tensors, scatter_lists, src = 0)
            dist.scatter(x_mb, scatter_list=scatter_lists[0], src = 0, async_op = True).wait()
            dist.scatter(y_mb, scatter_list=scatter_lists[1], src = 0, async_op = True).wait()
            yield tensors

def alladd_grad(model): 
    # NOT EMPIRCALLY FASTER TO DO ASYNC: does wait regardless
    # https://pytorch.org/docs/stable/_modules/torch/distributed/distributed_c10d.html#all_reduce
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    
def init_process(rank, size, device, backend="gloo"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=rank, world_size=size)

    # Comment this to run on one GPU
    if device == "cuda":
        device += ":" + str(rank)

    print("inited process group", rank, " on device ", device)

    model = load_model()
    model.train()
    model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=0.005)
    dataloader = DistributedDataLoader(rank=rank, size=size)
    loss_fn = t.nn.CrossEntropyLoss(reduction="sum")

    # train
    for epoch in range(40):
        for batch_num, (x, y) in enumerate(tqdm(dataloader)):
            # NOTE: look out for reduction == mean instead, whichj seems wrong
            loss = loss_fn(model(x.to(device)), y.to(device))
            #print(f"Loss before: {loss}, pid: {rank}")
            optimizer.zero_grad()
            loss.backward()
            alladd_grad(model) 
            optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss}")
    
    # test
    test_batches = load_data(train=False, bsz=200)
    with t.no_grad():
        model.eval()
        total_loss = 0 
        total = 0
        correct = 0
        for x, y in test_batches:
            x = x.to(device)
            y = y.to(device)
            total_loss += loss_fn(model(x.to(device)), y.to(device))
            y_hat = t.argmax(model(x.to(device)), dim=1)
            total += y_hat.shape[0]
            correct += t.sum(y_hat == y)
        print(f"Final Loss: {total_loss} and rank {rank} and prop correct {correct / total}")

    dist.all_reduce(t.zeros(2), op=dist.ReduceOp.SUM) # syncs processes

    if rank == 0:
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)

if __name__ == "__main__":  



    if len(sys.argv) < 3:
        local_parallelism = 2
        device = "cpu"
    else:
        local_parallelism = int(sys.argv[2])
        device = "cpu" if sys.argv[3] == "cpu" else "cuda"
    
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    mp.set_start_method("spawn") #breaks if removed
    processes = []
    for rank in range(local_parallelism): # for each process index
        p = mp.Process(target=init_process, args=(rank, local_parallelism, device))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()