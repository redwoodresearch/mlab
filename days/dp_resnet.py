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
            # load and shuffle x and y
            cifar_train = torchvision.datasets.CIFAR10("~/mlab/datasets/cifar10_train",
                            transform= torchvision.transforms.Compose([
                                torchvision.transforms.PILToTensor(),
                                torchvision.transforms.ConvertImageDtype(t.float),
                                torchvision.transforms.Resize((64, 64))]),
                            download=False, 
                            train=True)

            data = [t.stack([p[0] for p in cifar_train]), t.tensor([p[1] for p in cifar_train])]
            self.batches = to_batches(data, mini_batch_size * size, trim=True)

            #self.batches = t.utils.data.DataLoader(cifar_train, batch_size = mini_batch_size * size, drop_last=True)
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
            if self.batches is not None:
                #print(x.shape, y.shape)
                #x_y = [rearrange(tensor, "(s m) ... -> s m ...", s = self.size) for tensor in self.batches[i]]
                op_x = dist.scatter(x_mb, scatter_list=to_batches(self.batches[i][0], self.mini_batch_size) if self.rank == 0 else None, src = 0)
                op_y = dist.scatter(y_mb, scatter_list=to_batches(self.batches[i][1], self.mini_batch_size, trim=False), src = 0)
                
            else:
                
                # x_mb = t.zeros((self.size, self.mini_batch_size, 3, 64, 64), dtype=t.float32)
                # y_mb = t.zeros((self.size, self.mini_batch_size), dtype=t.int64)
                op_x = dist.scatter(x_mb, src = 0)
                op_y = dist.scatter(y_mb, src = 0)
                #x_y = [x_mb, y_mb]
            
            mini_batch_ops = [dist.scatter(tensor, src=0, async_op=True) for tensor in x_y]
            for op in mini_batch_ops:
                op.wait()
            my_batch = (x_y[0][self.rank], x_y[1][self.rank])
            yield my_batch
    def __iter__(self):
        for i in range(self.len):
            x_mb = t.zeros((self.mini_batch_size, 3, 64, 64), dtype=t.float32)
            y_mb = t.zeros((self.mini_batch_size), dtype=t.int64)
            op_x = dist.scatter(x_mb, scatter_list=to_batches(self.batches[i][0], self.mini_batch_size) if self.batches else None, src = 0)
            op_y = dist.scatter(y_mb, scatter_list=to_batches(self.batches[i][1], self.mini_batch_size) if self.batches else None, src = 0)
            op_x.wait()
            op_y.wait()
            yield x_mb, y_mb


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

    #print("test:", os.environ["test"])
    

    # init model, optim, data
    if device == "cuda":
        device += ":" + str(rank)

    print("inited process group", rank, " on device ", device)
    t.random.manual_seed(0)
    model = torchvision.models.resnet18()
    model.train()
    model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=0.005)
    dataloader = DistributedDataLoader(rank=rank, size=size)
    loss_fn = t.nn.CrossEntropyLoss(reduction="sum")

    # train
    for epoch in range(40):
        for batch_num, (x, y) in enumerate(tqdm(dataloader)):
            # print("batch", batch)
            # NOTE: look out for reduction = mean instead
            #print(x.shape, y.shape)
            loss = loss_fn(model(x.to(device)), y.to(device))
            #print(f"Loss before: {loss}, pid: {rank}")
            optimizer.zero_grad()
            loss.backward()
            alladd_grad(model) # broadcast gradients
            optimizer.step()
            
            #print(rank, "loss", loss.cpu().detach().numpy())
            #print(rank, batch_num)
        print(f"Epoch: {epoch}, Loss: {loss}")
    # print(rank, "done training")
    
    cifar_test = torchvision.datasets.CIFAR10("~/mlab/datasets/cifar10_test", 
                        transform=torchvision.transforms.Compose([
                                torchvision.transforms.PILToTensor(),
                                torchvision.transforms.ConvertImageDtype(t.float),
                                torchvision.transforms.Resize((64, 64))]), download=False, train=False)
    
    # print('ground truth', cifar_test[0][1], cifar_test[1][1])
    # assert False
    model.eval()
    data = [t.stack([p[0] for p in cifar_test]), t.Tensor([p[1] for p in cifar_test]).to(t.int64)]
    test_batches = to_batches(data, 200, trim=True)
    
    with t.no_grad():
        total_loss = 0 #CAREFUL WITH THIS!!!
        total = 0
        correct = 0
        for x, y in test_batches:
            #print(x.shape, y.shape, y_hat.shape, y_hat.dtype)
            x = x.to(device)
            y = y.to(device)
            total_loss += loss_fn(model(x.to(device)), y.to(device))
            y_hat = t.argmax(model(x.to(device)), dim=1)
            total += y_hat.shape[0]
            correct += t.sum(y_hat == y)

    print(f"Final Loss: {total_loss} and rank {rank} and prop correct {correct / total}")

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

    mp.set_start_method("spawn") #breaks if removed
    for rank in range(local_parallelism): # for each process index
        p = mp.Process(target=init_process, args=(rank, local_parallelism, device))
        p.start()


