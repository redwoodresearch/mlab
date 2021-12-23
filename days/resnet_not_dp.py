import enum
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
        mini_batch_size=20,
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
            if self.batches is not None:
                x, y = self.batches[i]

                mini_batches = t.stack(to_batches(self.batches[i], self.mini_batch_size, trim=False))
                if mini_batches[0] is None:
                    print("something wrong up here, size is ", self.size)
                if len(mini_batches) != self.size:
                    print("Found it! ", self.rank, self.size, len(mini_batches))
                print("Not none: ", i, self.rank, len(mini_batches), self.size, mini_batches[0])
                
            else:
                if self.rank == 0:
                    print("Somehow the rank 0 process got no batches, size is ", self.size)
                mini_batches = [None for _ in range(self.size)]
                mini_batches = t.zeros((self.size, ))
                print("None: ", i, self.rank, len(mini_batches), self.size, mini_batches[i])
            try:
                dist.broadcast_object_list(mini_batches, src=0)
            except RuntimeError:
                print(i, self.rank, len(self.batches)if self.batches is not None else None, len(mini_batches) if mini_batches is not None else None)
                raise AssertionError(":(")
            my_batch = mini_batches[self.rank]
            yield my_batch






if __name__ == "__main__":
    #print(t.cuda.get_device_capability("cuda:7"))    

    local_parallelism = 2 if len(sys.argv) < 3 else int(sys.argv[2])
    device = "cpu" if sys.argv[3] == "cpu" else "cuda"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"


    cifar_train = torchvision.datasets.CIFAR10("~/mlab/datasets/cifar10_train",
                            transform= torchvision.transforms.Compose([
                                torchvision.transforms.PILToTensor(),
                                torchvision.transforms.ConvertImageDtype(t.float),
                                torchvision.transforms.Resize((64, 64))]),
                            download=False, 
                            train=True)

    data = [t.stack([p[0] for p in cifar_train]), t.tensor([p[1] for p in cifar_train])]
    mini_batch_size = 1000
    size = local_parallelism
    batches = to_batches(data, mini_batch_size * size, trim=True)

    # init model, optim, data
    t.random.manual_seed(0)
    model = torchvision.models.resnet18()
    model.train()
    model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = t.nn.CrossEntropyLoss(reduction="sum")

    for epoch in range(5):
        for batch_no, batch in enumerate(tqdm(batches)):
            mini_batches = to_batches(batch, mini_batch_size, trim=False)

            optimizer.zero_grad()

            for x, y in mini_batches:
                loss = loss_fn(model(x.to(device)), y.to(device))
                loss.backward()
            optimizer.step()
        print("Epoch: ", epoch, " Loss: ", loss)
    
    cifar_test = torchvision.datasets.CIFAR10("~/mlab/datasets/cifar10_test", 
                        transform=torchvision.transforms.Compose([
                                torchvision.transforms.PILToTensor(),
                                torchvision.transforms.ConvertImageDtype(t.float),
                                torchvision.transforms.Resize((64, 64))]), download=False, train=False)
    
    data = [t.stack([p[0] for p in cifar_test]), t.tensor([p[1] for p in cifar_test])]
    test_batches = to_batches(data, 1000, trim=True)
    
    total_loss = 0
    for x, y in test_batches:
        total_loss += loss_fn(model(x.to(device)), y.to(device))
    print(f"Final Loss: {total_loss}")



