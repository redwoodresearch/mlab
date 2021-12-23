import os
import random
import torch.distributed as dist
import torch.multiprocessing as mp
import gin
import sys
from utils import import_object_from_qualified_name
import torch as t
import numpy as np
from sklearn.datasets import make_moons
import torchvision
from utils import *
import os
import signal
from tqdm import tqdm

DEVICE = "cpu"


def load_data():
    cifar_train = torchvision.datasets.CIFAR10("~/mlab/datasets/cifar10_train",
                            transform= torchvision.transforms.Compose([
                                torchvision.transforms.PILToTensor(),
                                torchvision.transforms.ConvertImageDtype(t.float),
                                torchvision.transforms.Resize((64, 64))]),
                            download=False, 
                            train=True)

    #print("LEN??", len(cifar_train))
    #assert False
    return t.stack([p[0] for p in cifar_train]), t.tensor([p[1] for p in cifar_train])


def init_model():
    t.random.manual_seed(0)
    model = torchvision.models.resnet18()
    return model


@gin.configurable
class DistributedDataLoader:
    def __init__(
        self,
        rank,
        size,
        data_fn="days.dataparallel_resnet_tao.load_data",
        mini_batch_size=1000,
        random_seed=0,
    ):
        self.rank = rank
        self.size = size
        if rank == 0:
            self.data_tensors = list(import_object_from_qualified_name(data_fn)())
            t.manual_seed(random_seed)
            perm = t.randperm(self.data_tensors[0].shape[0])
            for i, ten in enumerate(self.data_tensors):
                self.data_tensors[i] = ten[perm]

            self.batches = [
                to_batches(batch, mini_batch_size, trim=True)
                for batch in to_batches(self.data_tensors, mini_batch_size * size)
            ]

            # should be bignum, size, 2, 1000
            # print("DEMO: ")
            # print(len(self.data_tensors))
            # print(len(self.data_tensors[0]))
            # print(len(self.data_tensors[1]))
            # print(len(self.batches))
            # print(len(self.batches[0]))
            # print(len(self.batches[0][0]))
            # print(len(self.batches[0][0][0]))

            self.len = len(self.batches)
        else:
            self.len = -1
            self.batches = None
        blst = [self.len]
        print("broadcast length from", self.rank)
        dist.broadcast_object_list(blst, src=0)
        self.len = blst[0]

    def __len__(self):
        return self.len

    def __iter__(self):
        if self.batches is not None:
            for mini_batches in self.batches:
                dist.broadcast_object_list(mini_batches, src=0)
                my_batch = mini_batches[self.rank]
                yield my_batch
        else:
            for _ in range(self.len):
                mini_batches = [None for _ in range(self.size)]
                dist.broadcast_object_list(mini_batches, src=0)
                my_batch = mini_batches[self.rank]
                yield my_batch


def alladd_grad(model):
    reduce_ops = [
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        for param in model.parameters()
    ]
    for op in reduce_ops:
        op.wait()


def killgroup():
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


@gin.configurable()
def run(
    rank,
    size,
    model_init_fn_name="days.dataparallel.init_model",
):
    print("i'm rank", rank)
    # device = "cuda:" + str(rank)
    model = import_object_from_qualified_name(model_init_fn_name)()
    model.train()
    model.to(DEVICE)
    print()
    optimizer = t.optim.Adam(model.parameters(), lr=1e-4)
    dataloader = DistributedDataLoader(rank=rank, size=size)
    loss_fn = t.nn.CrossEntropyLoss(reduction="sum")

    for batch_num, batch in enumerate(tqdm(dataloader)):
        # print("batch", batch)
        out = model(batch[0].to(DEVICE))
        loss = loss_fn(out, batch[1].to(DEVICE))
        loss.backward()
        alladd_grad(model)
        optimizer.step()
        optimizer.zero_grad()
        # print(rank, "loss", loss.cpu().detach().numpy())
        # print(rank, batch_num)
    print(rank, "done training")

    cifar_test = torchvision.datasets.CIFAR10("~/mlab/datasets/cifar10_test", 
                        transform=torchvision.transforms.Compose([
                                torchvision.transforms.PILToTensor(),
                                torchvision.transforms.ConvertImageDtype(t.float),
                                torchvision.transforms.Resize((64, 64))]), download=False, train=False)
    
    data = [t.Tensor([p[0] for p in cifar_test]), t.Tensor([p[1] for p in cifar_test])]
    test_batches = to_batches(data, 1000, trim=True)

    total_loss = 0
    for x, y in test_batches:
        total_loss += loss_fn(model(x.to(DEVICE)), y.to(DEVICE))
    print(f"Final Loss: {total_loss} and rank {rank}")

    dist.all_reduce(t.zeros(2), op=dist.ReduceOp.SUM)

    if rank == 0:
        killgroup()


@gin.configurable
def init_process(rank, size, run, device, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    if device == "cuda":
        global DEVICE
        DEVICE = "cuda:" + str(rank)
    dist.init_process_group(backend, rank=rank, world_size=size)
    print("inited process group", rank)

    run(rank, size)


@gin.configurable
def create_processes(
    local_parallelism=2,
    device="cpu",
):
    # raise AssertionError(":)")
    processes = []
    mp.set_start_method("spawn")
    for rank in range(local_parallelism):
        p = mp.Process(target=init_process, args=(rank, local_parallelism, run, device))
        p.start()
        processes.append(p)
    # pytorch join requires you to join in order of completion!???


if __name__ == "__main__":
    local_parallelism = 2 if len(sys.argv) < 3 else int(sys.argv[2])
    device = "cpu" if sys.argv[3] == "cpu" else "cuda"
    if sys.argv[1] == "master":
        # gin.parse_config_file(sys.argv[2])
        create_processes(local_parallelism, device)
    else:
        tmpfilename = ".ginny_weasly"
        with open(tmpfilename, "w") as f:
            f.write(sys.argv[1])
        # gin.parse_config_file(tmpfilename)
        init_process()
