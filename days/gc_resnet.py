import os
import random
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision

# import gin
import sys
import torch as t
from days.utils import *
import signal
from tqdm import tqdm
from days.gradient_compression import LowRankCompressionDistributedSGD


def load_data(train, bsz):
    if os.path.exists("cifar_tensors.pt"):
        print("using cached cifar tensor")
        data = to_batches(t.load("cifar_tensors.pt"), bsz, trim=True)
        return data
    data = torchvision.datasets.CIFAR10(
        "~/mlab/datasets/cifar10_" + ("train" if train else "test"),
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.PILToTensor(),
                torchvision.transforms.ConvertImageDtype(t.float),
                torchvision.transforms.Resize((64, 64)),
            ]
        ),
        download=True,
        train=train,
    )
    data = [t.stack([p[0] for p in data]), t.tensor([p[1] for p in data])]
    t.save(data, "cifar_tensors.pt")
    return to_batches(data, bsz, trim=True)


def load_model():
    t.random.manual_seed(0)
    return torchvision.models.resnet18()


class DistributedDataLoader:
    def __init__(
        self,
        rank,
        size,
        device,
        mini_batch_size=512,
        random_seed=0,
    ):
        self.rank = rank
        self.size = size
        self.device = device
        self.mini_batch_size = mini_batch_size
        if rank == 0:
            self.batches = load_data(train=True, bsz=mini_batch_size * size)
            self.len = t.tensor(len(self.batches)).to(device)
        else:
            self.len = t.tensor(-1).to(device)
            self.batches = None

        print("broadcast length from", self.rank)
        dist.broadcast(self.len, src=0)  # everyone gets 0's len

    # Reason to do it this way: put as much data distribution as possibe as late as possible
    # because we want all our code paths to run as soon as possible
    # so our large distributed job fails cheap
    def __iter__(self):
        for i in range(self.len):
            if self.batches is not None:
                x_b = rearrange(
                    self.batches[i][0], "(s m) ... -> s m ...", s=self.size
                ).to(self.device)
                y_b = rearrange(
                    self.batches[i][1], "(s m) ... -> s m ...", s=self.size
                ).to(self.device)
            else:
                x_b = t.zeros(
                    (self.size, self.mini_batch_size, 3, 64, 64), dtype=t.float32
                ).to(self.device)
                y_b = t.zeros((self.size, self.mini_batch_size), dtype=t.int64).to(
                    self.device
                )
            dist.broadcast(x_b, src=0, async_op=True).wait()
            dist.broadcast(y_b, src=0, async_op=True).wait()
            yield [x_b[self.rank], y_b[self.rank]]


def init_process(rank, size, device, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=rank, world_size=size)

    # Comment this to run on one GPU
    if device == "cuda":
        device += ":" + str(rank)

    print("inited process group", rank, " on device ", device)

    model = load_model()
    model.train()
    model.to(device)
    # optimizer = LowRankCompressionDistributedSGD(
    #     model.parameters(), lr=0.005, compression_rank=4, momentum=0.8
    # )
    optimizer = LowRankCompressionDistributedSGD(
        model.parameters(), lr=0.005, dist_size=size, compression_rank=2, momentum=0.0
    )
    dataloader = DistributedDataLoader(rank=rank, size=size, device=device)
    loss_fn = t.nn.CrossEntropyLoss(reduction="mean")

    # train
    for epoch in range(40):
        for batch_num, (x, y) in enumerate(tqdm(dataloader)):
            # NOTE: look out for reduction == mean instead, whichj seems wrong
            loss = loss_fn(model(x.to(device)), y.to(device))
            # print(f"Loss before: {loss}, pid: {rank}")
            optimizer.zero_grad()
            loss.backward()
            if False:
                with t.no_grad():
                    reductions = []
                    for param in model.parameters():
                        if isinstance(param.grad, t.Tensor):
                            reductions.append(
                                dist.all_reduce(param.grad, async_op=True)
                            )
                            param.grad /= size
                    for reduction in reductions:
                        reduction.wait()
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
        print(
            f"Final Loss: {total_loss} and rank {rank} and prop correct {correct / total}"
        )

    dist.all_reduce(t.zeros(2).to(device), op=dist.ReduceOp.SUM)  # syncs processes

    if rank == 0:
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


if __name__ == "__main__":

    if len(sys.argv) < 3:
        local_parallelism = 2
        device = "cpu"
    else:
        local_parallelism = int(sys.argv[1])
        device = "cpu" if sys.argv[2] == "cpu" else "cuda"

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    mp.set_start_method("spawn")  # breaks if removed
    processes = []
    for rank in range(local_parallelism):  # for each process index
        p = mp.Process(target=init_process, args=(rank, local_parallelism, device))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
