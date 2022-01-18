import os
import torch as t
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch.distributed as dist
import torchtext
import random

DEVICES = [t.device("cuda:0"), t.device("cuda:1"), t.device("cuda:2"), t.device("cuda:3")]

class Extractor(t.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input):
        return self.model(input)[0]

def run(
    rank,
    world_size,
    sharded_optimizer=False,
):
    groups = [dist.new_group([i, i+1]) for i in range(world_size-1)] + [dist.new_group([0, world_size-1])]
        
    batch_size = 2
    seq_len = 128
    hidden_size = 4096
    print("i'm rank", rank)
    # device = "cuda:" + str(rank)
    model = t.load(f"model_{rank}.pt")
    model.train()
    model.to(DEVICES[rank])

    optim = t.optim.SGD(model.parameters(), lr=1e-4)

    tokens = None
    labels = None
    batches = t.tensor(0)
    if rank == 0:
        random.seed(123)
        data_train, data_test = torchtext.datasets.IMDB(root='.data', split=('train', 'test'))
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        tokenizer.pad_token = tokenizer.eos_token
        data_train = [x for x in data_train]
        random.shuffle(data_train)
        # data_train = data_train[:100]
        # data_train = [x for x in data_train]
        text = [data[1] for data in data_train]
        tokens = t.tensor([
            tokenizer.encode(p, truncation=True, padding='max_length', max_length=seq_len)
            for p in text
        ])
        tokens = tokens[:(tokens.shape[-2]//(world_size*batch_size))*world_size*batch_size]
        tokens = t.reshape(tokens, (-1, world_size, batch_size, seq_len)).to(DEVICES[rank])
        labels = t.tensor([int('pos' == data[0]) for data in data_train]).to(DEVICES[rank], dtype=t.int64)
        labels = labels[:(labels.shape[-1]//(world_size*batch_size))*world_size*batch_size]
        labels = t.reshape(labels, (-1, world_size, batch_size))
        batches = t.tensor(labels.shape[0])
    batches = batches.to(DEVICES[rank])
    dist.broadcast(batches, 0)
    batches = batches.item()

    if rank == world_size-1:
        labels = t.zeros(batches, world_size, batch_size, dtype=t.int64).to(DEVICES[rank])

    if rank == 0 or rank == world_size-1:
        dist.broadcast(labels, 0, groups[-1])

    in_data = None
    data = None
    for i in range(batches):
        optim.zero_grad()
        data = t.zeros(batch_size, seq_len, hidden_size).to(DEVICES[rank])
        if rank == 0:
            data = tokens[i, 0]
            in_data = data
            data = model(in_data)
        if rank in [0, 1]:
            dist.broadcast(data, 0, group=groups[0])
        for j in range(1, world_size - 1):
            if j == rank:
                in_data = data
                in_data.requires_grad = True
                data = model(in_data)
            if rank in [j, j+1]:
                dist.broadcast(data, j, group=groups[j])
        if rank == world_size - 1:
            in_data = data
            in_data.requires_grad = True
            data = model(in_data)
            loss = t.nn.functional.cross_entropy(data[:,:,-1], labels[i, 0])
            print(loss)
            loss.backward()
            optim.step()
        
        for j in range(world_size-1, 0, -1):
            grad = t.zeros(batch_size, seq_len, hidden_size).to(DEVICES[rank])
            if rank == j:
                grad = in_data.grad
            if rank in [j, j-1]:
                dist.broadcast(grad, j, group=groups[j-1])
            if rank == j-1:
                data.backward(gradient=grad)
                optim.step()
        



    # dist.broadcast(tokens, 0)
    # for i in range(batches):
    #     data = t.zeros(batch_size, seq_len, hidden_size).to(DEVICES[rank])
    #     if rank == 0:
    #         data = tokens[i, 0]
    #     for j in range(world_size):
    #         if j == rank:
    #             data = model(data)
    #         if j < world_size - 1:
    #             print("before b2")
    #             dist.broadcast(data, j)
    #             print(rank, "after b2")
    #             # print(data[0,0,0])
    #         elif rank == world_size-1:
    #             print("before loss")
    #             loss = t.nn.functional.cross_entropy(data[:,:,-1], labels[i, 0])
    #             print("after loss")
    #             print(loss.cpu())
    #             # loss.backward()



def init_process(
    rank, size, run, device, backend="nccl"
):  # gloo is algo for sharing gradients. nccl better?
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "2800"  # make the master available for mutual contact
    if device == "cuda":
        global DEVICE
        DEVICE = "cuda:" + str(DEVICES[rank])
    dist.init_process_group(backend, rank=rank, world_size=size)
    print("inited process group", rank)

    run(rank, size)

def create_processes(
    local_parallelism=len(DEVICES),
    device="cpu",
):
    processes = []
    mp.set_start_method("spawn")
    for rank in range(local_parallelism):  # process index = rank
        device = DEVICES[rank]
        p = mp.Process(target=init_process, args=(rank, local_parallelism, run, device))
        p.start()
        processes.append(p)
    # pytorch join requires you to join in order of completion!???


if __name__ == "__main__":
    # gin.parse_config_file(sys.argv[1])
    create_processes()
