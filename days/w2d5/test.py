e:
            transformers.GPT2LMHeadModel.from_pretrained("gpt2")
            all_data = json.load(open("/home/ubuntu/lw_corpus.json"))
            self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
            
            encode = lambda post : self.tokenizer.encode(post["text"] + "<|endoftext|>", return_tensors="pt")[0]
            self.texts = t.cat([encode(post) for post in all_data])
            self.texts = self.texts.cuda(device=DEVICES[rank])

            tokens_in_batch = self.mini_batch_size * self.world_size * self.seq_length
            self.batches = len(self.texts) // tokens_in_batch
            self.batched_texts = t.reshape(self.texts[:tokens_in_batch*self.batches],
                (self.batches, self.world_size, self.mini_batch_size, self.seq_length))
            self.batched_texts = self.batched_texts.to(dtype=t.int64, device=DEVICES[rank])
            self.batches = t.tensor(self.batches, device = DEVICES[rank])
        dist.broadcast(self.batches, 0)
        self.batches = self.batches.item()

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr >= self.batches:
            raise StopIteration
        x = self.batched_texts[self.ptr] if self.in_charge else t.zeros((self.world_size,self.mini_batch_size,self.seq_length), dtype=t.int64, device=DEVICES[self.rank])
        self.ptr += 1
        dist.broadcast(x, 0)
        x = x[self.rank]
        return x



def start(rank, size):
    """ Distributed function to be implemented later. """
    ddl = DistributedDataLoader(rank, size, 3)
    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(device=DEVICES[rank])
    if rank == 0:
        experiment = Experiment(
            api_key="EVwneUg4V62GWa4Vpu1JcjzpF",
            project_name="gpt_lesswrong",
            workspace="nixgd",
        )

    opt = t.optim.Adam(model.parameters(), lr = 1e-6)

    try:
        for epoch in range(4):
            i = 0
            if rank == 0:
                tbar = tqdm(total=ddl.batches)
            for batch in iter(ddl):
                opt.zero_grad()
                logits = model(batch).logits.reshape([batch.shape[0]*batch.shape[1], -1])
                batch = batch.reshape(-1)
                loss = t.nn.functional.cross_entropy(logits, batch)
                loss.backward()
                for p in model.parameters():
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                if rank == 0:
                    tbar.update(1)
                loss = loss.item()
                if rank == 0:
                    experiment.log_metric("loss", loss)
                    tbar.set_postfix({'loss': loss})
                opt.step()
                # broadcast to update model params
                i += 1
                if rank == 0 & (i%98 == 0):
                    t.save(model.state_dict(), "model.save")
                    experiment.log_model("finetuned model", "model.save")
    except Exception:
        pass
    if rank == 0:
        t.save(model.state_dict(), "model.save")
        experiment.log_model("finetuned model", "model.save")
        
def target(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = len(DEVICES)
    processes = []
    mp.set_start_method("spawn")
    for rank in range(3):
        p = mp.Process(target=target, args=(rank, size, start))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
#!/usr/bin/env python
from comet_ml import Experiment
import os
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional
import json
import transformers
from tqdm import tqdm


DEVICES=[t.device('cuda:1'),t.device('cuda:2'),t.device('cuda:3')]#,t.device('cuda:4'),t.device('cuda:5'),t.device('cuda:6')]
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '2900'


class DistributedDataLoader:
    def __init__(self, rank : int, world_size : int, mini_batch_size : int, random_seed : Optional[int] = 0):
        self.in_charge = rank == 0
        self.rank = rank
        self.mini_batch_size = mini_batch_size
        self.world_size = world_size
        self.ptr = 0
        self.batches = t.tensor(0, device = DEVICES[rank])
        self.seq_length = 1024
        if self.in_charge:
            transformers.GPT2LMHeadModel.from_pretrained("gpt2")
            all_data = json.load(open("/home/ubuntu/lw_corpus.json"))
            self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
            
            encode = lambda post : self.tokenizer.encode(post["text"] + "<|endoftext|>", return_tensors="pt")[0]
            self.texts = t.cat([encode(post) for post in all_data])
            self.texts = self.texts.cuda(device=DEVICES[rank])

            tokens_in_batch = self.mini_batch_size * self.world_size * self.seq_length
            self.batches = len(self.texts) // tokens_in_batch
            self.batched_texts = t.reshape(self.texts[:tokens_in_batch*self.batches],
                (self.batches, self.world_size, self.mini_batch_size, self.seq_length))
            self.batched_texts = self.batched_texts.to(dtype=t.int64, device=DEVICES[rank])
            self.batches = t.tensor(self.batches, device = DEVICES[rank])
        dist.broadcast(self.batches, 0)
        self.batches = self.batches.item()

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr >= self.batches:
            raise StopIteration
        x = self.batched_texts[self.ptr] if self.in_charge else t.zeros((self.world_size,self.mini_batch_size,self.seq_length), dtype=t.int64, device=DEVICES[self.rank])
        self.ptr += 1
        dist.broadcast(x, 0)
        x = x[self.rank]
        return x



def start(rank, size):
    """ Distributed function to be implemented later. """
    ddl = DistributedDataLoader(rank, size, 3)
    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(device=DEVICES[rank])
    if rank == 0:
        experiment = Experiment(
            api_key="EVwneUg4V62GWa4Vpu1JcjzpF",
            project_name="gpt_lesswrong",
            workspace="nixgd",
        )

    opt = t.optim.Adam(model.parameters(), lr = 1e-6)

    try:
        for epoch in range(4):
            i = 0
            if rank == 0:
                tbar = tqdm(total=ddl.batches)
            for batch in iter(ddl):
                opt.zero_grad()
                logits = model(batch).logits.reshape([batch.shape[0]*batch.shape[1], -1])
                batch = batch.reshape(-1)
                loss = t.nn.functional.cross_entropy(logits, batch)
                loss.backward()
                for p in model.parameters():
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                if rank == 0:
                    tbar.update(1)
                loss = loss.item()
                if rank == 0:
                    experiment.log_metric("loss", loss)
                    tbar.set_postfix({'loss': loss})
                opt.step()
                # broadcast to update model params
                i += 1
                if rank == 0 & (i%98 == 0):
                    t.save(model.state_dict(), "model.save")
                    experiment.log_model("finetuned model", "model.save")
    except Exception:
        pass
    if rank == 0:
        t.save(model.state_dict(), "model.save")
        experiment.log_model("finetuned model", "model.save")
        
def target(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = len(DEVICES)
    processes = []
    mp.set_start_method("spawn")
    for rank in range(3):
        p = mp.Process(target=target, args=(rank, size, start))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
