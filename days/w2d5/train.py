#!/usr/bin/env python3
import json
import os
import sys
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import transformers

devices = [f'cuda:{i}' for i in [2,3]]

def saved_tokenization_path_for_corpus(corpus_path):
    return f'{os.path.basename(corpus_path)}.saved-tokens'

def json_file_corpus_to_batches(json_path, batch_size, max_seq_len, device):
    all_tokens = []
    saved_path = saved_tokenization_path_for_corpus(corpus_path=json_path)
    try:
        with open(saved_path) as saved:
            all_tokens = json.load(fp=saved)
    except FileNotFoundError:
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        with open(json_path) as fp:
            json_contents = json.load(fp=fp)
        all_tokens = [
            token
            for article in tqdm(json_contents)
            for token in tokenizer(article["text"]).input_ids + [tokenizer.eos_token_id]
        ]
        with open(saved_path, 'w') as saved:
            json.dump(all_tokens, saved)
    num_batches = len(all_tokens) // (batch_size * max_seq_len) 
    all_tokens = all_tokens[:num_batches * batch_size * max_seq_len]
    batches = t.tensor(all_tokens, device=device, dtype=t.long).view(num_batches, batch_size, max_seq_len)
    return batches
        
class DistributedDataLoader:
    def __init__(
        self, rank : int, world_size : int, mini_batch_size : int, random_seed=0
    ):
        self.rank = rank
        self.device = devices[rank]
        self.world_size = world_size
        self.mini_batch_size = mini_batch_size
        self.leader_process = 0
        self.is_leader = self.rank == self.leader_process
        # self.process_group = list(range(self.world_size))
        self.num_batches = t.tensor([0], device=self.device)
        if self.is_leader:
            self.data = self.get_data()
            self.num_batches[0] = len(self.data)
        dist.broadcast(self.num_batches, src=self.leader_process)
        self.num_batches = self.num_batches.item()
        
    def get_data(self):
        json_path = sys.argv[1] if len(sys.argv) == 2 else "/home/ubuntu/lw_corpus.json"
        
        return json_file_corpus_to_batches(
            json_path=json_path,
            batch_size=self.mini_batch_size * self.world_size,
            max_seq_len=1024,
            device=self.device,
        )
            
    def agree_on_shapes(self, shape=None):
        rank_of_shape = t.zeros(1, device=self.device, dtype=t.long)
        if shape is not None:
            rank_of_shape[0] = len(shape)
        dist.broadcast(rank_of_shape, src=self.leader_process)
        shape_dst = t.zeros(rank_of_shape.item(), device=self.device, dtype=t.long)
        if shape is not None:
            shape_dst[:] = t.tensor(shape, device=self.device, dtype=t.long)
        dist.broadcast(shape_dst, src=self.leader_process)
        return shape_dst
        
    def __iter__(self):
        if self.is_leader:
            agreed_on_shapes = False
            # https://pytorch.org/docs/stable/distributed.html#torch.distributed.scatter
            for batch in tqdm(self.data):
                if not agreed_on_shapes:
                    self.agree_on_shapes(batch.shape)
                    agreed_on_shapes = True
                # print(f'leader thinks shape is {batch.shape}, batch is {batch}, dtype = {batch.dtype}')
                dist.broadcast(batch, src=self.leader_process)
                yield batch[self.rank * self.mini_batch_size : (self.rank + 1) * self.mini_batch_size]
        else:
            shape = self.agree_on_shapes()
            batch = t.zeros(tuple(shape), device=self.device, dtype=t.long)
            for _ in range(self.num_batches):
                # print(f'batch before broadcast is {batch}, dtype = {batch.dtype}')
                dist.broadcast(batch, src=self.leader_process)
                # print(f'non-leader thinks batch is {batch}, dtype = {batch.dtype}')
                yield batch[self.rank * self.mini_batch_size : (self.rank + 1) * self.mini_batch_size]
                
def run(rank, size):
    gpt2 = transformers.GPT2LMHeadModel.from_pretrained('gpt2').to(device=devices[rank])
    dataloader = DistributedDataLoader(
        rank=rank,
        world_size=size,
        mini_batch_size=8,
    )
    train_loop(
        model=gpt2,
        dataloader=dataloader,
        rank=rank,
    )

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '32647'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def main():
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
def train_loop(model, dataloader, rank):
    optim = t.optim.Adam(model.parameters())
    model.train()
    
    for i, batch in enumerate(dataloader):
        optim.zero_grad()
        output = model(input_ids=batch, labels=batch)
        # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.CausalLMOutput
        output.loss.backward()
        for param in model.parameters():
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM)
        optim.step()
        if i % 1000 == 999:
            t.save(model, f"gpt_trained.{i // 1000:03}.{rank}.zip")
    
    t.save(model, f"gpt_trained.final.{rank}.zip")
        
if __name__ == "__main__":
    main()
    