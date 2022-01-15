#!/usr/bin/env python
import os
import torch.distributed as dist
import torch.multiprocessing as mp

DEVICES = [
    "cuda:0", "cuda:1", "cuda:2"
]


import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import torchtext
from typing import Optional
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
MAX_SEQ_LEN = 10
train_data, valid_data, test_data = torchtext.datasets.WikiText2('data', split=('train', 'valid', 'test'))

# train_data = ' '.join([s for s in train_data])
# train_data = tokenizer(train_data)['input_ids']
# new_train_data = []
# for i in range(len(train_data) // MAX_SEQ_LEN):
#     new_train_data.append(train_data[i*MAX_SEQ_LEN: (i+1)*MAX_SEQ_LEN])
# new_train_data[0]
                                     
class DistributedDataLoader:
    def __init__(self, rank : int, world_size : int, 
                 mini_batch_size : int, 
                 random_seed : Optional[int] = 0) -> None:
        super().__init__()
        
        self.rank = rank
        self.world_size = world_size
        self.mini_batch_size = mini_batch_size
        self.random_seed = random_seed
        self.leader_rank = 0
        self.max_seq_len = MAX_SEQ_LEN
        
        # Only load data from leader
        if self.rank == self.leader_rank:
            batch_size = mini_batch_size * world_size
            self.train_dataloader = DataLoader(train_data,
                                               batch_size=batch_size)
            self.test_dataloader = DataLoader(test_data,
                                              batch_size=batch_size)

            
    def __iter__(self):
        minibatch = torch.zeros([self.mini_batch_size, self.max_seq_len], dtype= torch.long).to(DEVICES[self.rank])
        # If leader
        if self.rank == self.leader_rank:
            # Load 1 batch for each worker
            minibatches = []
            batch = next(iter(self.train_dataloader))
            for i in range(self.world_size):
                non_padded_seqs = tokenizer(batch[i*self.mini_batch_size:(i+1)*self.mini_batch_size])['input_ids'] # List of List of Tokens(int)

                non_padded_seqs = [seq[:self.max_seq_len] for seq in non_padded_seqs]

                padded_seqs = torch.LongTensor([sent + [tokenizer(" ")["input_ids"][0]]*(self.max_seq_len-len(sent)) for sent in non_padded_seqs])
                
                minibatches.append(padded_seqs.to(DEVICES[self.rank]))
        else:
            # Not load anything
            minibatches = None
                
        # Scatter minibatches
        dist.scatter(minibatch, minibatches, src=self.leader_rank)

        yield minibatch

    def tokenize(self, batch):
        non_padded_seqs = tokenizer(batch)['input_ids'] # List of List of Tokens(int)

        non_padded_seqs = [seq[:self.max_seq_len] for seq in non_padded_seqs]

        padded_seqs = torch.LongTensor([sent + [tokenizer(" ")["input_ids"][0]]*(self.max_seq_len-len(sent)) for sent in non_padded_seqs])
        
        return padded_seqs.to(DEVICES[self.rank])



def run(rank, size):
    """ Distributed function to be implemented later. """

    ddl = DistributedDataLoader(rank, len(DEVICES), 16, random_seed = 42)

    device = DEVICES[rank] 

    ddl.__iter__()

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