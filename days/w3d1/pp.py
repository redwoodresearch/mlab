from comet_ml import Experiment
import os
from re import L
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional
from days.utils import to_batches
import json
import transformers
from time import ctime, time

DEVICES=[f"cuda:{i}" for i in range(4)]
TRAINING_DATA_X_PATH=""
TRAINING_DATA_SIZE=25_000
SEQUENCE_LENGTH=1024
BATCH_SIZE=25
COMPONENT_PATHS=[f"component{i}" for i in range(4)]

"""
use t.mp to start up N processes and t.dist to create a process group. You can (and should) copy over your code. In each process, t.load one of your sub-models that you saved earlier onto the appropriate device and initialize a plain SGD optimizer (to save memory) on your model’s params. Also, have your rank-0 process load and batch the IMDB sentiment classification data.

In pipeline parallelism, we often send data between two nodes. Pyt Nccl doesn't support dist.send though (I think raw Nccl does), so we're going to have to create a new process group with dist.new_group for every pair of processes that ever need to communicate, and then broadcast in those groups.
"""

class DistributedDataLoader:
    def __init__(self, rank : int, world_size : int, mini_batch_size : int, group, seq_length=1024, random_seed : Optional[int] = 0) -> None:
        self.device = "cuda:" + str(rank + 3)
        self.group = group
        self.rank = rank
        self.world_size = world_size
        self.batch_num = 0
        self.mini_batch_size = mini_batch_size
        print("Num batches on", self.device)
        self.num_batches = t.tensor(0).to(self.device)
        self.seq_length = seq_length
        self.time_save = None

        if rank == 0:
            print(f"Init dataset")
            self.batch_size = mini_batch_size * world_size
            self.batches = get_processed_dataset(
                fname="/home/ubuntu/lw_corpus.json",
                num_minibatches=world_size, 
                seqs_per_minibatch=mini_batch_size, 
                seq_length=seq_length,
                truncate=-1,
            )

            self.batch_index = 0
            self.num_batches = t.tensor(len(self.batches)).to(self.device)

            print("NUMBER OF BATCHES:", self.num_batches)

        dist.broadcast(self.num_batches, src=0, group=group)

    def __iter__(self):
        for batch_num in range(self.num_batches):
            current_batch = t.zeros(self.world_size, self.mini_batch_size, 1024).long().to(self.device)

            if self.rank == 0:
                assert current_batch.shape == self.batches[self.batch_num].shape
                current_batch = self.batches[self.batch_num].to(self.device)
                self.batch_num += 1
                
                # start tracking stuff
                if self.batch_num % 100 == 0 or self.batch_num < 100:
                    if self.time_save is not None: print("#"*50, f"\nBATCH {self.batch_num} COMPLETED. TOOK {time() - self.time_save}")
                    self.time_save = time()
                # end tracking stuff

            dist.broadcast(current_batch, src=0, group=self.group)
            yield current_batch[self.rank]

def run(rank, size):
    group = dist.new_group(list(range(size)))
    device = f"cuda:{rank}"
    print(f"Beginning loop for rank {rank}")
    training_data = t.zeros(TRAINING_DATA_SIZE, SEQUENCE_LENGTH).cuda()

    if rank == 0:
        """
        Load and microbatch (?) the training data 
        """
        
        training_data = t.load(FPATH).cuda()
        assert training_data.shape == t.Size([TRAINING_DATA_SIZE, SEQUENCE_LENGTH]), f"{training_data.shape}"    
        
        assert TRAINING_DATA_SIZE % BATCH_SIZE == 0
        num_batches = TRAINING_DATA_SIZE // BATCH_SIZE

        training_data = training_data.reshape(num_batches, BATCH_SIZE, SEQUENCE_LENGTH)
        print("Finished loaded training data")

    print(f"Begin broadcast rank {rank}")
    dist.broadcast(training_data, src=0, group=group)
    print(f"End broadcast rank {rank}")

    component = t.load(COMPONENT_PATHS[rank])
    optimizer = t.optim.SGD(component.parameters())

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '5460'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()