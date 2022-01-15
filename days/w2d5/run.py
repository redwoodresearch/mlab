#!/usr/bin/env python
import json
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers
from einops import rearrange
import pickle

DEVICES=["cuda:3","cuda:4", "cuda:5"]
LEADER_RANK = 0
OPTIMIZER_STATE_SHARDING = False
SEQ_LEN = 1024

MODEL_FILENAME = "/home/ubuntu/alwin_tom_mlab/model.pt"
TOKENS_FILENAME = "/home/ubuntu/alwin_tom_mlab/lw_corpus_tokens_packed.pickle"

def packed_tokens():
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

    with open(LW_FILENAME) as json_file:
        # keys: id (of post), user_id, karma, af (alignment forum y/n), text
        json_data = json.load(json_file)

    texts = tokenizer([post["text"] for post in json_data]).input_ids
    # flatten texts
    texts = [token for text in texts for token in (text + [tokenizer.eos_token_id])]

    # pad text out to be a multiple of SEQ_LEN
    padding_amount = SEQ_LEN - (((len(texts) - 1) % SEQ_LEN) + 1)
    texts = texts + [tokenizer.eos_token_id] * padding_amount

    text_tensor = torch.tensor(texts, dtype=torch.long)
    return text_tensor.view(-1, SEQ_LEN)
    # shape: [97668, 1024]

class DistributedDataLoader:
    def __init__(self, rank, world_size, device, mini_batch_size, random_seed = 0):
        torch.manual_seed(random_seed)

        self.rank = rank
        self.world_size = world_size
        self.mini_batch_size = mini_batch_size
        self.device = device
        self.iter_num = 0
        self.iters_in_epoch = torch.zeros(1, dtype=torch.long, device=device)

        if rank == LEADER_RANK:
            # Instructions: If called from the process in charge, __init__
            # should load in the training data. Additionally, you may want to
            # convert the training data into batches (using either a DataLoader
            # or our handy utils.to_batches)

            # leader has to tokenize everything because torch.scatter
            # won't work on strings.

            if os.path.isfile(TOKENS_FILENAME):
                texts = pickle.load(open(TOKENS_FILENAME, "rb"))
            else:
                texts = packed_tokens()
                pickle.dump(texts, open(TOKENS_FILENAME, "wb"))

            train_loader = torch.utils.data.DataLoader(
                texts, 
                batch_size=world_size * mini_batch_size,
                shuffle=True, 
                # don't want to think about divisibility issues when
                # distributing minibatches
                drop_last=True,
                num_workers=2,
            )
            self.iters_in_epoch[0] = len(train_loader)
            self.loader = train_loader
        else:
            # Instructions: If __init__ is called from any other process, it
            # should not load in any data.  
            # Tip: you may want to calculate the number of batches (i.e.  the
            # number of training steps/epoch) and broadcast this value out to
            # all the processes.
            pass

        dist.broadcast(self.iters_in_epoch, LEADER_RANK)

    # Generates torch.tensor
    def __iter__(self):
        if self.rank == LEADER_RANK:
            loader_iter = iter(self.loader)
        # Instructions: If __iter__ is called from the leader process, it should
        # broadcast/scatter the whole batch to all the other processes. Then all
        # the processes extract their minibatch from the broadcasted batch. 
        # Tip: have __iter__ yield the minibatches.
        def generator():
            for _ in range(self.iters_in_epoch.item()):
                if self.rank == LEADER_RANK:
                    batch = next(loader_iter).to(self.device)
                else:
                    batch = torch.empty(self.world_size * self.mini_batch_size, SEQ_LEN, dtype=torch.long, device=self.device)
                dist.broadcast(batch, LEADER_RANK)

                mini_batch_start = self.rank * self.mini_batch_size
                mini_batch_end = (self.rank + 1) * self.mini_batch_size
                yield batch[mini_batch_start:mini_batch_end]

        return generator()

def average_gradients(model, world_size):
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= world_size

def next_token(model, input_ids, temperature, freq_penalty=2.0):
    logits = model(input_ids.unsqueeze(0)).logits[0,-1,:]
    id_freqs = torch.bincount(input_ids, minlength=logits.shape[-1])
    logits = logits / temperature - freq_penalty * id_freqs
    return torch.distributions.categorical.Categorical(logits=logits).sample()

def generate(model, device, tokenizer, text, max_length=512, temperature=1.0, freq_penalty=2.0):
    model.eval()
    input_ids = tokenizer(text).input_ids
    generated = []
    # TODO could do something smarter by caching outputs of previous runs but eh
    for i in range(max_length):
        new_token = next_token(model, torch.LongTensor(input_ids + generated).to(device),
                                    temperature=temperature, freq_penalty=freq_penalty)

        generated.append(new_token)
        if new_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids + generated)
        

def init_process(rank, num_processes, backend='nccl'):
    import transformers
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29528'
    dist.init_process_group(backend, rank=rank, world_size=num_processes)
    device = DEVICES[rank]

    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    loader = DistributedDataLoader(
            rank=rank, 
            world_size=num_processes, 
            device=device,
            mini_batch_size=3
    )

    params = list(model.parameters())
    my_params = []
    if OPTIMIZER_STATE_SHARDING:
        my_params = [params[i] for i in range(rank, len(params), num_processes)]
    else:
        my_params = params
    optimizer = torch.optim.Adam(my_params, lr=1e-4)

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1):
        for iter, mini_batch in enumerate(loader):
            optimizer.zero_grad()
            x = mini_batch[:,:-1]
            y = mini_batch[:,1:] # [batch_size, seq_len - 1]
            outputs = model(x) # [batch_size, seq_len - 1, vocab_size]
            logits = rearrange(outputs.logits, "b s v -> (b s) v")
            y = rearrange(y, "b s -> (b s)")
            loss = criterion(logits, y)
            loss.backward()
            average_gradients(model, num_processes)
            optimizer.step()
            if OPTIMIZER_STATE_SHARDING:
                for i, param in enumerate(params):
                    param_owner = i % num_processes
                    dist.broadcast(param, param_owner)

            if iter % 10 == 0 and rank == LEADER_RANK:
                print(f"{epoch=} {iter=}")

    if rank == LEADER_RANK:
        print("finished")
        print("saving model")
        torch.save(model.state_dict(), MODEL_FILENAME)
        print("model saved")
        tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        print(generate(model, device, tokenizer, "Making predictions is a good"))

if __name__ == "__main__":
    num_processes = len(DEVICES)
    processes = []
    mp.set_start_method("spawn")
    for rank in range(num_processes):
        p = mp.Process(target=init_process, args=(rank, num_processes))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
