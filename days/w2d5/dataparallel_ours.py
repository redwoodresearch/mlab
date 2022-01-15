from comet_ml import Experiment

import os
import torch as t
import json
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from os.path import exists
from tqdm import tqdm

DEVICES = ["cuda:0", "cuda:1", "cuda:2"]

TENSOR_PATH_PREFIX = "/home/ubuntu/dm_and_lukas/days/w2d5/lw_tensor"

def chunks(lst, n, skip_first=0):
    for i in range(skip_first, len(lst), n):
        yield lst[i:i + n]
        
def process_data_to_file(data_path='/home/ubuntu/lw_corpus.json', size=3):
    
    context_length = 512
    tensor_paths = [f"{TENSOR_PATH_PREFIX}_{rank}.pt" for rank in range(size)]
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    with open(data_path, 'r') as f:
        data = json.load(f)

        data = [post['text'] + tokenizer.eos_token for post in data]

        chunked_data = []

        for post in tqdm(data):
            post = tokenizer(post)['input_ids']
            
            n_missing_tokens = 0

            if len(chunked_data) > 0:
                n_missing_tokens = context_length - len(chunked_data[-1])
                chunked_data[-1].extend(post[:n_missing_tokens])

            post_chunks = chunks(post, context_length, skip_first=n_missing_tokens)

            chunked_data.extend(post_chunks)

        if len(chunked_data[-1]) < context_length:
            chunked_data.pop(-1)
        
        data_tensor = t.tensor(chunked_data)
        
        split_data_tensors = data_tensor.split(data_tensor.shape[0] // size)
        
        for save_path, split_data_tensor in zip(tensor_paths, split_data_tensors):
            t.save(split_data_tensor, save_path)


def get_data_loaders(rank, world_size, mini_batch_size, random_seed=0):
    if t.manual_seed is not None:
        t.manual_seed(random_seed)
    
    device = DEVICES[rank]

    tensor = t.load(f"{TENSOR_PATH_PREFIX}_{rank}.pt")
    # tensor = tensor.to(device)

    train_tensor = tensor[:int(len(tensor) * 0.8)]
    test_tensor = tensor[int(len(tensor) * 0.8):]

    train_loader = t.utils.data.DataLoader(train_tensor, batch_size=mini_batch_size, shuffle=True, pin_memory=True)
    test_loader = t.utils.data.DataLoader(test_tensor, batch_size=mini_batch_size, shuffle=True, pin_memory=True)
    return train_loader, test_loader
        

def train(rank, world_size, model, train_loader, num_epochs, lr):
    
    device = DEVICES[rank]
    
    if rank == 0:
        experiment = Experiment(
            api_key='oGcm04SiEeJM89dRyU0vcOFzd',
            project_name='dm_and_lukas_gpt2'
        )
        experiment.log_parameters({'lr':lr, 'epochs':num_epochs})
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    loss_fn = t.nn.CrossEntropyLoss()
    model.train()
    for e in range(num_epochs):
        pbar = tqdm(train_loader)
        for mini_batch in pbar:
            
            mini_batch = mini_batch.to(device)
            
            optimizer.zero_grad()
            
            out = model(mini_batch[:,:-1])
            target = mini_batch[:,1:].flatten()
                        
            logits = t.reshape(out.logits, (len(target), -1))
                        
            loss = loss_fn(logits, target)
            loss.backward()
            reduce_ops = [
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
                for param in model.parameters()
            ]
            for op in reduce_ops:
                op.wait()

            optimizer.step()
            
            pbar.set_description(f'loss={loss.item()}')
            
    return model

def evaluate(model, test_loader):

    model.eval()
    
    pbar = tqdm(test_loader)
    
    test_loss = []
    
    with t.no_grad():
        for mini_batch in pbar:

            mini_batch = mini_batch.to(device)

            optimizer.zero_grad()

            out = model(mini_batch[:,:-1])
            target = mini_batch[:,1:].flatten()

            logits = t.reshape(out.logits, (len(target), -1))

            loss = loss_fn(logits, target)
            
            test_loss.append(test_loss)
            
    print('Overall test loss:', np.mean(test_loss))
            
    return np.mean(test_loss)
            
def get_gpt2():
    return GPT2LMHeadModel.from_pretrained('gpt2')
    
            
def run(rank, size):
    """ Distributed function to be implemented later. """        
    device = DEVICES[rank]
    
    train_loader, test_loader = get_data_loaders(rank, size, mini_batch_size=4)
    
    model = get_gpt2().to(device)
    
    model = train(rank, size, model, train_loader, num_epochs=1, lr=1e-4)
    
    if rank == 0:
        t.save(model, '/home/ubuntu/dm_and_lukas/days/w2d5/model.pt')
        t.save(model.state_dict(), '/home/ubuntu/dm_and_lukas/days/w2d5/state_dict.pt')
    
    evaluate(model, test_loader)
    

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29503'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    # process_data_to_file(size=3)
    
    size = 3
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
