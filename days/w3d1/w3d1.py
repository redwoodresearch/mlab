import os
import time
import random
import gc
from typing import Optional
import transformers
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torchtext
import torchvision
import einops

DEVICES = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
BATCH_SIZE = 2
MAX_SEQ_LEN = 128

# model = transformers.AutoModelForSequenceClassification.from_pretrained("EleutherAI/gpt-j-6B")



class WrappedGPTJBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        # Taking output out from one-element tuple
        [activations] = self.block(x)
        return activations


def save_gptj_blocks(model):
    block_0 = torch.nn.Sequential(
        model.transformer.wte,
        model.transformer.drop,
        *[WrappedGPTJBlock(model.transformer.h[i]) for i in range(7)]
    )

    block_1 = torch.nn.Sequential(
        *[WrappedGPTJBlock(model.transformer.h[i]) for i in range(7, 7*2)]
    )

    block_2 = torch.nn.Sequential(
        *[WrappedGPTJBlock(model.transformer.h[i]) for i in range(7*2, 7*3)]
    )

    block_3 = torch.nn.Sequential(
        *[WrappedGPTJBlock(model.transformer.h[i]) for i in range(7*3, 7*4)],
        model.transformer.ln_f,
        model.score,
    )

    blocks = [block_0, block_1, block_2, block_3]

    for i, block in enumerate(blocks):
        torch.save(block, f"gptj_block_{i}.pt")

        
def compare_our_model_to_theirs(our_model, their_model):
    our_model.eval()
    their_model.eval()
    
    with torch.no_grad():
        our_model = MultiGPUGPTJ(model)
        inp = torch.randint(0, 100, (1, 2))
        expected_outputs = their_model(inp).logits # shape: 1,2 -- batch num_class
        actual_outputs = our_model(inp) # shape: 1,2,2 -- batch seq num_class

        assert torch.allclose(expected_outputs, actual_outputs), f"Got {actual_outputs} but expected {expected_outputs}"


def to_batches(data, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, num_batches=None):
    sorted_data = sorted(data, key=lambda d: len(d[1]))
    if num_batches is None:
        num_batches = (len(data) + batch_size - 1) // batch_size
    batched_data = []
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        batch = sorted_data[batch_start:batch_end]
        sentiments = torch.tensor([1 if s == 'pos' else 0 for s, r in batch])
        reviews = [r for s, r in batch]
        tokenization = tokenizer(
            reviews,
            padding='max_length',
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt"
        )
        review_tokens = tokenization.input_ids
        batched_data.append((review_tokens, sentiments))
    random.shuffle(batched_data)
    print(f"batched_data len: {len(batched_data)}")
    print(f"batched_data[0] len: {len(batched_data[0])}")
    return batched_data

        
def train(model):
    model.train()
    optimizer = t.optim.Adam(model.parameters(), lr=1e-5)

    for i, (input, target) in enumerate(data_batches):
        optimizer.zero_grad()
        logits, class_logits = model(input.to(device))
        loss = t.nn.functional.cross_entropy(class_logits, target.to(device))
        loss.backward()
        optimizer.step()
        print(f"{i} {loss}")        


def eval_on_imdb(model, num_batches_to_use=None):
    model.eval()
    test_batches = to_batches(data_test, batch_size=8)
    total_correct = 0
    total_samples = 0
    if num_batches_to_use is None:
        num_batches_to_use = len(test_batches)
    print(f'evaluating using {num_batches_to_use} batches')
    for i, (input, target) in enumerate(test_batches):
        logits, class_logits = model(input.to(device))
        answers = t.argmax(class_logits, dim=-1)
        total_correct += t.sum(answers == target.to(device))
        total_samples += answers.shape[0]
        print(i, total_correct, total_samples)
        if i >= num_batches_to_use:
            break
    return total_correct / total_samples



# SHARD_OPTIMIZER_STATE=False
"""
mini_batch_size 350: both pass
mini_batch_size 375: sharding passes
mini_batch_size 400: both fail
"""

LEADER = 0
HIDDEN_SIZE = 4096


def load_block(rank):
    return torch.load(f"gptj_block_{rank}.pt")


def run(rank, size, data_batches):
    """ Distributed function to be implemented later. """
    
    print(f"Rank {rank} started")
    
    device = DEVICES[rank]
    block = load_block(rank).to(device)
    print(f"Block {rank} loaded")
    
    groups = {(i,j): dist.new_group([i, j]) for i, j in [(0, 1), (1, 2), (2, 3), (0, 3)]}
    
    for batch_idx in range(len(data_batches)):
    
        if rank == LEADER:
            # If there's still data, fetch them
            inps, labels = data_batches[batch_idx]
            print(f"Rank {rank} fetched new batch")
            inps = inps.to(device)
            labels = labels.to(device)
        else:
            # Initialise inps: batch, seq_len, hidden_size (I'm guessing we don't consider num_heads, head_size??)
            # We are fetching outputs from the last block
            inps = torch.zeros(BATCH_SIZE, MAX_SEQ_LEN, HIDDEN_SIZE).to(device)
            print(f"Rank {rank} creating new group")
            group = groups[rank-1, rank]
            print(f"Rank {rank} waiting for input of size {inps.shape}")
            dist.broadcast(tensor=inps, src=rank-1, group=group)
            print(f"Rank {rank} received input of size {inps.shape}")
        
        # Put it through block
        with torch.no_grad():
            out = block(inps)
        
        print(f"Rank {rank} output {out.shape}")

        # Send output on to next block
        if rank < len(DEVICES) - 1:
            print(f"Rank {rank} creating new group")
            group = groups[rank, rank+1]
            print(f"Rank {rank} waiting to send output of size {inps.shape}")
            dist.broadcast(tensor=out, src=rank, group=group)
            print(f"Rank {rank} sent output of size {inps.shape}")
            if rank == LEADER:
                # Send labels to last block
                group = groups[rank, size-1]
                print(f"Rank {rank} waiting to send labels of size {inps.shape}")
                dist.broadcast(tensor=labels, src=rank, group=group)
                print(f"Rank {rank} sent labels of size {inps.shape}")
        else:
            # Get labels from the first block
            labels = torch.zeros(BATCH_SIZE, dtype=torch.int64).to(device)
            group = groups[LEADER, size - 1]
            print(f"Rank {rank} waiting for labels")
            dist.broadcast(tensor=labels, src=LEADER, group=group)

            # Handle loss and backprop at last block
            classification_logits = out[:, -1]

            # inputs [N, C] and targets [N]
            loss = torch.nn.functional.cross_entropy(classification_logits, labels)
            # loss.backward()
            print(loss.detach().item())

#     if rank == 0:
#         start_time = time.time()


#     model = transformers.GPT2LMHeadModel.from_pretrained('gpt2').to(device)

#     if SHARD_OPTIMIZER_STATE:
#         params_to_optimize = []
#         for i, param in enumerate(model.parameters()):
#             if i % size == 0:
#                 params_to_optimize.append(param)
#         optimizer = torch.optim.Adam(params_to_optimize, lr=1e-5)
#     else:
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

#     # Iterate over minibatches
#     model.train()
#     for epoch in range(4):
#         print(epoch)
#         ddl = DistributedDataLoader(rank, len(DEVICES), 375, random_seed = epoch)
#         for minibatch_data in ddl:
#             optimizer.zero_grad()
#             # Normal training loop
#             # FIXME
#             minibatch_data = {'input_ids': minibatch_data,
#                             'attention_mask': torch.ones_like(minibatch_data, dtype=torch.long)}
#             outputs = model(**minibatch_data, labels=minibatch_data['input_ids']) 
#             loss = outputs.loss
#             loss.backward()
#             print(loss.detach())
#             # All-reduce to share gradients, for each parameter
#             for param in model.parameters():
#                 old_grad = param.grad.detach().clone()
#                 # Taking the mean over the gradients
#                 dist.all_reduce(param.grad, dist.ReduceOp.SUM)
#                 # assert not torch.allclose(old_grad, param.grad.detach())
#                 param.grad = param.grad / size
#             # Does it take real long? Maybe time optimizer.step() and dist.broadcast, and compare them
#             optimizer.step()
#             if SHARD_OPTIMIZER_STATE:
#                 for i, param in enumerate(model.parameters()):
#                     dist.broadcast(param.data, src=i % 3)

#     print('Training completed')
#     loss = 0.

#     ddl = DistributedDataLoader(rank, len(DEVICES), 32, random_seed = epoch)

#     if rank == 0:
#         model.eval()
#         test_data = ddl.test_dataloader
#         c = 0
#         for test_datum in test_data:
#             test_datum = ddl.tokenize(test_datum)
#             test_datum = {'input_ids': test_datum,
#                         'attention_mask': torch.ones_like(test_datum, dtype=torch.long)}
#             outputs = model(**test_datum, labels=test_datum['input_ids']) 
#             loss += outputs.loss.detach()
#             c +=1
#             if c % 10 == 0:
#                 print(loss)
#             if c > 100:
#                 break    
#         print("eval loss: ", loss / len(test_data) / c)  

#         print("time: ", time.time() - start_time)          
    # After 4 epochs evaluate on test set

def init_process(rank, size, data_batches, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    # Get port offset based on our GPU ids
    os.environ['MASTER_PORT'] = str(29500 + sum([0,1,2,3]))
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, data_batches)
    device = DEVICES[rank]


def superstitious_button():
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    superstitious_button()
    
    # save_gptj_blocks(model)
    # compare_our_model_to_theirs(our_model, their_model)
    # eval_on_imdb(my_bert, num_batches_to_use=50)
    
    # TODO: find out why the following breaks `tokenizer(['hi how are you', 'something'], padding='longest').input_ids`
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = tokenizer.eos_token
    
    data_train, data_test = torchtext.datasets.IMDB(root='.data', split=('train', 'test'))

    data_train = list(data_train)
    data_test = list(data_test)
    print(len(data_train), len(data_test))

    data_batches = to_batches(data_train, batch_size=BATCH_SIZE, num_batches=4)

    size = len(DEVICES)
    processes = []
    # mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, data_batches, run, "gloo"))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

