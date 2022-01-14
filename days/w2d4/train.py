from comet_ml import Experiment
import torch as t
import torch.nn as nn
from sklearn.datasets import make_moons
import gin
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import torchtext
import transformers
import random
import os
from tqdm import tqdm



# @gin.configurable
# class MyModel(nn.Module):
#     def __init__(self, hidden_size, n_layers, input_size, output_size):
#         super().__init__()
#         self.mlps = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             *[nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)],
#             nn.Linear(hidden_size, output_size),
#         )

#     def forward(self, x):
#         return self.mlps(x)

def extract_batches(dataset, batch_size=4, max_seq_length=512):
    
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

    dataset.sort(key=lambda x: len(x[1]))

    number_of_batches = math.ceil(len(dataset) / batch_size)

    dataset = np.array_split(dataset, number_of_batches)

    batches = []

    for batch in dataset:
        scores, reviews = zip(*batch)

        tokens = tokenizer(list(reviews))

        # print(len(tokens['input_ids'][0]))
        tokens = tokens["input_ids"]

        tokens = [review_token[:max_seq_length] for review_token in tokens]
        tokens = [review_token if len(review_token) == max_seq_length else review_token + (max_seq_length - len(review_token))*[tokenizer.pad_token_id] for review_token in tokens]

        X = t.tensor(tokens)
        y = t.tensor([1 if s == "pos" else 0 for s in scores])

        assert X.shape[0] <= batch_size and X.shape[1] == max_seq_length, str(X.shape)
        batches.append((X,y))

    random.shuffle(batches)
    return batches




@gin.configurable
def train(experiment, 
          batch_size, 
          lr, 
          num_epochs, 
          max_seq_length = 512, 
          vocab_size=28996, 
          hidden_size=768, 
          type_vocab_size=2,
          dropout=0.1,
          intermediate_size=3072,
          num_heads=12,
          num_layers=12,
          num_classes=2,
         ):
    
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    
    from days.w2d1.bert_tests import get_pretrained_bert
    from days.w2d1.bert_sol import BertWithClassify, mapkey
    
    model = BertWithClassify(
        vocab_size=vocab_size, hidden_size=hidden_size,
        max_position_embeddings=max_seq_length, 
        type_vocab_size=type_vocab_size, dropout=dropout,
        intermediate_size=intermediate_size, 
        num_heads=num_heads, num_layers=num_layers,
        num_classes=num_classes
    )
    pretrained_bert = get_pretrained_bert()
    mapped_params = {mapkey(k): v for k, v in pretrained_bert.state_dict().items()}
    model.load_state_dict(mapped_params)

    #print("hi")

    optimizer = t.optim.Adam(model.parameters(), lr)
    data_train = list(torchtext.datasets.IMDB(root='.data',split='train'))
    data_test = list(torchtext.datasets.IMDB(root='.data',split='test'))
    
    #print("i'm here now")


    dataset = extract_batches(data_train, batch_size, max_seq_length)
    test_dataset = extract_batches(data_test, batch_size, max_seq_length)
    
    #print("data is batched and ready")
    
    experiment.log_parameter("learning rate", lr)
    experiment.log_parameter("batch_size", batch_size)
    experiment.log_parameter("num_epochs", num_epochs)
    
    #print(dataset[0][0].shape, dataset[0][1].shape)
    
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        
        a = random.randint(0, len(test_dataset)-1000)
        for batch in dataset[a:a+1000]:
            b0 = batch[0].to(device)
            b1 = batch[1].to(device)
            optimizer.zero_grad()
            x = model(b0)[1]
            loss = t.nn.CrossEntropyLoss()(x, b1)
            loss.backward()
            optimizer.step()
            
            # del b0
            # del b1
            # del x
            # t.cuda.empty_cache()

        optimizer.zero_grad()
        model.eval()
        total = 0
        cnt_num_batches = 0
        
        a = random.randint(0, len(test_dataset)-100)
        # bar.set_description('validation')
        with t.no_grad():
            for batch in test_dataset[a:a+100]:
                b0 = batch[0].to(device)
                b1 = batch[1].to(device)
                total += t.nn.CrossEntropyLoss()(model(b0)[1], b1)
                cnt_num_batches += 1

                # del b0
                # del b1
                # t.cuda.empty_cache()

            
        average_test_loss = total / cnt_num_batches
        experiment.log_metric("test_loss", average_test_loss)
    