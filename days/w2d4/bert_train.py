from comet_ml import Experiment
import torch as t
import torch.nn as nn
from sklearn.datasets import make_moons
import gin
from torch.utils.data import DataLoader, TensorDataset, Dataset
from days.w2d1.bert_sol import BertWithClassify, mapkey
from days.w2d1 import bert_tests
import torchtext
import random
import transformers


@gin.configurable
def train(experiment, batch_size, lr, num_epochs):
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    
    my_bert = BertWithClassify(
            vocab_size=28996, hidden_size=768, max_position_embeddings=512, 
            type_vocab_size=2, dropout=0.1, intermediate_size=3072, 
            num_heads=12, num_layers=12, num_classes=2)
    pretrained_bert = bert_tests.get_pretrained_bert()
    mapped_params = {mapkey(k): v for k, v in pretrained_bert.state_dict().items()}
    my_bert.load_state_dict(mapped_params)
    # bert_tests.test_same_output(my_bert, pretrained_bert) # we don't need this - tim
    my_bert.to('cuda')
    my_bert.train()
    
    loss = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(my_bert.parameters(), lr)
    data_train, data_test = torchtext.datasets.IMDB(root='.data', split=('train', 'test'))
    train_dataloader =  DataLoader(list(data_train), shuffle=True, batch_size=batch_size)
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            input_ids = t.tensor(tokenizer(list(batch[1]), padding='longest', max_length=512, truncation=True)['input_ids']).to('cuda')
            target = t.tensor([1 if x == 'pos' else 0 for x in batch[0]], dtype=t.long).to('cuda')
            output = my_bert(input_ids)[1]
            l = loss(output, target)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            
    
    

            