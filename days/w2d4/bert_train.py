from comet_ml import Experiment
import torch as t
import torch.nn as nn
from sklearn.datasets import make_moons
import gin
from torch.utils.data import DataLoader, TensorDataset
from w2d1.bert_sol import BertWithClassify, mapkey
from w2d1 import bert_tests
import torchtext


@gin.configurable
def train(experiment, batch_size, lr, num_epochs):
    
    my_bert = BertWithClassify(
            vocab_size=28996, hidden_size=768, max_position_embeddings=512, 
            type_vocab_size=2, dropout=0.1, intermediate_size=3072, 
            num_heads=12, num_layers=12)
    pretrained_bert = bert_tests.get_pretrained_bert()
    mapped_params = {mapkey(k): v for k, v in pretrained_bert.state_dict().items()}
    my_bert.load_state_dict(mapped_params)
    bert_tests.test_same_output(my_bert, pretrained_bert)

    optimizer = t.optim.Adam(my_bert.parameters(), lr)
    data_train, data_test = torchtext.datasets.IMDB(root='.data', split=('train', 'test'))
    train_dataloader =  DataLoader(data_train, batch_size=batch_size)
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            loss = t.binary_cross_entropy_with_logits(my_bert(batch[0]), batch[1]).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
