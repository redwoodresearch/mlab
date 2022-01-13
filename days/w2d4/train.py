from comet_ml import Experiment
from days.w2d4.bert import BertWithClassify, mapkey
from days.w2d1.bert_tests import get_pretrained_bert
from torch.utils.data import DataLoader

import gin
import transformers
import torchtext
import torch as t
import torch.nn as nn
import torch.nn.functional as F

def tensorify_batch(batch, tokenizer):
    labels, reviews = batch
    labels = t.tensor([label == 'pos' for label in labels]).long()
    reviews = t.tensor(tokenizer(
        [*reviews],
        padding='longest',
        max_length=512,
        truncation=True
    )['input_ids'])
    return (labels, reviews)

def load_weights(model):
    pretrained_bert = get_pretrained_bert()
    mapped_params = {mapkey(k): v for k, v in pretrained_bert.state_dict().items()
                    if not k.startswith('classification_head')}
    model.load_state_dict(mapped_params, strict=False)
    
@gin.configurable
def train(experiment, batch_size, lr, num_epochs):
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased", return_tensors="pt")
    data_train, data_test = torchtext.datasets.IMDB(root='.data', split=('train', 'test'))
    data_train = DataLoader(list(data_train), batch_size=batch_size, shuffle=True) 
    model = BertWithClassify()
    load_weights(model)
    optimizer = t.optim.Adam(model.parameters(), lr)
    for epoch in range(num_epochs):
        for batch in data_train:
            optimizer.zero_grad()
            labels, reviews = tensorify_batch(batch, tokenizer)
            labels, reviews = labels.cuda(), reviews.cuda()
            model.cuda()
            logits, classifications = model(reviews)
            loss = F.cross_entropy(classifications, labels)
            loss.backward()
            optimizer.step()
