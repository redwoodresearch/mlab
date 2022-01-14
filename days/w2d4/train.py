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

def tensorify_batch(batch, tokenizer, max_len):
    labels, reviews = batch
    labels = t.tensor([label == 'pos' for label in labels]).long()
    reviews = t.tensor(tokenizer(
        [*reviews],
        padding='longest',
        max_length=max_len,
        truncation=True
    )['input_ids'])
    return (labels, reviews)

def load_weights(model):
    pretrained_bert = get_pretrained_bert()
    mapped_params = {mapkey(k): v for k, v in pretrained_bert.state_dict().items()
                    if not k.startswith('classification_head')}
    model.load_state_dict(mapped_params, strict=False)
    
@gin.configurable
def train(experiment, batch_size, lr, num_epochs, max_len):
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased", return_tensors="pt")
    data_train, data_test = torchtext.datasets.IMDB(root='.data', split=('train', 'test'))
    data_train = DataLoader(list(data_train), batch_size=batch_size, shuffle=True) 
    model = BertWithClassify()
    load_weights(model)
    optimizer = t.optim.Adam(model.parameters(), lr)
    for epoch in range(num_epochs):
        for step, batch in enumerate(data_train):
            optimizer.zero_grad()
            labels, reviews = tensorify_batch(batch, tokenizer, max_len)
            labels, reviews = labels.cuda(), reviews.cuda()
            model.cuda()
            
            logits, classifications = model(reviews)
            loss = F.cross_entropy(classifications, labels)
            loss.backward()
            optimizer.step()
            
            predictions = t.argmax(classifications.detach(), dim=1)
            accuracy = t.eq(predictions, labels).float().mean()
            experiment.log_metric("loss", loss)
            experiment.log_metric("accuracy", accuracy)
            
            if step >= 2000: break
    experiment.log_parameters("train.lr", lr)
    t.save(model.state_dict(), "model.pt")
    experiment.log_model("FINETUNED_BERT", "./model.pt")
