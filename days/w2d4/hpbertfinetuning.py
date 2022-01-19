from comet_ml import Experiment

from days.w2d1.bert_sol import BertWithClassify, mapkey
import days.w2d1.bert_tests as bert_tests
from einops import rearrange, reduce, repeat
import math
import re
import torch as t
from torch import einsum
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import *
import transformers
import torchtext
import gin


# @gin.configurable
def load_model(
    hidden_size=768, max_position_embeddings=512,
    dropout=0.1, intermediate_size=3072, 
    num_heads=12, num_layers=12
):
    my_bert = BertWithClassify(
        vocab_size=28996, hidden_size=hidden_size, max_position_embeddings=max_position_embeddings, 
        type_vocab_size=2, dropout=dropout, intermediate_size=intermediate_size, 
        num_heads=num_heads, num_layers=num_layers, num_classes=2,
    )

    pretrained_bert = bert_tests.get_pretrained_bert()
    mapped_params = {mapkey(k): v for k, v in pretrained_bert.state_dict().items()
                    if not k.startswith('classification_head')}
    my_bert.load_state_dict(mapped_params, strict=False)
    # bert_tests.test_same_output(my_bert, pretrained_bert)
    return my_bert

def get_imdb_collate_fn(
    max_seq_length: int,
    tokenizer: transformers.AutoTokenizer,
    device: str,
):
    def fn(raw_xs: List[Tuple[str, str]]) -> Tuple[t.Tensor, t.Tensor]:
        labels: Tuple[str, ...]
        texts: Tuple[str, ...]
        labels, texts = zip(*raw_xs)

        xs = t.tensor(
            tokenizer(
                list(texts),
                padding="longest",
                max_length=max_seq_length,
                truncation=True,
            )['input_ids'],
            dtype=t.long,
            device=device,
        )

        ys = t.tensor([int(l == "pos") for l in labels], dtype=t.long, device=device)

        return xs, ys

    return fn


def test(model, dl_test, num_batches=math.inf):
    model.eval()
    n_accurate = 0
    n_total = 0
    with t.no_grad():
        for i,(x,y) in zip(range(num_batches), dl_test):
            _, out = model(x)
            n_accurate += t.sum(t.argmax(out, dim=-1) == y)
            n_total += x.shape[0]
    return n_accurate / n_total
        

def train(experiment, model, dl_train, dl_test, num_epochs, lr):
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    num_steps = 0
    for epoch in range(num_epochs):
        for batch_num,(x, y) in enumerate(dl_train):
            optimizer.zero_grad()
            _, out = model(x)
            loss = loss_fn(input=out, target=y)
            loss.backward()
            optimizer.step()
            num_steps += 1
            
            if batch_num % 32 == 0:
                acc = test(model, dl_test, 8)
                experiment.log_metric("val accuracy", acc, step=num_steps)
                model.train()
            
    return model
            

@gin.configurable
def run(experiment, batch_size, num_epochs, lr, seed):
    t.manual_seed(seed)
    
    experiment.log_parameters({'batch_size': batch_size, 'num_epochs': num_epochs, 'lr': lr})
    device = "cuda"
    model = load_model()
    model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    data_train, data_test = torchtext.datasets.IMDB(root='.data', split=('train', 'test'))
    data_train = list(data_train)
    # data_train = list(data_train)
    data_test = list(data_test)
    # data_test = list(data_test)
    
    collate_fn = get_imdb_collate_fn(512, tokenizer, device)
    dl_train = DataLoader(
        data_train,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    dl_test = DataLoader(
        data_test,
        batch_size=batch_size*2,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    train(experiment, model, dl_train, dl_test, num_epochs, lr)


if __name__ == "__main__":
    run()