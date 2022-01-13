import os

from comet_ml import Experiment
import torch as t
import torch.nn as nn
from sklearn.datasets import make_moons
import gin
from torch.utils.data import DataLoader, TensorDataset

import torch as t
from torch import einsum
from einops import rearrange, reduce, repeat
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
print(os.getcwd())
print(sys.path)
from days.w2d1 import bert_tests


def raw_attention_pattern(token_activations, num_heads, project_query, project_key):
    query = project_query(token_activations)
    query = rearrange(query, "b s (n h) -> b n s h", n=num_heads)
    key = project_key(token_activations)
    key = rearrange(key, "b s (n h) -> b n s h", n=num_heads)
    head_size = key.shape[-1]
    dot_prod = t.einsum("bsnh,bsmh->bsmn", query, key)
    dot_prod /= np.sqrt(head_size)
    return dot_prod

def bert_attention(token_activations, num_heads, attention_pattern, project_value, project_output):
    attention_score = t.softmax(attention_pattern, dim=-2)
    value = project_value(token_activations)
    value = rearrange(value, "batch seq (head size) -> batch seq size head", head=num_heads)
    attention_value = t.einsum("bhkq,bksh->bshq", attention_score, value)
    attention_value = rearrange(attention_value, "batch size head seq-> batch seq (head size)", head=num_heads)
    output = project_output(attention_value)
    return output


class MultiHeadedSelfAttention(t.nn.Module):
    def __init__(self, num_heads: int, hidden_size:int):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.query = t.nn.Linear(hidden_size, hidden_size)
        self.key = t.nn.Linear(hidden_size, hidden_size)
        self.value = t.nn.Linear(hidden_size, hidden_size)
        self.output = t.nn.Linear(hidden_size, hidden_size)


    def forward(self, input):
        attn_pattern = raw_attention_pattern(input, self.num_heads, self.query, self.key)
        bert_attn = bert_attention(input, self.num_heads, attn_pattern, self.value, self.output)
        return bert_attn

def bert_mlp(token_activations, linear_1, linear_2):
    x = linear_1(token_activations)
    y = t.nn.functional.gelu(x)
    return linear_2(y)

class BertMLP(t.nn.Module):
    def __init__(self, input_size: int, intermediate_size: int):
        super().__init__()
        self.input_size = input_size
        self.intermediate_size = intermediate_size
        self.linear1 = t.nn.Linear(input_size, intermediate_size)
        self.linear2 = t.nn.Linear(intermediate_size, input_size)
    def forward(self, input):
        return bert_mlp(input, self.linear1, self.linear2)

class LayerNorm(t.nn.Module):
    def __init__(self, normalized_dim: int):
        super().__init__()
        self.weight = t.nn.Parameter(t.ones((normalized_dim,)))
        self.bias = t.nn.Parameter(t.zeros((normalized_dim,)))

    def forward(self, input):
        mean = t.mean(input, dim=-1, keepdims=True).detach()
        var = t.std(input, dim=-1, unbiased=False, keepdims=True).detach()
        input = (input - mean) / var
        input = input * self.weight + self.bias
        return input

class BertBlock(t.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size)
        self.attn = MultiHeadedSelfAttention(num_heads, hidden_size)
        self.mlp = BertMLP(hidden_size, intermediate_size)
        self.norm2 = LayerNorm(hidden_size)
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, input):
        x = self.attn(input)
        y = self.norm1(x + input)
        x = self.mlp(y)
        x = self.dropout(x)
        return self.norm2(x + y)

class Embedding(t.nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embedding = t.nn.Parameter(t.randn((vocab_size, embed_size)))

    def forward(self, inputs):
        return self.embedding[inputs, :] #TODO look at solution

def bert_embedding(input_ids, token_type_ids, position_embedding, token_embedding, token_type_embedding, layer_norm, dropout):
    x = position_embedding(t.arange(input_ids.shape[-1])) + token_embedding(input_ids) + token_type_embedding(token_type_ids)
    x = layer_norm(x)
    return dropout(x)

class BertEmbedding(t.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int, type_vocab_size: int, dropout: float):
        super().__init__()
        self.token_em = Embedding(vocab_size, hidden_size)
        self.pos_em = Embedding(max_position_embeddings, hidden_size)
        self.token_type_em = Embedding(2, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids):
        return bert_embedding(input_ids, token_type_ids, self.pos_em, self.token_em, self.token_type_em, self.layer_norm, self.dropout)

class Bert(t.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int,
                 max_position_embeddings: int, type_vocab_size: int,
                 dropout: float, intermediate_size: int, num_heads: int,
                 num_layers: int, shared_embedding = False):
        super().__init__()
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.embedding = BertEmbedding(vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout)
        self.layers = t.nn.Sequential(
            t.nn.Sequential(*[BertBlock(hidden_size, intermediate_size, num_heads, dropout) for _ in range(num_layers)]),
            t.nn.Linear(hidden_size, hidden_size),
            t.nn.GELU(),
            LayerNorm(hidden_size),
        )
        self.shared_embedding = shared_embedding
        if shared_embedding:
            self.bias = t.nn.Parameter(t.zeros(vocab_size))
        else:
            self.linear = t.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.layers(self.embedding(input_ids, t.zeros(1, dtype=t.long, device=input_ids.device)))
        if self.shared_embedding:
            x = (x @ self.embedding.token_em.embedding.T) + self.bias
        else:
            x = self.linear(x)
        return x

my_bert = Bert(
    vocab_size=28996, hidden_size=768, max_position_embeddings=512,
    type_vocab_size=2, dropout=0.1, intermediate_size=3072,
    num_heads=12, num_layers=12
)
pretrained_bert = bert_tests.get_pretrained_bert()
pretrained_bert_dict = pretrained_bert.state_dict()
del pretrained_bert_dict['classification_head.weight']
del pretrained_bert_dict['classification_head.bias']
mapped_weights = {v[0]: pretrained_bert_dict[v[1]] for v in zip(my_bert.state_dict(), pretrained_bert_dict)}
mapping_replacements = {
    'layers.3.weight' : 'linear.weight',
    'linear.weight' : 'layers.3.weight',
    'layers.3.bias' : 'linear.bias',
    'linear.bias' : 'layers.3.bias',
}

# mapped_weights = {'a':2,'b':3}
# mapping_replacements = {'a':'b','b':'a'}
another_mapped_weights = {mapping_replacements[i] if i in mapping_replacements else i : mapped_weights[i] for i in mapped_weights}
# another_mapped_weights

import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
bad_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

class BertSentiment(t.nn.Module):
    def __init__(self, bert_model, num_classes: int):
        super().__init__()
        self.bert_model = bert_model
        self.num_classes = num_classes
        self.dropout = t.nn.Dropout(bert_model.dropout)
        self.linear = t.nn.Linear(bert_model.hidden_size, num_classes)

    def forward(self, input_ids):
        return self.linear(self.dropout(self.bert_model.layers(self.bert_model.embedding(input_ids, t.zeros(1, dtype=t.long, device=input_ids.device)))))[:,0,:]

import torchtext
import random

data_train, data_test = torchtext.datasets.IMDB(root='./data2', split=('train', 'test'))
data_train = list(data_train)
random.shuffle(data_train)
data_test = list(data_test)
random.shuffle(data_test)

def get_batches(data, batch_size, tokenizer, max_seq_len = 2048):
    assert len(data) > batch_size
    n_batches = len(data) // batch_size
    batched_data = t.zeros(n_batches, batch_size, max_seq_len, dtype=t.long)
    batched_labels = t.zeros(n_batches, batch_size, dtype=t.long)
    for batch in range(n_batches):
        for i in range(batch_size):
            label, text = data[batch*batch_size + i]
            tokens = tokenizer.encode(text)
            l = min(len(tokens), max_seq_len)
            batched_data[batch,i,:l] = t.tensor(tokens)[:l]
            batched_labels[batch,i] = int(label == 'pos')
    return batched_data, batched_labels

@gin.configurable
def train_sentiments(experiment, data_train, data_test, model, batch_size, tokenizer, max_seq_len=512, lr=1e-5, epochs=1):
    batched_data, batched_labels = get_batches(data_train, batch_size, tokenizer, max_seq_len=max_seq_len)
    #print(batched_data.shape)
    model = model.cuda()
    optimizer = t.optim.Adam(model.parameters(),lr=lr)
    losses = []
    try:
        for _ in range(epochs):
            tbar = tqdm(range(batched_data.shape[0]))
            for batch in tbar:
                batched_data_cuda = batched_data[batch].cuda()
                batched_labels_cuda = batched_labels[batch].cuda()
                #print(batched_data_cuda.shape)
                #print(batched_labels_cuda.shape)
                optimizer.zero_grad()
                pred = model(batched_data_cuda)
                #print(pred.shape)
                #print(batched_labels[batch].shape)
                loss = t.nn.functional.cross_entropy(pred, batched_labels_cuda)
                loss.backward()

                loss = loss.item()
                tbar.set_postfix({"loss": loss})
                losses.append(loss)
                experiment.log_metric("loss", loss)

                optimizer.step()
    except KeyboardInterrupt:
        print("ctrl c")
    return losses

@gin.configurable
def train(experiment, max_seq_len=265,  batch_size=16, lr=1e-5, num_epochs=1):
    sentiment_model = BertSentiment(my_bert, 2)
    losses = train_sentiments(experiment, data_train, data_test, sentiment_model, batch_size=batch_size, tokenizer=tokenizer, lr=lr, max_seq_len=max_seq_len, epochs=num_epochs)
    t.save(sentiment_model.state_dict(), "sentiment_model.save")
    experiment.log_model("trained sentiment bert", "sentiment_model.save")





