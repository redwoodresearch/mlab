from typing import *
import torch as t
import numpy as np
from torch.nn import Module, Parameter, Sequential
import math
from torchtyping import TensorType, patch_typeguard


def softmax(tensor: t.Tensor, dim: int = 0):
    exps = math.e ** tensor
    exp_sums = exps.sum(dim=dim, keepdim=True)
    result = exps / exp_sums
    return result


def relu(tensor: t.Tensor) -> t.Tensor:
    tensor[tensor < 0] = 0
    return tensor


def gelu(x):
    return 0.5 * x * (1 + t.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * t.pow(x, 3))))


# TODO: figure out what this should actually be
def normalize(tensor: t.Tensor, dim: int = -1, eps=1e-12):
    norm = t.norm(tensor, dim=dim, keepdim=True)
    norm[norm < eps] = eps
    tensor = tensor / norm
    return tensor


class LayerNorm(Module):
    def __init__(self, shape, eps=1e-05):
        super(LayerNorm, self).__init__()
        self.bias = Parameter(t.zeros(shape))
        self.weight = Parameter(t.ones(shape))
        self.eps = eps
        # indexes to normalize over
        self.idx_list = [-i - 1 for i, _ in enumerate(shape)]

    def forward(self, tensor):
        tensor = (tensor - tensor.mean(*self.idx_list, keepdim=True)) / t.sqrt(
            tensor.var(*self.idx_list, keepdim=True) + self.eps
        )
        print("shapes", tensor.shape, self.weight.shape, self.bias.shape)
        tensor = tensor * self.weight + self.bias
        return tensor


class Dropout(Module):
    def __init__(self, fraction):
        super(Dropout, self).__init__()
        self.fraction = fraction

    def forward(self, input):
        if self.training:
            mask = t.random.uniform() > self.fraction
            return mask * input
        return input


class Linear(Module):
    def __init__(self, x, y):
        super(Linear, self).__init__()
        weight_bound = 1 / np.sqrt(x)
        self.weight = Parameter(t.FloatTensor(x, y).uniform_(-weight_bound, weight_bound))
        bias_bound = 1 / np.sqrt(y)
        self.bias = Parameter(t.FloatTensor(y).uniform_(-bias_bound, bias_bound))

    def forward(self, x: TensorType[..., "channels"]) -> TensorType[..., "channels"]:
        return t.matmul(x, self.weight) + self.bias


class Embedding(Module):
    def __init__(self, vocab_size: int, embedding_size: int):
        super(Embedding, self).__init__()
        # this needs to be initialized as a bunch of normalized vector,
        # as opposed to a linear layer, which is initialized to _produce_ normalized vectors
        self.embedding = Parameter(t.FloatTensor(vocab_size, embedding_size).normal_(0, 1))

        # then zero out padding tokens and/or whatever
        self.unembed_layer_norm = LayerNorm((embedding_size,))

    def embed(self, x: TensorType[..., t.long]):
        return self.embedding[x]

    def unembed(self, x: TensorType["...", "embed_dim"]):
        return t.einsum("...j,kj->...k", self.unembed_layer_norm(x), self.embedding)


class NormedResidualLayer(Module):
    def __init__(self, size):
        super(NormedResidualLayer, self).__init__()
        self.mlp = Linear(size, size)
        self.layer_norm = LayerNorm((size,))

    def forward(self, input):
        return self.layer_norm(input + relu(self.mlp(input)))


def multi_head_self_attention(
    token_activations,
    attention_masks,
    num_heads,
    project_query,
    project_key,
    project_value,
):
    query = t.einsum("...j,jk->...k", token_activations, project_query)
    query = t.stack(t.split(query, num_heads, dim=-1), dim=0)

    key = t.einsum("...j,jk->...k", token_activations, project_key)
    key = t.stack(t.split(key, num_heads, dim=-1), dim=0)

    value = t.einsum("...j,jk->...k", token_activations, project_value)
    value = t.stack(t.split(value, num_heads, dim=-1), dim=0)

    attention_raw = t.einsum("hbsc,hbsc->hbs", query, key) / math.sqrt(token_activations.shape[-1] / num_heads)
    if attention_masks:
        attention_raw = attention_raw * attention_masks
    attention_patterns = softmax(attention_raw)

    attention_values = t.einsum("hbs,hbsc->hbsc", attention_patterns, value)
    output = token_activations + t.cat(t.split(attention_values, num_heads, dim=0), dim=-1)
    return output


class SelfAttentionLayer(Module):
    def __init__(self, hidden_size, num_heads):
        super(SelfAttentionLayer, self).__init__()

        self.num_heads = num_heads
        self.head_size = hidden_size / num_heads
        if int(self.head_size) != self.head_size:
            raise AssertionError("head num must divide hidden size")

        self.project_query = Linear(hidden_size, hidden_size)
        self.project_key = Linear(hidden_size, hidden_size)
        self.project_value = Linear(hidden_size, hidden_size)
        self.layer_norm = LayerNorm((hidden_size,))

    def forward(self, token_activations, attention_masks=None):
        # should this function include layer norm?
        return self.layer_norm(
            multi_head_self_attention(
                token_activations,
                attention_masks,
                self.project_query,
                self.project_key,
                self.project_value,
            )
        )


class BidirectionalTransformerLayer(Module):
    def __init__(self, hidden_size, num_heads):
        super(BidirectionalTransformerLayer, self).__init__()

        self.self_attention_layer = SelfAttentionLayer(hidden_size, num_heads)

        self.residual_layer = NormedResidualLayer(hidden_size, hidden_size)

    def forward(self, token_activations, attention_masks):
        attention_output = self.self_attention_layer(token_activations, attention_masks)
        output = self.residual_layer(attention_output)

        return output


class Bert(Module):
    def __init__(self, config):
        super(Bert, self).__init__()

        default_config = {
            "vocab_size": 5024,
            "embedding_size": 5024,
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
        }
        default_config.update(config)
        config = default_config

        self.embedding = Embedding(config.vocab_size, config.embedding_size)
        self.transformer = Sequential(
            [BidirectionalTransformerLayer(config.embedding_size, config.head_num, config.dropout)] * config.layer_num
        )

    def forward(self, input_token_ids):
        embeddings = self.embedding.embed(input_token_ids)
        encodings = self.transformer(embeddings)
        output_ids = self.embedding.unembed(encodings)
        return output_ids


def bert_from_pytorch_save(file_path):
    import transformers

    pass
