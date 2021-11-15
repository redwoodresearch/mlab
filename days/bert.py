from typing import *
import torch as t
import numpy as np
from torch.nn import Module, Parameter, Sequential
import math
from torchtyping import TensorType, patch_typeguard


def softmax(tensor: t.Tensor, dim: int = 0):
    exps = math.e ** tensor
    exp_sums = exps.sum(dim=dim)
    result = exps / exp_sums
    return result


def relu(tensor: t.Tensor) -> t.Tensor:
    tensor[tensor < 0] = 0
    return tensor


def gelu(x):
    return 0.5 * x * (1 + t.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * t.pow(x, 3))))


# TODO: figure out what this should actually be
def normalize(tensor: t.Tensor, ndims: int = 2):
    mean = tensor
    variance = tensor
    for dim in ndims:
        mean = mean.mean(-1)
        variance = tensor.variance(-1)
    tensor = tensor / variance + mean
    return tensor


class LayerNorm(Module):
    def __init__(self, shape, normalize_dims, eps=1e-05):
        self.bias = Parameter(t.empty(shape))
        self.weight = Parameter(t.ones(shape))
        self.eps = eps
        self.normalize_dims = normalize_dims

    def forward(self, tensor):
        tensor = normalize(tensor, self.normalize_dims)
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
        weight_bound = 1 / t.sqrt(x)
        self.weight = Parameter(t.random.uniform(-weight_bound, weight_bound, (x, y)))
        bias_bound = 1 / math.sqrt(y)
        self.bias = Parameter(t.random.uniform(-bias_bound, bias_bound, (y,)))

    def forward(self, x: TensorType[..., "channels"]) -> TensorType[..., "channels"]:
        return t.einsum("ij,jk->ik", x, self.weight) + self.bias


class Embedding(Module):
    def __init__(self, vocab_size: int, embedding_size: int):
        # this needs to be initialized as a bunch of normalized vector,
        # as opposed to a linear layer, which is initialized to _produce_ normalized vectors
        self.embedding = Parameter(t.random.normal(0, 1, (vocab_size, embedding_size)))

        # then zero out padding tokens and/or whatever

    def embed(self, x: TensorType[..., t.long]):
        return self.embedding[x]

    def unembed(self, x: TensorType["...", "embed_dim"]):
        return t.einsum("ij,kj->ik", x, self.embedding)


class NormedResidualLayer(Module):
    def __init__(self, size):
        super(NormedResidualLayer, self).__init__()
        self.mlp = Linear(size, size)
        self.layer_norm = LayerNorm((size,))

    def forward(self, input):
        return self.layer_norm(input + self.mlp(input))


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

    def forward(self, token_activations, attention_masks):
        query = t.einsum("ij,jk->ik", token_activations, self.project_query)
        query = t.stack(t.split(query, self.num_heads, dim=-1), dim=0)

        key = t.einsum("ij,jk->ik", token_activations, self.project_key)
        key = t.stack(t.split(key, self.num_heads, dim=-1), dim=0)

        value = t.einsum("ij,jk->ik", token_activations, self.project_value)
        value = t.stack(t.split(value, self.num_heads, dim=-1), dim=0)

        attention_raw = t.einsum("hbsc,hbsc->hbs", query, key) / math.sqrt(self.head_size)

        attention_masked = attention_raw * attention_masks
        attention_patterns = softmax(attention_masked)

        attention_values = t.einsum("hbs,hbsc->hbsc", attention_patterns, value)
        output = token_activations + t.cat(t.split(attention_values, dim=0), dim=-1)
        return output


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
    pass
