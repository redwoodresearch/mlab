import torch as t
import numpy as np
from torch.nn import Module, Parameter, Sequential
import math

from torchtyping import TensorType, patch_typeguard
from einops import rearrange


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
        tensor = tensor * self.weight + self.bias
        return tensor


class Dropout(Module):
    def __init__(self, fraction=0.1):
        super(Dropout, self).__init__()
        self.fraction = fraction

    def forward(self, input):
        if self.training:
            mask = t.random.uniform() > self.fraction
            return mask * input
        return input


class Linear(Module):
    def __init__(self, x, y, bias):
        super(Linear, self).__init__()
        weight_bound = 1 / np.sqrt(x)
        self.weight = Parameter(t.FloatTensor(x, y).uniform_(-weight_bound, weight_bound))
        if bias:
            bias_bound = 1 / np.sqrt(y)
            self.bias = Parameter(t.FloatTensor(y).uniform_(-bias_bound, bias_bound))
        else:
            self.bias = None

    def forward(self, x: TensorType[..., "channels"]) -> TensorType[..., "channels"]:
        x = t.matmul(x, self.weight)
        if self.bias is not None:
            x += self.bias
        return x


class Embedding(Module):
    def __init__(self, vocab_size: int, embedding_size: int):
        super(Embedding, self).__init__()
        self.weight = Parameter(t.FloatTensor(vocab_size, embedding_size).normal_(0, 1))
        self.embedding_size = embedding_size

    def embed(self, ids: t.LongTensor):
        return self.weight[ids]

    def unembed(self, embeddings):
        return t.einsum("...j,kj->...k", embeddings, self.weight)


class BertEmbedding(Module):
    def __init__(self, config):
        super(BertEmbedding, self).__init__()
        self.config = config
        embedding_size = config["hidden_size"]
        # this needs to be initialized as a bunch of normalized vector,
        # as opposed to a linear layer, which is initialized to _produce_ normalized vectors
        self.token_embedding = Embedding(config["vocab_size"], embedding_size)
        self.position_embedding = Embedding(config["max_position_embeddings"], embedding_size)
        self.token_type_embedding = Embedding(config["type_vocab_size"], embedding_size)

        # then zero out padding tokens and/or whatever
        self.unembed_layer_norm = LayerNorm((embedding_size,))
        self.layer_norm = LayerNorm((embedding_size,))
        self.dropout = Dropout(config["dropout"])

    def embed(self, token_ids: TensorType[..., t.long], token_type_ids):
        seq_length = token_ids.shape[1]
        token_embeddings = self.token_embedding.embed(token_ids)
        token_type_embeddings = self.token_type_embedding.embed(token_type_ids)
        position_embeddings = self.position_embedding.embed(t.arange(seq_length))
        embeddings = token_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def unembed(self, embeddings: TensorType["...", "embed_dim"]):
        return self.token_embedding.unembed(self.unembed_layer_norm(embeddings))


class NormedResidualLayer(Module):
    def __init__(self, size, intermediate_size, dropout):
        super(NormedResidualLayer, self).__init__()
        self.mlp1 = Linear(size, intermediate_size, bias=True)
        self.mlp2 = Linear(intermediate_size, size, bias=True)
        self.layer_norm = LayerNorm((size,))
        self.dropout = Dropout(dropout)

    def forward(self, input):
        intermediate = gelu(self.mlp1(input))
        output = self.mlp2(intermediate) + input
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output


def multi_head_self_attention(
    token_activations, attention_masks, num_heads, project_query, project_key, project_value, mlp
):
    head_size = token_activations.shape[-1] / num_heads
    query = project_query(token_activations)
    query = t.stack(t.split(query, num_heads, dim=-1), dim=0)

    key = project_key(token_activations)
    key = t.stack(t.split(key, num_heads, dim=-1), dim=0)

    value = project_value(token_activations)
    value = t.stack(t.split(value, num_heads, dim=-1), dim=0)

    attention_raw = t.einsum("hbsc,hbsc->hbs", query, key) / math.sqrt(head_size)
    if attention_masks:
        attention_raw = attention_raw * attention_masks
    attention_patterns = softmax(attention_raw)

    attention_values = t.einsum("hbs,hbsc->hbsc", attention_patterns, value)
    output = rearrange(attention_values, "h b s c -> b s (c h)")
    return mlp(output)


class SelfAttentionLayer(Module):
    def __init__(self, config):
        super(SelfAttentionLayer, self).__init__()
        self.config = config
        if config["hidden_size"] % config["num_heads"] != 0:
            raise AssertionError("head num must divide hidden size")
        hidden_size = config["hidden_size"]
        self.project_query = Linear(hidden_size, hidden_size, bias=True)
        self.project_key = Linear(hidden_size, hidden_size, bias=True)
        self.project_value = Linear(hidden_size, hidden_size, bias=True)
        self.mlp = Linear(hidden_size, hidden_size, bias=True)
        self.layer_norm = LayerNorm((hidden_size,))
        self.dropout = Dropout()

    def forward(self, token_activations, attention_masks=None):
        # should this function include layer norm?
        return self.layer_norm(
            token_activations
            + self.dropout(
                multi_head_self_attention(
                    token_activations,
                    attention_masks,
                    self.config["num_heads"],
                    self.project_query,
                    self.project_key,
                    self.project_value,
                    self.mlp,
                )
            )
        )


class BertLayer(Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()

        self.attention = SelfAttentionLayer(config)

        self.residual = NormedResidualLayer(config["hidden_size"], config["intermediate_size"], config["dropout"])

    def forward(self, token_activations, attention_masks=None):
        attention_output = self.attention(token_activations, attention_masks)

        return self.residual(attention_output)


class Bert(Module):
    def __init__(self, config):
        super(Bert, self).__init__()

        default_config = {
            "vocab_size": 28996,
            "embedding_size": 5024,
            "intermediate_size": 3072,
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "max_position_embeddings": 512,
            "dropout": 0.1,
            "type_vocab_size": 2,
        }
        self.config = {**default_config, **config}

        self.embedding = BertEmbedding(self.config)
        self.transformer = Sequential(*[BertLayer(self.config) for _ in range(self.config["num_layers"])])

    def forward(self, token_ids, token_type_ids):
        embeddings = self.embedding.embed(token_ids=token_ids, token_type_ids=token_type_ids)
        encodings = self.transformer(embeddings)
        output_ids = self.embedding.unembed(encodings)
        return output_ids


def bert_from_pytorch_save():
    import transformers
    from transformers import AutoModel

    bert_default_config = {
        "position_embedding_type": "absolute",
        "hidden_act": "gelu",
        "attention_probs_dropout_prob": 0.1,
        "classifier_dropout": None,
        "gradient_checkpointing": False,
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "transformers_version": "4.11.3",
        "type_vocab_size": 2,
        "use_cache": True,
        "vocab_size": 28996,
    }
    model: transformers.models.bert.modeling_bert.BertModel = AutoModel.from_pretrained("bert-base-cased")
    my_model = Bert(bert_default_config)

    def has_not_null(obj, prop):
        return hasattr(obj, prop) and (getattr(obj, prop) is not None)

    def copy_maybe_transpose(mine, theirs):
        if tuple(mine.weight.shape) == tuple(theirs.weight.shape):
            mine.weight = theirs.weight
        else:
            mine.weight = Parameter(t.transpose(theirs.weight, 1, 0))

    def copy_weight_bias(mine, theirs):
        copy_maybe_transpose(mine, theirs)
        theirs_has_bias = has_not_null(theirs, "bias")
        mine_has_bias = has_not_null(mine, "bias")
        if theirs_has_bias != mine_has_bias:
            print(mine.bias)
            raise AssertionError("yikes")
        if mine_has_bias and theirs_has_bias:
            mine.bias = theirs.bias

    # copy embeddings
    my_model.embedding.position_embedding.weight = model.embeddings.position_embeddings.weight
    my_model.embedding.token_embedding.weight = model.embeddings.word_embeddings.weight
    my_model.embedding.token_type_embedding.weight = model.embeddings.token_type_embeddings.weight
    copy_weight_bias(model.embeddings.LayerNorm, my_model.embedding.layer_norm)

    my_layers = list(my_model.transformer)
    official_layers = list(model.encoder.layer)
    for my_layer, their_layer in zip(my_layers, official_layers):
        my_layer: BertLayer
        # their_layer:transformers.

        copy_weight_bias(my_layer.attention.project_key, their_layer.attention.self.key)
        copy_weight_bias(my_layer.attention.project_query, their_layer.attention.self.query)
        copy_weight_bias(my_layer.attention.project_value, their_layer.attention.self.value)

        copy_weight_bias(my_layer.attention.mlp, their_layer.attention.output.dense)

        copy_weight_bias(my_layer.attention.layer_norm, their_layer.attention.output.LayerNorm)

        copy_weight_bias(my_layer.residual.mlp1, their_layer.intermediate.dense)
        copy_weight_bias(my_layer.residual.mlp2, their_layer.output.dense)
        copy_weight_bias(my_layer.residual.layer_norm, their_layer.output.LayerNorm)

    return my_model, model
