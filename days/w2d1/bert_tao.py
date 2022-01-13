import torch as t
import numpy as np
from torch.nn import (
    Module,
    Parameter,
    Sequential,
)  # not allowed to use other stuff from nn
from transformers import AutoTokenizer

# from days.modules import gelu, Embedding, Dropout, LayerNorm, softmax, Linear
from torch.nn import Embedding, Dropout, LayerNorm, Linear
from torch.nn.functional import gelu, softmax

from einops import rearrange
from days.utils import tpeek, copy_weight_bias
from dataclasses import dataclass


class BertEmbedding(Module):
    def __init__(self, config):
        super(BertEmbedding, self).__init__()
        self.config = config
        embedding_size = config["hidden_size"]
        # this needs to be initialized as a bunch of normalized vector,
        # as opposed to a linear layer, which is initialized to _produce_ normalized vectors
        self.token_embedding = Embedding(config["vocab_size"], embedding_size)
        self.position_embedding = Embedding(
            config["max_position_embeddings"], embedding_size
        )
        self.token_type_embedding = Embedding(config["type_vocab_size"], embedding_size)

        self.layer_norm = LayerNorm((embedding_size,))
        self.dropout = Dropout(config["dropout"])

    def embed(self, input_ids: t.LongTensor, token_type_ids):
        seq_length = input_ids.shape[1]
        token_embeddings = self.token_embedding(input_ids)
        token_type_embeddings = self.token_type_embedding(token_type_ids)
        position_embeddings = self.position_embedding(
            t.arange(seq_length).to(next(self.parameters()).device)
        )
        embeddings = token_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def forward(self, **kwargs):
        return self.embed(**kwargs)

    def unembed(self, embeddings: t.Tensor):
        return self.token_embedding.unembed(embeddings)


def bert_mlp(token_activations, linear_1, linear_2, dropout):
    return dropout(linear_2(gelu(linear_1(token_activations))))


class NormedResidualLayer(Module):
    def __init__(self, size, intermediate_size, dropout):
        super(NormedResidualLayer, self).__init__()
        self.mlp1 = Linear(size, intermediate_size, bias=True)
        self.mlp2 = Linear(intermediate_size, size, bias=True)
        self.layer_norm = LayerNorm((size,))
        self.dropout = Dropout(dropout)

    def forward(self, input):
        intermediate = gelu(self.mlp1(input))
        output = self.dropout(self.mlp2(intermediate)) + input
        output = self.layer_norm(output)
        return output


def raw_attention_pattern(
    token_activations, num_heads, project_query, project_key, attention_mask=None
):
    head_size = token_activations.shape[-1] // num_heads

    query = project_query(token_activations)
    query = rearrange(query, "b s (h c) -> b h s c", h=num_heads)

    key = project_key(token_activations)
    key = rearrange(key, "b s (h c) -> b h s c", h=num_heads)

    # my attention raw has twice the mean and half the variance of theirs
    attention_raw = t.einsum("bhtc,bhfc->bhft", query, key) / np.sqrt(head_size)
    if attention_mask is not None:
        attention_raw -= (1 - attention_mask) * 10000
    return attention_raw


def multi_head_self_attention(
    token_activations, num_heads, attention_pattern, project_value, project_out, dropout
):

    # if attention_masks is not None:
    #     attention_raw = attention_raw * attention_masks
    attention_patterns = softmax(attention_pattern, dim=-2)
    attention_patterns = dropout(attention_patterns)

    value = project_value(token_activations)
    value = rearrange(value, "b s (h c) -> b h s c", h=num_heads)

    context_layer = t.einsum("bhft,bhfc->bhtc", attention_patterns, value)
    attention_values = rearrange(context_layer, "b h s c -> b s (h c)")
    attention_values = project_out(attention_values)
    return attention_values


class AttentionPattern(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config["hidden_size"]
        self.project_query = Linear(hidden_size, hidden_size, bias=True)
        self.project_key = Linear(hidden_size, hidden_size, bias=True)

    def forward(self, token_activations, attention_masks):
        return raw_attention_pattern(
            token_activations=token_activations,
            num_heads=self.config["num_heads"],
            project_key=self.project_key,
            project_query=self.project_query,
        )


class SelfAttentionLayer(Module):
    def __init__(self, config):
        super(SelfAttentionLayer, self).__init__()
        self.config = config
        if config["hidden_size"] % config["num_heads"] != 0:
            raise AssertionError("head num must divide hidden size")
        hidden_size = config["hidden_size"]
        self.pattern = AttentionPattern(config)
        self.project_value = Linear(hidden_size, hidden_size, bias=True)
        self.project_out = Linear(hidden_size, hidden_size, bias=True)
        self.dropout = Dropout(config["dropout"])

    def forward(self, token_activations, attention_masks=None):
        return multi_head_self_attention(
            token_activations,
            # attention_masks,
            self.config["num_heads"],
            self.pattern(token_activations, attention_masks),
            self.project_value,
            self.project_out,
            self.dropout,
        )


class BertBlock(Module):
    def __init__(self, config):
        super(BertBlock, self).__init__()

        self.config = config
        hidden_size = config["hidden_size"]
        self.layer_norm = LayerNorm((hidden_size,))
        self.dropout = Dropout()
        self.attention = SelfAttentionLayer(config)

        self.residual = NormedResidualLayer(
            config["hidden_size"], config["intermediate_size"], config["dropout"]
        )

    def forward(self, token_activations, attention_masks=None):
        attention_output = self.layer_norm(
            token_activations
            + self.dropout(self.attention(token_activations, attention_masks))
        )

        return self.residual(attention_output)


class BertLMHead(Module):
    def __init__(self, config):
        super(BertLMHead, self).__init__()
        hidden_size = config["hidden_size"]
        self.mlp = Linear(hidden_size, hidden_size, bias=True)
        self.unembedding = Linear(hidden_size, config["vocab_size"], bias=True)
        self.layer_norm = LayerNorm((hidden_size,))

    def forward(self, activations):
        return self.unembedding(self.layer_norm(gelu(self.mlp(activations))))


@dataclass
class BertOutput:
    logits: t.Tensor
    encodings: t.Tensor
    classification: t.Tensor


class Bert(Module):
    def __init__(self, config, tokenizer=None):
        super(Bert, self).__init__()

        default_config = {
            "vocab_size": 28996,
            "intermediate_size": 3072,
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "max_position_embeddings": 512,
            "dropout": 0.1,
            "type_vocab_size": 2,
            "num_classes": 2,
        }
        config = {**default_config, **config}
        self.tokenizer = tokenizer
        self.config = config

        self.embedding = BertEmbedding(self.config)
        self.transformer = Sequential(
            *[BertBlock(self.config) for _ in range(self.config["num_layers"])]
        )
        self.lm_head = BertLMHead(config)
        self.classification_head = Linear(config["hidden_size"], config["num_classes"])
        self.classification_dropout = Dropout(config["dropout"])

    def forward(self, input_ids, token_type_ids=None):

        if token_type_ids is None:
            token_type_ids = t.zeros_like(input_ids).to(next(self.parameters()).device)

        embeddings = self.embedding.embed(
            input_ids=input_ids, token_type_ids=token_type_ids
        )
        encodings = self.transformer(embeddings)
        logits = self.lm_head(encodings)
        classification = self.classification_head(
            self.classification_dropout(encodings[:, 0])
        )
        return BertOutput(
            logits=logits, encodings=encodings, classification=classification
        )


def my_bert_from_hf_weights(their_lm_bert=None, config={}):
    import transformers

    if their_lm_bert is None:
        their_lm_bert: transformers.models.bert.modeling_bert.BertModel = (
            transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
        )
    model = their_lm_bert.bert
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    my_model = Bert({**their_lm_bert.config.to_dict(), **config}, tokenizer)
    # copy embeddings
    my_model.embedding.position_embedding.weight = (
        model.embeddings.position_embeddings.weight
    )
    my_model.embedding.token_embedding.weight = model.embeddings.word_embeddings.weight
    my_model.embedding.token_type_embedding.weight = (
        model.embeddings.token_type_embeddings.weight
    )
    copy_weight_bias(my_model.embedding.layer_norm, model.embeddings.LayerNorm)

    my_layers = list(my_model.transformer)
    official_layers = list(model.encoder.layer)
    for my_layer, their_layer in zip(my_layers, official_layers):
        my_layer: BertBlock

        copy_weight_bias(
            my_layer.attention.pattern.project_key, their_layer.attention.self.key
        )
        copy_weight_bias(
            my_layer.attention.pattern.project_query, their_layer.attention.self.query
        )
        copy_weight_bias(
            my_layer.attention.project_value, their_layer.attention.self.value
        )

        copy_weight_bias(
            my_layer.attention.project_out, their_layer.attention.output.dense
        )

        copy_weight_bias(my_layer.layer_norm, their_layer.attention.output.LayerNorm)

        copy_weight_bias(my_layer.residual.mlp1, their_layer.intermediate.dense)
        copy_weight_bias(my_layer.residual.mlp2, their_layer.output.dense)
        copy_weight_bias(my_layer.residual.layer_norm, their_layer.output.LayerNorm)

    copy_weight_bias(
        my_model.lm_head.mlp, their_lm_bert.cls.predictions.transform.dense
    )
    copy_weight_bias(
        my_model.lm_head.layer_norm, their_lm_bert.cls.predictions.transform.LayerNorm
    )

    # bias is output_specific, weight is from embedding
    my_model.lm_head.unembedding.bias = their_lm_bert.cls.predictions.decoder.bias
    my_model.lm_head.unembedding.weight = model.embeddings.word_embeddings.weight
    return my_model, their_lm_bert


if __name__ == "__main__":
    my_bert_from_hf_weights()
