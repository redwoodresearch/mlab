import torch as t
import numpy as np
from torch.nn import Module, Parameter, ModuleList, Sequential  # not allowed to use other stuff from nn
from transformers import AutoTokenizer

from days.modules import gelu, Embedding, Dropout, LayerNorm, softmax, Linear

# from torch.nn import Embedding, Dropout, LayerNorm, Linear
# from torch.nn.functional import gelu, softmax
from einops import rearrange
from utils import tpeek, tstat, copy_weight_bias
from dataclasses import dataclass
import transformers

from torch import nn


class GPT2Attention(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config["hidden_size"]
        max_positions = config["max_position_embeddings"]
        self.c_attn = Linear(hidden_size, hidden_size * 3)
        self.c_proj = Linear(hidden_size, hidden_size)
        self.dropout = Dropout(config["dropout"])
        self.register_buffer(
            "mask",
            t.tril(t.ones((max_positions, max_positions), dtype=t.bool)).view(1, 1, max_positions, max_positions),
        )
        self.masked_bias = t.tensor(-1e4)

    def forward(self, encodings, attention_masks=None, past_key_values=None):
        num_heads = self.config["num_heads"]
        head_size = self.config["hidden_size"] // num_heads
        sequence_length = encodings.shape[1]
        cattn = self.c_attn(encodings)
        query, key, value = t.split(cattn, self.config["hidden_size"], -1)
        query = rearrange(query, "b s (h c) -> b h s c", h=num_heads)
        key = rearrange(key, "b s (h c) -> b h s c", h=num_heads)
        value = rearrange(value, "b s (h c) -> b h s c", h=num_heads)

        attention_raw = t.einsum("bhfc,bhtc->bhft", query, key) / np.sqrt(head_size)

        unidirectional_mask = self.mask[:, :, :sequence_length, :sequence_length]
        attention_raw = t.where(unidirectional_mask, attention_raw, self.masked_bias)
        # if attention_masks is not None:
        #     attention_raw = attention_raw * attention_masks
        tpeek("my pre softmax", attention_raw)
        attention_patterns = nn.Softmax(dim=-1)(attention_raw)
        tpeek("my patterns", attention_patterns)

        context_layer = t.einsum("bhft,bhtc->bhfc", attention_patterns, value)
        attention_values = rearrange(context_layer, "b h s c -> b s (h c)")
        tpeek("my ctx", attention_values)
        return attention_values  # a, present, (attentions)


class GPT2Layer(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config["hidden_size"]
        self.layer_norm_1 = LayerNorm((hidden_size,), eps=config["layer_norm_eps"])
        self.attention = GPT2Attention(config)
        # gpt2 calls these fully connected layers "Linear", but they're actually just linear layers, not Linear layers
        self.fc1 = Linear(hidden_size, hidden_size * 4)
        self.fc2 = Linear(hidden_size * 4, hidden_size)
        self.layer_norm_2 = LayerNorm((hidden_size,), eps=config["layer_norm_eps"])
        self.dropout = Dropout(config["dropout"])

    def forward(self, x: t.Tensor, past_key_values=None):
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.dropout(self.fc2(gelu(self.fc1(self.layer_norm_2(x)))))
        return x


@dataclass
class GPT2Output:
    logits: t.Tensor
    final_embedding: t.Tensor


class GPT2Core(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = ModuleList([GPT2Layer(config) for _ in range(config["num_layers"])])
        # cache is map from inputs as tuples to sequences of key/values
        self.cache = {}

    def forward(self, embeddings, use_cache=False):
        if use_cache:
            return Sequential(*self.blocks)(embeddings)
        else:
            return Sequential(*self.blocks)(embeddings)


class GPT2(Module):
    def __init__(self, config):
        super().__init__()
        default_config = {
            "bos_token_id": 50256,
            "eos_token_id": 50256,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-05,
            "ctx_size": 1024,
            "hidden_size": 768,
            "num_heads": 12,
            "num_layers": 12,
            "max_position_embeddings": 1024,
            "dropout": 0.1,
            "scale_attn_weights": True,
            "use_cache": True,
            "vocab_size": 50257,
        }
        config = {**default_config, **config}
        config["use_cache"] = False
        self.config = config
        self.token_embedding = Embedding(config["vocab_size"], config["hidden_size"])
        self.position_embedding = Embedding(config["max_position_embeddings"], config["hidden_size"])
        self.dropout = Dropout(config["dropout"])
        self.transformer = GPT2Core(config)
        self.layer_norm_final = LayerNorm((config["hidden_size"],), eps=config["layer_norm_eps"])

    def forward(self, input_ids: t.LongTensor):
        seq_length = input_ids.shape[1]
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(t.arange(seq_length).to(next(self.parameters()).device))
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        encodings = self.transformer(embeddings)
        encodings = self.layer_norm_final(encodings)
        final_encoding = encodings[:, -1]
        logits = t.einsum("...i,ji->...j", final_encoding, self.token_embedding.weight)
        return GPT2Output(logits=logits, final_embedding=final_encoding)


def my_gpt_from_hf_weights():
    their_lm_model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    their_model: transformers.models.gpt2.modeling_gpt2.GPT2Model = their_lm_model.transformer
    my_model = GPT2({})

    my_model.token_embedding.weight = their_model.wte.weight
    my_model.token_embedding.weight = their_model.wpe.weight
    for their_layer, my_layer in zip(their_model.h, my_model.transformer.blocks):
        copy_weight_bias(my_layer.fc1, their_layer.mlp.c_fc, transpose=True)
        copy_weight_bias(my_layer.fc2, their_layer.mlp.c_proj, transpose=True)

        copy_weight_bias(my_layer.layer_norm_1, their_layer.ln_1)
        copy_weight_bias(my_layer.layer_norm_2, their_layer.ln_2)

        copy_weight_bias(my_layer.attention.c_attn, their_layer.attn.c_attn, transpose=True)
        copy_weight_bias(my_layer.attention.c_proj, their_layer.attn.c_proj, transpose=True)

    copy_weight_bias(my_model.layer_norm_final, their_model.ln_f)
    # not supporting cross attention
    return my_model, their_lm_model


if __name__ == "__main__":

    my_gpt = GPT2({})
    my_gpt(input_ids=t.LongTensor(2, 2).fill_(1))

    their_model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
