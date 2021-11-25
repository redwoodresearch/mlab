from typing import *
import torch as t
import numpy as np
from torch.nn import Module, Parameter, ModuleList, Sequential  # not allowed to use other stuff from nn
from transformers import AutoTokenizer

from days.modules import Embedding, Dropout, Linear, LayerNorm, gelu, softmax
from torchtyping import TensorType

# from torch.nn import LayerNorm
# from torch.nn.functional import gelu, softmax
from einops import rearrange
from utils import tpeek, tstat, copy_weight_bias
from dataclasses import dataclass
import transformers

from torch import nn


class GPT2Attention(Module):
    def __init__(self, config):
        super().__init__()
        config = convert_hf_to_my_config(config)
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

    def forward(
        self,
        encodings,
        attention_masks=None,
        past_key_values: Optional[Tuple[TensorType["h", "s", "head_size"], ...]] = None,
    ):
        num_heads = self.config["num_heads"]
        head_size = self.config["hidden_size"] // num_heads
        output_sequence_length = encodings.shape[1]
        input_sequence_length = encodings.shape[1]
        cattn = self.c_attn(encodings)
        query, key, value = t.split(cattn, self.config["hidden_size"], -1)
        query = rearrange(query, "b s (h c) -> b h s c", h=num_heads)

        key = rearrange(key, "b s (h c) -> b h s c", h=num_heads)
        value = rearrange(value, "b s (h c) -> b h s c", h=num_heads)

        if past_key_values is not None:
            new_kvs = (key[0], value[0])
            print(past_key_values[0].shape, key.shape)
            key = t.cat([past_key_values[0].unsqueeze(0), key], dim=2)
            value = t.cat([past_key_values[1].unsqueeze(0), value], dim=2)
            input_sequence_length += past_key_values[0].shape[1]

        attention_raw = t.einsum("bhtc,bhfc->bhft", query, key) / np.sqrt(head_size)

        unidirectional_mask = self.mask[:, :, :input_sequence_length, :output_sequence_length]
        attention_raw = t.where(unidirectional_mask, attention_raw, self.masked_bias)
        if attention_masks is not None:
            attention_raw = attention_raw * attention_masks
        attention_patterns = nn.Softmax(dim=-2)(attention_raw)

        # print("value shape", value.shape)
        # print("attention shape", attention_raw.shape)
        context_layer = t.einsum("bhft,bhfc->bhtc", attention_patterns, value)
        attention_values = rearrange(context_layer, "b h s c -> b s (h c)")
        attention_values = self.c_proj(attention_values)
        if past_key_values is not None:
            return attention_values, new_kvs
        return attention_values


class GPT2Layer(Module):
    def __init__(self, config):
        super().__init__()
        config = convert_hf_to_my_config(config)
        self.config = config
        hidden_size = config["hidden_size"]
        self.layer_norm_1 = LayerNorm((hidden_size,), eps=config["layer_norm_epsilon"])
        self.attention = GPT2Attention(config)
        # gpt2 calls these fully connected layers "conv1d", but they're actually 100% identical to linear layers
        self.fc1 = Linear(hidden_size, hidden_size * 4)
        self.fc2 = Linear(hidden_size * 4, hidden_size)
        self.layer_norm_2 = LayerNorm((hidden_size,), eps=config["layer_norm_epsilon"])
        self.dropout = Dropout(config["dropout"])

    def forward(self, x: t.Tensor, past_key_values=None):
        if past_key_values is not None:
            attention_output, new_kvs = self.attention(self.layer_norm_1(x), past_key_values=past_key_values)
        else:
            attention_output = self.attention(self.layer_norm_1(x))
        x = x + attention_output
        after_1_mlp = gelu(self.fc1(self.layer_norm_2(x)))
        mlpout = self.dropout(self.fc2(after_1_mlp))
        x = x + mlpout
        if past_key_values is not None:
            return x, new_kvs
        return x


@dataclass
class GPT2Output:
    logits: t.Tensor
    final_encoding: t.Tensor


class GPT2(Module):
    def __init__(self, config):
        super().__init__()
        config = convert_hf_to_my_config(config)
        default_config = {
            "bos_token_id": 50256,
            "eos_token_id": 50256,
            "initializer_range": 0.02,
            "layer_norm_epsilon": 1e-05,
            "hidden_size": 768,
            "num_heads": 12,
            "num_layers": 12,
            "max_position_embeddings": 1024,
            "dropout": 0.1,
            "scale_attn_weights": True,
            "use_cache": False,
            "vocab_size": 50257,
        }
        # convert config params from HF
        config = {**default_config, **config}
        self.config = config
        self.token_embedding = Embedding(config["vocab_size"], config["hidden_size"])
        self.position_embedding = Embedding(config["max_position_embeddings"], config["hidden_size"])
        self.dropout = Dropout(config["dropout"])
        self.blocks = Sequential(*[GPT2Layer(config) for _ in range(config["num_layers"])])
        self.layer_norm_final = LayerNorm((config["hidden_size"],), eps=config["layer_norm_epsilon"])
        # cache is some sequences of tokens
        # how much space per token? hidden_size*seq_length*(layers*2+1)
        # 768*100*25 = 10M, aka very little. Definitely have space for 10 sequences

        # I'm only caching batch size one rn because simple for teaching
        self.cache = []

    def forward(self, input_ids: t.LongTensor):
        seq_length = input_ids.shape[1]
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(t.arange(seq_length).to(next(self.parameters()).device))

        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)

        # cache only works for batch size 1
        if self.config["use_cache"] and input_ids.shape[0] == 1:
            input_ids = input_ids[0]
            for layer_num, (current_id, (id, layer_kvs, _last_embedding)) in enumerate(zip(input_ids, self.cache)):
                if current_id != id:
                    self.cache = self.cache[:layer_num]
                    break
            print("using cache")
            encodings = embeddings[:, len(self.cache) :]
            new_cache_stuff = [[id, [], None] for id in input_ids[len(encodings) :]]
            for layer_num, block in enumerate(self.blocks):
                # this silly if because t.cat doesn't work for 0 length list (which kinda makes sense)
                if len(self.cache) > 0:
                    past_keys = t.cat([layer_kvs[layer_num][0] for id, layer_kvs in self.cache])
                    past_values = t.cat([layer_kvs[layer_num][1] for id, layer_kvs in self.cache])
                else:
                    past_keys = t.zeros(12, 1, self.config["hidden_size"] // self.config["num_heads"])
                    past_values = t.zeros(12, 1, self.config["hidden_size"] // self.config["num_heads"])
                encodings, kvs = block(encodings, past_key_values=(past_keys, past_values))
                print("len kvs", len(kvs))
                print(len(kvs[0]))
                tpeek("kvs 0", kvs[0][0])
                tpeek("kvs 1", kvs[0][1])
                for id_num, (id, layer_kvs, _last_embedding) in enumerate(new_cache_stuff):
                    layer_kvs.append((kvs[0][:, id_num], kvs[1][:, id_num]))
            for layer_num, thing in enumerate(new_cache_stuff):
                thing[2] = encodings[:, 0, layer_num]
            self.cache.extend(new_cache_stuff)
            print(self.cache)
        else:
            print("not using cache")
            encodings = self.blocks(embeddings)

        encodings = self.layer_norm_final(encodings)
        final_encoding = encodings[:, -1, :]
        logits = t.einsum("...i,ji->...j", encodings, self.token_embedding.weight)
        return GPT2Output(logits=logits, final_encoding=final_encoding)


def convert_hf_to_my_config(hf_config):
    if isinstance(hf_config, dict):
        return hf_config
    hf_config = hf_config.to_dict()
    key_map = {
        "resid_pdrop": "dropout",  # mine doesn't confiure attention and resid dropout seperately
        "n_layer": "num_layers",
        "n_embd": "hidden_size",
        "n_head": "num_heads",
        "n_ctx": "max_position_embeddings",
    }
    return {(key_map.get(k, k)): v for k, v in hf_config.items()}


def copy_gpt2_attention_weights(my_attn, their_attn):
    copy_weight_bias(my_attn.c_attn, their_attn.c_attn, transpose=True)
    copy_weight_bias(my_attn.c_proj, their_attn.c_proj, transpose=True)


def copy_gpt2_layer_weights(my_layer, their_layer):
    copy_weight_bias(my_layer.fc1, their_layer.mlp.c_fc, transpose=True)
    copy_weight_bias(my_layer.fc2, their_layer.mlp.c_proj, transpose=True)

    copy_weight_bias(my_layer.layer_norm_1, their_layer.ln_1)
    copy_weight_bias(my_layer.layer_norm_2, their_layer.ln_2)
    copy_gpt2_attention_weights(my_layer.attention, their_layer.attn)


def copy_gpt2_weights(to_model, from_model):
    their_model: transformers.models.gpt2.modeling_gpt2.GPT2Model = from_model.transformer
    print(their_model.config)
    to_model.token_embedding.weight = their_model.wte.weight
    to_model.position_embedding.weight = their_model.wpe.weight
    for their_layer, my_layer in zip(their_model.h, to_model.blocks):
        copy_gpt2_layer_weights(my_layer, their_layer)

    copy_weight_bias(to_model.layer_norm_final, their_model.ln_f)


def my_gpt_from_hf_weights():
    their_lm_model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    my_model = GPT2({"use_cache": False})
    copy_gpt2_weights(my_model, their_lm_model)
    # not supporting cross attention
    return my_model, their_lm_model
