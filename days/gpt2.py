import torch as t
import numpy as np
from torch.nn import Module, Parameter, ModuleList, Sequential  # not allowed to use other stuff from nn
from transformers import AutoTokenizer

# from days.modules import gelu, Embedding, Dropout, LayerNorm, softmax, Linear
from torch.nn import Embedding, Dropout, LayerNorm, Linear, Conv1d
from torch.nn.functional import gelu, softmax
from einops import rearrange
from utils import tpeek, tstat, copy_weight_bias
from dataclasses import dataclass


class ThatWhichOaiCallsConv1d(Module):
    def __init__(self, n_out, n_in):
        super().__init__()
        self.weight = Parameter(t.FloatTensor(n_out, n_in).normal_(0, 0.02))
        self.bias = Parameter(t.FloatTensor(n_out))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = t.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class GPT2Attention(Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        self.c_attn = Conv1d(hidden_size * 3, hidden_size)
        self.c_proj = Conv1d(hidden_size, hidden_size)
        self.dropout = Dropout(config["dropout"])

    def forward(self, encodings):
        query, key, value = self.c_attn(encodings)
        tpeek(query)
        query = rearrange()
        key = rearrange()
        value = rearrange()

        attn_output, attn_weights = self._attn(query, key, value)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.dropout(attn_output)

        return attn_output  # a, present, (attentions)


class GPT2Layer(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_norm_1 = LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.attention = GPT2Attention(config)
        self.fc1 = Conv1d()
        self.fc2 = Conv1d()
        self.layer_norm_2 = LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])

    def forward(self, encodings: t.Tensor):
        attention_output = self.attention(self.layer_norm_1(encodings))
        mlp_input = self.layer_norm_2(encodings + attention_output)
        mlp_output = self.fc2(gelu(self.fc1(mlp_input)))
        return mlp_input + mlp_output


@dataclass
class GPT2Output:
    logits: t.Tensor
    final_embedding: t.Tensor


class GPT2Core(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = ModuleList([GPT2Layer(config) for _ in range(config["num_layers"])])

    def forward(self, embeddings):
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
        self.config = config
        self.token_embedding = Embedding(config["vocab_size"], config["hidden_size"])
        self.position_embedding = Embedding(config["max_position_embeddings"], config["hidden_size"])
        self.dropout = Dropout(config["dropout"])
        self.transformer = GPT2Core(config)
        self.layer_norm_final = LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])

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
    their_model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    my_model = GPT2(their_model.config)
    my_model
    # not supporting cross attention


if __name__ == "__main__":
    import transformers

    my_gpt = GPT2({})
    my_gpt(input_ids=t.LongTensor(2, 2).fill_(1))

    their_model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    print(their_model)
    print(their_model.config)
