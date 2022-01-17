from dataclasses import dataclass
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange


@dataclass
class GPT2TPConfig:
    hidden_size = 4096
    n_heads = 16
    tp_size = 2
    max_positions = 2048
    dropout = 0.1
    pos_enc_type = "RoPE"

    tp_rank = None
    part_hidden_size = None
    tp_dist_group = None

    def __init__(self):
        assert (self.hidden_size % self.tp_size) == 0
        assert (self.n_heads % self.tp_size) == 0
        self.part_hidden_size = self.hidden_size // self.tp_size


class GPT2AttentionTP(nn.Module):
    def __init__(self, config: GPT2TPConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.n_heads // config.tp_size
        self.qkv_projection = nn.Linear(config.hidden_size, 3 * config.hidden_size // config.tp_size)
        self.dropout = nn.Dropout(config.dropout)
        self.o_projection = nn.Linear(config.hidden_size // config.tp_size, config.hidden_size)

        self.register_buffer(
            "mask",
            t.triu(t.ones((config.max_positions, config.max_positions), dtype=t.bool)).unsqueeze(0).unsqueeze(0),
        )
        self.register_buffer("masked_bias", t.tensor(-1e4))

    def forward(self, x):
        b_len, s_len, _ = x.shape
        qkv = t.split(self.qkv_projection(x), 3, dim=-1)
        qkv = [rearrange(a, "b s (h c) -> b h s c", h=self.num_heads) for a in qkv]
        q, k, v = qkv
        attention_pattern = t.einsum("bhfc,bhtc->bhft", k, q)
        attention_pattern = t.where(self.mask[:, :, :s_len, :s_len], attention_pattern, self.masked_bias)
        attention_pattern = t.softmax(attention_pattern, dim=-2)
        attention_pattern = self.dropout(attention_pattern)
        values = t.einsum("bhft,bhfc->bhtc", attention_pattern, v)
        x = self.o_projection(rearrange(values, "b h s c -> b s (h c)"))
        dist.all_reduce(x, group=self.config.tp_dist_group)
        return x


class GPT2MLPTP(nn.Module):
    def __init__(self, config: GPT2TPConfig):
        super().__init__()
        self.config = config
        self.intermediate_size = config.hidden_size * 4 // config.tp_size
        self.linear1 = nn.Linear(config.hidden_size, self.intermediate_size)
        self.linear2 = nn.Linear(self.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        intermediate = F.gelu(self.dropout(self.linear1(x)))
        x = self.linear2(intermediate)
        dist.all_reduce(x, group=self.config.tp_dist_group)
        return x


class GPT2LayerTP(nn.Module):
    def __init__(self, config: GPT2TPConfig):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm((config.part_hidden_size))

        self.ln2 = nn.LayerNorm((config.part_hidden_size))
        self.attention = GPT2AttentionTP(config)
        self.attention = GPT2MLPTP(config)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
