import torch
import einops
import torch.nn as nn
import gpt_tests
from math import sqrt


class AttnHeads(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super(AttnHeads, self).__init__()
        self.head_size = hidden_size // num_heads
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # TODO: check if this is the right dim
        self.softmax = nn.Softmax(dim=-2)

        self.qkv = nn.Parameter(torch.rand(hidden_size, hidden_size * 3))
        self.out_weight = nn.Parameter(torch.rand(hidden_size, hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor: # [batch, seq_len, hidden_size]
        # x : [batch, seq_len, hidden_size]

        # [batch, seq_length, hidden_size] (for each)
        q_pre, k_pre, v_pre = torch.split(torch.einsum('bsh,hj->bsj', x, self.qkv), self.hidden_size, dim=-1)

        # [batch, num_heads, seq_length, head_size]
        pattern = 'b s (n d) -> b n s d'
        args = {'n': self.num_heads, 'd': self.head_size}
        q = einops.rearrange(q_pre, pattern, **args)
        k = einops.rearrange(k_pre, pattern, **args)
        v = einops.rearrange(v_pre, pattern, **args)

        # [num_heads, hidden_size, head_size]
        out = einops.rearrange(self.out_weight, '(n d) h -> n d h', **args)

        # [batch, num_heads, seq_len, seq_len]
        # attn_score: torch.Tensor = torch.einsum('ab,bcd,ced,ef->afd', x, self.K_weight, self.Q_weight, x)
        attn_score: torch.Tensor = torch.einsum('bnsd,bntd->bnst', k, q)
        attn: torch.Tensor = self.softmax(attn_score)

        # TODO: enforce unidirectionality

        # TODO: adjust attn score to -1e4

        # TODO: check that this is the right order for attn indices
        # [batch, seq_len, hidden_size]
        output: torch.Tensor = torch.einsum('bnsd,nde,bnst->bte', v, out, attn)

        return output / sqrt(self.head_size)


gpt_tests.test_unidirectional_attn(AttnHeads)