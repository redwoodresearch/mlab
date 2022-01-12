"""Implementation of GPT-2."""

import torch as t
import torch.nn as nn

from einops import rearrange


class UnidirAttention(nn.Module):
    """Each token can only attend to itself and previous tokens."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        masked_logit_fill: float = -1e4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.masked_logit_fill = masked_logit_fill

        assert hidden_size % num_heads == 0
        self.head_size = hidden_size // num_heads

        self.QKV_linear = nn.Linear(hidden_size, 3 * hidden_size)
        self.O_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: Tensor[batch, seq_len, hidden_size]
        """

        QKV: t.Tensor = self.QKV_linear(x)
        Qb, Kb, Vb = QKV.split(self.hidden_size, dim=-1)
        Q = rearrange(Qb, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_size)
        K = rearrange(Kb, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_size)
        V = rearrange(Vb, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_size)

        attn_logits = t.einsum("bhqi, bhki -> bhqk", Q, K) / (self.head_size ** 0.5)
        attn_logits_tril = t.tril(attn_logits)
        attn_logits_masked = attn_logits_tril.masked_fill(
            attn_logits_tril == 0,
            self.masked_logit_fill,
        )

        attn_probs = t.softmax(attn_logits_masked, dim=-1)

        pre_O = t.einsum("bhqv, bhvd -> bhqd", attn_probs, V)
        pre_Ob = rearrange(
            pre_O, "b h s d -> b s (h d)", h=self.num_heads, d=self.head_size
        )
        return self.O_linear(pre_Ob)


if __name__ == "__main__":
    import gpt_tests

    gpt_tests.test_unidirectional_attn(UnidirAttention)
