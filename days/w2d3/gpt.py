"""Implementation of GPT-2."""

from math import log
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
        Qh, Kh, Vh = QKV.split(self.hidden_size, dim=-1)
        Q = rearrange(Qh, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_size)
        K = rearrange(Kh, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_size)
        V = rearrange(Vh, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_size)

        attn_logits = t.einsum("bhqi, bhki -> bhqk", Q, K) / (self.head_size ** 0.5)
        attn_logits_tril = t.tril(attn_logits)
        attn_logits_masked = attn_logits_tril.masked_fill(
            attn_logits_tril == 0,
            self.masked_logit_fill,
        )

        attn_probs = t.softmax(attn_logits_masked, dim=-1)

        pre_O = t.einsum("bhqv, bhvd -> bhqd", attn_probs, V)
        pre_Oh = rearrange(
            pre_O, "b h s d -> b s (h d)", h=self.num_heads, d=self.head_size
        )
        return self.O_linear(pre_Oh)


class GPT2Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        layer_norm_epsilon: float,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.attn = UnidirAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )
        # residual
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=layer_norm_epsilon),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        # residual

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: Tensor[batch, seq_len, hidden_size]
        """
        # do we need to scale the weights of the residual layers by sqrt(N=12)?
        residual1 = self.attn(self.ln1(x))
        mlp_input = x + residual1
        residual2 = self.mlp(mlp_input)
        return mlp_input + residual2


from dataclasses import dataclass
from torchtyping import TensorType


@dataclass
class GPT2Output:
    logits: TensorType["batch_size", "vocab_size"]
    final_encoding: TensorType["batch_size", "hidden_size"]


class GPT2(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        dropout,
        layer_norm_epsilon,
    ) -> None:
        super().__init__()
        self.vocab_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=hidden_size
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=max_position_embeddings,
            embedding_dim=hidden_size,
        )
        self.dropout = nn.Dropout(dropout)
        self.gpt_blocks = nn.Sequential(
            *[
                GPT2Block(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    layer_norm_epsilon=layer_norm_epsilon,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm((hidden_size,), eps=layer_norm_epsilon)

    def forward(self, input_ids: t.Tensor) -> GPT2Output:
        """
        input_ids: Tensor[batch, seq_len]
        """
        embeddings = self.vocab_embedding(input_ids)
        embeddings += self.position_embedding(t.arange(input_ids.shape[-1]))
        embeddings_do = self.dropout(embeddings)
        gpt_blocks_result = self.gpt_blocks(embeddings_do)
        final_encoding = self.layer_norm(gpt_blocks_result)[:, -1, :]
        logits = t.einsum(
            "...i, ji -> ...j", final_encoding, self.vocab_embedding.weight
        )
        return GPT2Output(logits=logits, final_encoding=final_encoding)


if __name__ == "__main__":
    import gpt_tests

    gpt_tests.test_unidirectional_attn(UnidirAttention)
    gpt_tests.test_gpt_block(GPT2Block)
    gpt_tests.test_gpt(GPT2)
