"""Implementation of GPT-2."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch as t
import torch.nn as nn
import transformers
from einops import rearrange
from torchtyping import TensorType

import gpt_tests


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

    def forward(
        self,
        x: t.Tensor,
        past_key_values: Optional[
            t.Tensor
        ] = None,  # [num_heads, seq_len, 2 * head_size]
        return_key_values: bool = False,
    ) -> t.Tensor:
        """
        x: Tensor[batch, seq_len, hidden_size]

        batch = 1 if past_key_values is not None
        """
        if past_key_values is not None:
            return self.forward_with_cached_values(
                x, past_key_values, return_key_values
            )

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

        if return_key_values:
            return self.O_linear(pre_Oh), t.cat((K, V), dim=-1)
        else:
            return self.O_linear(pre_Oh)

    def forward_with_cached_values(
        self,
        x: t.Tensor,
        past_key_values: t.Tensor,  # [num_heads, seq_len, 2 * head_size]
        return_key_values: bool,
    ):
        print(past_key_values.shape, "past key values")
        QKV_new: t.Tensor = self.QKV_linear(x[0, -1:])
        Qh_new, Kh_new, Vh_new = QKV_new.split(self.hidden_size, dim=-1)
        K_old, V_old = t.split(
            past_key_values, split_size_or_sections=self.head_size, dim=-1
        )
        K_new = rearrange(
            Kh_new, "s (h d) -> h s d", h=self.num_heads, d=self.head_size
        )
        V_new = rearrange(
            Vh_new, "s (h d) -> h s d", h=self.num_heads, d=self.head_size
        )
        K = t.concat((K_old, K_new), dim=-2)
        V = t.concat((V_old, V_new), dim=-2)

        Q_new = rearrange(
            Qh_new, "s (h d) -> h s d", h=self.num_heads, d=self.head_size
        )

        attn_logits_new = t.einsum("hqi, hki -> hqk", Q_new, K) / (
            self.head_size ** 0.5
        )

        attn_probs_new = t.softmax(attn_logits_new, dim=-1)

        pre_O = t.einsum("hsv, hvd -> hsd", attn_probs_new, V)
        pre_Oh = rearrange(
            pre_O, "h s d -> s (h d)", h=self.num_heads, d=self.head_size
        )

        if return_key_values:
            return (
                self.O_linear(pre_Oh).unsqueeze(0),
                t.cat((K_new, V_new), dim=-1).unsqueeze(0),
            )
        else:
            return self.O_linear(pre_Oh).unsqueeze(0)


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
        tokenizer: Optional[transformers.AutoTokenizer] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        # transformers.AutoTokenizer.from_pretrained("gpt2")

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
        # cache shape: [num_layers, num_heads, seq_len, 2 * head_size]

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids: t.Tensor) -> GPT2Output:
        """
        input_ids: Tensor[batch, seq_len]
        """
        embeddings = self.vocab_embedding(input_ids)
        embeddings += self.position_embedding(
            t.arange(input_ids.shape[-1], device=self.device)
        )
        embeddings_do = self.dropout(embeddings)
        gpt_blocks_result = self.gpt_blocks(embeddings_do)
        final_encoding = self.layer_norm(gpt_blocks_result)[:, -1, :]
        logits = t.einsum(
            "...i, ji -> ...j", final_encoding, self.vocab_embedding.weight
        )
        return GPT2Output(logits=logits, final_encoding=final_encoding)

    def get_next_word_logits(self, text: str) -> t.Tensor:

        input_ids = t.tensor(
            self.tokenizer([text])["input_ids"], dtype=t.long, device=self.device
        )
        output = self.forward(input_ids)
        return output.logits[-1]

    def get_top_predictions(self, text: str, top_k: int) -> Tuple[List[str], t.Tensor]:
        logits = self.get_next_word_logits(text)

        top_pred_ids = t.sort(logits, descending=True).indices[:top_k]
        top_probs = t.softmax(logits, dim=-1)[top_pred_ids]

        words = [
            text + self.tokenizer.decode([top_pred_id]) for top_pred_id in top_pred_ids
        ]
        return zip(words, top_probs)

    def gen_next_word(self, text: str) -> str:
        self.eval()
        with t.no_grad():
            logits = self.get_next_word_logits(text)
        pred_id = logits.argmax()
        return self.tokenizer.decode(pred_id.item())


def transfer_weights(my_gpt: GPT2, o_gpt: nn.Module):
    my_keys, _ = zip(*my_gpt.state_dict().items())
    _, o_vals = zip(*o_gpt.state_dict().items())
    assert len(my_keys) == len(o_vals)
    my_gpt.load_state_dict(dict(zip(my_keys, o_vals)))
    my_gpt.tokenizer = o_gpt.tokenizer


def get_gpt_with_pretrained_weights() -> GPT2:
    my_gpt = GPT2(
        num_layers=12,
        num_heads=12,
        vocab_size=50257,
        hidden_size=768,
        max_position_embeddings=1024,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
    )
    pretrained_gpt = gpt_tests.get_pretrained_gpt()
    transfer_weights(my_gpt, pretrained_gpt)

    return my_gpt


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # test vanilla implementation
    # gpt_tests.test_unidirectional_attn(UnidirAttention)
    # gpt_tests.test_gpt_block(GPT2Block)
    # gpt_tests.test_gpt(GPT2)

    # test caching implementation
    gpt_tests.test_attn_cache(UnidirAttention)

    # my_gpt = get_gpt_with_pretrained_weights()
    # for word, prob in my_gpt.get_top_predictions("My life motto is to", top_k=10):
    #     print(repr(word), prob.item())
