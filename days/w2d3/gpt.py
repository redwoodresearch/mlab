"""Implementation of GPT-2."""

import time
from collections import OrderedDict
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
        QKV_new: t.Tensor = self.QKV_linear(x[0, -1:])
        Qh_new, Kh_new, Vh_new = QKV_new.split(self.hidden_size, dim=-1)
        K_old, V_old = t.split(
            past_key_values, split_size_or_sections=self.head_size, dim=-1
        )
        K_new = rearrange(
            Kh_new, "1 (h d) -> h 1 d", h=self.num_heads, d=self.head_size
        )
        V_new = rearrange(
            Vh_new, "1 (h d) -> h 1 d", h=self.num_heads, d=self.head_size
        )

        K = t.concat((K_old, K_new), dim=-2)
        V = t.concat((V_old, V_new), dim=-2)

        Q_new = rearrange(
            Qh_new, "1 (h d) -> h 1 d", h=self.num_heads, d=self.head_size
        )

        attn_logits_new = t.einsum("hqi, hki -> hqk", Q_new, K) / (
            self.head_size ** 0.5
        )

        attn_probs_new = t.softmax(attn_logits_new, dim=-1)

        pre_O = t.einsum("hsv, hvd -> hsd", attn_probs_new, V)
        pre_Oh = rearrange(
            pre_O, "h 1 d -> 1 (h d)", h=self.num_heads, d=self.head_size
        )

        if return_key_values:
            return (
                self.O_linear(pre_Oh).unsqueeze(0),  # [1, 1, hidden_size]
                t.cat((K_new, V_new), dim=-1).unsqueeze(
                    0
                ),  # [1, num_heads, 2*head_size]
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

    def forward(
        self,
        x: t.Tensor,
        past_key_values: Optional[t.Tensor] = None,
        return_key_values: bool = False,
    ) -> t.Tensor:
        """
        x: Tensor[batch, seq_len, hidden_size]
        """
        attn_ret = self.attn.forward(
            self.ln1(x),
            past_key_values=past_key_values,
            return_key_values=return_key_values,
        )

        if not return_key_values:
            residual1 = attn_ret
        else:
            residual1, new_key_values = attn_ret

        mlp_input = x + residual1
        residual2 = self.mlp(mlp_input)
        output = mlp_input + residual2

        if not return_key_values:
            return output
        else:
            return output, new_key_values


@dataclass
class GPT2Output:
    logits: TensorType["batch_size", "vocab_size"]
    final_encoding: TensorType["batch_size", "hidden_size"]


class LRUDict:
    def __init__(self, max_size) -> None:
        self.max_size = max_size
        self.values = OrderedDict()

    def __setitem__(self, key, value):
        self.values[key] = value
        if len(self.values) > self.max_size:
            self.values.popitem()

    def __getitem__(self, key):
        if key in self.values:
            self.values.move_to_end(key)
        return self.values[key]

    def __contains__(self, key):
        return key in self.values


class GPT2(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        dropout: float,
        layer_norm_epsilon: float,
        use_cache: bool = False,
        tokenizer: Optional[transformers.AutoTokenizer] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.head_size = hidden_size // num_heads
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

        self.use_cache = use_cache
        self.cached_key_values = LRUDict(50)
        # if use_cache:
        #     self.cached_key_values = t.zeros(
        #         size=(num_layers, num_heads, 0, 2 * self.head_size),
        #         device=self.device,
        #     )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids: t.Tensor) -> GPT2Output:
        """
        input_ids: Tensor[batch, seq_len]
        """
        if self.use_cache:
            return self.forward_with_cache(input_ids)

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

    def forward_with_cache(self, input_ids: t.Tensor) -> GPT2Output:
        cache_key = tuple(input_ids[0, :-1].tolist())
        next_cache_key = tuple(input_ids[0, :].tolist())

        assert input_ids.shape[0] == 1

        embeddings = self.vocab_embedding(input_ids)
        embeddings += self.position_embedding(
            t.arange(input_ids.shape[-1], device=self.device)
        )
        embeddings_do = self.dropout(embeddings)

        if cache_key not in self.cached_key_values:
            # print("cache miss")
            # Cache is empty
            key_values_to_cache = []
            x = embeddings_do
            block: GPT2Block
            for block in self.gpt_blocks:
                x, block_key_values = block.forward(
                    x, past_key_values=None, return_key_values=True
                )
                key_values_to_cache.append(block_key_values.squeeze(0))

            self.cached_key_values[next_cache_key] = t.stack(key_values_to_cache, dim=0)

            final_encoding = self.layer_norm(x)[:, -1, :]
            logits = t.einsum(
                "...i, ji -> ...j", final_encoding, self.vocab_embedding.weight
            )
            return GPT2Output(logits=logits, final_encoding=final_encoding)

        # cache is not empty
        assert input_ids.shape[1] == self.cached_key_values[cache_key].shape[2] + 1

        key_values_to_cache = []
        x = embeddings_do
        block: GPT2Block
        for block_idx, block in enumerate(self.gpt_blocks):
            x, block_key_values = block.forward(
                x,
                past_key_values=self.cached_key_values[cache_key][block_idx],
                return_key_values=True,
            )
            key_values_to_cache.append(block_key_values.squeeze(0))

        self.cached_key_values[next_cache_key] = t.cat(
            (self.cached_key_values[cache_key], t.stack(key_values_to_cache, dim=0)),
            dim=2,
        )
        final_encoding = self.layer_norm(x)[:, -1, :]
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

        print("42 prob:", t.softmax(logits, dim=-1)[5433].item())

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

    def next_token(
        self,
        input_ids: t.Tensor,  # [seq_len]
        temperature: float,
        freq_penalty: float = 2.0,
    ) -> t.long:
        self.eval()
        id_frequencies = t.bincount(input_ids, minlength=self.vocab_size)
        with t.no_grad():
            gpt_output = self.forward(input_ids.unsqueeze(0))
            logits = gpt_output.logits[0]
            probs = t.softmax(
                logits / temperature - id_frequencies * freq_penalty, dim=-1
            )
        return t.multinomial(probs, 1)[0]

    def generate(
        self,
        text: str,
        max_length: int = 30,
        temperature: float = 1.0,
        freq_penalty: float = 2.0,
    ) -> str:
        self.cached_key_values = None
        token_ids = self.tokenizer(text)["input_ids"]
        eos_token_id = self.tokenizer.eos_token_id
        while len(token_ids) < max_length and token_ids[-1] != eos_token_id:
            input_ids = t.tensor(token_ids, dtype=t.long, device=self.device)
            token_ids.append(
                self.next_token(
                    input_ids=input_ids,
                    temperature=temperature,
                    freq_penalty=freq_penalty,
                ).item()
            )
        return self.tokenizer.decode(token_ids)

    def beam_next_token(self):
        pass

    def beam_generate(
        self,
        base_text: str,
        max_length: int = 30,
        beam_size: int = 5,
        max_completions_to_return: int = 10,
        temperature: float = 1.0,
        freq_penalty: float = 2.0,
    ) -> List[Tuple[List[str], float]]:
        self.eval()

        input_ids: List[int] = self.tokenizer(base_text)["input_ids"]
        eos_token_id: int = self.tokenizer.eos_token_id

        def seq_is_done(seq: List[int]) -> bool:
            return len(seq) == max_length or seq[-1] == eos_token_id

        active_seqs: List[Tuple[List[int], float]] = [(input_ids, 0.0)]
        while True:

            # Check if we are finished
            if all(seq_is_done(s) for s, _ in active_seqs):
                break

            # Expand forward
            new_seqs: List[Tuple[List[int], float]] = []
            for seq, seq_log_prob in active_seqs:
                if seq_is_done(seq):
                    new_seqs.append(seq)
                    continue
                input_ids = t.tensor(
                    [seq],
                    dtype=t.long,
                    device=self.device,
                )
                with t.no_grad():
                    gpt_out = self.forward(input_ids)
                    logits = gpt_out.logits[0]
                    id_frequencies = t.bincount(input_ids[0], minlength=self.vocab_size)
                    log_probs = t.log_softmax(
                        logits / temperature - id_frequencies * freq_penalty, dim=-1
                    )

                top_next_ids = t.argsort(log_probs, descending=True)
                for next_id in top_next_ids[:beam_size]:
                    new_seqs.append(
                        (
                            seq + [next_id.item()],
                            seq_log_prob + log_probs[next_id],
                        )
                    )

            # Filter new_seqs to top beam_size sequences
            active_seqs = sorted(
                new_seqs,
                key=lambda sp: sp[1],
                reverse=True,
            )[:beam_size]

        # Convert sequences of ids to strings
        return [
            (self.tokenizer.decode(seq), log_p)
            for seq, log_p in active_seqs[:max_completions_to_return]
        ]


def transfer_weights(my_gpt: GPT2, o_gpt: nn.Module):
    my_keys, _ = zip(*my_gpt.state_dict().items())
    _, o_vals = zip(*o_gpt.state_dict().items())
    assert len(my_keys) == len(o_vals)
    my_gpt.load_state_dict(dict(zip(my_keys, o_vals)))
    my_gpt.tokenizer = o_gpt.tokenizer


def get_gpt_with_pretrained_weights(use_cache=True) -> GPT2:
    my_gpt = GPT2(
        num_layers=12,
        num_heads=12,
        vocab_size=50257,
        hidden_size=768,
        max_position_embeddings=1024,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
        use_cache=use_cache,
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
    # gpt_tests.test_attn_cache(UnidirAttention)
    # gpt_tests.test_gpt_cache(GPT2)

    # my_gpt = get_gpt_with_pretrained_weights(use_cache=False)
    # for word, prob in my_gpt.get_top_predictions(
    #     "The meaning of life, the universe, and everything is", top_k=20
    # ):
    #     print(repr(word), prob.item())
    # for word, prob in my_gpt.get_top_predictions("The meaning of life is", top_k=20):
    #     print(repr(word), prob.item())

    # print(my_gpt.tokenizer.encode("42"))
    # print(my_gpt.tokenizer.encode(" 42"))
    # print(my_gpt.tokenizer.encode("43"))
    # print(my_gpt.tokenizer.encode(" 43"))

    # print(
    #     my_gpt.tokenizer.decode(
    #         my_gpt.tokenizer.encode(
    #             "The meaning of life, the universe, and everything is"
    #         )
    #         + [5433]
    #     )
    # )

    # my_gpt = get_gpt_with_pretrained_weights()
    # for _ in range(3):
    #     print(
    #         my_gpt.generate(
    #             "After a long day’s work, I like to",
    #             temperature=1.0,
    #             max_length=30,
    #         )
    #     )

    print("************************************")

    start = time.time()
    my_gpt = get_gpt_with_pretrained_weights(use_cache=False)
    beam_ret = my_gpt.beam_generate(
        base_text="After a long day’s work, I like to",
        temperature=1.0,
        max_length=30,
        beam_size=10,
        max_completions_to_return=3,
    )
    for seq, _ in beam_ret:
        print(seq)
    print(time.time() - start)

    start = time.time()
    my_gpt = get_gpt_with_pretrained_weights(use_cache=True)
    beam_ret = my_gpt.beam_generate(
        base_text="After a long day’s work, I like to",
        temperature=1.0,
        max_length=30,
        beam_size=10,
        max_completions_to_return=3,
    )
    for seq, _ in beam_ret:
        print(seq)
    print(time.time() - start)
