"""
Bert by Lukas and Tony.
Adapted from days/w2d1/w2d2.ipynb on the branch lukas-tony.
"""

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, Union

import torch as t
import torch.nn.functional as F
from days.w2d1 import bert_tests
from einops import rearrange, reduce, repeat
from torch import einsum, nn
from torchtyping import TensorType as T
from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


@typechecked
def raw_attention_pattern(
    token_activations: T["batch", "seq_len", "hidden"],
    num_heads: int,
    project_query: Callable[[T["_":..., "hidden"]], T["_":..., "qk"]],
    project_key: Callable[[T["_":..., "hidden"]], T["_":..., "qk"]],
) -> T["batch", "num_heads", "key":"seq_len", "query":"seq_len"]:

    queries = rearrange(
        project_query(token_activations), "b s (head d) -> b head s d", head=num_heads
    )
    keys = rearrange(
        project_key(token_activations), "b s (head d) -> b head s d", head=num_heads
    )

    head_size = t.tensor(keys.shape[-1])
    return einsum("bhid, bhjd -> bhij", keys, queries) / t.sqrt(head_size)


if __name__ == "__main__":
    bert_tests.test_attention_pattern_fn(raw_attention_pattern)


@typechecked
def bert_attention(
    token_activations: T["batch", "seq_len", "d_in"],
    num_heads: int,
    attention_pattern: T["batch", "num_heads", "key":"seq_len", "query":"seq_len"],
    project_value: Callable[[T["_":..., "hidden"]], T["_":..., "d_v"]],
    project_output: Callable[[T["_":..., "d_v"]], T["_":..., "d_out"]],
) -> T["batch", "seq_len", "d_out"]:

    attention_prob = t.softmax(attention_pattern, dim=-2)  # dim: b head s s
    values = rearrange(
        project_value(token_activations), "b s (head d) -> b head s d", head=num_heads
    )

    output_by_head = einsum("bhis, bhid -> bhsd", attention_prob, values)
    concatenated = rearrange(output_by_head, "b h s d -> b s (h d)")

    return project_output(concatenated)


if __name__ == "__main__":
    bert_tests.test_attention_fn(bert_attention)


@typechecked
class MultiHeadedSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        attention_dim: int = 64,
        per_head_output_dim: int = 64,
        output_dim: Optional[int] = None,
    ):
        super().__init__()

        if output_dim is None:
            output_dim = hidden_size

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        self.per_head_output_dim = per_head_output_dim
        self.output_dim: int = output_dim

        self.Q = nn.Linear(
            in_features=hidden_size, out_features=num_heads * attention_dim
        )
        self.K = nn.Linear(
            in_features=hidden_size, out_features=num_heads * attention_dim
        )
        self.V = nn.Linear(
            in_features=hidden_size, out_features=num_heads * per_head_output_dim
        )
        self.O = nn.Linear(
            in_features=num_heads * per_head_output_dim, out_features=output_dim
        )

    def forward(self, input: T["batch", "seq_len", "hidden"]) -> t.Tensor:

        attention_pattern = raw_attention_pattern(
            input,
            self.num_heads,
            project_key=self.K,
            project_query=self.Q,
        )

        return bert_attention(
            token_activations=input,
            num_heads=self.num_heads,
            attention_pattern=attention_pattern,
            project_value=self.V,
            project_output=self.O,
        )


if __name__ == "__main__":
    bert_tests.test_bert_attention(MultiHeadedSelfAttention)
    mhsa = MultiHeadedSelfAttention(
        num_heads=17,
        hidden_size=73,
        attention_dim=37,
        per_head_output_dim=89,
        output_dim=2,
    )
    mhsa(t.ones((10, 117, 73))).shape


@typechecked
def bert_mlp(
    token_activations: T["batch", "seq_len", "hidden"],
    linear_1: nn.Module,
    linear_2: nn.Module,
) -> T["batch", "seq_len", "hidden"]:
    x = linear_1(token_activations)
    x = F.gelu(x)
    x = linear_2(x)
    return x


if __name__ == "__main__":
    bert_tests.test_bert_mlp(bert_mlp)


class BertMLP(nn.Module):
    def __init__(self, input_size: int, intermediate_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=input_size, out_features=intermediate_size
        )
        self.linear_2 = nn.Linear(
            in_features=intermediate_size, out_features=input_size
        )

    def forward(self, input: t.Tensor) -> t.Tensor:
        return bert_mlp(input, self.linear_1, self.linear_2)


class LayerNorm(nn.Module):
    def __init__(self, normalized_dim: int, eps=1e-5):
        super().__init__()
        self.weight = t.nn.Parameter(t.ones((normalized_dim,)))
        self.bias = t.nn.Parameter(t.zeros((normalized_dim,)))
        self.eps = eps

    def forward(self, input: t.Tensor):
        m = t.mean(input, dim=-1, keepdim=True).detach()
        v = t.var(input, dim=-1, keepdim=True, unbiased=False).detach()
        input = (input - m) / t.sqrt(v + self.eps)
        return input * self.weight + self.bias


if __name__ == "__main__":
    bert_tests.test_layer_norm(LayerNorm)


class BertBlock(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, num_heads: int, dropout: float
    ):
        super().__init__()

        self.mha = MultiHeadedSelfAttention(
            num_heads=num_heads, hidden_size=hidden_size
        )
        self.ln1 = LayerNorm(normalized_dim=hidden_size)
        self.bmlp = BertMLP(input_size=hidden_size, intermediate_size=intermediate_size)
        self.ln2 = LayerNorm(normalized_dim=hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input: t.Tensor) -> t.Tensor:
        x1 = self.ln1(self.mha(input) + input)
        return self.ln2(self.dropout(self.bmlp(x1)) + x1)


if __name__ == "__main__":
    bert_tests.test_bert_block(BertBlock)


@typechecked
class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.embedding = nn.Parameter(
            t.randn(
                (vocab_size, embed_size),
            )
        )

    def forward(self, input: T["_":..., int]) -> T["_":..., "embed_size", float]:
        return self.embedding[input]


if __name__ == "__main__":
    bert_tests.test_embedding(Embedding)


def bert_embedding(
    input_ids: T["batch", "seq_len"],
    token_type_ids: T["batch", "seq_len"],
    position_embedding: Embedding,
    token_embedding: Embedding,
    token_type_embedding: Embedding,
    layer_norm: LayerNorm,
    dropout: nn.Dropout,
) -> T["batch", "embed_size"]:
    seq_len = input_ids.shape[-1]
    device = input_ids.device

    inputs = token_embedding(input_ids)
    tokens = token_type_embedding(token_type_ids)
    positions = position_embedding(t.arange(seq_len, dtype=t.long, device=device))

    return dropout(layer_norm(inputs + tokens + positions))


if __name__ == "__main__":
    bert_tests.test_bert_embedding_fn(bert_embedding)


@typechecked
class BertEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        type_vocab_size: int,
        dropout: float,
    ):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, hidden_size)
        self.position_embedding = Embedding(max_position_embeddings, hidden_size)
        self.token_type_embedding = Embedding(type_vocab_size, hidden_size)

        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: t.Tensor, token_type_ids: t.Tensor) -> t.Tensor:
        return bert_embedding(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_embedding=self.position_embedding,
            token_embedding=self.token_embedding,
            token_type_embedding=self.token_type_embedding,
            layer_norm=self.layer_norm,
            dropout=self.dropout,
        )


if __name__ == "__main__":
    bert_tests.test_bert_embedding(BertEmbedding)


@typechecked
class Bert(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        type_vocab_size: int,
        dropout: float,
        intermediate_size: int,
        num_heads: int,
        num_layers: int,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.embedding = BertEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            dropout=dropout,
        )

        self.transformer = nn.Sequential(
            *[
                BertBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.lm_head = nn.Sequential(
            OrderedDict(
                [
                    (
                        "mlp",
                        nn.Linear(in_features=hidden_size, out_features=hidden_size),
                    ),
                    ("gelu", nn.GELU()),
                    ("layer_norm", LayerNorm(hidden_size)),
                    (
                        "unembedding",
                        nn.Linear(in_features=hidden_size, out_features=vocab_size),
                    ),
                ]
            )
        )
        # Tie embedding and unembedding weight matrices
        # self.lm_head.unembedding.weight = self.embedding.token_embedding.embedding

        self.classification_head = (
            None
            if num_classes is None
            else nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features=hidden_size, out_features=num_classes),
            )
        )

    def transformer_output(self, input_ids: t.Tensor) -> t.Tensor:
        token_type_ids = t.zeros_like(input_ids)
        embed = self.embedding(input_ids, token_type_ids)
        return self.transformer(embed)

    def forward(
        self, input_ids: t.Tensor
    ) -> Union[t.Tensor, Tuple[t.Tensor, t.Tensor]]:
        transformer_output = self.transformer_output(input_ids)
        lm_output = self.lm_head(transformer_output)
        if self.classification_head is not None:
            return (lm_output, self.classification_head(transformer_output[:, 0]))
        return lm_output


if __name__ == "__main__":
    bert_tests.test_bert(Bert)


def hf_to_our_state_dict(hf_dict: Dict[str, t.Tensor]) -> Dict[str, t.Tensor]:
    def include_key(key: str) -> bool:
        if key.startswith("classification_head"):
            return False
        return True

    def transform_key(key: str) -> str:
        subkeys = key.split(".")

        if key.startswith("embedding") and key.endswith("_embedding.weight"):
            subkeys[-1] = "embedding"
            return ".".join(subkeys)

        if subkeys[0] == "transformer":
            if subkeys[2] == "attention":
                subkeys[2] = "mha"

                if subkeys[3] == "pattern":
                    subkeys.pop(3)

                subkeys[3] = {
                    "project_value": "V",
                    "project_query": "Q",
                    "project_key": "K",
                    "project_out": "O",
                }[subkeys[3]]

                return ".".join(subkeys)

            if subkeys[2] == "residual":
                if subkeys[3] != "layer_norm":
                    subkeys[2] = "bmlp"
                    subkeys[3] = {"mlp1": "linear_1", "mlp2": "linear_2"}[subkeys[3]]
                    return ".".join(subkeys)

                if subkeys[3] == "layer_norm":
                    subkeys.pop(2)
                    subkeys[2] = "ln2"
                    return ".".join(subkeys)

            if subkeys[2] == "layer_norm":
                subkeys[2] = "ln1"
                return ".".join(subkeys)

        return key

    return {transform_key(k): v for k, v in hf_dict.items() if include_key(k)}


def load_pretrained_bert(num_classes: Optional[int] = None):
    my_bert = Bert(
        vocab_size=28996,
        hidden_size=768,
        max_position_embeddings=512,
        type_vocab_size=2,
        dropout=0.1,
        intermediate_size=3072,
        num_heads=12,
        num_layers=12,
        num_classes=num_classes,
    )
    pretrained_bert = bert_tests.get_pretrained_bert()

    my_bert.load_state_dict(
        hf_to_our_state_dict(pretrained_bert.state_dict()),
        strict=True,
    )

    return my_bert, pretrained_bert


if __name__ == "__main__":
    my_bert, pretrained_bert = load_pretrained_bert()
    bert_tests.test_same_output(my_bert, pretrained_bert, tol=1e-4)
