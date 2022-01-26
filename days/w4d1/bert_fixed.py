"""
Bert by Lukas and Tony.
Adapted from days/w2d1/w2d2.ipynb on the branch lukas-tony.
"""

from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import torch as t
import torch.nn.functional as F
from days.w2d1 import bert_tests
from torch import nn
from torchtyping import TensorType as T
from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked

from days.w2d1.bert_sol import (
    MultiHeadedSelfAttention,
    BertMLP,
    BertEmbedding,
    BertBlock,
)


class LayerNorm(nn.Module):
    def __init__(self, normalized_dim: int, eps=1e-5):
        super().__init__()
        self.weight = t.nn.Parameter(t.ones((normalized_dim,)))
        self.bias = t.nn.Parameter(t.zeros((normalized_dim,)))
        self.eps = eps

    # Working code
    # This version removes "grokking"
    # def forward(self, input: t.Tensor):
    #     m = t.mean(input, dim=-1, keepdim=True)
    #     v = t.var(input, dim=-1, keepdim=True, unbiased=False)
    #     input = (input - m) / t.sqrt(v + self.eps)
    #     return input * self.weight + self.bias

    # Broken code
    def forward(self, input: t.Tensor):
        m = t.mean(input, dim=-1, keepdim=True).detach()
        v = t.var(input, dim=-1, keepdim=True, unbiased=False).detach()
        input = (input - m) / t.sqrt(v + self.eps)
        return input * self.weight + self.bias


if __name__ == "__main__":
    bert_tests.test_layer_norm(LayerNorm)


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
            if subkeys[1] == "position_embedding":
                subkeys[1] = "pos_embedding"
            # subkeys[-1] = "embedding"
            return ".".join(subkeys)

        if subkeys[0] == "transformer":
            if subkeys[2] == "attention":
                # subkeys[2] = "mha"

                if subkeys[3] == "pattern":
                    subkeys.pop(3)

                subkeys[3] = {
                    "project_value": "project_value",
                    "project_query": "project_query",
                    "project_key": "project_key",
                    "project_out": "project_output",
                }[subkeys[3]]

                return ".".join(subkeys)

            if subkeys[2] == "residual":
                if subkeys[3] != "layer_norm":
                    subkeys[2] = "mlp"
                    subkeys[3] = {"mlp1": "lin1", "mlp2": "lin2"}[subkeys[3]]
                    return ".".join(subkeys)

                if subkeys[3] == "layer_norm":
                    subkeys.pop(2)
                    subkeys[2] = "layernorm2"
                    return ".".join(subkeys)

            if subkeys[2] == "layer_norm":
                subkeys[2] = "layernorm1"
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
