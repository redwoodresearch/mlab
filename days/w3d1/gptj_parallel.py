import torch as t
from torch import nn
from transformers.models.gptj.modeling_gptj import (
    GPTJBlock,
    GPTJForSequenceClassification,
)


class GPTJBlockWrapper(nn.Module):
    def __init__(
        self,
        block: GPTJBlock,
    ):
        super().__init__()
        self.block = block

    def forward(self, x: t.Tensor):
        return self.block.forward(x)[0]


class GPTJComponent(nn.Module):
    def __init__(
        self,
        idx: int,
        gptj: GPTJForSequenceClassification,
    ):
        super().__init__()
        self.idx = idx

        if idx == 0:
            self.seq = nn.Sequential(
                gptj.transformer.wte,
                gptj.transformer.drop,
                *(GPTJBlockWrapper(gptj.transformer.h[i]) for i in range(7)),
            )

        if idx == 1:
            self.seq = nn.Sequential(
                *(GPTJBlockWrapper(gptj.transformer.h[i]) for i in range(7, 14))
            )

        if idx == 2:
            self.seq = nn.Sequential(
                *(GPTJBlockWrapper(gptj.transformer.h[i]) for i in range(14, 21))
            )

        if idx == 3:
            self.seq = nn.Sequential(
                *(GPTJBlockWrapper(gptj.transformer.h[i]) for i in range(21, 28)),
                gptj.transformer.ln_f,
            )
            self.score = gptj.score

    def forward(self, x):
        if self.idx == 3:
            hidden_state = self.seq(x)
            return hidden_state, self.score(hidden_state)[:, -1]

        return self.seq(x)
