import torch as t
import transformers
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
        gptj: GPTJForSequenceClassification,
        num_components: int,
        idx: int,
    ):
        super().__init__()
        self.num_components = num_components
        self.blocks_per_component = (28 + num_components - 1) // num_components
        self.idx = idx

        if idx == 0:
            self.seq = nn.Sequential(
                gptj.transformer.wte,
                gptj.transformer.drop,
                *(
                    GPTJBlockWrapper(gptj.transformer.h[i])
                    for i in range(self.blocks_per_component)
                ),
            )

        if 1 <= idx < num_components:
            self.seq = nn.Sequential(
                *(
                    GPTJBlockWrapper(gptj.transformer.h[i])
                    for i in range(
                        idx * self.blocks_per_component,
                        max((idx + 1) * self.blocks_per_component, 28),
                    )
                ),
            )

        if idx == num_components - 1:
            self.score = gptj.score

    def forward(self, x):
        if self.idx == self.num_components - 1:
            hidden_state = self.seq(x)
            return hidden_state, self.score(hidden_state)[:, -1]

        return self.seq(x)

    @classmethod
    def save_component_checkpoints(
        cls,
        gptj: GPTJForSequenceClassification,
        num_components: int,
        save_dir: str,
    ):
        for i in range(num_components):
            component = cls(
                gptj=gptj,
                num_components=num_components,
                idx=i,
            )

            print(f"Saving component{i}.pt ...", end="", flush=True)
            t.save(component, f"{save_dir}/component{i}.pt")
            print("done")


if __name__ == "__main__":
    print("Loading gptj...", end="", flush=True)
    gptj = transformers.AutoModelForSequenceClassification.from_pretrained(
        "EleutherAI/gpt-j-6B",
    )
    print("done")

    GPTJComponent.save_component_checkpoints(
        gptj=gptj,
        num_components=7,
        save_dir=".data/gptj_components/",
    )
