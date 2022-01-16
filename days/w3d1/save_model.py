import torch as t
from torch import nn
import os

# HuggingFace models return tuples in the middle (things like activation patterns), thus the [0]


class HFBlockSequence(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)  # treat like a list

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x


def make_gptj_and_save_pieces(chunks=[4, 5, 5, 5, 5, 4]):
    print(f"Saving model into chunks: {chunks}")
    import transformers

    model_lm = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    print("Model loaded successfully")
    model = model_lm.transformer
    num_layers = len(model.h)
    assert num_layers == 28
    # less at ends due to embeddings/unembed
    assert sum(chunks) == num_layers
    chunk_cumsum = t.cumsum(t.tensor(chunks), dim=0).tolist()
    print("cumsum", chunk_cumsum)
    models = [HFBlockSequence(*model.h[start - size : start]) for start, size in zip(chunk_cumsum, chunks)]
    models[0] = nn.Sequential(model.wte, model.drop, models[0])
    models[-1] = nn.Sequential(models[-1], model.ln_f, model_lm.lm_head)
    for i, model_part in enumerate(models):
        path = os.path.abspath(f"gpt-j-6b_part{i}.pt")
        print(f"Saving model part {i} to {path}")
        t.save(model_part, path)
    return models


def gptj_from_pieces(base_path="gpt-j-6b"):
    ...


if __name__ == "__main__":
    make_gptj_and_save_pieces()
