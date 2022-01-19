from einops import reduce, rearrange, repeat
import days.gpt2 as gpt2
import days.bert as bert
import torch as t
from days.utils import tpeek
import matplotlib.pyplot as plt


def hook_attentions(model):
    result = []

    def hook(module, _input, output):
        nonlocal result
        # tpeek(output)
        output = output[0]
        output = t.softmax(output, dim=-2)
        result.append(output)

    for block in model.transformer:
        block.attention.pattern.register_forward_hook(hook)
    return result


def show_aggregate_attention(model, text):
    attention_buffer = hook_attentions(model)
    model(model.tokenizer(text, return_tensors="pt")["input_ids"])
    attention_vals = t.stack(attention_buffer, dim=0)
    tpeek("attention buffer", attention_vals)
    attention_aggregate = reduce(
        attention_vals, "layer head from to -> from to", "mean"
    )
    plt.imshow(attention_aggregate.detach().numpy())
    plt.imshow(rearrange(attention_vals, "l h f t -> (l f) (h t)").detach().numpy())


if __name__ == "__main__":
    model, _ = bert.my_bert_from_hf_weights()
    show_aggregate_attention(model, "The firetruck was painted bright red")
