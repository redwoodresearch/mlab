import torch as t
import transformers
import numpy as np
from IPython.core.display import HTML, display
from utils import *

attention_only_model = t.load("ao2.pt", map_location=t.device("cpu"))
print(attention_only_model)
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")


def test_text_to_attentions(their_text_to_attentions):
    input_text = "Hi, my name is Adam"
    mine = text_to_attentions(input_text)
    theirs = their_text_to_attentions(input_text)
    allclose(mine, theirs)


def text_to_attentions(text, model=attention_only_model):
    attentions = []

    def hook(module, _input, output):
        nonlocal attentions
        # tpeek(output)
        output = output[0]
        output = t.softmax(output, dim=-2)
        attentions.append(output)

    for block in model:
        block.attention.pattern.register_forward_hook(hook)
    tokens = tokenizer(text, return_tensors="pt")["input_ids"]
    model(tokens)
    attentions = t.stack(attentions, dim=0)
    return attentions


def display_attention(attn_matrix, tokens):
    attn_matrix = t.tensor(attn_matrix)
    attn_matrix = (attn_matrix).log()
    attn_matrix = t.nan_to_num(attn_matrix, neginf=np.pi)
    attn_matrix[attn_matrix == np.pi] = t.min(attn_matrix)
    attn_matrix = (attn_matrix - t.min(attn_matrix)) / (
        t.max(attn_matrix) - t.min(attn_matrix)
    )

    def get_color(scalar):
        np.log(scalar)
        h = int(scalar * 100)
        return f"hsl({h}, 100%,50%)"

    htmlstring = f"""
    <style>
    .cell{{
    width:22px;
    height:22px;
    }}</style>
    <div style="display:flex; flex-direction:'row';">
        <div style="text-align:right;">
            {''.join([f'<div>{token}</div>' for token in tokens])}
        </div>
        <div>
            {''.join(['<div style="display:flex;">'+''.join([f'<div class="cell" style="background-color:{get_color(attn)};"></div>'
for attn in row])+'</div>' for row in attn_matrix])}
<div style="writing-mode: vertical-rl; text-orientation: mixed; display:flex; flex-direction:column-reverse;">{''.join([f'<div>{token}</div>' for token in tokens])}</div>
        </div>
    </div>
    """
    # print(htmlstring)
    return HTML(htmlstring)
