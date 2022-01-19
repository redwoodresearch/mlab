import collections
import einops
import math
import numpy as np
import torch as t
from torch import nn
import transformers
from IPython.core.display import HTML, display
from days.utils import *


tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")


def get_minigpt(fname):
    minigpt = MiniGPT()
    weights = t.load(fname, map_location=t.device("cpu"))
    
    out = collections.OrderedDict()
    out['token_embedding.weight'] = \
        weights['embedding.token_embedding.weight']
    out['pos_embedding.weight'] = \
        weights['embedding.position_embedding.weight']
    for i in [0, 1]:
        out[f'blocks.{i}.project_qkv.weight'] = \
            weights[f'blocks.{i}.attention.attention_weights.weight']
        out[f'blocks.{i}.project_output.weight'] = \
            weights[f'blocks.{i}.attention.project_output.weight']
    minigpt.load_state_dict(out)
    return minigpt


class UniAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.project_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.project_output = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.n_heads = num_heads
        
    def value_headwise(self, x, pos_embedding):
        batch, seq_len = x.shape[:2]
        pos_ids = t.arange(x.shape[1]).unsqueeze(0).to(x.device)
        pos_emb = pos_embedding(pos_ids)

        q, k, _ = t.split(self.project_qkv(x + pos_emb), self.hidden_size, dim=-1)
        _, _, v = t.split(self.project_qkv(x), self.hidden_size, dim=-1)
        
        q = einops.rearrange(q, 'b n (h l) -> b h n l', l=self.head_size)
        k = einops.rearrange(k, 'b n (h l) -> b h n l', l=self.head_size)
        v = einops.rearrange(v, 'b n (h l) -> b h n l', l=self.head_size)
        
        neg_inf = t.tensor(-1e4).to(x.device)
        q_ind = t.arange(seq_len).unsqueeze(1)
        k_ind = t.arange(seq_len).unsqueeze(0)
        mask = (q_ind < k_ind).to(x.device)
        attn_scores = t.einsum('bhql, bhkl -> bhqk', q, k) / math.sqrt(self.head_size)
        attn_scores = t.where(mask, neg_inf, attn_scores)

        self._attn_scores = attn_scores.detach()[0]
        probs = attn_scores.softmax(dim=-1)
        combined_v = t.einsum('bhqk, bhkl -> bhql', probs, v)
        # combined_v: batch num_heads seq_len head_size
        return combined_v
        
        
    def forward_headwise(self, x, pos_embedding):
        combined_v = self.value_headwise(x, pos_embedding)       
        output_weight_matrix = self.project_output.weight
        output_weight_matrix = einops.rearrange(output_weight_matrix, "embed (heads headsize) -> heads embed headsize", heads=self.n_heads)
        out = t.einsum("hel,bhql->bhqe", output_weight_matrix, combined_v)
        return out

    def forward(self, x: t.Tensor, pos_embedding):
        combined_v = self.value_headwise(x, pos_embedding)
        combined_v = einops.rearrange(combined_v, 'b h q l -> b q (h l)')
        out = self.project_output(combined_v)
        return out


class MiniGPT(nn.Module):
    def __init__(self, num_heads=8, vocab_size=50259, hidden_size=256, max_position_embeddings=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.blocks = nn.Sequential(
            UniAttention(hidden_size, num_heads),
            UniAttention(hidden_size, num_heads),
        )
        
    def forward(self, input_ids):
        emb = self.token_embedding(input_ids)
        for block in self.blocks:
            emb = emb + block(emb, self.pos_embedding)
        return t.einsum('bnl, vl -> bnv', emb, self.token_embedding.weight)
    
        
def test_text_to_attentions(their_text_to_attentions):
    input_text = "Hi, my name is Adam"
    mine = text_to_attentions(input_text)
    theirs = their_text_to_attentions(input_text)
    allclose(mine, theirs)


def text_to_attentions(text, model=None):
    if model is None:
        model = get_minigpt_weights()
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
