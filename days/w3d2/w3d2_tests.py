import collections
import einops
import matplotlib.pyplot as plt
import math
import numpy as np
import torch as t
from torch import nn
import transformers
from IPython.core.display import HTML, display
from days.utils import *


tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
tokenizer._add_tokens(["[BEGIN]", "[END]"])
tokenizer.pad_token = "[END]"
tokenizer.eos_token = "[END]"


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

    def forward(self, x: t.Tensor, pos_embedding):
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
        combined_v = einops.rearrange(combined_v, 'b h q l -> b q (h l)')
        self._combined_v = combined_v
        out = self.project_output(combined_v)
        return out

    def weight_matrix(self, qkvo: str, head: int):
        if qkvo in 'qkv':
            q, k, v = t.split(self.project_qkv.weight, self.hidden_size, dim=0)
            ret = {'q': q, 'k': k, 'v': v}[qkvo]
            return ret[head*self.head_size: (head+1)*self.head_size]
        elif qkvo == 'o':
            return self.project_output.weight[:, head*self.head_size: (head+1)*self.head_size]


class MiniGPT(nn.Module):
    def __init__(self, num_heads=8, vocab_size=50259, hidden_size=256,
                 max_position_embeddings=512):
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

    def layer0_embedding_contributions(self, input_ids):
        emb = [self.token_embedding(input_ids)]

        Wo = self.blocks[0].project_output.weight
        n_heads = self.blocks[0].n_heads
        head_size = Wo.shape[1] // n_heads
        self(input_ids)

        for head in range(n_heads):
            v_h = self.blocks[0]._combined_v[:, :, head*head_size: (head+1)*head_size]
            Wo_h = Wo[:, head*head_size: (head+1)*head_size]
            o_h = t.einsum('bnh, lh -> bnl', v_h, Wo_h)
            emb.append(o_h)
        return emb

    def layer1_embedding_contributions(self, input_ids):
        # -> list[i][j], where i is layer-0 index and j is layer-1 index
        n_heads = self.blocks[0].n_heads
        l0_terms = self.layer0_embedding_contributions(input_ids)
        emb = [[None]*(n_heads+1) for _ in range(n_heads+1)]
        for i in range(n_heads+1):
            emb[i][0] = l0_terms[i]

        attn = self.attention_softmaxed(input_ids)
        
        for i in range(n_heads + 1):        
            for j in range(n_heads):
                Wv = self.weight_matrix('v', 1, j)
                V = t.einsum('hl, bnl -> bnh', Wv, emb[i][0])
                combined_V = t.einsum('qk, bkh -> bqh', attn[1, j],  V)
                Wo = self.weight_matrix('o', 1, j)
                emb[i][j+1] = t.einsum('bqh, lh -> bql', combined_V, Wo)
        return emb

    def weight_matrix(self, qkvo: str, layer: int, head: int):
        return self.blocks[layer].weight_matrix(qkvo, head)

    def _replace_upper_triangle(self, matrix, neg_inf=-1e4):
        q_ind = t.arange(matrix.shape[-2]).unsqueeze(1)
        k_ind = t.arange(matrix.shape[-1]).unsqueeze(0)
        mask = (q_ind < k_ind).to(matrix.device)
        neg_inf = t.tensor(neg_inf).to(matrix.device)
        return t.where(mask, neg_inf, matrix)

    def vis_attention(self, layer, head, input_ids, tokenizer=tokenizer):
        assert input_ids.shape[0] == 1
        self(input_ids)
        attn = self.blocks[layer]._attn_scores[head].softmax(dim=-1)
        plt.imshow(attn)
        if input_ids is not None:
            wordpieces = [tokenizer.decode(i.item()) for i in input_ids[0]]
            plt.xticks(range(attn.shape[0]), wordpieces, rotation=90)
            plt.yticks(range(attn.shape[0]), wordpieces)
            plt.xlabel("Key")
            plt.ylabel("Query")        

    def attention_softmaxed(self, input_ids):
        # -> [n_layers, n_heads, seq_len(q), seq_len(k)]
        assert input_ids.shape[0] == 1
        self(input_ids)
        return t.stack([self.blocks[i]._attn_scores
                        for i in range(len(self.blocks))], dim=0).softmax(dim=-1)

    def attention_presoftmax(self, input_ids, neg_inf=-1e4):
        # -> [n_layers, n_heads, seq_len(q), seq_len(k)]
        assert input_ids.shape[0] == 1
        self(input_ids)
        return t.stack([self._replace_upper_triangle(self.blocks[i]._attn_scores, neg_inf)
                        for i in range(len(self.blocks))], dim=0)


def _check_equal(tensor1, tensor2, print_congrats=True):
    if isinstance(tensor1, list):
        assert isinstance(tensor2, list)
        assert len(tensor1) == len(tensor2)
        for t1, t2 in zip(tensor1, tensor2):
            _check_equal(t1, t2, print_congrats=False)            
    else:
        assert t.allclose(tensor1, tensor2, atol=1e-4, rtol=1e-4)
    if print_congrats:
        print("Congrats! You've passed the test!")

    
def test_weight_matrix(weight_matrix):
    model = MiniGPT()
    matrix_type = dict(q='query', k='key', v='value', o='output')
    for qkvo in 'qkvo':
        print(f'Checking {matrix_type[qkvo]} weight matrices...')
        for layer in range(2):
            for head in range(8):
                _check_equal(model.weight_matrix(qkvo, layer, head),
                             weight_matrix(model, qkvo, layer, head))

                
def test_attention_softmaxed(attention_softmaxed):
    model = MiniGPT()
    input_ids = t.randint(0, 50000, (1, 10))
    _check_equal(model.attention_softmaxed(input_ids),
                 attention_softmaxed(model, input_ids))


def test_attention_presoftmax(attention_presoftmax):
    model = MiniGPT()
    input_ids = t.randint(0, 50000, (1, 10))
    _check_equal(model.attention_presoftmax(input_ids, -10.0),
                 attention_presoftmax(model, input_ids, -10.0))


def test_layer0_embedding_contributions(f):
    model = MiniGPT()
    input_ids = t.randint(0, 50000, (1, 10))
    _check_equal(model.layer0_embedding_contributions(input_ids), f(model, input_ids))


def test_layer1_embedding_contributions(f):
    model = MiniGPT()
    input_ids = t.randint(0, 50000, (1, 10))
    _check_equal(model.layer1_embedding_contributions(input_ids), f(model, input_ids))
    
    
###########################################################################    
        
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
