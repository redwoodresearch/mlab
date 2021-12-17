from dataclasses import dataclass
import einops
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchtyping import TensorType
from typing import Optional


def _check_equal(tensor1, tensor2):
    if torch.allclose(tensor1, tensor2, atol=1e-4, rtol=1e-4):
        print("Congrats! You've passed the test!")
    else:
        print("Your solution doesn't match ours.")
        

class _UnidirectionalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, max_position_embeddings: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.attn_lin = nn.Linear(hidden_size, hidden_size * 3)
        self.out_lin = nn.Linear(hidden_size, hidden_size)
        N = max_position_embeddings
        self.register_buffer(
            'mask', (torch.arange(N).unsqueeze(1) <= torch.arange(N).unsqueeze(0)))
        self.register_buffer(
            'neg_inf', torch.tensor(-1e4))
    
    def forward(self, encodings, attention_masks=None,
                past_key_values: Optional[torch.Tensor] = None):
        qkv = self.attn_lin(encodings)
        qkv = einops.rearrange(qkv, 'b n (h l) -> b h n l', h=self.n_heads)
        q, k, v = torch.split(qkv, self.head_size, dim=-1)
        
        n = qkv.shape[2]
        mask = self.mask[:n, :n]
        if attention_masks is not None:
            mask = mask * attention_masks

        if past_key_values is not None:
            assert encodings.shape[0] == encodings.shape[1] == 1
            assert attention_masks is None
            new_key_values =  torch.cat([k, v], dim=-1)[0]
            prev_k, prev_v = torch.split(past_key_values, self.head_size, dim=-1)
            k = torch.cat([prev_k.unsqueeze(0), k], dim=2)
            v = torch.cat([prev_v.unsqueeze(0), v], dim=2)
            mask = torch.tensor([True])
            
        attn_scores = torch.einsum('bhki, bhqi -> bhkq', k, q) / math.sqrt(self.head_size)
        attn_scores = torch.where(mask, attn_scores, self.neg_inf)
        attn_prob = attn_scores.softmax(dim=2)
        
        combined_v = torch.einsum('bhkq, bhkl -> bhql', attn_prob, v)
        out = einops.rearrange(combined_v, 'b h n l -> b n (h l)')
        if past_key_values is not None:
            return self.out_lin(out), new_key_values
        return self.out_lin(out)


def test_unidirectional_attn(Attention):
    kwargs = dict(hidden_size=24, num_heads=4, max_position_embeddings=10)
    encodings = torch.randn(1, 5, kwargs['hidden_size'])
    
    torch.manual_seed(545)
    _attn = _UnidirectionalAttention(**kwargs)
    _out = _attn(encodings)
    
    torch.manual_seed(545)
    attn = Attention(**kwargs)
    out = attn(encodings)
    
    _check_equal(out, _out)


def test_attn_cache(Attention):
    kwargs = dict(hidden_size=24, num_heads=4, max_position_embeddings=10)
    head_size = kwargs['hidden_size'] // kwargs['num_heads']
    encodings = torch.randn(1, 1, kwargs['hidden_size'])
    past_key_values = torch.randn(kwargs['num_heads'], 3, 2 * head_size)

    torch.manual_seed(945)
    _attn = _UnidirectionalAttention(**kwargs)
    _out = _attn(encodings, past_key_values=past_key_values)

    torch.manual_seed(945)
    attn = Attention(**kwargs)
    out = attn(encodings, past_key_values=past_key_values)

    print('Checking encoding:')
    _check_equal(out[0], _out[0])
    print('Checking new key and value:')
    _check_equal(out[1], _out[1])
    
    

class _GPT2Block(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, layer_norm_epsilon,
                 max_position_embeddings):
        super().__init__()
        self.ln1 = nn.LayerNorm((hidden_size,), eps=layer_norm_epsilon)
        self.attn = _UnidirectionalAttention(hidden_size, num_heads,
                                             max_position_embeddings)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.ln2 = nn.LayerNorm((hidden_size,), eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, past_key_values=None):
        if past_key_values is None:
            res1 = self.attn(self.ln1(x)) + x
            return self.dropout(self.linear2(
                F.gelu(self.linear1(self.ln2(res1))))) + res1
        else:
            res1, new_key_values = self.attn(self.ln1(x), past_key_values=past_key_values)
            res1 += x
            return self.dropout(self.linear2(
                F.gelu(self.linear1(self.ln2(res1))))) + res1, new_key_values


def test_gpt_block(GPT2Block):
    kwargs = dict(hidden_size=48, layer_norm_epsilon=1e-4, dropout=0.0, num_heads=4,
                  max_position_embeddings=10)
    x = torch.randn(1, 5, 48)
    
    torch.manual_seed(710)
    _block = _GPT2Block(**kwargs)
    _out = _block(x)
    
    torch.manual_seed(710)
    block = GPT2Block(**kwargs)
    out = block(x)
    
    _check_equal(out, _out)


@dataclass
class GPT2Output:
    logits: TensorType["batch_size", "seq_length", "vocab_size"]
    final_encoding: TensorType["batch_size", "hidden_size"]
    

class _GPT2(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size, hidden_size,
                 max_position_embeddings, dropout, layer_norm_epsilon, use_cache=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            _GPT2Block(hidden_size, num_heads, dropout, layer_norm_epsilon,
                       max_position_embeddings) for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm((hidden_size,), eps=layer_norm_epsilon)
        head_size = hidden_size // num_heads

        self.use_cache = use_cache
        self.cache_kv_shape = (num_layers, num_heads, 0, 2 * head_size)
        self.cache_enc_shape = (0, hidden_size)
        self.empty_cache()

    def empty_cache(self):
        self.cache_kv = torch.zeros(self.cache_kv_shape).to(self.ln.weight.device)
        self.cache_enc = torch.zeros(self.cache_enc_shape).to(self.ln.weight.device)
        
    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        pos = torch.arange(seq_len).unsqueeze(0).to(input_ids.device)
        tok_embed = self.token_embedding(input_ids)
        pos_embed = self.pos_embedding(pos)
        x = self.dropout(tok_embed + pos_embed)

        if self.use_cache:
            assert input_ids.shape[0] == 1
            cache_len = self.cache_enc.shape[0]
            x = x[:, cache_len: cache_len+1, :]
            new_key_vals = []
            for l, block in enumerate(self.blocks):
                x, new_key_values = block(x, past_key_values=self.cache_kv[l])
                new_key_vals += [new_key_values]
            new_key_vals = torch.stack(new_key_vals)
            self.cache_kv = torch.cat([self.cache_kv, new_key_vals], dim=2)
            self.cache_enc = torch.cat([self.cache_enc, x[0]], dim=0)
            x = self.cache_enc.unsqueeze(0)
        else:
            x = self.blocks(x)

        x = self.ln(x)
        logits = torch.einsum('bnl, vl -> bnv', x, self.token_embedding.weight)
        return GPT2Output(logits=logits, final_encoding=x[:, -1, :])


def test_gpt(GPT2):
    config = dict(num_layers=2, num_heads=4, vocab_size=100, hidden_size=64,
                  max_position_embeddings=32, dropout=0.0, layer_norm_epsilon=1e-4)
    x = torch.randint(0, config['vocab_size'], (1, 5))
    
    torch.manual_seed(1010)
    _gpt = _GPT2(**config)
    _output = _gpt(x)
    
    torch.manual_seed(1010)
    gpt = GPT2(**config)
    output = gpt(x)
    
    print('Checking logits:')
    _check_equal(_output.logits, output.logits)
    print('Checking final encodings:')
    _check_equal(_output.final_encoding, output.final_encoding)    
