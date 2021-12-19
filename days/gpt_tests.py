from dataclasses import dataclass
import einops
import math
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from torchtyping import TensorType
import transformers
from typing import Optional


def _check_equal(tensor1, tensor2):
    if torch.allclose(tensor1, tensor2, atol=1e-4, rtol=1e-4):
        print("Congrats! You've passed the test!")
    else:
        print("Your solution doesn't match ours.")
        

class _UnidirectionalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.attn_lin = nn.Linear(hidden_size, hidden_size * 3)
        self.out_lin = nn.Linear(hidden_size, hidden_size)
        self.register_buffer(
            'neg_inf', torch.tensor(-1e4))
    
    def forward(self, encodings, past_key_values: Optional[torch.Tensor] = None,
                return_key_values=False):
        qkv = self.attn_lin(encodings)
        q, k, v = torch.split(qkv, self.hidden_size, dim=-1)
        q = einops.rearrange(q, 'b n (h l) -> b h n l', h=self.n_heads)
        k = einops.rearrange(k, 'b n (h l) -> b h n l', h=self.n_heads)
        v = einops.rearrange(v, 'b n (h l) -> b h n l', h=self.n_heads)
        
        n = qkv.shape[1]
        mask = (torch.arange(n).unsqueeze(1) <= torch.arange(n).unsqueeze(0))
        
        if return_key_values:
            new_key_values =  torch.cat([k, v], dim=-1)

        if past_key_values is not None:
            assert encodings.shape[0] == encodings.shape[1] == 1
            prev_k, prev_v = torch.split(past_key_values, self.head_size, dim=-1)
            k = torch.cat([prev_k.unsqueeze(0), k], dim=2)
            v = torch.cat([prev_v.unsqueeze(0), v], dim=2)
            mask = torch.tensor([True])

        attn_scores = torch.einsum('bhki, bhqi -> bhkq', k, q) / math.sqrt(self.head_size)
        attn_scores = torch.where(mask, attn_scores, self.neg_inf)
        attn_prob = attn_scores.softmax(dim=2)
        
        combined_v = torch.einsum('bhkq, bhkl -> bhql', attn_prob, v)
        out = einops.rearrange(combined_v, 'b h n l -> b n (h l)')
        if return_key_values:
            return self.out_lin(out), new_key_values
        return self.out_lin(out)


def test_unidirectional_attn(Attention):
    kwargs = dict(hidden_size=24, num_heads=4)
    encodings = torch.randn(1, 5, kwargs['hidden_size'])
    
    torch.manual_seed(545)
    _attn = _UnidirectionalAttention(**kwargs)
    _out = _attn(encodings)
    
    torch.manual_seed(545)
    attn = Attention(**kwargs)
    out = attn(encodings)
    
    _check_equal(out, _out)


def test_attn_cache(Attention):
    kwargs = dict(hidden_size=24, num_heads=4)
    head_size = kwargs['hidden_size'] // kwargs['num_heads']
    encodings = torch.randn(1, 1, kwargs['hidden_size'])
    past_key_values = torch.randn(kwargs['num_heads'], 3, 2 * head_size)

    torch.manual_seed(945)
    _attn = _UnidirectionalAttention(**kwargs)
    _out = _attn(encodings, past_key_values=past_key_values, return_key_values=True)

    torch.manual_seed(945)
    attn = Attention(**kwargs)
    out = attn(encodings, past_key_values=past_key_values, return_key_values=True)

    print('Checking encoding:')
    _check_equal(out[0], _out[0])
    print('Checking new key and value:')
    _check_equal(out[1], _out[1])
    
    

class _GPT2Block(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, layer_norm_epsilon):
        super().__init__()
        self.ln1 = nn.LayerNorm((hidden_size,), eps=layer_norm_epsilon)
        self.attn = _UnidirectionalAttention(hidden_size, num_heads)
        self.ln2 = nn.LayerNorm((hidden_size,), eps=layer_norm_epsilon)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, past_key_values=None, return_key_values=False):
        if return_key_values == False:
            res1 = x + self.attn(self.ln1(x), past_key_values=past_key_values) 
            return self.dropout(self.linear2(
                F.gelu(self.linear1(self.ln2(res1))))) + res1
        else:
            res1, new_key_values = self.attn(self.ln1(x), past_key_values=past_key_values,
                                             return_key_values=return_key_values)
            res1 += x
            return self.dropout(self.linear2(
                F.gelu(self.linear1(self.ln2(res1))))) + res1, new_key_values


def test_gpt_block(GPT2Block):
    kwargs = dict(hidden_size=48, layer_norm_epsilon=1e-4, dropout=0.0, num_heads=4)
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
                 max_position_embeddings, dropout, layer_norm_epsilon, use_cache=False,
                 tokenizer=None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            _GPT2Block(hidden_size, num_heads, dropout, layer_norm_epsilon)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm((hidden_size,), eps=layer_norm_epsilon)
        head_size = hidden_size // num_heads

        self.use_cache = use_cache
        self.cache_kv_shape = (num_layers, num_heads, 0, 2 * head_size)
        self.cache_enc_shape = (0, hidden_size)
        self.empty_cache()

        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        if self.tokenizer:
            self.special_token_ids = {self.tokenizer.sep_token_id, self.tokenizer.unk_token_id,
                                      self.tokenizer.pad_token_id, self.tokenizer.cls_token_id,
                                      self.tokenizer.mask_token_id}

    def empty_cache(self):
        self.cache_kv = torch.zeros(self.cache_kv_shape).to(self.ln.weight.device)
        self.cache_enc = torch.zeros(self.cache_enc_shape).to(self.ln.weight.device)
        
    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        pos = torch.arange(seq_len).unsqueeze(0).to(input_ids.device)

        if not self.use_cache:
            tok_embed = self.token_embedding(input_ids)
            pos_embed = self.pos_embedding(pos)
            x = self.dropout(tok_embed + pos_embed)
            x = self.ln(self.blocks(x))

        elif self.cache_enc.shape[0] == 0:
            assert input_ids.shape[0] == 1
            tok_embed = self.token_embedding(input_ids)
            pos_embed = self.pos_embedding(pos)
            x = self.dropout(tok_embed + pos_embed)
            
            new_key_vals = []
            for l, block in enumerate(self.blocks):
                x, new_key_values = block(x, return_key_values=True)
                new_key_vals += [new_key_values]
                
            x = self.ln(x)
            self.cache_kv = torch.cat(new_key_vals, dim=0)
            self.cache_enc = x[0]
            
        else:
            assert input_ids.shape[0] == 1
            tok_embed = self.token_embedding(input_ids[:, -1:])
            pos_embed = self.pos_embedding(pos[:, -1:])
            x = self.dropout(tok_embed + pos_embed)
            
            new_key_vals = []
            for l, block in enumerate(self.blocks):
                x, new_key_values = block(x, past_key_values=self.cache_kv[l],
                                          return_key_values=True)
                new_key_vals += [new_key_values]
                
            x = self.ln(x)
            new_key_vals = torch.cat(new_key_vals, dim=0)
            self.cache_kv = torch.cat([self.cache_kv, new_key_vals], dim=2)
            self.cache_enc = torch.cat([self.cache_enc, x[0]], dim=0)
            x = self.cache_enc.unsqueeze(0)
            
        logits = torch.einsum('bnl, vl -> bnv', x, self.token_embedding.weight)
        return GPT2Output(logits=logits, final_encoding=x[:, -1, :])

    def next_token(self, input_ids, temperature, freq_penalty=2.0):
        logits = self(input_ids.unsqueeze(0)).logits[0,-1]
        id_freqs = torch.bincount(input_ids, minlength=self.vocab_size)
        probs = (logits / temperature - id_freqs * freq_penalty).softmax(dim=-1)
        return torch.distributions.Categorical(probs).sample()

    def generate_ids(self, input_ids, temperature, freq_penalty=2.0, max_length=30):
        prompt_len = len(input_ids)
        for i in range(max_length):
            new_token = self.next_token(input_ids, temperature, freq_penalty)
            input_ids = torch.cat([input_ids, new_token.unsqueeze(0)], dim=0)
            if self.tokenizer and new_token == self.tokenizer.eos_token_id:
                break            
        return input_ids[prompt_len:]

    def generate(self, text, max_length=30, temperature=1.0, freq_penalty=2.0):
        assert self.tokenizer, 'No assigned tokenizer'
        self.empty_cache()
        input_ids = self.tokenizer(text)['input_ids']
        i = len(input_ids)
        while input_ids[i-1] in self.special_token_ids:
            i -= 1
        input_ids = torch.tensor(input_ids[:i])
        new_ids = self.generate_ids(input_ids, temperature, freq_penalty, max_length)
        return self.tokenizer.decode(new_ids)


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


def test_gpt_cache(GPT2):
    config = dict(num_layers=2, num_heads=4, vocab_size=100, hidden_size=64,
                  max_position_embeddings=500, dropout=0.0, layer_norm_epsilon=1e-4)
    x = torch.randint(0, config['vocab_size'], (1, 500))

    torch.manual_seed(1010)
    gpt = GPT2(**config)
    t = time.time()
    for i in range(1, x.shape[1]+1):
        output_nocache = gpt(x[:, :i])
    t1 = time.time() - t

    torch.manual_seed(1010)
    gpt = GPT2(**config, use_cache=True)
    t = time.time()
    for i in range(1, x.shape[1]+1):
        output_cache = gpt(x[:, :i])
    t2 = time.time() - t

    if torch.allclose(output_cache.logits, output_nocache.logits, rtol=1e-4, atol=1e-4):
        print('Congrats! Your GPT returns the same results with and without cache.') 
        print(f'It took {t1:.3f}s to generate a 500-token sentence without cache and '
              f'{t2:.3f}s with cache.')
    else:
        print('Your GPT returns different results when using cache.')


def _copy_weight_bias(mine, theirs, transpose=False):
    if transpose:
        mine.weight.copy_(theirs.weight.T)
    else:
        mine.weight.copy_(theirs.weight)
    if mine.bias is not None:
        mine.bias.copy_(theirs.bias)
    

def load_gpt_weights():
    pretrained_gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    config = dict(num_layers=12, num_heads=12, vocab_size=50257, hidden_size=768,
                  max_position_embeddings=1024, dropout=0.1, layer_norm_epsilon=1e-5)
    my_gpt = _GPT2(**config, tokenizer=tokenizer)
    for p in my_gpt.parameters():
        p.requires_grad = False

    my_gpt.token_embedding.weight.copy_(pretrained_gpt.transformer.wte.weight)
    my_gpt.pos_embedding.weight.copy_(pretrained_gpt.transformer.wpe.weight)
    _copy_weight_bias(my_gpt.ln, pretrained_gpt.transformer.ln_f)

    for my_block, hf_block in zip(my_gpt.blocks, pretrained_gpt.transformer.h):
        _copy_weight_bias(my_block.ln1, hf_block.ln_1)
        _copy_weight_bias(my_block.attn.attn_lin, hf_block.attn.c_attn, transpose=True)
        _copy_weight_bias(my_block.attn.out_lin, hf_block.attn.c_proj, transpose=True)
        _copy_weight_bias(my_block.ln2, hf_block.ln_2)
        _copy_weight_bias(my_block.linear1, hf_block.mlp.c_fc, transpose=True)
        _copy_weight_bias(my_block.linear2, hf_block.mlp.c_proj, transpose=True)
    return my_gpt

        
    
    
