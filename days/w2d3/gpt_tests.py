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

from _gpt_sol import _UnidirectionalAttention, _GPT2Block, _GPT2


def _check_equal(tensor1, tensor2):
    assert torch.allclose(tensor1, tensor2, atol=1e-4, rtol=1e-4)
    print("Congrats! You've passed the test!")
        

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
    

def get_pretrained_gpt():
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

        
    
    
