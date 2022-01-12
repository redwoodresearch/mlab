from dataclasses import dataclass
import einops
import math
import torch
from torch import nn
import torch.nn. functional as F
from torchtyping import TensorType
import transformers
from typing import Optional

import gpt_tests


class UniAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.n_heads = num_heads

    def forward(self, x: torch.Tensor, past_key_values: Optional[torch.Tensor]=None, return_key_values=False):
        batch, seq_len = x.shape[:2]
        q, k, v = torch.split(self.qkv_proj(x), self.hidden_size, dim=-1)
        q = einops.rearrange(q, 'b n (h l) -> b h n l', l=self.head_size)
        k = einops.rearrange(k, 'b n (h l) -> b h n l', l=self.head_size)
        v = einops.rearrange(v, 'b n (h l) -> b h n l', l=self.head_size)
        new_k, new_v = k, v
        
        if past_key_values is not None:
            assert x.shape == (1, 1, self.hidden_size)
            past_k, past_v = torch.split(past_key_values.unsqueeze(0), self.head_size, dim=-1)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            attn_scores = torch.einsum('bhql, bhkl -> bhqk', q, k) / math.sqrt(self.head_size)
        else:
            neg_inf = torch.tensor(-1e4).to(x.device)
            q_ind = torch.arange(seq_len).unsqueeze(1)
            k_ind = torch.arange(seq_len).unsqueeze(0)
            mask = (q_ind < k_ind).to(x.device)
            attn_scores = torch.einsum('bhql, bhkl -> bhqk', q, k) / math.sqrt(self.head_size)
            attn_scores = torch.where(mask, neg_inf, attn_scores)

        probs = attn_scores.softmax(dim=-1)
        combined_v = torch.einsum('bhqk, bhkl -> bhql', probs, v)
        combined_v = einops.rearrange(combined_v, 'b h q l -> b q (h l)')
        out = self.output_proj(combined_v)
        if return_key_values:
            return out, torch.cat([new_k, new_v], dim=-1)
        return out


gpt_tests.test_unidirectional_attn(UniAttention)
gpt_tests.test_attn_cache(UniAttention)


class GPT2Block(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float, layer_norm_epsilon: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.attn = UniAttention(hidden_size, num_heads)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, past_key_values=None, return_key_values=False):
        if return_key_values:
            attn_output, new_key_values = self.attn(self.ln1(x), past_key_values=past_key_values,
                                                    return_key_values=return_key_values)
            x = x + attn_output
            x = x + self.dropout(self.linear2(F.gelu(self.linear1(self.ln2(x)))))
            return x, new_key_values
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.dropout(self.linear2(F.gelu(self.linear1(self.ln2(x)))))
            return x


gpt_tests.test_gpt_block(GPT2Block)    


@dataclass
class GPT2Output:
    logits: TensorType["batch_size", "vocab_size"]
    final_encoding: TensorType["batch_size", "hidden_size"]


class GPT2(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size, hidden_size, max_position_embeddings, dropout,
                 layer_norm_epsilon, use_cache=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            GPT2Block(hidden_size, num_heads, dropout, layer_norm_epsilon) for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        self.use_cache = use_cache
        head_size = hidden_size // num_heads
        self.cache_size = (num_layers, num_heads, 0, 2 * head_size)
        self.clear_cache()
        
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        self.vocab_size = vocab_size

    def clear_cache(self):
        self._cache_kv = torch.zeros(self.cache_size).to(self.ln.weight.device)

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        pos = torch.arange(seq_len).to(input_ids.device)

        if not self.use_cache:
            enc = self.dropout(self.token_embedding(input_ids) + self.pos_embedding(pos))
            enc = self.blocks(enc)
        
        elif self._cache_kv.shape[2] == 0:
            assert input_ids.shape[0] == 1
            enc = self.dropout(self.token_embedding(input_ids) + self.pos_embedding(pos))
            new_key_values = []
            for i, block in enumerate(self.blocks):
                enc, new_kv = block(enc, return_key_values=True)
                new_key_values.append(new_kv)
            self._cache_kv = torch.cat(new_key_values, dim=0)
            
        else:
            assert input_ids.shape[0] == 1
            enc = self.dropout(self.token_embedding(input_ids[:,-1:]) + self.pos_embedding(pos[-1:]))
            new_key_values = []
            for i, block in enumerate(self.blocks):
                enc, new_kv = block(enc, return_key_values=True, past_key_values=self._cache_kv[i])
                new_key_values.append(new_kv)
            last_token_cache = torch.cat(new_key_values, dim=0)
            self._cache_kv = torch.cat([self._cache_kv, last_token_cache], dim=2)

        self._enc = enc
        enc = self.ln(enc)
        logits = torch.einsum('bnl, vl -> bnv', enc, self.token_embedding.weight)
        return GPT2Output(logits=logits[:,-1,:], final_encoding=enc[:,-1,:])

    def next_token(self, input_ids, temperature, freq_penalty=2.0):
        logits = self(input_ids.unsqueeze(0)).logits[0]
        id_freqs = torch.bincount(input_ids, minlength=self.vocab_size)
        logits = logits / temperature - freq_penalty * id_freqs
        return torch.distributions.categorical.Categorical(logits=logits).sample()

    def generate(self, text, max_length=30, temperature=1.0, freq_penalty=2.0):
        self.empty_cache()
        input_ids = self.tokenizer(text).input_ids
        generated = []
        for i in range(max_length):
            new_token = self.next_token(torch.LongTensor(input_ids + generated),
                                        temperature=temperature, freq_penalty=freq_penalty)
            generated.append(new_token)
            if new_token == self.tokenizer.eos_token_id:
                break
        return self.tokenizer.decode(input_ids + generated)
        
    
gpt_tests.test_gpt(GPT2)    
gpt_tests.test_gpt_cache(GPT2)

# gpt = load_weights(GPT2)
# gpt.generate("After a long day's work, I like to")


def load_weights(GPT2Class):
    pretrained_gpt = gpt_tests.get_pretrained_gpt()
    my_gpt = GPT2Class(num_layers=12, num_heads=12, vocab_size=50257, hidden_size=768, max_position_embeddings=1024,
                       dropout=0.1, layer_norm_epsilon=1e-5)

    state_dict = {mykey: v for (k, v), mykey in zip(pretrained_gpt.state_dict().items(), my_gpt.state_dict().keys())}
    my_gpt.load_state_dict(state_dict)
    return my_gpt


def bert_vs_gpt(gpt, bert):
    sentences = [
        "My life motto:",
        "My life motto: Fortune",
        "My life motto: Fortune favors",
        "My life motto: Fortune favors the",
        "My life motto: Fortune favors the bold",
    ]

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer(sentences).input_ids
    maxlen = max(len(sent) for sent in input_ids)
    input_ids = torch.LongTensor([sent + [tokenizer.pad_token_id]*(maxlen-len(sent)) for sent in input_ids])

    gpt.eval()(input_ids);
    gpt._enc[:, 3]  # assumes GPT saves encodings in self._enc

    bert.eval()(input_ids);
    bert._enc[:, 3]  # assumes Bert saves encodings in self._enc


    
