import torch as t
import gpt_tests
from einops import rearrange
import torch.nn.functional as F
import transformers
import days.w2d1.bert_sol as bert_sol
import days.w2d1.bert_tests as bert_tests
VOCAB_SIZE = 50257
BERT_VOCAB_SIZE = 28996

class UniModule(t.nn.Module):
    def __init__(self, hidden_size, num_heads):
        assert hidden_size % num_heads == 0
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.attention = t.nn.Linear(hidden_size, hidden_size * 3)
        self.output_proj = t.nn.Linear(hidden_size, hidden_size)

    def forward(self, x, past_key_values=None, return_key_values=False): # b seq_len hidden_size
        if past_key_values is None:
            qkv = self.attention(x)
            q, k, v = t.split(qkv, self.hidden_size, dim=-1)
            q = rearrange(q, 'b s (n h) -> b n s h', n=self.num_heads)
            k = rearrange(k, 'b s (n h) -> b n s h', n=self.num_heads)
            v = rearrange(v, 'b s (n h) -> b n s h', n=self.num_heads)
            qk = t.einsum('bnil,bnjl->bnij', q, k)
            qk /= self.head_size**0.5
            qk = t.tril(qk)
            # setting everything qk can't attend to
            qk[t.triu(t.ones_like(qk), diagonal=1).bool()] = -1e4
            qk = F.softmax(qk, dim=-1)
            # qk: batch num_heads seq_len seq_len
            # v: batch seq_len num_heads head_size
            # out: batch num_heads seq_len head_size
            combined = t.einsum('bnij,bnjh->bnih', qk, v)
            combined = rearrange(combined, 'b n i h -> b i (n h)')
            if return_key_values:
                return self.output_proj(combined), t.cat((k, v), dim=-1)
            else:
                return self.output_proj(combined)
        else:
            expected_input_shape = (1, 1, self.hidden_size)
            assert x.shape == expected_input_shape, (x.shape, expected_input_shape)
            # k and v: batch_size num_heads seq_len head_size
            k, v = t.split(past_key_values.unsqueeze(0), self.head_size, dim=-1)

            qkv_x = self.attention(x)
            # each of these: (batch_size 1 hidden_size) == (1, 1, hidden_size)
            q_x, k_x, v_x = t.split(qkv_x, self.hidden_size, dim=-1)
            # rearrange to  (1, num_heads, 1, head_size)
            q_x = rearrange(q_x, 'b s (n h) -> b n s h', n=self.num_heads)
            k_x = rearrange(k_x, 'b s (n h) -> b n s h', n=self.num_heads)
            v_x = rearrange(v_x, 'b s (n h) -> b n s h', n=self.num_heads)

            # q_x: (1, 1, num_heads, head_size)
            # k: batch_size num_heads seq_len_so_far head_size
            # qk: batch_size num_heads 1 seq_len_so_far
            qk = t.einsum('bnil,bnjl->bnij', q_x, k)
            q_x_with_k_x = t.einsum('bnil,bnjl->bni', q_x, k_x)
            qk = t.cat((qk, q_x_with_k_x.unsqueeze(-1)), dim=-1)
            qk /= self.head_size**0.5
            qk = F.softmax(qk, dim=-1)
            v = t.cat((v, v_x), dim=-2)
            combined = t.einsum('bnij,bnjh->bnih', qk, v)
            combined = rearrange(combined, 'b n i h -> b i (n h)')
            if return_key_values:
                return self.output_proj(combined), t.cat((k_x, v_x), dim=-1)
            else:
                return self.output_proj(combined)

# gpt_tests.test_unidirectional_attn(UniModule)
# gpt_tests.test_attn_cache(UniModule)

class GPT2Block(t.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float, layer_norm_epsilon: float):
        super().__init__()
        self.layer_norm1 = t.nn.LayerNorm(normalized_shape=hidden_size, eps=layer_norm_epsilon)
        self.attention = UniModule(hidden_size, num_heads)
        self.layer_norm2 = t.nn.LayerNorm(normalized_shape=hidden_size, eps=layer_norm_epsilon)
        self.linear1 = t.nn.Linear(hidden_size, 4 * hidden_size)
        self.linear2 = t.nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, x, past_key_values=None, return_key_values=False): # [batch, seq_len, hidden_size]
        layer_normed_1 = self.layer_norm1(x)
        if return_key_values:
            attentioned, new_key_values = self.attention(layer_normed_1, past_key_values, return_key_values)
        else:
            attentioned = self.attention(layer_normed_1, past_key_values, return_key_values)
            
        attentioned_x = x + attentioned
        layer_normed_2 = self.layer_norm2(attentioned_x)
        mlped = self.dropout(self.linear2(F.gelu(self.linear1(layer_normed_2))))

        if return_key_values:
            return attentioned_x + mlped, new_key_values
        else:
            return attentioned_x + mlped

# t.backends.cuda.matmul.allow_tf32 = False
# t.backends.cudnn.allow_tf32 = False
# # gpt_tests.test_gpt_block(GPT2Block)
# input("Done")

from dataclasses import dataclass
from torchtyping import TensorType

@dataclass
class GPT2Output:
    logits: TensorType["batch_size", "vocab_size"]
    # encoding of the last word in the sequence
    final_encoding: TensorType["batch_size", "hidden_size"]

class GPT2Module(t.nn.Module):
    def __init__(self, 
        num_layers, 
        num_heads, 
        vocab_size, 
        hidden_size,
        max_position_embeddings, 
        dropout, 
        layer_norm_epsilon,
        use_cache=False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = t.nn.Parameter(t.randn(vocab_size, hidden_size))
        self.position_embedding = t.nn.Parameter(t.randn(max_position_embeddings, hidden_size))
        self.dropout = t.nn.Dropout(dropout)
        self.blocks = t.nn.ModuleList([
            GPT2Block(
                hidden_size,
                num_heads,
                dropout,
                layer_norm_epsilon
            ) for _  in range(num_layers)
        ])
        self.layer_norm = t.nn.LayerNorm(normalized_shape=hidden_size, eps=layer_norm_epsilon)
        self.use_cache = use_cache

        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads 
        self.num_layers = num_layers

        self.register_buffer('cache', t.zeros((num_layers, num_heads, 0, 2 * self.head_size)), persistent=False)

        self.id_frequencies = t.zeros(self.vocab_size)
        self.tokenizer = None

    def forward(self, input_ids): # [batch, seq_len]
        is_cache_empty = self.cache.shape[2] == 0
        result = None
        if self.use_cache and not is_cache_empty:
            # assume that if the cache isn't empty, then we're continuing from
            # the previous forward() call with one extra token
            position = input_ids.shape[1] - 1
            input_ids = input_ids[:,-1:]
            seq_len = input_ids.shape[1]
            result = self.token_embedding[input_ids] + self.position_embedding[position]
        else:
            seq_len = input_ids.shape[1]
            result = self.token_embedding[input_ids] + self.position_embedding[t.arange(seq_len)]

        result = self.dropout(result)

        if self.use_cache:
            self.cache = t.cat((self.cache, t.zeros((self.num_layers, self.num_heads, seq_len, 2 * self.head_size))), dim=-2)
            if is_cache_empty:
                for i, block in enumerate(self.blocks):
                    result, kv = block(result, past_key_values=None, return_key_values=True)
                    self.cache[i,:,:,:] = kv
            else:
                for i, block in enumerate(self.blocks):
                    result, kv_x = block(result, past_key_values=self.cache[i,:,:-1,:], return_key_values=True)
                    self.cache[i,:,-1:,:] = kv_x
        else:
            for block in self.blocks:
                result = block(result)
        self._enc = result

        all_encodings = self.layer_norm(result)
        final_encoding = all_encodings[:,-1,:]
        logits = t.einsum('bh,vh->bv', final_encoding, self.token_embedding)
        return GPT2Output(logits, final_encoding)

    def next_token(self, input_ids, temperature, freq_penalty=2.0):
        gpt2_output = self.forward(input_ids)
        logits = gpt2_output.logits[0]
        encoding = gpt2_output.final_encoding
        
        new_logits = logits / temperature - self.id_frequencies * freq_penalty
        probability_dist = F.softmax(new_logits, dim=0)

        p = rearrange(probability_dist, '(d1 d2) -> d1 d2', d1=1)
        p = t.cumsum(p, dim=1) - t.rand(1, 1)
        p = (p > 0).float()
        result = t.argmax(p, dim=1).item()
        return result

    def generate(self, text, max_length=30, temperature=1.0, freq_penalty=2.0):
        if self.tokenizer is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")        

        self.cache = t.zeros(self.num_layers, self.num_heads, 0, 2 * self.head_size)
        self.id_frequencies = t.zeros(self.vocab_size)
        tokens = t.tensor(self.tokenizer.encode(text)).long()

        for token in tokens: 
            self.id_frequencies[token.item()] += 1

        while tokens.shape[0] < max_length and tokens[-1] != self.tokenizer.eos_token_id:
            next_token = self.next_token(tokens.unsqueeze(0), temperature=temperature, freq_penalty=freq_penalty)
            new_tokens = t.zeros(tokens.shape[0] + 1).long()
            new_tokens[:-1] = tokens
            new_tokens[-1] = next_token
            tokens = new_tokens
            self.id_frequencies[next_token] += 1
        return self.tokenizer.decode(tokens.long())

# gpt2_tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
# gpt2_tokenized = gpt2_tokenizer.decode(raw_tokens + [token])
# print(gpt2_tokenized)
# gpt2_tokenized = gpt2_tokenizer.decode(raw_tokens)
# print(gpt2_tokenized)

# gpt_tests.test_gpt(GPT2Module)
# gpt_tests.test_gpt_cache(GPT2Module)

def load_gpt():
    # Loading weights
    gpt = GPT2Module(num_layers=12, num_heads=12, vocab_size=VOCAB_SIZE, hidden_size=768, max_position_embeddings=1024, dropout=0.1, layer_norm_epsilon=1e-5)
    pretrained_gpt = gpt_tests.get_pretrained_gpt()
    gpt_keys = list(gpt.state_dict().keys())  # listifying this so that we can index into it
    pretrained_values = pretrained_gpt.state_dict().values()
    assert len(gpt_keys) == len(pretrained_values)

    state_dict = {}
    # assumes gpt.state_dict has same ordering as pretrained_gpt.state_dict
    for i, value in enumerate(pretrained_gpt.state_dict().values()):
        state_dict[gpt_keys[i]] = value
    gpt.load_state_dict(state_dict)
    return gpt

def load_bert():
    bert = bert_sol.Bert(
        vocab_size=BERT_VOCAB_SIZE,
        hidden_size=768,
        max_position_embeddings=512,
        type_vocab_size=2,
        dropout=0.1,
        intermediate_size=3072,
        num_heads=12,
        num_layers=12,
    )
    pretrained_bert = bert_tests.get_pretrained_bert()
    mapped_params = {bert_sol.mapkey(k): v for k, v in pretrained_bert.state_dict().items() if not k.startswith('classification_head')}
    bert.load_state_dict(mapped_params)
    return bert

def look_at_encodings():
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    gpt2_tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    gpt2_tokenizer.pad_token = 0

    SENTENCES = ['My life motto:']
    for suffix in ['Fortune', 'favors', 'the', 'bold']:
        SENTENCES.append(SENTENCES[-1] + ' ' +suffix)

    bert_tokenized = bert_tokenizer(SENTENCES, padding="longest")["input_ids"]
    bert_tokenized = t.tensor(bert_tokenized)
    bert = load_bert()
    bert.eval()
    bert(bert_tokenized)
    bert_encodings = bert._enc
    print("Bert motto?", bert_tokenizer.decode([13658]))
    print("GPT motto?", gpt2_tokenizer.decode([33600]))

    gpt2_tokenized = gpt2_tokenizer(SENTENCES, padding="longest")["input_ids"]
    gpt2_tokenized = t.tensor(gpt2_tokenized)
    gpt2 = load_gpt()
    gpt2.eval()
    gpt2(gpt2_tokenized)
    gpt2_encodings = gpt2._enc

    print(bert_tokenized)
    print()
    print(gpt2_tokenized)

    print("BERT ########################")
    print(bert_encodings[:,3])
    print("\n\nGPT ########################")
    print(gpt2_encodings[:,2])

# config = dict(num_layers=2, num_heads=4, vocab_size=VOCAB_SIZE, hidden_size=64,
#     max_position_embeddings=500, dropout=0.0, layer_norm_epsilon=1e-4)
# gpt2 = GPT2Module(**config)

gpt2 = load_gpt()
for _ in range(10):
    print("-------BEGIN GENERATION--------")
    print(gpt2.generate('There was a young man named Martin with an offer to read history at Queen\'s college, Oxford', max_length=60, temperature=1.0))
    print("--------END GENERATION---------")
