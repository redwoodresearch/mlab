import days.w2d1.bert_tests as bert_tests
import torch as t
from torch import einsum
from torch.nn import Module
from einops import rearrange, reduce, repeat
from typing import Callable
from torchtyping import patch_typeguard, TensorType
import math

'''
def raw_attention_pattern(
    token_activations: TensorType["batch", "seq_len", "hidden"],
    num_heads: int, 
    project_query: Callable[
        [TensorType["batch", "seq_len", "hidden"]],
        TensorType["batch", "seq_len", "h_times_d"]
    ],
    project_key: Callable[
        [TensorType["batch", "seq_len", "hidden"]],
        TensorType["batch", "seq_len", "h_times_d"]
    ]
) -> TensorType["batch", "h", "seq_len", "seq_len"]:

    # Multiply Q, K learned weights by tokens to get matrices for each token
    qs = project_query(token_activations)
    ks = project_key(token_activations)

    # Unflatten the (h d) dimension to two dimensions of sizes h, d
    qs = rearrange(qs, "b s (h d) -> b h s d", h=num_heads)
    ks = rearrange(ks, "b s (h d) -> b h s d", h=num_heads)

    # Add a fake dimension t of size seq_len so that we can get all pairwise combinations of tokens in the sequence
    seq_len = token_activations.shape[1] 
    qs = repeat(qs, "b h s d -> b h s t d", t=seq_len)
    ks = repeat(ks, "b h s d -> b h s t d", t=seq_len)q

    qk = einsum("b h s t d, b h t s d -> b h t s", qs, ks) / t.sqrt(t.tensor(qs.shape[-1]))

    return qk
'''

def raw_attention_pattern(
    token_activations,  # Tensor[batch_size, seq_length, hidden_size(768)],
    num_heads,
    project_query,      # nn.Module, (Tensor[..., 768]) -> Tensor[..., 768],
    project_key,        # nn.Module, (Tensor[..., 768]) -> Tensor[..., 768] 
): # -> Tensor[batch_size, head_num, key_token: seq_length, query_token: seq_length]:
    Q = project_query(token_activations)
    Q = rearrange(Q, 'b seqlen (headnum headsize) -> b headnum seqlen headsize',
                  headnum=num_heads)
    K = project_key(token_activations)
    K = rearrange(K, 'b seqlen (headnum headsize) -> b headnum seqlen headsize',
                  headnum=num_heads)
    headsize = K.shape[-1]
    scores = einsum('bhql, bhkl -> bhkq', Q, K) / math.sqrt(headsize)
    return scores

def bert_attention(
    token_activations: TensorType["batch", "seq_len", "hidden"],
    num_heads: int, 
    attention_pattern: TensorType["batch", "num_heads", "seq_len", "seq_len"],
    project_value: Callable[
        [TensorType["batch", "seq_len", "hidden"]],
        TensorType["batch", "seq_len", "h_times_d"]
    ],
    project_output: Callable[
        [TensorType["batch", "seq_len", "h_times_d"]],
        TensorType["batch", "seq_len", "hidden"]
    ]
) -> TensorType["batch", "seq_len", "hidden"]:
    sm = t.softmax(attention_pattern, dim= -2)

    vs = project_value(token_activations) # ["batch", "seq_len", "h_times_d"]
    
    vs = rearrange(vs, "b s (h d) -> b h s d", d = vs.shape[-1]//num_heads)

    sm_v = einsum("b h s t, b h s d -> b h t d", sm, vs)
    sm_v = rearrange(sm_v, "b h t d -> b t (h d)")
    return project_output(sm_v)

    b, h, s, t_ = sm.shape
    d = vs.shape[-1]//num_heads
    sm = t.as_strided(
        sm, size=[b, h, d, s, t_],
        stride=list(sm.stride()[:2])+[0]+list(sm.stride()[2:])
    )
    # the above is a memory efficient version of:
    # repeat(sm, "b h s t -> b h d s t", d = vs.shape[-1]//num_heads)
    vs = rearrange(vs, "b s (h d) -> b h s d", d = vs.shape[-1]//num_heads)
    o = einsum("b h d s t, b h s d -> b t h d", sm, vs)
    o = rearrange(o, "b t h d -> b t (h d)")

    # o = einsum("b m t s, b t m -> b s m", sm, vs)



class MultiHeadedSelfAttention(Module):
    def __init__(self,
        num_heads: int,
        hidden_size:int):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size


        self.project_query = t.nn.Linear(hidden_size, hidden_size)
        self.project_key = t.nn.Linear(hidden_size, hidden_size)    
        self.project_value = t.nn.Linear(hidden_size, hidden_size)
        self.project_out = t.nn.Linear(hidden_size, hidden_size)
        

    def forward(self, input: TensorType["batch_size", "seq_length", "hidden_size"]):
        rap = raw_attention_pattern(input , self.num_heads, self.project_query, self.project_key)

        full_attention = bert_attention(input, self.num_heads, rap, self.project_value, self.project_out)

        return full_attention


def bert_mlp(
    token_activations: TensorType["batch", "seq_len", "hidden"],
    linear_1: Module,
    linear_2: Module
) -> TensorType["batch", "seq_len", "hidden"]:
    gelu = t.nn.GELU()
    return linear_2(gelu(linear_1(token_activations)))



class BertMLP(Module):
    def __init__(self, input_size: int, intermediate_size: int):
        super().__init__()
        self.linear_1 = t.nn.Linear(input_size, intermediate_size)
        self.linear_2 = t.nn.Linear(intermediate_size, input_size)

    def forward(self, x: TensorType["batch", "seq_len", "input_size"]):
        return bert_mlp(x, self.linear_1, self.linear_2)


class LayerNorm(Module):
    def __init__(self, normalized_dim: int):
        super().__init__()
        self.weight = t.nn.parameter.Parameter(t.ones(normalized_dim))
        self.bias = t.nn.parameter.Parameter(t.zeros(normalized_dim))
        self.epsilon = 1e-5

    def forward(self, x: t.Tensor):
        centered = x - t.unsqueeze(t.mean(x, dim=-1).detach(), dim=-1)
        normalized = centered/t.sqrt(t.unsqueeze(t.var(x, dim=-1, unbiased=False).detach(), dim=-1) + self.epsilon)
        return normalized*self.weight + self.bias


class BertBlock(Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, dropout: float):
        super().__init__()
 
        self.norm1 = LayerNorm(hidden_size)
        self.attention = MultiHeadedSelfAttention(num_heads, hidden_size)
        self.feedforward = BertMLP(hidden_size, intermediate_size)
        self.dropout = t.nn.Dropout(dropout)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x):
        x = self.attention(x) + x
        x = self.norm1(x)
        x = self.dropout(self.feedforward(x)) + x
        x = self.norm2(x)

        return x


class Embedding(Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.weight = t.nn.parameter.Parameter(t.randn((vocab_size, embed_size)))

    def forward(self, input):
        a = t.nn.functional.one_hot(input, self.vocab_size)
        return einsum("... v, v e -> ... e", a.float(), self.weight)


def bert_embedding(
    input_ids: TensorType['batch', 'seqlen'], 
    token_type_ids: TensorType['batch', 'seqlen'], 
    position_embedding: Embedding,
    token_embedding: Embedding, 
    token_type_embedding: Embedding, 
    layer_norm: LayerNorm, 
    dropout: t.nn.Dropout):

    te = token_embedding(input_ids)
    tte = token_type_embedding(token_type_ids)
    indices = t.arange(0,input_ids.shape[-1]).to(input_ids.device)
    pe = repeat(position_embedding(indices), "s h-> b s h", b = input_ids.shape[0])

    e = te + tte + pe

    return dropout(layer_norm(e))


class BertEmbedding(Module):
    def __init__(self,
    vocab_size: int,
    hidden_size: int,
    max_position_embeddings: int,
    type_vocab_size: int, 
    dropout: float):
        super().__init__()

        self.token_embedding =  Embedding(vocab_size, hidden_size)
        self.position_embedding = Embedding(max_position_embeddings, hidden_size)
        self.token_type_embedding = Embedding(type_vocab_size, hidden_size) 
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids):
        x = bert_embedding(
            input_ids,
            token_type_ids,
            self.position_embedding,
            self.token_embedding,
            self.token_type_embedding,
            self.layer_norm,
            self.dropout,
        )

        return x

class BaseBert(Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int, 
        max_position_embeddings: int,
        type_vocab_size: int, 
        dropout: float, 
        intermediate_size: int,
        num_heads: int, 
        num_layers: int
        ):
        super().__init__()

        self.embed = BertEmbedding(
            vocab_size,
            hidden_size,
            max_position_embeddings,
            type_vocab_size, 
            dropout
        )

        self.transformer = t.nn.Sequential(*[BertBlock(hidden_size, intermediate_size, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, input_ids):
        x = input_ids
        fake_token_types = t.zeros_like(x)

        x = self.embed(x, fake_token_types)
        x = self.transformer(x)

        return x

class LMBertHead(Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int, 
        ):
        super().__init__()

        self.mlp = t.nn.Linear(hidden_size, hidden_size)

        self.gelu = t.nn.GELU()

        self.unembedding = t.nn.Linear(hidden_size, vocab_size)

        self.layer_norm = LayerNorm(hidden_size)

    def forward(self, input_ids):
        x = input_ids
        
        x = self.mlp(x)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.unembedding(x)
        
        return x


class LMBert(Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int, 
        **kwargs
        ):
        super().__init__()

        self.base_bert = BaseBert(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            **kwargs
        )

        self.lmbert = LMBertHead(vocab_size=vocab_size, hidden_size=hidden_size)

    def forward(self, input_ids):
        x = input_ids
        
        x = self.base_bert(x)
        x = self.lmbert(x)
        
        return x


def load_pretrained_bert(my_bert):
    pretrained_bert = bert_tests.get_pretrained_bert()

    my_keys = my_bert.state_dict().keys()
    their_keys = pretrained_bert.state_dict().keys()

    key_fixed_state_dict = {}
    for my_key, their_key in zip(my_keys, their_keys):
        key_fixed_state_dict[my_key] = pretrained_bert.state_dict()[their_key]

    my_bert.load_state_dict(key_fixed_state_dict)


def guess_words(model, masked_sentence, tokenizer):
    input_ids = tokenizer(masked_sentence, return_tensors="pt")["input_ids"]
    guess_ids = t.argmax(model(input_ids), dim=-1).flatten()
    final_tokens = input_ids[0].numpy()

    for pos, token in enumerate(input_ids[0]):
        if token==103:
            final_tokens[pos] = guess_ids[pos]

    return(tokenizer.decode(final_tokens))
