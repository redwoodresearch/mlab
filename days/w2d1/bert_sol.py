import days.w2d1.bert_tests as bert_tests
from einops import rearrange, reduce, repeat
import math
import re
import torch as t
from torch import einsum
from torch.nn import functional as F
from torch import nn


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

if __name__ == "__main__":
    bert_tests.test_attention_pattern_fn(raw_attention_pattern)
    

def bert_attention(
    token_activations, #: Tensor[batch_size, seq_length, hidden_size (768)], 
    num_heads: int, 
    attention_pattern, #: Tensor[batch_size,num_heads, seq_length, seq_length], 
    project_value, #: function( (Tensor[..., 768]) -> Tensor[..., 768] ), 
    project_output, #: function( (Tensor[..., 768]) -> Tensor[..., 768] )
): # -> Tensor[batch_size, seq_length, hidden_size]
    softmaxed_attention = attention_pattern.softmax(dim=-2)
    V = project_value(token_activations)
    V = rearrange(V, 'b seqlen (headnum headsize) -> b headnum seqlen headsize',
                  headnum=num_heads)
    combined_values = einsum('bhkl, bhkq -> bhql', V, softmaxed_attention)
    out = project_output(rearrange(combined_values, 'b h q l -> b q (h l)'))
    return out

if __name__ == "__main__":
    bert_tests.test_attention_fn(bert_attention)


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, num_heads, hidden_size):
        super().__init__()
        self.num_heads = num_heads
        self.project_query = nn.Linear(hidden_size, hidden_size)
        self.project_key = nn.Linear(hidden_size, hidden_size)
        self.project_value = nn.Linear(hidden_size, hidden_size)
        self.project_output = nn.Linear(hidden_size, hidden_size)

    def forward(self, input):  # b n l
        attention_pattern = raw_attention_pattern(
            input, self.num_heads, self.project_query, self.project_key)
        return bert_attention(
            input, self.num_heads, attention_pattern, self.project_value, self.project_output)
    
if __name__ == "__main__":
    bert_tests.test_bert_attention(MultiHeadedSelfAttention)


def bert_mlp(token_activations, #: torch.Tensor[batch_size,seq_length,768],
             linear_1: nn.Module, linear_2: nn.Module
): #-> torch.Tensor[batch_size, seq_length, 768]
    return linear_2(F.gelu(linear_1(token_activations)))

if __name__ == "__main__":
    bert_tests.test_bert_mlp(bert_mlp)


class BertMLP(nn.Module):
    def __init__(self, input_size: int, intermediate_size: int):
        super().__init__()
        self.lin1 = nn.Linear(input_size, intermediate_size)
        self.lin2 = nn.Linear(intermediate_size, input_size)

    def forward(self, input):
        return bert_mlp(input, self.lin1, self.lin2)


class LayerNorm(nn.Module):
    def __init__(self, normalized_dim: int):
        super().__init__()
        self.weight = nn.Parameter(t.ones(normalized_dim))
        self.bias = nn.Parameter(t.zeros(normalized_dim))

    def forward(self, input):
        input_m0 = input - input.mean(dim=-1, keepdim=True).detach()
        input_m0v1 = input_m0 / input_m0.std(dim=-1, keepdim=True, unbiased=False).detach()
        return input_m0v1 * self.weight + self.bias

if __name__ == "__main__":
    bert_tests.test_layer_norm(LayerNorm)

    
class BertBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout: float):
        super().__init__()
        self.attention = MultiHeadedSelfAttention(num_heads, hidden_size)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.mlp = BertMLP(hidden_size, intermediate_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        out = self.layernorm1(input + self.attention(input))
        return self.layernorm2(self.dropout(self.mlp(out)) + out)

if __name__ == "__main__":
    bert_tests.test_bert_block(BertBlock)    


# import transformers
# tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
# print(tokenizer(['Hello, I am a sentence.']))

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embedding_matrix = nn.Parameter(t.randn(vocab_size, embed_size))

    def forward(self, input):
        return self.embedding_matrix[input]

if __name__ == "__main__":
    bert_tests.test_embedding(Embedding)


def bert_embedding(
    input_ids, #: [batch, seqlen], 
    token_type_ids, # [batch, seqlen], 
    position_embedding, #: nn.Embedding,
    token_embedding, #: nn.Embedding, 
    token_type_embedding, #: nn.Embedding, 
    layer_norm, #: nn.Module, 
    dropout #: nn.Module
):
    position = t.arange(input_ids.shape[1]).to(input_ids.device)
    position = repeat(position, 'n -> b n', b = input_ids.shape[0])
    out = (token_embedding(input_ids) + token_type_embedding(token_type_ids) +
           position_embedding(position))
    return dropout(layer_norm(out))

if __name__ == "__main__":
    bert_tests.test_bert_embedding_fn(bert_embedding)


class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size,
                 dropout: float):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embedding = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids):
        return bert_embedding(
            input_ids, token_type_ids, self.pos_embedding, self.token_embedding,
            self.token_type_embedding, self.layer_norm, self.dropout)

if __name__ == "__main__":
    bert_tests.test_bert_embedding(BertEmbedding)


class Bert(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size,
                 dropout, intermediate_size, num_heads, num_layers):
        super().__init__()
        self.embed = BertEmbedding(vocab_size, hidden_size, max_position_embeddings,
                                   type_vocab_size, dropout)
        self.blocks = nn.Sequential(*[
            BertBlock(hidden_size, intermediate_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.lin = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.unembed = nn.Linear(hidden_size, vocab_size)
        

    def forward(self, input_ids):
        token_type_ids = t.zeros_like(input_ids, dtype=int)
        emb = self.embed(input_ids, token_type_ids)
        self._enc = enc = self.blocks(emb)
        enc = self.lin(enc)
        return self.unembed(self.layer_norm(F.gelu(enc)))


class BertWithClassify(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size,
                 dropout, intermediate_size, num_heads, num_layers, num_classes):
        super().__init__()
        self.embed = BertEmbedding(vocab_size, hidden_size, max_position_embeddings,
                                   type_vocab_size, dropout)
        self.blocks = nn.Sequential(*[
            BertBlock(hidden_size, intermediate_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.lin = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.unembed = nn.Linear(hidden_size, vocab_size)
        self.classification_head = nn.Linear(hidden_size, num_classes)
        self.classification_dropout = nn.Dropout(dropout)
        

    def forward(self, input_ids):
        token_type_ids = t.zeros_like(input_ids, dtype=int)
        emb = self.blocks(self.embed(input_ids, token_type_ids))
        logits = self.unembed(self.layer_norm(F.gelu((self.lin(emb)))))
        classifs = self.classification_head(self.classification_dropout(emb[:, 0]))
        return logits, classifs


def mapkey(key):
    key = re.sub('^embedding\.', 'embed.', key)
    key = re.sub('\.position_embedding\.', '.pos_embedding.', key)
    key = re.sub('^lm_head\.mlp\.', 'lin.', key)
    key = re.sub('^lm_head\.unembedding\.', 'unembed.', key)
    key = re.sub('^lm_head\.layer_norm\.', 'layer_norm.', key)
    key = re.sub('^transformer\.([0-9]+)\.layer_norm', 'blocks.\\1.layernorm1', key)
    key = re.sub('^transformer\.([0-9]+)\.attention\.pattern\.',
                 'blocks.\\1.attention.', key)
    key = re.sub('^transformer\.([0-9]+)\.residual\.layer_norm\.',
                 'blocks.\\1.layernorm2.', key)
    
    key = re.sub('^transformer\.', 'blocks.', key)
    key = re.sub('\.project_out\.', '.project_output.', key)
    key = re.sub('\.residual\.mlp', '.mlp.lin', key)
    return key

if __name__ == "__main__":
    bert_tests.test_bert(Bert)    
    bert_tests.test_bert_classification(BertWithClassify) 
    my_bert = Bert(
        vocab_size=28996, hidden_size=768, max_position_embeddings=512, 
        type_vocab_size=2, dropout=0.1, intermediate_size=3072, 
        num_heads=12, num_layers=12
    )
    pretrained_bert = bert_tests.get_pretrained_bert()
    mapped_params = {mapkey(k): v for k, v in pretrained_bert.state_dict().items()
                    if not k.startswith('classification_head')}
    my_bert.load_state_dict(mapped_params)
    bert_tests.test_same_output(my_bert, pretrained_bert)




