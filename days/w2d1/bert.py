import torch as t
import torch.nn as nn
from torch import einsum
from einops import rearrange, reduce, repeat
import einops
import bert_tests
from torchtext.datasets import WikiText2
import random
import transformers

def raw_attention_pattern(token_activations, num_heads, project_query, project_key):
    queries = project_query(token_activations)
    keys = project_key(token_activations)
    keys_reshaped = rearrange(keys, 'b l (h p) -> b h l p', h = num_heads)
    queries_reshaped = rearrange(queries, 'b l (h p) -> b h l p', h = num_heads)
    keys_times_queries = t.einsum('b h l p, b h m p -> b h l m', keys_reshaped, queries_reshaped) / t.sqrt(t.tensor(keys.shape[-1]//num_heads))
    return keys_times_queries

def bert_attention(token_activations, num_heads, attention_pattern, project_value, project_output):
    projected_input = project_value(token_activations)
    soft_max = t.nn.functional.softmax(attention_pattern, dim=-2)
    activations_reshaped = rearrange(projected_input, 'b l (h p) -> b h l p', h = num_heads)
    weighted_activations = t.einsum('b h l m, b h l p -> b h m p', soft_max, activations_reshaped)
    return project_output(rearrange(weighted_activations, 'b h m p -> b m (h p)'))

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, num_heads, hidden_size):
        super(MultiHeadedSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.keys = nn.Linear(hidden_size, hidden_size)
        self.project_value = nn.Linear(hidden_size, hidden_size)
        self.project_out = nn.Linear(hidden_size, hidden_size)

    
    def forward(self, token_activations):
        pattern = raw_attention_pattern(token_activations, self.num_heads, lambda a: self.query(a), lambda a: self.keys(a))
        attn = bert_attention(token_activations, self.num_heads, pattern, lambda a: self.project_value(a), lambda a: self.project_out(a))
        return attn
    
def bert_mlp(token_activations, linear_1, linear_2):
    return linear_2(nn.functional.gelu(linear_1(token_activations)))

class BertMLP(nn.Module):
    def __init__(self, input_size, intermediate_size):
        super(BertMLP, self).__init__()
        self.linear1 = nn.Linear(input_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, input_size)
        
    def forward(self, x):
        return bert_mlp(x, self.linear1, self.linear2)
    
class LayerNorm(nn.Module):
    def __init__(self, normalized_dim: int):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(t.ones(normalized_dim))
        self.bias = nn.Parameter(t.zeros(normalized_dim))
        
    def forward(self, x):
        x = (x - t.mean(x, dim=-1).unsqueeze(-1))/t.std(x, dim = -1, unbiased=False).unsqueeze(-1)
        x = x * self.weight + self.bias
        return x
    
class BertBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout):
        super(BertBlock, self).__init__()
        self.attention = MultiHeadedSelfAttention(num_heads, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.mlp = BertMLP(hidden_size, intermediate_size)
        self.ln2 = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x_mhsa = self.attention(x)
        x_ln1 = self.layer_norm(x_mhsa + x)
        x_mlp = self.mlp(x_ln1)
        x_dropout = self.dropout(x_mlp)
        x_ln2 = self.ln2(x_dropout + x_ln1) 
        return x_ln2
    
class Embedding(nn.Module):
    
    def __init__(self, vocab_size, embed_size):
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(t.randn(vocab_size, embed_size))
        
    def forward(self, x):
        return self.weight[x, :]
    
def bert_embedding(input_ids, token_type_ids, position_embedding, token_embedding, token_type_embedding, layer_norm, dropout):
    device = "cuda" if input_ids.is_cuda else "cpu"
    tens = t.arange(0, input_ids.shape[1])
    tens.to(device)
    pos_emb = position_embedding(tens)
    tok_emb = token_embedding(input_ids)
    typ_emb = token_type_embedding(token_type_ids)
    emb = pos_emb + tok_emb + typ_emb
    return dropout(layer_norm(emb))

class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout):
        super(BertEmbedding, self).__init__()
        self.token_embedding = Embedding(vocab_size, hidden_size)
        self.position_embedding = Embedding(max_position_embeddings, hidden_size)
        self.token_type_embedding = Embedding(type_vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(hidden_size)
    
    def forward(self, input_ids, token_type_ids):
        return bert_embedding(input_ids, token_type_ids, self.position_embedding, self.token_embedding, self.token_type_embedding, self.layer_norm, self.dropout)
    
class Bert(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout, intermediate_size, num_heads, num_layers):
        super(Bert, self).__init__()
        self.embedding = BertEmbedding(vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout)
        self.transformer = t.nn.Sequential(*[BertBlock(hidden_size, intermediate_size, num_heads, dropout) for _ in range(num_layers)])
        self.mlp = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.layer_norm = LayerNorm(hidden_size)
        self.unembedding = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        token_type_ids = t.zeros(input_ids.shape, dtype=t.long)
        token_type_ids.to("cuda" if input_ids.is_cuda else "cpu")
        embedding = self.embedding(input_ids, token_type_ids)
        output = self.transformer(embedding)
        lin = self.mlp(output)
        gelu = self.gelu(lin)
        layernorm = self.layer_norm(gelu)
        return self.unembedding(layernorm)
    
def repl(st):
    st = st.replace("pattern.project_key", "keys")
    st = st.replace("pattern.project_query", "query")
    st = st.replace("residual.mlp", "mlp.linear")
    st = st.replace("residual.layer_norm", "ln2")
    st = st.replace("lm_head.", "")
    return st

def mask_wiki(data, tokenizer, max_seq_len):
    orig_data = []
    all_data = []
    lengths = []
    for text in data:
        if len(text.strip().split(" ")) < 10:
            continue
        tokenized_text = tokenizer([text], max_length=max_seq_len, truncation=True)["input_ids"][0]
        lengths.append(len(tokenized_text))
        all_data += tokenized_text
        orig_data.append(tokenized_text)
        # if len(tokenized_text) < max_seq_len:
        #     tokenized_text += [0] * (max_seq_len - len(tokenized_text))
    for i in range(len(all_data)):
        if random.random() < 0.15:
            if random.random() < 0.8:
                all_data[i] = 103 # [MASK] token
            elif random.random() < 0.5:
                idx = random.randint(0, len(all_data))
                all_data[i] = all_data[idx]
    reshaped_data = []
    idx = 0
    for i in range(len(lengths)):
        reshaped_data.append(all_data[idx:idx + lengths[i]])
        idx += lengths[i]
    return reshaped_data, orig_data

def main():
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

    data_train, data_test = WikiText2(root='.data', split=('train', 'test'))
    wiki_train = list(data_train).copy()
    wiki_test = list(data_test).copy()
    
    vocab_size = 28996
    max_seq_len = 512
    tiny_bert = Bert(
        vocab_size=vocab_size, hidden_size=384, max_position_embeddings=max_seq_len, 
        type_vocab_size=2, dropout=0.1, intermediate_size=1536, 
        num_heads=12, num_layers=2
    )
    tiny_bert.eval()
    
    # pretrained_bert = bert_tests.get_pretrained_bert()
    # d = OrderedDict([(repl(k), v) for k,v in pretrained_bert.state_dict().items()])
    # tiny_bert.load_state_dict(d)
    
    masked_wiki, wiki = mask_wiki(wiki_train, tokenizer, max_seq_len)
    for i in range(len(masked_wiki)):
        if len(masked_wiki[i]) < max_seq_len:
            masked_wiki[i] += [0] * (max_seq_len - len(masked_wiki[i]))
            wiki[i] += [0] * (max_seq_len - len(wiki[i]))
    masked_wiki = t.Tensor(masked_wiki).long()
    wiki = t.Tensor(wiki).long()
    
    batch_size = 64
    batched_masked_wiki = einops.rearrange(masked_wiki[masked_wiki.shape[0]%batch_size:], "(k b) l -> k b l", b = batch_size)
    batched_wiki = einops.rearrange(wiki[wiki.shape[0]%batch_size:], "(k b) l -> k b l", b = batch_size)

    tiny_bert.train()
    tiny_bert.cuda()
    adam = t.optim.Adam(tiny_bert.parameters(), 0.01)
    num_batches = batched_masked_wiki.shape[0]
    batch_size = batched_masked_wiki.shape[1]
    for epoch in range(3):
        print("epoch", epoch)
        for batch_num in range(num_batches):
            adam.zero_grad()
            b = batched_masked_wiki[batch_num].cuda()
            l = batched_wiki[batch_num].cuda()
            model_output = tiny_bert(b)
            out = model_output
            # out = t.softmax(model_output, dim=-1)
            is_mask_token = (b == 103).detach()
            masked_tokens = l.masked_select(is_mask_token)
            predictions = einops.rearrange(out.masked_select(is_mask_token.unsqueeze(-1)), "(h w) -> h w", w= vocab_size)
            # actual = t.zeros_like(predictions.detach())
            # for i in range(masked_tokens.shape[0]):
            #     actual[i, masked_tokens[i].int()] = 1
            # print("Predictions:")
            # print(predictions)
            # print("Actual:")
            # print(actual)
            # print("p(correct)")
            # print(predictions[actual == 1])
            out_loss = nn.functional.cross_entropy(predictions, masked_tokens)
            out_loss.backward()    
            adam.step()
            if batch_num % 5 == 0:
                print("batch", batch_num, "loss", out_loss.item())

if __name__ == "__main__":
    main()
    

