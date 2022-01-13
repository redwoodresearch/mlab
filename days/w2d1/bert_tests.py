import torch as t
import transformers
import days.w2d1.bert_tao as bert
import torch.nn as nn
import torch.nn.functional as F
from days.utils import tpeek


def allclose(my_out, their_out, name, tol=1e-5):

    if not t.allclose(my_out, their_out, rtol=1e-4, atol=tol):
        errstring = f'error in {name}\n{tpeek("", my_out, ret=True)} \n!=\n{tpeek("", their_out, ret=True)}'
        raise AssertionError(errstring)
    else:
        tpeek(f"{name} MATCH!!!!!!!!\n", my_out)


def get_pretrained_bert():
    pretrained_bert, _ = bert.my_bert_from_hf_weights()
    return pretrained_bert


def test_back_forth_my_name_is_bert(string):
    assert string == "[CLS] colleges 天 largest happened smile donation 夫 [SEP]"


def test_attention_fn(fn):
    reference = bert.multi_head_self_attention
    hidden_size = 768
    batch_size = 2
    seq_length = 3
    num_heads = 12
    token_activations = t.empty(batch_size, seq_length, hidden_size).uniform_(-1, 1)
    attention_pattern = t.rand(batch_size, num_heads, seq_length, seq_length)
    project_value = nn.Linear(hidden_size, hidden_size)
    project_output = nn.Linear(hidden_size, hidden_size)
    dropout = t.nn.Dropout(0.1)
    dropout.eval()
    allclose(
        fn(
            token_activations=token_activations,
            num_heads=num_heads,
            attention_pattern=attention_pattern,
            project_value=project_value,
            # project_out=project_output,
            project_output=project_output,
            # dropout=dropout,
        ),
        reference(
            token_activations=token_activations,
            num_heads=num_heads,
            attention_pattern=attention_pattern,
            project_value=project_value,
            project_out=project_output,
            dropout=dropout,
        ),
        "attention",
    )


def test_attention_pattern_fn(fn):
    reference = bert.raw_attention_pattern
    hidden_size = 768
    token_activations = t.empty(2, 3, hidden_size).uniform_(-1, 1)
    num_heads = 12
    project_query = nn.Linear(hidden_size, hidden_size)
    project_key = nn.Linear(hidden_size, hidden_size)
    allclose(
        fn(
            token_activations=token_activations,
            num_heads=num_heads,
            project_query=project_query,
            project_key=project_key,
        ),
        reference(
            token_activations=token_activations,
            num_heads=num_heads,
            project_query=project_query,
            project_key=project_key,
        ),
        "attention pattern raw",
    )


def test_attention_pattern_single_head(fn):
    reference = bert.raw_attention_pattern
    hidden_size = 768
    token_activations = t.empty(2, 3, hidden_size).uniform_(-1, 1)
    num_heads = 12
    project_query = nn.Linear(hidden_size, hidden_size)
    project_key = nn.Linear(hidden_size, hidden_size)
    head_size = hidden_size // num_heads
    project_query_ub = nn.Linear(hidden_size, head_size)
    project_query_ub.weight = nn.Parameter(project_query.weight[:head_size])
    project_query_ub.bias = nn.Parameter(project_query.bias[:head_size])
    project_key_ub = nn.Linear(hidden_size, head_size)
    project_key_ub.weight = nn.Parameter(project_key.weight[:head_size])
    project_key_ub.bias = nn.Parameter(project_key.bias[:head_size])
    allclose(
        fn(
            token_activations=token_activations[0],
            project_query=project_query_ub,
            project_key=project_key_ub,
        ),
        reference(
            token_activations=token_activations,
            num_heads=num_heads,
            project_query=project_query,
            project_key=project_key,
        )[0, 0, :, :],
        "attention pattern raw",
    )


def test_bert_mlp(fn):
    reference = bert.bert_mlp
    hidden_size = 768
    intermediate_size = 4 * hidden_size

    token_activations = t.empty(2, 3, hidden_size).uniform_(-1, 1)
    mlp_1 = nn.Linear(hidden_size, intermediate_size)
    mlp_2 = nn.Linear(intermediate_size, hidden_size)
    dropout = t.nn.Dropout(0.1)
    dropout.eval()
    allclose(
        fn(token_activations=token_activations, linear_1=mlp_1, linear_2=mlp_2),
        reference(
            token_activations=token_activations,
            linear_1=mlp_1,
            linear_2=mlp_2,
            dropout=dropout,
        ),
        "bert mlp",
    )


def test_layer_norm(LayerNorm):
    ln1 = LayerNorm(10)
    ln2 = nn.LayerNorm(10)
    tensor = t.randn(20, 10)
    allclose(ln1(tensor), ln2(tensor), "layer norm")


# TODO write this
def test_bert(your_module):
    config = {
        "vocab_size": 28996,
        "intermediate_size": 3072,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_position_embeddings": 512,
        "dropout": 0.1,
        "type_vocab_size": 2,
    }
    t.random.manual_seed(0)
    reference = bert.Bert(config)
    reference.eval()
    t.random.manual_seed(0)
    theirs = your_module(**config)
    theirs.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer("hello there", return_tensors="pt")["input_ids"]
    allclose(
        theirs(input_ids=input_ids),
        reference(input_ids=input_ids).logits,
        "bert",
    )


def test_bert_classification(your_module):
    config = {
        "vocab_size": 28996,
        "intermediate_size": 3072,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_position_embeddings": 512,
        "dropout": 0.1,
        "type_vocab_size": 2,
        "num_classes": 2,
    }
    t.random.manual_seed(0)
    reference = bert.Bert(config)
    reference.eval()
    t.random.manual_seed(0)
    theirs = your_module(**config)
    theirs.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer("hello there", return_tensors="pt")["input_ids"]
    logits, classifs = theirs(input_ids=input_ids)
    allclose(
        logits,
        reference(input_ids=input_ids).logits,
        "bert",
    )

    allclose(
        classifs,
        reference(input_ids=input_ids).classification,
        "bert",
    )


def test_same_output(your_bert, pretrained_bert, tol=1e-4):
    vocab_size = pretrained_bert.embedding.token_embedding.weight.shape[0]
    input_ids = t.randint(0, vocab_size, (10, 20))
    allclose(
        your_bert.eval()(input_ids),
        pretrained_bert.eval()(input_ids).logits,
        "comparing Berts",
        tol=tol,
    )


def test_bert_block(your_module):
    config = {
        "vocab_size": 28996,
        "intermediate_size": 3072,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_position_embeddings": 512,
        "dropout": 0.1,
        "type_vocab_size": 2,
    }
    t.random.manual_seed(0)
    reference = bert.BertBlock(config)
    reference.eval()
    t.random.manual_seed(0)
    theirs = your_module(
        intermediate_size=config["intermediate_size"],
        hidden_size=config["hidden_size"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
    )
    theirs.eval()
    input_activations = t.rand((2, 3, 768))
    allclose(
        theirs(input_activations),
        reference(input_activations),
        "bert",
    )


def test_embedding(Embedding):
    input = t.randint(0, 10, (2, 3))
    t.manual_seed(1157)
    emb1 = Embedding(10, 5)
    t.manual_seed(1157)
    emb2 = nn.Embedding(10, 5)
    allclose(emb1(input), emb2(input), "embedding")


def test_bert_embedding_fn(your_fn):
    config = {
        "vocab_size": 28996,
        "hidden_size": 768,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "dropout": 0.1,
    }
    input_ids = t.randint(0, 2900, (2, 3))
    tt_ids = t.randint(0, 2, (2, 3))
    reference = bert.BertEmbedding(config)
    reference.eval()
    allclose(
        your_fn(
            input_ids=input_ids,
            token_type_ids=tt_ids,
            token_embedding=reference.token_embedding,
            token_type_embedding=reference.token_type_embedding,
            position_embedding=reference.position_embedding,
            layer_norm=reference.layer_norm,
            dropout=reference.dropout,
        ),
        reference(input_ids=input_ids, token_type_ids=tt_ids),
        "bert embedding",
    )


def test_bert_embedding(your_module):
    config = {
        "vocab_size": 28996,
        "hidden_size": 768,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "dropout": 0.1,
    }
    input_ids = t.randint(0, 2900, (2, 3))
    tt_ids = t.randint(0, 2, (2, 3))
    t.random.manual_seed(0)
    reference = bert.BertEmbedding(config)
    reference.eval()
    t.random.manual_seed(0)
    yours = your_module(**config)
    yours.eval()
    allclose(
        yours(input_ids=input_ids, token_type_ids=tt_ids),
        reference(input_ids=input_ids, token_type_ids=tt_ids),
        "bert embedding",
    )


def test_bert_attention(your_module):
    config = {
        "vocab_size": 28996,
        "intermediate_size": 3072,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_position_embeddings": 512,
        "dropout": 0.0,  # not testing dropout!!
        "type_vocab_size": 2,
    }
    t.random.manual_seed(0)
    reference = bert.SelfAttentionLayer(config)
    reference.eval()
    t.random.manual_seed(0)
    theirs = your_module(
        hidden_size=config["hidden_size"],
        num_heads=config["num_heads"],
        # dropout=config["dropout"],
    )
    theirs.eval()
    input_activations = t.rand((2, 3, 768))
    allclose(
        theirs(input_activations),
        reference(input_activations),
        "bert",
    )


def test_bert_attention_pattern(your_module):
    config = {
        "vocab_size": 28996,
        "intermediate_size": 3072,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_position_embeddings": 512,
        "dropout": 0.1,
        "type_vocab_size": 2,
    }
    t.random.manual_seed(0)
    reference = bert.AttentionPattern(config)
    reference.eval()
    t.random.manual_seed(0)
    theirs = your_module(
        hidden_size=config["hidden_size"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
    )
    theirs.eval()
    input_activations = t.rand((2, 3, 768))
    allclose(
        theirs(input_activations),
        reference(input_activations),
        "bert",
    )
