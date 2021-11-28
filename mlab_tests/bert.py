import torch as t
import transformers
import days.bert as bert
from test_all import allclose
import torch.nn as nn
import torch.nn.functional as F


def test_back_forth_my_name_is_bert(string):
    assert string == "[CLS] colleges 天 largest happened smile donation 夫 [SEP]"


def test_attention_layer(fn):
    reference = bert.multi_head_self_attention
    hidden_size = 768
    token_activations = t.empty(2, 3, hidden_size).uniform_(-1, 1)
    num_heads = 12
    project_query = nn.Linear(hidden_size, hidden_size)
    project_key = nn.Linear(hidden_size, hidden_size)
    project_value = nn.Linear(hidden_size, hidden_size)
    project_output = nn.Linear(hidden_size, hidden_size)
    dropout = t.nn.Dropout(0.1)
    dropout.eval()
    allclose(
        fn(
            token_activations=token_activations,
            num_heads=num_heads,
            project_query=project_query,
            project_key=project_key,
            project_value=project_value,
            project_output=project_output,
            dropout=dropout,
        ),
        reference(
            token_activations=token_activations,
            num_heads=num_heads,
            project_query=project_query,
            project_key=project_key,
            project_value=project_value,
            project_output=project_output,
            dropout=dropout,
        ),
        "attention",
    )


def test_attention_pattern_raw(fn):
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
        reference(token_activations=token_activations, linear_1=mlp_1, linear_2=mlp_2),
        "bert mlp",
    )


# TODO write this
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
    reference = bert.Bert(config)
    reference.eval()
    t.random.manual_seed(0)
    theirs = your_module(**config)
    theirs.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    input_ids = tokenizer("hello there", return_tensors="pt")["input_ids"]
    allclose(
        theirs(input_ids=input_ids),
        reference(input_ids=input_ids),
        "bert",
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


# TODO write this
def test_bert(bert_module):
    raise NotImplementedError()
