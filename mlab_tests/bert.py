import torch as t
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
def test_bert_block(fn):
    raise NotImplementedError()
    reference = bert.d
    hidden_size = 768
    intermediate_size = 4 * hidden_size

    token_activations = t.empty(2, 3, hidden_size).uniform_(-1, 1)
    attention = bert.SelfAttentionLayer(config)
    mlp_2 = nn.Linear(intermediate_size, hidden_size)
    dropout = t.nn.Dropout(0.1)
    dropout.eval()
    allclose(
        fn(token_activations=token_activations, linear_1=mlp_1, linear_2=mlp_2),
        reference(token_activations=token_activations, linear_1=mlp_1, linear_2=mlp_2),
        "bert mlp",
    )


# TODO write this
def test_bert(bert_module):
    raise NotImplementedError()
