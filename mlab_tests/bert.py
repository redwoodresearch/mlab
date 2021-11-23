import torch as t
import days.bert as bert
from test_all import allclose


def test_attention_layer(fn):
    reference = bert.multi_head_self_attention
    hidden_size = 768
    token_activations = t.empty(2, 3, hidden_size).uniform_(-1, 1)
    num_heads = 12
    project_query = t.empty(hidden_size, hidden_size).uniform_(-0.02, 0.02)
    project_key = t.empty(hidden_size, hidden_size).uniform_(-0.02, 0.02)
    project_value = t.empty(hidden_size, hidden_size).uniform_(-0.02, 0.02)
    project_output = t.empty(hidden_size, hidden_size).uniform_(-0.02, 0.02)
    dropout = t.nn.Dropout(0.1)
    dropout.eval()
    allclose(
        fn(token_activations, num_heads, project_query, project_key, project_value, project_output, dropout),
        reference(token_activations, num_heads, project_query, project_key, project_value, project_output, dropout),
    )


def test_attention_pattern_raw(fn):
    reference = bert.raw_attention_pattern
    hidden_size = 768
    token_activations = t.empty(2, 3, hidden_size).uniform_(-1, 1)
    num_heads = 12
    project_query = t.empty(hidden_size, hidden_size).uniform_(-0.02, 0.02)
    project_key = t.empty(hidden_size, hidden_size).uniform_(-0.02, 0.02)
    project_value = t.empty(hidden_size, hidden_size).uniform_(-0.02, 0.02)
    allclose(
        fn(token_activations, num_heads, project_query, project_key, project_value),
        reference(token_activations, num_heads, project_query, project_key, project_value),
    )


def test_mlp_layer(fn):
    reference = bert.multi_head_self_attention
    hidden_size = 768
    token_activations = t.empty(2, 3, hidden_size).uniform_(-1, 1)
    num_heads = 12
    project_query = t.empty(hidden_size, hidden_size).uniform_(-0.02, 0.02)
    project_key = t.empty(hidden_size, hidden_size).uniform_(-0.02, 0.02)
    project_value = t.empty(hidden_size, hidden_size).uniform_(-0.02, 0.02)
    project_output = t.empty(hidden_size, hidden_size).uniform_(-0.02, 0.02)
    dropout = t.nn.Dropout(0.1)
    dropout.eval()
    allclose(
        fn(token_activations, num_heads, project_query, project_key, project_value, project_output, dropout),
        reference(token_activations, num_heads, project_query, project_key, project_value, project_output, dropout),
    )
