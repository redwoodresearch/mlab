import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import days.modules as modules
from transformers.utils.dummy_sentencepiece_objects import PegasusTokenizer
import days.bert as bert
import days.gpt2 as gpt2
import days.resnet as resnet
import pytest
import transformers
from utils import tpeek, tstat


import sys, traceback
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter


def setmyexcepthook():

    lexer = get_lexer_by_name("pytb" if sys.version_info.major < 3 else "py3tb")
    formatter = TerminalFormatter()

    def myexcepthook(type, value, tb):
        tbtext = "".join(traceback.format_exception(type, value, tb))
        tbtext = tbtext.replace("/home/tao/.asdf/installs/python/3.9.6/lib/python3.9/site-packages/", "[PKG]")
        sys.stderr.write(highlight(tbtext, lexer, formatter))

    sys.excepthook = myexcepthook


setmyexcepthook()


def init_both(my_class, their_class, *args, **kwargs):
    t.random.manual_seed(0)
    my_class = my_class(*args, **kwargs)
    t.random.manual_seed(0)
    their_class = their_class(*args, **kwargs)
    return my_class, their_class


def allclose(my_out, their_out, name, tol=1e-8):

    if not t.allclose(my_out, their_out, rtol=tol * 1000, atol=tol):
        errstring = f'error in {name}\n{tpeek("", my_out, ret=True)} \n!=\n{tpeek("", their_out, ret=True)}'
        raise AssertionError(errstring)
    else:
        print("they match!")
        print(my_out)


def test_relu():
    input = t.FloatTensor(435, 234).uniform_(-10, 10)
    my_out = modules.relu(input)
    their_out = F.relu(input)
    allclose(my_out, their_out, "relu")


def test_gelu():
    hf_gelu = transformers.activations.gelu_new
    input = t.FloatTensor(435, 234).uniform_(-10, 10)
    my_out = modules.gelu(input)
    their_out = hf_gelu(input)
    allclose(my_out, their_out, "gelu")


def test_softmax():
    input = t.FloatTensor(435, 234).uniform_(-10, 10)
    my_out = modules.softmax(input, dim=1)
    their_out = F.softmax(input, dim=1)
    allclose(my_out, their_out, "softmax")


def test_normalize():
    input = t.FloatTensor(435, 234).uniform_(-10, 10)
    my_out = modules.normalize(input, dim=1)
    their_out = F.normalize(input, dim=1)
    allclose(my_out, their_out, "normalize")


def test_layer_norm():
    input = t.FloatTensor(435, 234).uniform_(-10, 10)
    t.random.manual_seed(0)
    my_layer_norm = bert.LayerNorm((234,))
    t.random.manual_seed(0)
    their_layer_norm = nn.LayerNorm(234)

    my_out = my_layer_norm(input)
    their_out = their_layer_norm(input)
    allclose(my_out, their_out, "layer_norm")


def test_linear():
    input = t.FloatTensor(435, 435, 234).uniform_(-10, 10)
    my_linear, their_linear = init_both(bert.Linear, nn.Linear, 234, 111)
    tstat("my weight", my_linear.weight)
    tstat("their weight", their_linear.weight)

    my_out = my_linear(input)
    their_out = their_linear(input)
    allclose(my_out, their_out, "linear")


def test_dropout():
    # use a large input so its aggregates will be consistent
    input = t.empty(10000, 1000).uniform_(-1, 1)
    my_dropout = modules.Dropout(0.1)
    their_dropout = nn.Dropout(0.1)
    t.random.manual_seed(0)
    their_output = their_dropout(input)
    t.random.manual_seed(0)
    my_output = my_dropout(input)

    allclose(my_output.mean(), their_output.mean(), "dropout mean", 0.001)
    allclose(my_output.var(), their_output.var(), "dropout var", 0.001)
    my_fraction_zero = t.mean(my_output == 0)
    allclose(my_fraction_zero, t.Tensor(0.1), "dropout frac zero")


def test_embedding():
    embed_input = t.LongTensor([[1, 2, 3], [7, 8, 9]])
    my_embedding, their_embedding = init_both(bert.Embedding, nn.Embedding, 234, 111)
    my_output = my_embedding(embed_input)
    their_output = their_embedding(embed_input)
    allclose(my_output, their_output, "embedding")


def test_bert_attention():
    their_bert = transformers.AutoModelForMaskedLM.from_pretrained("bert-base-cased")
    their_layer = their_bert.bert.encoder.layer[0].attention
    my_layer = bert.SelfAttentionLayer(their_bert.config)
    bert.copy_bert_attention(my_layer, their_layer)
    their_layer.eval()
    my_layer.eval()
    input_encoding = t.FloatTensor(2, 2, 768).uniform_(-0.2, 0.2)
    my_output = my_layer(input_encoding)
    their_output = their_layer(input_encoding)[0]
    tpeek("my output", my_output)
    tpeek("their output", their_output)
    allclose(my_output, their_output, "bert attention", tol=0.001)


def test_bert_layer():
    their_bert = transformers.AutoModelForMaskedLM.from_pretrained("bert-base-cased")
    their_layer = their_bert.bert.encoder.layer[0]
    my_layer = bert.BertLayer(their_bert.config)
    bert.copy_bert_layer(my_layer, their_layer)
    their_layer.eval()
    my_layer.eval()
    input_encoding = t.FloatTensor(2, 2, 768).uniform_(-0.2, 0.2)
    my_output = my_layer(input_encoding)
    their_output = their_layer(input_encoding)[0]
    tpeek("my output", my_output)
    tpeek("their output", their_output)
    allclose(my_output, their_output, "bert layer", tol=0.001)


def test_bert():
    my_bert, their_bert = bert.my_bert_from_hf_weights()
    my_bert.eval()
    their_bert.eval()

    inputs = {
        "token_type_ids": t.LongTensor([[0, 0, 0, 0], [0, 0, 0, 0]]),
        "input_ids": t.LongTensor([[0, 1, 2, 3], [5, 6, 7, 8]]),
    }
    my_output = my_bert(**inputs)
    their_output = their_bert(**inputs)
    print(their_bert.bert(**inputs))
    my_logits = my_output.logits
    their_logits = their_output.logits
    tpeek("my logits", my_logits)
    tpeek("their logits", their_logits)
    my_encodings = my_output.encodings
    their_encodings = their_bert.bert(**inputs).last_hidden_state
    tpeek("my encodings", my_encodings)
    tpeek("their encodings", their_encodings)
    allclose(my_logits, their_logits, "bert", tol=0.1)


def test_gpt2_layer():
    their_lm_model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    my_layer = gpt2.GPT2Layer(their_lm_model.config)
    their_lm_model.eval()
    my_layer.eval()
    their_layer = their_lm_model.transformer.h[0]
    gpt2.copy_gpt2_layer_weights(my_layer, their_layer)

    example_encoding = t.FloatTensor(2, 2, 768).uniform_(-0.2, 0.2)
    my_output = my_layer(example_encoding)
    their_output = their_layer(example_encoding)[0]
    tpeek("my layer", my_output)
    tpeek("their layer", their_output)
    allclose(my_output, their_output, "gpt2 layer", tol=0.01)


def test_gpt2_attention():
    their_lm_model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    my_attn = gpt2.GPT2Attention(their_lm_model.config)
    their_lm_model.eval()
    my_attn.eval()
    their_attention = their_lm_model.transformer.h[0].attn
    gpt2.copy_gpt2_attention_weights(my_attn, their_attention)

    example_encoding = t.FloatTensor(2, 2, 768).uniform_(-0.2, 0.2)
    my_output = my_attn(example_encoding)
    their_output = their_attention(example_encoding)[0]
    tpeek("my attention", my_output)
    tpeek("their attention", their_output)
    allclose(my_output, their_output, "gpt2 attetnion", tol=0.001)


def test_gpt2():
    my_gpt2, their_gpt2 = gpt2.my_gpt_from_hf_weights()
    my_gpt2.eval()
    their_gpt2.eval()

    inputs = {
        "input_ids": t.LongTensor([[0, 1], [2, 3]]),
    }
    my_output = my_gpt2(**inputs).logits
    their_output = their_gpt2(**inputs).logits
    tpeek("my layer", my_output)
    print(their_output)
    tpeek("their layer", their_output)
    allclose(my_output, their_output, "gpt2 logits", tol=0.1)


def test_gpt2_cache_is_correct():
    short_input_ids = t.LongTensor([[0, 1, 2]])
    long_input_ids = t.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    other_input_ids = t.LongTensor([[88, 323, 134]])

    t.random.manual_seed(0)
    model_no_cache = gpt2.GPT2({"use_cache": False})
    model_no_cache.eval()
    short_no_cache = model_no_cache(short_input_ids).logits
    short_no_cache_2 = model_no_cache(short_input_ids).logits
    long_no_cache = model_no_cache(long_input_ids).logits
    t.random.manual_seed(0)

    model = model_no_cache
    model.config["use_cache"] = True
    model.eval()
    print("short cache")
    short_cache = model(short_input_ids).logits
    long_cache = model(long_input_ids).logits

    other_no_cache = model_no_cache(other_input_ids).logits
    other_cache = model(other_input_ids).logits

    allclose(short_no_cache, short_no_cache_2, "determinism")
    allclose(short_no_cache, short_cache, "cache short")
    allclose(long_no_cache, long_cache, "cache long")
    allclose(other_no_cache, other_cache, "cache other")


def test_resnet():
    resnet.resnet34_with_pretrained_weights()


if __name__ == "__main__":
    # test_gpt2_attention()
    # test_gpt2_layer()
    test_gpt2()
    test_gpt2_cache_is_correct()

    # test_bert_attention()
    # test_bert_layer()
    test_bert()

    test_dropout()
    test_relu()
    test_gelu()
    test_softmax()
    test_normalize()
    test_layer_norm()
    test_embedding()
    test_linear()

    # test_resnet()
