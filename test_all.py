import sys, traceback
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter
import re

lexer = get_lexer_by_name("pytb" if sys.version_info.major < 3 else "py3tb")
formatter = TerminalFormatter()


def format_traceback(tbtext):
    # replace long python internal package path
    tbtext = re.sub(
        r"/home/[a-zA-Z/.]+/.asdf/installs/python/[0-9.]+/lib/python[0-9.]+/site-packages/",
        "<PKG>/",
        tbtext,
    )

    # remove gin stuff
    tbtext = re.sub(r".*File .*/gin/.+line \d+, in .*\n[^\n]*\n", "", tbtext)
    tbtext = re.sub(r".*In call to configurable.*\n[^\n]*\n", "", tbtext)

    return highlight(tbtext, lexer, formatter)


def setmyexcepthook():
    def myexcepthook(type, value, tb):
        print("type")
        tbtext = "".join(traceback.format_exception(type, value, tb))

        sys.stderr.write(format_traceback(tbtext))

    sys.excepthook = myexcepthook


setmyexcepthook()

import time
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import days.modules as modules
from transformers.utils.dummy_sentencepiece_objects import PegasusTokenizer
import days.bert as bert
import days.gpt2 as gpt2
import days.old_resnet as old_resnet
import pytest
import transformers
from days.utils import tpeek
import days.w2d5.dataparallel as dp


def init_both(my_class, their_class, *args, **kwargs):
    t.random.manual_seed(0)
    my_class = my_class(*args, **kwargs)
    t.random.manual_seed(0)
    their_class = their_class(*args, **kwargs)
    return my_class, their_class


def allclose(my_out, their_out, name, tol=1e-5):

    if not t.allclose(my_out, their_out, rtol=1e-4, atol=tol):
        errstring = f'error in {name}\n{tpeek("", my_out, ret=True)} \n!=\n{tpeek("", their_out, ret=True)}'
        raise AssertionError(errstring)
    else:
        tpeek(f"{name} MATCH!!!!!!!!\n", my_out)


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
    tpeek("my weight", my_linear.weight)
    tpeek("their weight", their_linear.weight)

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
    my_fraction_zero = t.mean((my_output == 0).float())
    print(my_fraction_zero)
    allclose(my_fraction_zero, t.tensor(0.1), "dropout frac zero", tol=0.001)


def test_log_softmax():
    input = t.FloatTensor(435, 234).uniform_(-10, 10)
    my_out = modules.log_softmax(input, dim=1)
    their_out = F.log_softmax(input, dim=1)
    allclose(my_out, their_out, "log_softmax")


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
    my_layer = bert.BertBlock(their_bert.config)
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

    inputs = {
        "input_ids": t.LongTensor(
            my_gpt2.tokenizer(["I'm Alex Rider, i'm a writer"])["input_ids"]
        ),
    }
    my_output = my_gpt2(**inputs).logits
    their_output = their_gpt2(**inputs).logits
    tpeek("my layer", my_output)
    tpeek("their layer", their_output)
    allclose(my_output, their_output, "gpt2 logits", tol=0.01)


def test_gpt2_cache_is_correct():
    short_input_ids = t.arange(0, 398).unsqueeze(0)
    long_input_ids = t.arange(0, 400).unsqueeze(0)
    other_input_ids = t.LongTensor([[88, 323, 134]])

    t.random.manual_seed(0)
    model_no_cache = gpt2.GPT2({"use_cache": False})
    model_no_cache.eval()
    short_no_cache = model_no_cache(short_input_ids).logits
    tstart = time.time()
    long_no_cache = model_no_cache(long_input_ids).logits
    print("no cache took", time.time() - tstart)
    t.random.manual_seed(0)

    model = model_no_cache
    model.config["use_cache"] = True
    print("short cache")
    short_cache = model(short_input_ids).logits
    print("long cache")
    tstart = time.time()
    long_cache = model(long_input_ids).logits
    print("with cache took", time.time() - tstart)
    other_no_cache = model_no_cache(other_input_ids).logits
    other_cache = model(other_input_ids).logits

    allclose(short_no_cache, short_cache, "cache short")
    allclose(other_no_cache, other_cache, "cache other")
    allclose(long_no_cache, long_cache, "cache long", tol=0.01)


def test_gpt2_generation():
    my_gpt2, their_gpt2 = gpt2.my_gpt_from_hf_weights()
    my_gpt2.config["use_cache"] = False
    prompt = "I'm Alex Rider,"
    print("generating")
    their_generated_text = my_gpt2.tokenizer.decode(
        their_gpt2.generate(
            input_ids=my_gpt2.tokenizer([prompt], return_tensors="pt")["input_ids"],
            max_length=10,
        )[0]
        .cpu()
        .tolist()
    )
    print("their generated text", their_generated_text)
    generated_text = my_gpt2.generate(
        prompt, max_length=10, freq_penalty=1000, temperature=1
    )
    print("generated text", generated_text)


def test_gpt2_generation_beam():
    my_gpt2, their_gpt2 = gpt2.my_gpt_from_hf_weights()
    my_gpt2.config["use_cache"] = False
    prompt = "I'm Alex Rider,"
    print("generating")
    their_generated_text = my_gpt2.tokenizer.decode(
        their_gpt2.generate(
            input_ids=my_gpt2.tokenizer([prompt], return_tensors="pt")["input_ids"],
            max_length=10,
        )[0]
        .cpu()
        .tolist()
    )
    print("their generated text", their_generated_text)
    generated_text = my_gpt2.generate_beam_search(
        prompt, beam_width=3, max_length=10, freq_penalty=1000
    )
    generated_text_2 = my_gpt2.generate_beam_search(
        prompt, beam_width=3, max_length=10, freq_penalty=1000
    )
    print("generated text", generated_text)
    assert generated_text_2 == generated_text


def test_gpt2_specific_prob():
    my_gpt2, their_gpt2 = gpt2.my_gpt_from_hf_weights()
    my_gpt2.config["use_cache"] = False
    prompt = "I just ate my favorite food."
    completions = [
        " It was so good!",
        " I loved it!",
        " It was the best food ever!",
        " I threw up afterward.",
    ]
    # prompt = "I'm Alex Rider,"
    # completions = [" and I'm an MI6 agent", "and I'm a secret agent", "and I'm a writer", "gnwlkno63", "enxoilke"]
    print("generating")
    probs = my_gpt2.specific_completion_probs(prompt, completions)
    print("probs", {x: y for x, y in zip(completions, probs)})
    prompt = "I just ate some awful food."
    completions = [
        " It was so good!",
        " I loved it!",
        " It was the best food ever!",
        " I threw up afterward.",
    ]
    probs = my_gpt2.specific_completion_probs(prompt, completions)
    print("probs", {x: y for x, y in zip(completions, probs)})


def test_dp():
    dp.create_processes()


def test_resnet():
    old_resnet.resnet34_with_pretrained_weights()


if __name__ == "__main__":
    test_dp()
    test_gpt2_generation_beam()
    test_bert()
    test_gpt2_generation()
    raise AssertionError("hi")
    test_gpt2_cache_is_correct()
    test_gpt2_specific_prob()
    # test_gpt2_attention()
    # test_gpt2_layer()
    test_gpt2()

    # test_bert_attention()
    # test_bert_layer()

    test_relu()
    test_gelu()
    test_log_softmax()
    test_softmax()
    test_normalize()
    test_layer_norm()
    test_embedding()
    test_linear()
    test_dropout()

    # test_resnet()
