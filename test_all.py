import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules import activation
from transformers.utils.dummy_sentencepiece_objects import PegasusTokenizer
import days.bert as bert
import pytest
import transformers
from utils import tpeek, tstat


def setmyexcepthook():
    import sys, traceback
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import TerminalFormatter

    lexer = get_lexer_by_name("pytb" if sys.version_info.major < 3 else "py3tb")
    formatter = TerminalFormatter()

    def myexcepthook(type, value, tb):
        tbtext = "".join(traceback.format_exception(type, value, tb))
        tbtext = tbtext.replace("/home/tao/.asdf/installs/python/3.9.6/lib/python3.9/site-packages/", "[PKG]")
        sys.stderr.write(highlight(tbtext, lexer, formatter))

    sys.excepthook = myexcepthook


setmyexcepthook()

MAX_TOLERANCE = 5e-3
AVG_TOLERANCE = 1e-4


def init_both(my_class, their_class, *args, **kwargs):
    t.random.manual_seed(0)
    my_class = my_class(*args, **kwargs)
    t.random.manual_seed(0)
    their_class = their_class(*args, **kwargs)
    return my_class, their_class


def testy(my_out, their_out, name):
    maxdif = t.max(t.abs(my_out - their_out))
    avgdif = t.mean(my_out - their_out)
    if maxdif > MAX_TOLERANCE or avgdif > AVG_TOLERANCE:
        tpeek("mine", my_out)
        tpeek("theirs", their_out)
        raise AssertionError(f"{name} avgdif {float(avgdif)} maxdif {float(maxdif)}")
    print(f"{name} avgdif {float(avgdif)} maxdif {float(maxdif)}")


def test_relu():
    input = t.FloatTensor(435, 234).uniform_(-10, 10)
    my_out = bert.relu(input)
    their_out = F.relu(input)
    testy(my_out, their_out, "relu")


def test_gelu():
    input = t.FloatTensor(435, 234).uniform_(-10, 10)
    my_out = bert.gelu(input)
    their_out = F.gelu(input)
    testy(my_out, their_out, "gelu")


def test_softmax():
    input = t.FloatTensor(435, 234).uniform_(-10, 10)
    my_out = bert.softmax(input, dim=1)
    their_out = F.softmax(input, dim=1)
    testy(my_out, their_out, "softmax")


def test_normalize():
    input = t.FloatTensor(435, 234).uniform_(-10, 10)
    my_out = bert.normalize(input, dim=1)
    their_out = F.normalize(input, dim=1)
    testy(my_out, their_out, "normalize")


def test_layer_norm():
    input = t.FloatTensor(435, 234).uniform_(-10, 10)
    t.random.manual_seed(0)
    my_layer_norm = bert.LayerNorm((234,))
    t.random.manual_seed(0)
    their_layer_norm = nn.LayerNorm(234)

    my_out = my_layer_norm(input)
    their_out = their_layer_norm(input)
    testy(my_out, their_out, "layer_norm")


def test_linear():
    input = t.FloatTensor(435, 435, 234).uniform_(-10, 10)
    my_linear, their_linear = init_both(bert.Linear, nn.Linear, 234, 111)
    tstat("my weight", my_linear.weight)
    tstat("their weight", their_linear.weight)

    my_out = my_linear(input)
    their_out = their_linear(input)
    testy(my_out, their_out, "linear")


def test_embedding():
    unembed_input = t.FloatTensor(643, 23, 111).uniform_(-1, 1)
    embed_input = t.LongTensor([[1, 2, 3], [7, 8, 9]])
    my_embedding, their_embedding = init_both(bert.Embedding, nn.Embedding, 234, 111)
    tstat("my embedding", my_embedding.weight)
    tstat("their embedding", their_embedding.weight)

    my_out = my_embedding.embed(embed_input)
    their_out = their_embedding(embed_input)
    testy(my_out, their_out, "embedding")

    my_out = my_embedding.unembed(unembed_input)


def test_self_attention_fundamentals():
    width = 16
    activations = t.FloatTensor(10, 10, width).normal_(0, 1)
    # activations = t.FloatTensor(10, 10, width).fill_(1)
    q = t.eye(width)
    k = t.eye(width)
    v = t.eye(width)
    tpeek("activations", activations)
    output = bert.multi_head_self_attention(
        activations, attention_masks=None, num_heads=4, project_query=q, project_key=k, project_value=v
    )
    print("shape", output.shape)
    tpeek("output", output)
    tstat("output", output)


def test_bert():
    my_bert, their_bert = bert.bert_from_pytorch_save()
    their_bert: transformers.models.bert.modeling_bert.BertModel
    my_bert.eval()
    their_bert.eval()

    inputs = {"token_type_ids": t.LongTensor([[0, 0, 0, 1]]), "token_ids": t.LongTensor([[0, 1, 2, 3]])}

    my_embedded = my_bert.embedding.embed(**inputs)
    their_embedded = their_bert.embeddings(input_ids=inputs["token_ids"], token_type_ids=inputs["token_type_ids"])
    # testy(my_embedded, their_embedded, "embeds")

    embedding_inputs = my_embedded

    my_encoded = my_bert.transformer[0].attention.layer_norm(embedding_inputs)
    their_encoded = their_bert.encoder.layer[0].attention.output.LayerNorm(embedding_inputs)
    tpeek("my layer norm", my_encoded)
    tpeek("their layer norm", their_encoded)
    print()

    my_encoded = my_bert.transformer[0].attention.attention(embedding_inputs)
    their_encoded = their_bert.encoder.layer[0].attention.self(embedding_inputs)[0]
    tpeek("my pure attention", my_encoded)
    tpeek("their pure attention", their_encoded)
    print()

    # my_encoded = my_bert.transformer[0].attention(embedding_inputs)
    # their_encoded = their_bert.encoder.layer[0].attention(embedding_inputs)[0]
    # tpeek("my attention", my_encoded)
    # tpeek("their attention", their_encoded)
    # print()

    # my_encoded = my_bert.transformer[0](embedding_inputs)
    # their_encoded = their_bert.encoder.layer[0](
    #     embedding_inputs[:, :, :100],
    # )[0]
    # tpeek("my encoded", my_encoded)
    # tpeek("their encoded", their_encoded)
    # print()

    my_encoded = my_bert.transformer(embedding_inputs)
    their_encoded = their_bert.encoder(embedding_inputs).last_hidden_state
    tpeek("my encoded", my_encoded)
    tpeek("their encoded", their_encoded)
    print()


if __name__ == "__main__":
    # test_relu()
    # test_gelu()
    # test_softmax()
    # test_normalize()
    # test_layer_norm()
    # test_embedding()
    test_bert()

    # test_self_attention_fundamentals() # this looks okay?
    # test_linear()  # idk why this isn't producing same result
