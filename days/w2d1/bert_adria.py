import functools
from typing import Callable, List, Optional, Tuple, Union, Iterable, Dict

import days.w2d1.bert_tests as btests
import days.w2d1.bert_tao as bert
import haiku as hk
import jax.dlpack
import jax.nn
import jax.numpy as jnp
import torch
import torch.nn.functional as F
import torch.utils.dlpack
from jax import lax, jit
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map
from jaxtyping import JaxArray, patch_typeguard
from typeguard import typechecked
import transformers

patch_typeguard()


def _torch_to_jax(arg: torch.Tensor):
    if torch.is_tensor(arg):
        return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(arg))
    if isinstance(arg, Callable):
        return jax_wrap_torch(arg)
    return arg

def _jax_to_torch(arg):
    if isinstance(arg, jnp.ndarray):
        return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(arg))
    if isinstance(arg, Callable):
        return torch_wrap_jax(arg)
    return arg


def torch_wrap_jax(jax_fn):
    @functools.wraps(jax_fn)
    def torch_fn(*args, **kwargs):
        new_args = map(_torch_to_jax, args)
        new_kwargs = {k: _torch_to_jax(v) for (k, v) in kwargs.items()}

        out = jax_fn(*new_args, **new_kwargs)
        torch_out = tree_map(_jax_to_torch, out)
        return torch_out
    return torch_fn

def jax_wrap_torch(torch_fn):
    @functools.wraps(torch_fn)
    def jax_fn(*args, **kwargs):
        new_args = map(_jax_to_torch, args)
        new_kwargs = {k: _jax_to_torch(v) for (k, v) in kwargs.items()}

        out = torch_fn(*new_args, **new_kwargs)
        jax_out = tree_map(_torch_to_jax, out)
        return jax_out
    return jax_fn


@typechecked
def raw_attention_scores(token_activations: JaxArray["batch_size", "seq_length", "hidden_size"],
                         num_heads: int,
                         project_query: Callable[[JaxArray[..., "hidden_size"]], JaxArray[..., "num_heads*head_size"]],
                         project_key  : Callable[[JaxArray[..., "hidden_size"]], JaxArray[..., "num_heads*head_size"]],
                         ) -> JaxArray["batch_size", "num_heads", "key_token": "seq_length", "query_token": "seq_length"]:
    queries = project_query(token_activations)
    keys = project_key(token_activations)
    batch_size, seq_length, num_heads__head_size = queries.shape
    head_size = num_heads__head_size//num_heads

    queries = jnp.reshape(queries, (batch_size, seq_length, num_heads, head_size))
    keys = jnp.reshape(keys, (batch_size, seq_length, num_heads, head_size))

    QK_norm = jnp.einsum("bsnh,bznh->bnsz", keys, queries) * head_size**-.5
    return QK_norm

btests.test_attention_pattern_fn(torch_wrap_jax(raw_attention_scores))

@typechecked
def bert_attention(token_activations: JaxArray["batch_size", "seq_length", "hidden_size"],
                   num_heads: int,
                   attention_pattern: JaxArray["batch_size", "num_heads", "seq_length", "seq_length"],
                   project_value: Callable[[JaxArray[..., "hidden_size"]], JaxArray[..., "num_heads*head_size"]],
                   project_output: Callable[[JaxArray[..., "num_heads*head_size"]], JaxArray[..., "hidden_size"]],
                   ) -> JaxArray["batch_size", "seq_length", "hidden_size"]:
    values = project_value(token_activations)
    batch_size, seq_length, num_heads__head_size = values.shape
    head_size = num_heads__head_size//num_heads

    values = jnp.reshape(values, (batch_size, seq_length, num_heads, head_size))

    # Seems inefficient -- it's better if the last dimension is summed over.
    QKv = jnp.einsum("bnzs,bznh->bsnh", jax.nn.softmax(attention_pattern, -2), values)
    out = project_output(jnp.reshape(QKv, (batch_size, seq_length, num_heads__head_size)))
    return out

btests.test_attention_fn(torch_wrap_jax(bert_attention))


class TorchLikeLinear(hk.Linear):
    """Clone of hk.Linear but uses `weight` and `bias` instead of `w` and `b`
    for parameter names."""
    @typechecked
    def __call__(
        self,
        inputs: JaxArray["b": ..., "i_sz"],
        *,
        precision: Optional[lax.Precision] = None,
    ) -> JaxArray["b": ..., "o_sz"]:
        """Computes a linear transform of the input."""
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = self.input_size ** -0.5
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("weight", [output_size, input_size], dtype, init=w_init)

        out = jnp.dot(inputs, w.T, precision=precision)

        if self.with_bias:
            b = hk.get_parameter("bias", [self.output_size], dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        return out


class MultiHeadedSelfAttention(hk.Module):
    @typechecked
    def __init__(self, num_heads: int, hidden_size: int, name: Optional[str]=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        output_size = self.hidden_size
        self.query = TorchLikeLinear(output_size, name="project_query")
        self.key = TorchLikeLinear(output_size, name="project_key")
        self.value = TorchLikeLinear(output_size, name="project_value")
        self.output = TorchLikeLinear(self.hidden_size, name="project_output")


    @typechecked
    def __call__(self, x: JaxArray["batch_size", "seq_length", "hidden_size"]
                 ) -> JaxArray["batch_size", "seq_length", "hidden_size"]:
        attn = raw_attention_scores(x, self.num_heads, self.query, self.key)
        out = bert_attention(x, self.num_heads, attn, self.value, self.output)
        return out


@typechecked
def bert_mlp(token_activations: JaxArray["bdims": ..., "input_size"],
             linear_1: Callable[[JaxArray["bdims": ..., "input_size"]], JaxArray["bdims": ..., "intermediate_size"]],
             linear_2: Callable[[JaxArray["bdims": ..., "intermediate_size"]], JaxArray["bdims": ..., "input_size"]],
             approximate_gelu: bool=False,
             ) -> JaxArray["bdims": ..., "input_size"]:
    out = token_activations
    out = linear_1(out)
    out = jax.nn.gelu(out, approximate=approximate_gelu)
    out = linear_2(out)
    return out

btests.test_bert_mlp(torch_wrap_jax(bert_mlp))

class BertMLP(hk.Module):
    @typechecked
    def __init__(self, input_size: int, intermediate_size: int, approximate_gelu: bool=False,
                 name: Optional[str]=None):
        super().__init__(name=name)
        self.linear_1 = TorchLikeLinear(intermediate_size, name="linear_1")
        self.linear_2 = TorchLikeLinear(input_size, name="linear_2")
        self.approximate_gelu = approximate_gelu

    @typechecked
    def __call__(self, x: JaxArray["bdims": ..., "hidden_size"],
                 ) -> JaxArray["bdims": ..., "hidden_size"]:
        return bert_mlp(x, linear_1=self.linear_1, linear_2=self.linear_2,
                        approximate_gelu=self.approximate_gelu)



class LayerNorm(hk.Module):
    @typechecked
    def __init__(self, norm_size: int, epsilon: float=1e-5, name: Optional[str]=None):
        super().__init__(name=name)
        self.norm_size = norm_size
        self.epsilon = epsilon

    @typechecked
    def __call__(self, inputs: JaxArray["bdims": ..., "norm_size"],
                 ) -> JaxArray["bdims": ..., "norm_size"]:
        dtype = inputs.dtype
        weight = hk.get_parameter("weight", [self.norm_size], dtype, init=jnp.ones)
        bias = hk.get_parameter("bias", [self.norm_size], dtype, init=jnp.zeros)

        norm_inputs = (inputs - jnp.mean(inputs, axis=-1, keepdims=True)) * (self.epsilon + jnp.var(inputs, axis=-1, keepdims=True))**-.5
        return norm_inputs*weight + bias


def torch_like_lnorm(norm_size: int):
    lnorm = hk.without_apply_rng(hk.transform(lambda x: LayerNorm(norm_size)(x)))
    params = lnorm.init(jax.random.PRNGKey(0), jnp.ones([1, 1, norm_size], dtype=jnp.float32))
    return torch_wrap_jax(functools.partial(lnorm.apply, params))

btests.test_layer_norm(torch_like_lnorm)


class BertBlock(hk.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, dropout: float, approximate_gelu: bool=False, name: Optional[str]=None):
        super().__init__(name=name)
        self.mhsa = MultiHeadedSelfAttention(num_heads=num_heads, hidden_size=hidden_size, name="mhsa")
        self.mlp = BertMLP(input_size=hidden_size, intermediate_size=intermediate_size, approximate_gelu=approximate_gelu, name="mlp")
        self.ln1 = LayerNorm(hidden_size, name="ln1")
        self.ln2 = LayerNorm(hidden_size, name="ln2")
        self.dropout = dropout

    @typechecked
    def __call__(self, x: JaxArray["s": ..., "h"], training: bool) -> JaxArray["s": ..., "h"]:
        residual = x
        x = self.mhsa(x)
        x = self.ln1(x + residual)
        residual = x
        x = self.mlp(x)
        if training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = self.ln2(x + residual)
        return x

@typechecked
def copy_params(
        param_map: List[Tuple[str, str]],
        params: Dict[str, Dict[str, JaxArray]],
        prefix_params: Iterable[str],
        bert_sd: Dict[str, torch.Tensor],
        prefix_torch: Iterable[str],
        ) -> Dict[str, Dict[str, JaxArray]]:
    out = {}
    for jax_key, torch_key in param_map:
        jaxk = "/~/".join([*prefix_params, jax_key])
        out[jaxk] = {}
        for pname in params[jaxk].keys():
            tk = ".".join([*prefix_torch, torch_key, pname])
            tensor = bert_sd[tk].clone()
            assert tensor.shape == params[jaxk][pname].shape, f"{jaxk} {pname} {tensor.shape} {params[jaxk][pname].shape}"
            out[jaxk][pname] = _torch_to_jax(tensor)
    return out


copy_bert_block = functools.partial(
    copy_params,
    [('ln1', 'layer_norm'),
     ('mhsa/~/project_query', 'attention.pattern.project_query'),
     ('mhsa/~/project_key', 'attention.pattern.project_key'),
     ('mhsa/~/project_value', 'attention.project_value', ),
     ('mhsa/~/project_output', 'attention.project_out',),
     ('mlp/~/linear_1', 'residual.mlp1'),
     ('mlp/~/linear_2', 'residual.mlp2'),
     ('ln2', 'residual.layer_norm')
     ])


def test_bert_block():
    config = {
        "vocab_size": 28996,
        "intermediate_size": 3072,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_position_embeddings": 512,
        "dropout": 0.0,  # No dropout in eval mode
        "type_vocab_size": 2,
    }
    reference = bert.BertBlock(config)
    reference.eval()

    theirs = hk.transform(lambda *a, **kw: BertBlock(
        intermediate_size=config["intermediate_size"],
        hidden_size=config["hidden_size"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        approximate_gelu=False,
        name="bert_block"
    )(*a, **kw))
    params = theirs.init(jax.random.PRNGKey(123), jnp.ones((1, 1, config["hidden_size"])), training=False)
    params = copy_bert_block(params, ["bert_block"], reference.state_dict(), [])
    fn = torch_wrap_jax(functools.partial(theirs.apply, params, jax.random.PRNGKey(123)))

    input_activations = torch.rand((2, 3, 768))
    btests.allclose(
        fn(input_activations, training=False),
        reference(input_activations),
        "bert",
    )
test_bert_block()


class Embedding(hk.Module):
    @typechecked
    def __init__(self, vocab_size: int, embed_size: int, name: Optional[str]=None):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.embed_size = embed_size

    @typechecked
    def __call__(self, inputs: Union[slice, JaxArray["bs": ..., "seq_length", jnp.int64]],
                 dtype: jnp.dtype=jnp.float32,
                 ) -> JaxArray["bs": ..., "seq_length", "embed_size"]:
        embedding = hk.get_parameter("weight", [self.vocab_size, self.embed_size], dtype=dtype,
                                     init=hk.initializers.RandomNormal(stddev=self.embed_size**-.5))
        return embedding[inputs, :]


class BertEmbedding(hk.Module):
    @typechecked
    def __init__(self, vocab_size: int, hidden_size: int,
                 max_position_embeddings: int, type_vocab_size: int, dropout: float,
                 name: Optional[str]=None):
        super().__init__(name=name)
        self.vocab_embed = Embedding(vocab_size, hidden_size, name="token_embedding")
        self.position_embed = Embedding(max_position_embeddings, hidden_size, name="position_embedding")
        self.type_embed = Embedding(type_vocab_size, hidden_size, name="token_type_embedding")
        self.ln = LayerNorm(hidden_size, name="layer_norm")
        self.dropout = dropout

    @typechecked
    def __call__(self, input_ids: JaxArray["bs": ..., "seq_length", jnp.int64],
                 token_type_ids: Union[slice, JaxArray["bs": ..., "seq_length", jnp.int64]],
                 training: bool,
                 ) -> JaxArray["bs": ..., "seq_length", "hidden_size"]:
        seq_length = input_ids.shape[-1]
        embeds = (self.vocab_embed(input_ids)
                  + self.position_embed(slice(seq_length))
                  + self.type_embed(token_type_ids))
        out = self.ln(embeds)
        if training:
            out = hk.dropout(hk.next_rng_key(), self.dropout, out)
        return out


def embedding_copy_params(params, reference):
    sd = reference.state_dict()
    for module_key in params.keys():
        for param_key in params[module_key].keys():
            sd_key = ".".join([*module_key.split("/~/")[1:], param_key])
            assert sd[sd_key].shape == params[module_key][param_key].shape
            params[module_key][param_key] = _torch_to_jax( sd[sd_key].clone())
    return params


def test_bert_embedding():
    config = {
        "vocab_size": 28996,
        "hidden_size": 768,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "dropout": 0.1,
    }
    input_ids = torch.randint(0, 2900, (2, 3))
    tt_ids = torch.randint(0, 2, (2, 3))
    torch.random.manual_seed(0)
    reference = bert.BertEmbedding(config)
    reference.eval()

    theirs = hk.transform(lambda *a, **kw: BertEmbedding(**config)(*a, **kw))
    params = theirs.init(jax.random.PRNGKey(123), _torch_to_jax(input_ids), _torch_to_jax(tt_ids), training=False)
    params = embedding_copy_params(params, reference)
    fn = torch_wrap_jax(functools.partial(theirs.apply, params, jax.random.PRNGKey(123)))

    btests.allclose(
        fn(input_ids=input_ids, token_type_ids=tt_ids, training=False),
        reference(input_ids=input_ids, token_type_ids=tt_ids),
        "bert embedding",
    )

test_bert_embedding()


class Bert(hk.Module):
    @typechecked
    def __init__(self, vocab_size: int, hidden_size: int,
                 max_position_embeddings: int, type_vocab_size: int,
                 dropout: float, intermediate_size: int, num_heads: int,
                 num_layers: int,
                 approximate_gelu: bool=False,
                 name: Optional[str]=None):
        super().__init__(name=name)
        self.embedding = BertEmbedding(
            vocab_size=vocab_size, hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size, dropout=dropout, name="embedding")

        self.layers = tuple(
            BertBlock(hidden_size=hidden_size,
                      intermediate_size=intermediate_size, num_heads=num_heads,
                      dropout=dropout, name=f"block_{i}",
                      approximate_gelu=approximate_gelu)
            for i in range(num_layers)
        )
        self.approximate_gelu = approximate_gelu

        self.mlp = TorchLikeLinear(hidden_size, name="mlp")
        self.layer_norm = LayerNorm(hidden_size, name="layer_norm")
        self.unembedding = TorchLikeLinear(vocab_size, name="unembedding")


    @typechecked
    def __call__(self, input_ids: JaxArray["bs": ..., "seq_length", jnp.int64],
                 training: bool=True,
                 ) -> JaxArray["bs": ..., "seq_length", "vocab_size"]:
        out = self.embedding(input_ids, token_type_ids=slice(1), training=training)
        for layer in self.layers:
            out = layer(out, training=training)
        out = self.mlp(out)
        out = jax.nn.gelu(out, approximate=self.approximate_gelu)
        out = self.layer_norm(out)
        out = self.unembedding(out)
        return out


my_bert = hk.transform(lambda input_ids, training=True: Bert(
    vocab_size=28996, hidden_size=768, max_position_embeddings=512,
    type_vocab_size=2, dropout=0.1, intermediate_size=3072,
    num_heads=12, num_layers=12, approximate_gelu=False,
)(input_ids, training))
params = my_bert.init(jax.random.PRNGKey(123), jnp.zeros((1, 512), dtype=jnp.int64), training=False)
# pretrained_bert = btests.get_pretrained_bert()

copy_embedding = functools.partial(
    copy_params,
    [("token_embedding",)*2,
     ("position_embedding",)*2,
     ("token_type_embedding",)*2,
     ("layer_norm",)*2,
     ])

copy_lm_head = functools.partial(
    copy_params,
    [("mlp",)*2,
     ("unembedding",)*2,
     ("layer_norm",)*2,
     ])

@typechecked
def copy_bert(
        params: Dict[str, Dict[str, JaxArray]],
        prefix_params: Iterable[str],
        bert_sd: Dict[str, torch.Tensor],
        prefix_torch: Iterable[str],
        ) -> Dict[str, Dict[str, JaxArray]]:

    out1 = copy_embedding(params, [*prefix_params, "embedding"], bert_sd, [*prefix_torch, "embedding"])
    out2 = copy_lm_head(params, prefix_params, bert_sd, [*prefix_torch, "lm_head"])
    out = {**out1, **out2}
    for i in range(100):
        try:
            outn = copy_bert_block(params, [*prefix_params, f"block_{i}"], bert_sd, ["transformer", str(i)])
        except KeyError:
            break
        out = {**out, **outn}
    return out


params = copy_bert(params, ["bert"], pretrained_bert.state_dict(), [])
your_bert = functools.partial(my_bert.apply, params, jax.random.PRNGKey(123), training=False)
class your_bert_module:
    def eval():
        return torch_wrap_jax(your_bert)

btests.test_same_output(your_bert_module, pretrained_bert, tol=1e-4)
