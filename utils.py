from einops.einops import rearrange
import torch as t
import numpy as np
import time
import importlib


def tstat(name, tensor):
    print(
        name, "mean", "{0:.4g}".format(t.mean(tensor).cpu().item()), "var", "{0:.4g}".format(t.var(tensor).cpu().item())
    )


def chunk(list, chunk_size):
    return [list[i * chunk_size : (i + 1) * chunk_size] for i in range(len(list) // chunk_size)]


def itpeek(tensor: t.Tensor):
    contains_nan = t.any(t.isnan(tensor)).item()
    contains_inf = t.any(t.isinf(tensor)).item()
    string = f"SHAPE {tuple(tensor.shape)} MEAN: {'{0:.4g}'.format(t.mean(tensor.float()).cpu().item())} VAR: {'{0:.4g}'.format(t.var(tensor.float()).cpu().item())} {'CONTAINS_NAN! ' if contains_nan else ''}{'CONTAINS_INF! ' if contains_inf else ''}VALS [{' '.join(['{0:.4g}'.format(x) for x in t.flatten(tensor)[:10].cpu().tolist()])}...]"
    return string


def tpeek(name: str, tensor: t.Tensor, ret: bool = False):
    string = f"{name} {itpeek(tensor)}"
    if ret:
        return string
    print(string)


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print("took", self.interval)


def getprops(obj):
    props = set(dir(obj))
    strprops = set(dir(" "))
    justones = sorted(list(props - strprops))
    print("props: \n", "\n    ".join(justones))


def has_not_null(obj, prop):
    return hasattr(obj, prop) and (getattr(obj, prop) is not None)


def copy_state_identical(from_module, to_module):
    state_dict = from_module.state_dict()
    to_state_keys = set(to_module.state_dict().keys())
    from_state_keys = set(state_dict.keys())
    from_only = from_state_keys - to_state_keys
    to_only = to_state_keys - from_state_keys
    if len(to_only) > 0 or len(from_only) > 0:
        raise AssertionError(f"{from_only} only in from module, {to_only} only in to module".replace('"', ""))
    to_module.load_state_dict(from_state_keys)


def copy_weight_bias(mine, theirs, transpose=False):
    if transpose:
        mine.weight.copy_(rearrange(theirs.weight, "a b -> b a"))
    else:
        mine.weight.copy_(theirs.weight)

    theirs_has_bias = has_not_null(theirs, "bias")
    mine_has_bias = has_not_null(mine, "bias")
    if theirs_has_bias and not mine_has_bias:
        mine.bias = theirs.bias
        print("theirs has bias, mine doesn't")
    elif mine_has_bias and not theirs_has_bias:
        mine.bias.zero_()
        print("mine has bias, theirs doesnt")
    if mine_has_bias and theirs_has_bias:
        mine.bias.copy_(theirs.bias)


def import_object_from_qualified_name(qname: str):
    last_period = qname.rindex(".")
    module_name = qname[:last_period]
    object_name = qname[last_period + 1 :]
    module = importlib.import_module(module_name)
    out = getattr(module, object_name)
    assert out is not None
    return out
