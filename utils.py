from einops.einops import rearrange
import torch as t
import numpy as np
import time


def tstat(name, tensor):
    print(
        name, "mean", "{0:.4g}".format(t.mean(tensor).cpu().item()), "var", "{0:.4g}".format(t.var(tensor).cpu().item())
    )


def tpeek(name: str, tensor: t.Tensor, ret: bool = False):
    contains_nan = t.any(t.isnan(tensor)).item()
    contains_inf = t.any(t.isinf(tensor)).item()
    string = f"{name} SHAPE {tuple(tensor.shape)} MEAN: {'{0:.4g}'.format(t.mean(tensor.float()).cpu().item())} VAR: {'{0:.4g}'.format(t.var(tensor.float()).cpu().item())} {'CONTAINS_NAN! ' if contains_nan else ''}{'CONTAINS_INF! ' if contains_inf else ''}VALS [{' '.join(['{0:.4g}'.format(x) for x in t.flatten(tensor)[:10].cpu().tolist()])}...]"
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
    from_state_keys = state_dict.keys()
    shared_keys = to_state_keys * from_state_keys
    to_module.load_state_dict(from_state_keys)


def copy_weight_bias(mine, theirs, transpose=False):
    if transpose:
        mine.weight = t.nn.Parameter(rearrange(theirs.weight, "a b -> b a"))
    else:
        mine.weight = theirs.weight

    theirs_has_bias = has_not_null(theirs, "bias")
    mine_has_bias = has_not_null(mine, "bias")
    if theirs_has_bias != mine_has_bias:
        print(mine.bias)
        raise AssertionError("yikes")
    if mine_has_bias and theirs_has_bias:
        mine.bias = theirs.bias
