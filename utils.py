from einops.einops import rearrange
import torch as t
import numpy as np
import time


def tstat(name, tensor):
    print(
        name, "mean", "{0:.4g}".format(t.mean(tensor).cpu().item()), "var", "{0:.4g}".format(t.var(tensor).cpu().item())
    )


def tpeek(name, tensor):
    print(
        f"{name} MEAN: {'{0:.4g}'.format(t.mean(tensor).cpu().item())} VAR: {'{0:.4g}'.format(t.var(tensor).cpu().item())} SHAPE {tuple(tensor.shape)} VALS [{' '.join(['{0:.4g}'.format(x) for x in t.flatten(tensor)[:10].cpu().tolist()])}]"
    )


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


def copy_weight_bias(mine, theirs, transpose=False):
    # support weights called 'w' because sometimes oai does that
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
