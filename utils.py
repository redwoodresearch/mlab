import itertools
import re
from einops.einops import rearrange
import torch as t
import numpy as np
import time
import importlib


def tstat(name, tensor):
    print(
        name, "mean", "{0:.4g}".format(t.mean(tensor).cpu().item()), "var", "{0:.4g}".format(t.var(tensor).cpu().item())
    )


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
        raise AssertionError(f"{from_only} only in from module, {to_only} only in to module")
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


def import_object_from_qualified_name(qname: str):
    last_period = qname.rindex(".")
    module_name = qname[:last_period]
    object_name = qname[last_period + 1 :]
    module = importlib.import_module(module_name)
    out = getattr(module, object_name)
    assert out is not None
    return out


def einsum(string, *tensors):
    inputs, output = string.split("->")
    inputs = [x.split() for x in inputs.split(",")]
    output = output.split()

    if len(tensors) != len(inputs):
        raise AssertionError("einsum string must have same number of inputs as input tensors")
    dim_sizes = {}
    for args, tensor in zip(inputs, tensors):
        if len(args) != len(tensor.shape):
            raise AssertionError(f"arg string {args} has the wrong length for shape {tensor.shape}.")
        for arg, dim in zip(args, tensor.shape):
            if arg in dim_sizes and dim_sizes[arg] != dim:
                raise AssertionError(f"dim {arg} has incompatible dimensions {dim_sizes[arg]} and {dim}")
            dim_sizes[arg] = dim
    for arg in output:
        if arg not in dim_sizes:
            raise AssertionError(f"output string has a name that doesn't appear in the input: {arg}")
    valid_regex = r"^[a-zA-Z0-9_]+$"
    for arg in itertools.chain(output, *inputs):
        if not re.match(valid_regex, arg):
            raise AssertionError(f"invalid identifier {arg}")

    dim_order = list(output)
    for dim in dim_sizes.keys():
        if dim not in dim_order:
            dim_order.append(dim)
    wide = t.ones([dim_sizes[x] for x in dim_order])
    for arg, input in zip(inputs, tensors):
        input_ordered = rearrange(input, f"{' '.join(arg)} -> {' '.join([x if x in arg else '1' for x in dim_order])}")
        wide *= input_ordered
    return wide.sum(dim=tuple(range(len(output), len(dim_order))))


if __name__ == "__main__":
    i1, i2 = t.rand(10, 20, 40), t.rand(10, 40, 50)
    print(
        t.allclose(
            einsum("batch seq chan, batch chan chan2 -> batch seq chan2", i1, i2), t.einsum("bsc,bct->bst", i1, i2)
        )
    )
