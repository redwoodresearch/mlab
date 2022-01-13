import itertools
import re
from einops.einops import rearrange
import torch as t
import numpy as np
import time
import importlib
from torchtyping import patch_typeguard, TensorType
import math

from PIL import Image
import requests
from io import BytesIO
import torchvision


def load_image(url):
    response = requests.get(url)
    return torchvision.transforms.ToTensor(Image.open(BytesIO(response.content)))


def to_batches(list_of_tensors, batch_size, trim=False):
    len = list_of_tensors[0].shape[0]
    batches = []
    for i in range(0, len, batch_size):
        if i + batch_size > len and trim:
            break
        batches.append([x[i : i + batch_size] for x in list_of_tensors])
    return batches


def itpeek(tensor: t.Tensor):
    contains_nan = t.any(t.isnan(tensor)).item()
    contains_inf = t.any(t.isinf(tensor)).item()
    string = f"SHAPE {tuple(tensor.shape)} MEAN: {'{0:.4g}'.format(t.mean(tensor.float()).cpu().item())} STD: {'{0:.4g}'.format(t.std(tensor.float()).cpu().item())} {'CONTAINS_NAN! ' if contains_nan else ''}{'CONTAINS_INF! ' if contains_inf else ''}VALS [{' '.join(['{0:.4g}'.format(x) for x in t.flatten(tensor)[:10].cpu().tolist()])}{'...' if tensor.numel()>10 else ''}]"
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
        raise AssertionError(
            f"{from_only} only in from module, {to_only} only in to module"
        )
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
        raise AssertionError(
            "einsum string must have same number of inputs as input tensors"
        )
    dim_sizes = {}
    for args, tensor in zip(inputs, tensors):
        if len(args) != len(tensor.shape):
            raise AssertionError(
                f"arg string {args} has the wrong length for shape {tensor.shape}."
            )
        for arg, dim in zip(args, tensor.shape):
            if arg in dim_sizes and dim_sizes[arg] != dim:
                raise AssertionError(
                    f"dim {arg} has incompatible dimensions {dim_sizes[arg]} and {dim}"
                )
            dim_sizes[arg] = dim
    for arg in output:
        if arg not in dim_sizes:
            raise AssertionError(
                f"output string has a name that doesn't appear in the input: {arg}"
            )
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
        input_ordered = rearrange(
            input,
            f"{' '.join(arg)} -> {' '.join([x if x in arg else '1' for x in dim_order])}",
        )
        wide *= input_ordered
    return wide.sum(dim=tuple(range(len(output), len(dim_order))))


def allclose(my_out, their_out, name, tol=1e-5):

    if not t.allclose(my_out, their_out, rtol=1e-4, atol=tol):
        errstring = f'error in {name}\n{tpeek("", my_out, ret=True)} \n!=\n{tpeek("", their_out, ret=True)}'
        raise AssertionError(errstring)
    else:
        tpeek(f"{name} MATCH!!!!!!!!\n", my_out)


if __name__ == "__main__":
    i1, i2 = t.rand(10, 20, 40), t.rand(10, 40, 50)
    print(
        t.allclose(
            einsum("batch seq chan, batch chan chan2 -> batch seq chan2", i1, i2),
            t.einsum("bsc,bct->bst", i1, i2),
        )
    )

# prior over sequence lenghts is exponentially decaying, 50% chance below 32 - e^(-x/32)
# p(b) = e^(-x/32)/32
# D(x) = 1-e^(-x/32)
# 1-D(x) = e^(-x/32)
# -x/32 = ln(1-D(x))
# x = -32ln(1-D(x))

from einops import *


def find_max_batch_size(fn, example_dp):
    def test_fn(bs):
        dp = t.as_strided(
            example_dp, (int(bs), *example_dp.shape), (0, *example_dp.stride())
        )
        try:
            fn(dp)
            return True
        except Exception as e:
            if "CUDA out of memory" in e:
                return False
            raise e

    return dist_bisect(test_fn)


def exp_dist(decay=32):
    return lambda p: -32 * math.log(1 - max(min(p, 0.999), 0.001))


# dist is a function from percentile to value
def dist_bisect(fn, distribution=exp_dist(decay=32), min_difference=1):
    lo, hi = 0, 1
    counter = 0
    while distribution(hi) - distribution(lo) > min_difference and counter < 20:
        mid = (hi - lo) / 2 + lo
        mid_place = distribution(mid)
        worked = fn(mid_place)
        if worked:
            lo = mid
        else:
            hi = mid
        print("testing ", mid_place, worked)
        counter += 1
    return distribution(lo)
