from functools import reduce
import torch as t
import numpy as np
from torch._C import dtype
from pycuda_utils import Holder, load_module, ceil_divide

mod = load_module("sum_shared.cu")
sum_512 = mod.get_function("sum_512")


def shared_sum(xs: t.Tensor):
    def reduce_once(arr: t.Tensor) -> t.Tensor:
        size = int(arr.shape[0])
        num_outputs = ceil_divide(size, 512)
        totals = t.zeros(num_outputs, dtype=t.float32).cuda()

        sum_512(
            Holder(arr),
            np.uint32(size),
            Holder(totals),
            block=(512, 1, 1),
            grid=(num_outputs, 1),
        )

        t.cuda.synchronize()

        return totals

    while xs.shape[0] > 1:
        xs = reduce_once(xs)

    return xs[0]


def test1():
    arr = t.randn(2, dtype=t.float32).cuda()
    size = int(arr.shape[0])
    num_outputs = ceil_divide(size, 512)
    totals = t.zeros(num_outputs, dtype=t.float32).cuda()

    sum_512(
        Holder(arr),
        np.uint32(size),
        Holder(totals),
        block=(512, 1, 1),
        grid=(num_outputs, 1),
    )

    # Kernels run async by default, so we call synchronize() before using values.
    t.cuda.synchronize()

    print(totals)
    print(t.nn.functional.pad(arr, (0, (- size) % 512)).reshape(num_outputs, -1).sum(dim=-1))

def test2():
    arr = t.randn(2121321, dtype=t.float32).cuda()
    print(shared_sum(arr))
    print(arr.sum())

if __name__ == "__main__":
    test2()
