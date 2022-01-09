import torch as t
import numpy as np
import pycuda.autoinit as _

from pycuda_utils import Holder, load_module, ceil_divide

BLOCK_SIZE = 512

mod = load_module('sum_shared.cu')
sum_kernel = mod.get_function('sumShared')

def sum_shared(values):
    size = len(values)
    num_blocks = ceil_divide(size, BLOCK_SIZE)
    dest = t.zeros(1).cuda()
    sum_kernel(Holder(dest), Holder(values),
               block=(BLOCK_SIZE, 1, 1), grid=(num_blocks, 1))
    t.cuda.synchronize()
    return dest[0]

if __name__ == "__main__":
    values = t.randn(512).cuda()
    my_ans = sum_shared(values)
    torch_ans = values.sum()
    t.cuda.synchronize()
    print(f"My answer: {my_ans:.5f}, Torch answer: {torch_ans:.5f}")