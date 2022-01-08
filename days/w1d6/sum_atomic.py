import torch
import numpy as np
import pycuda.autoinit as _

from pycuda_utils import Holder, load_module, ceil_divide


mod = load_module('sum_atomic.cu')
sum_kernel = mod.get_function('sumAtomic')

def sum_atomic(values):
    SIZE = len(values)
    sum = torch.zeros(1, dtype=torch.float32).cuda()
    num_blocks = ceil_divide(SIZE, 512)
    sum_kernel(Holder(values), Holder(sum), np.int32(SIZE), block=(512, 1, 1), grid=(num_blocks, 1))
    torch.cuda.synchronize()
    return sum[0]

if __name__ == "__main__":
    values = torch.randn(7821).cuda()
    our_ans = sum_atomic(values)
    torch_ans = values.to("cpu").sum()

    # Kernels run async by default, so we call synchronize() before using values.
    torch.cuda.synchronize()
    print(our_ans)
    print(torch_ans)
    print(torch.isclose(our_ans, torch_ans))