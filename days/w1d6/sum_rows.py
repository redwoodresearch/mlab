import torch
import numpy as np
import pycuda.autoinit as _

from pycuda_utils import Holder, load_module, ceil_divide


mod = load_module('sum_rows.cu')
sum_kernel = mod.get_function('sumRows')

def sum_rows(values):
    rows, cols = values.shape
    sum = torch.zeros(rows, dtype=torch.float32).cuda()
    num_blocks = ceil_divide(rows, 512)
    sum_kernel(Holder(sum), Holder(values), np.int32(rows), np.int32(cols),block=(512, 1, 1), grid=(num_blocks, 1))
    torch.cuda.synchronize()
    return sum

if __name__ == "__main__":
    values = torch.randn(782,234).cuda()
    our_ans = sum_rows(values)
    torch_ans = values.sum(dim=1)

    # Kernels run async by default, so we call synchronize() before using values.
    torch.cuda.synchronize()
    print(our_ans)
    print(torch_ans)
    print(torch.max(our_ans-torch_ans))
    # print(torch.isclose(our_ans, torch_ans))