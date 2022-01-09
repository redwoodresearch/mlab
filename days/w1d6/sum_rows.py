import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from torch._C import dtype

from pycuda_utils import Holder, load_module, ceil_divide

mod = load_module("sum_rows.cu")
sum_rows = mod.get_function("sum_rows")

num_cols = np.uint32(10)
num_rows = np.uint32(2000)
src = torch.randn((num_cols, num_rows), dtype=torch.float32).cuda()
dst = torch.zeros(num_cols, dtype=torch.float32).cuda()

src = src.contiguous()
dst = dst.contiguous()

sum_rows(
    Holder(src),
    Holder(dst),
    num_cols,
    num_rows,
    block=(1024, 1, 1),
    grid=(int(ceil_divide(num_cols, 1024)), 1),
)

# Kernels run async by default, so we call synchronize() before using values.
torch.cuda.synchronize()

print((dst - src.sum(dim=-1)).abs().max())
# 0 2 6 12 20 30
