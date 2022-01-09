import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from torch._C import dtype

from pycuda_utils import Holder, load_module, ceil_divide

mod = load_module("sum_atomic.cu")
sum_atomic = mod.get_function("sum")
# arr = torch.randint(
#     low=0,
#     high=100,
#     size=(10214112,),
#     dtype=torch.int32,
# ).cuda()
arr = torch.randn(10214112, dtype=torch.float32).cuda()
total = torch.zeros(1, dtype=torch.float32).cuda()

# size = np.array(arr.size(), dtype=np.int32)[0]
size = np.uint32(arr.size()[0])
# print(type(size))

sum_atomic(
    Holder(arr),
    size,
    Holder(total),
    block=(512, 1, 1),
    grid=(int(ceil_divide(size, 512)), 1),
)

# Kernels run async by default, so we call synchronize() before using values.
torch.cuda.synchronize()

check = torch.sum(arr)
shuffled_arr = arr[torch.randperm(size)]

print(shuffled_arr.sum())
print(check)
print(total)
# 0 2 6 12 20 30
