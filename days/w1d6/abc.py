import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from pycuda_utils import Holder, load_module

mod = load_module("abc.cu")
abc = mod.get_function("abc")
dest = torch.ones(128, dtype=torch.float32).cuda()
a = torch.arange(128, dtype=torch.float32).cuda()
b = torch.arange(128, dtype=torch.float32).cuda()
c = torch.arange(128, dtype=torch.float32).cuda()
abc(
    Holder(dest),
    Holder(a),
    Holder(b),
    Holder(c),
    block=(128, 1, 1),
    grid=(1, 1),
)


# Kernels run async by default, so we call synchronize() before using values.
torch.cuda.synchronize()
print(dest)
# 0 2 6 12 20 30
