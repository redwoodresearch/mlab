import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import torch

# must be called before using pycuda. (theoretically torch.cuda.init() is
# supposed to work, but it doesn't)
x = torch.tensor(1).cuda()


class Holder(drv.PointerHolderBase):

    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()


def ceil_divide(a, b):
    return (a + b - 1) // b


def load_module(filename):
    with open(filename) as f:
        return SourceModule(f.read())