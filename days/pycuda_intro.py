# TODO: turn into jupyter notebook (I vastly prefer developing outside of a
# notebook)

# First, let's look at a simple example of using pycuda to operate on
# gpu tensors.
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import torch

# must be called before using pycuda. (theoretically torch.cuda.init() is
# supposed to work, but it doesn't)
x = torch.tensor(1).cuda()

zero_mod = SourceModule("""
__global__ void zero(float *dest)
{
  dest[threadIdx.x] = 0.f;
}
""")
zero_kernel = zero_mod.get_function("zero")


# some glue for interfacing with pycuda
class Holder(drv.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()


dest = torch.randn(128).cuda()

# Recall that cuda allows for executing 3D blocks of threads in the grid
# arrangement of our choice.
#
# In this case, we'll just use a single 1d block with size 128 to match the
# tensor. To execute just one block, we can set the values of grid to 1 in both
# dimensions.
#
# WARNING!!! PyCuda does NOT check the arguments to kernels! If you pass too
# few arguments the remaining arguments will be unintialized (garbage values)!
# If you pass too many, pycuda won't catch the issue (but it doesn't seem to
# cause any issues)
zero_kernel(Holder(dest), block=(128, 1, 1), grid=(1, 1))

# Kernels run async by default, so we call synchronize() before using values.
torch.cuda.synchronize()

print(dest)

# Note that pycuda *isn't* a great way to write maintainable pytorch cuda code
# in practice (typically you'd write cuda c++ code and use pybind11 with the
# pytorch c++ api, but it's convenient for experimenting here.)
#
# Next, let's create a kernel which computes out = a * b + c with float inputs
# out, a, b, and c. Use the kernel on gpu tensor inputs of size 128 with values
# of your choice (you can keep using a block size of 128).
#
# Note that the kernel function takes (args..., block=., grid=.), see
# https://documen.tician.de/pycuda/driver.html#pycuda.driver.Function for
# details
#
# Getting 'illegal memory access'? You might have forgotten to copy your
# tensors to the gpu.

# solution (not in student copy ofc):
mul_add_mod = SourceModule("""
__global__ void mul_add(float *dest, float *a, float *b, float *c)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i] + c[i];
}
""")
mul_add_kernel = mul_add_mod.get_function("mul_add")

size = 128
dest = torch.empty(size, dtype=torch.float32).cuda()
a = torch.arange(size).to(torch.float32).cuda()
b = torch.arange(0, 4 * size, 4).to(torch.float32).cuda()
c = torch.arange(3, size + 3).to(torch.float32).cuda()
mul_add_kernel(Holder(dest),
               Holder(a),
               Holder(b),
               Holder(c),
               block=(size, 1, 1),
               grid=(1, 1))
torch.cuda.synchronize()
print(dest)

# Next we'll change our code and kernel invocation so that we can handle inputs
# of any size with a fixed 1D block size. Let's use 512 as the block size*. This
# is where the grid parameter
# comes in to play. We'll need to set the grid such that we end up running a
# thread for each location in the array.
#
# *In general, efficient block sizes are some power of 2  which is greater than
# or equal to 128 found via benchmarking.
#
# From within a cuda kernel:
# - This index of the thread within the block can be found with `threadIdx.x` (or
# `.y`/`.z` for other dimensions). So, because the block size is 512, this value
# will be from 0 to 511.
# - the block size can be found with `blockDim.x`
# - and the index of the current block in the grid can be found with `blockIdx.x`
#
# (See
# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-variables
# for full details)
#
# Note that the total number of threads must be a multiple of 512. However, we
# want to handle inputs of any size! So if the index for a given thread is too
# large, we should just return instead of indexing out of bounds. We can pass
# the size of our tensor into the kernel as an argument with type `int`. PyCuda
# that requires that non-pointer arguments have a numpy dtype so it can convert
# them to the c type under the hood. In this case you want `np.int32`.
#
# Test your kernel on tensors of a variety of lengths including lengths which
# aren't a multiple of 512.
#
# Remember to torch.cuda.synchronize() after each kernel invocation!
# (Ker


# Here's a function you might find useful:
def ceil_divide(a, b):
    return (a + b - 1) // b


# solution (not in student copy ofc):
import numpy as np

mul_add_mod = SourceModule("""
__global__ void mul_add(float *dest, float *a, float *b, float *c, int size)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= size) return;
  dest[i] = a[i] * b[i] + c[i];
}
""")
mul_add_kernel = mul_add_mod.get_function("mul_add")


def mul_add(a, b, c):
    block_size = 512
    dest = torch.empty(a.nelement(), dtype=torch.float32).cuda()
    mul_add_kernel(Holder(dest),
                   Holder(a),
                   Holder(b),
                   Holder(c),
                   np.int32(a.nelement()),
                   block=(block_size, 1, 1),
                   grid=(ceil_divide(a.nelement(), block_size), 1))
    torch.cuda.synchronize()

    return dest


a = torch.tensor(2.0).cuda()
b = torch.tensor(7.0).cuda()
c = torch.tensor(3.0).cuda()
print(mul_add(a, b, c))

size = 50000
a = torch.arange(size).to(torch.float32).cuda()
b = torch.arange(0, 4 * size, 4).to(torch.float32).cuda()
c = torch.arange(3, size + 3).to(torch.float32).cuda()
print(mul_add(a, b, c))

# If you remove the size check and run the kernel on a tensor of length 1,
# what happens?

# solution (not in student copy ofc):
mul_add_no_size_check_mod = SourceModule("""
__global__ void mul_add(float *dest, float *a, float *b, float *c, int size)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  // if (i >= size) return;
  dest[i] = a[i] * b[i] + c[i];
}
""")
mul_add_no_size_check_kernel = mul_add_no_size_check_mod.get_function(
    "mul_add")


def mul_add_no_size_check(a, b, c):
    block_size = 512
    dest = torch.empty(a.nelement(), dtype=torch.float32).cuda()
    mul_add_no_size_check_kernel(Holder(dest),
                                 Holder(a),
                                 Holder(b),
                                 Holder(c),
                                 block=(block_size, 1, 1),
                                 grid=(ceil_divide(a.nelement(),
                                                   block_size), 1))
    torch.cuda.synchronize()

    return dest


# despite out of bounds access, this doesn't crash!

size = 1
a = torch.randn(size).cuda()
b = torch.randn(size).cuda()
c = torch.randn(size).cuda()
# print(mul_add_no_size_check(a, b, c))

# Next, write a kernel which sums over the last dimension of a 2d tensor. For
# simplicity, just sum over the last dimension at a given first dimension index
# in each thread. Output to a
#
# The code should handle 2d tensors of any size, so you'll have to pass both
# sizes as an input.
#
# Note that the kernel will depend on the memory layout of the tensor! So
# operations like `.transpose(0, 1)` which take a non-contigous view would
# cause issues. You can get a contigous tensor to avoid this issue (copying
# when needed) using `.contiguous()`. See
# https://pytorch.org/docs/stable/tensor_view.html for details.
#
# When will this kernel be slow and fail to use the parallelism of the gpu?
#
# This is our last kernel with pycuda before transitioning to writing our
# kernels independently from python (in just c++).

# solution (not in student copy ofc):
reduce_mod = SourceModule("""
__global__ void reduce(float *dest, float *inp, int dim0, int dim1)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= dim0) return;
  float x = 0.f;
  for(int j = 0; j < dim1; ++j) {
    x += inp[i * dim1 + j];
  }
  dest[i] = x;
}
""")
reduce_kernel = reduce_mod.get_function("reduce")


def reduce(inp):
    block_size = 512
    inp = inp.contiguous()
    dest = torch.empty(inp.size(0), dtype=torch.float32).cuda()
    reduce_kernel(Holder(dest),
                  Holder(inp),
                  np.int32(inp.size(0)),
                  np.int32(inp.size(1)),
                  block=(block_size, 1, 1),
                  grid=(ceil_divide(inp.size(0), block_size), 1))
    torch.cuda.synchronize()
    return dest


print(reduce(torch.tensor([[1., 6.]]).cuda()))
print(reduce(torch.arange(8).to(torch.float32).reshape(4, 2).cuda()))
print(reduce(torch.arange(100000).to(torch.float32).reshape(10000, -1).cuda()))
print(
    reduce(
        torch.arange(100000).to(torch.float32).reshape(10000, -1).transpose(
            0, 1).cuda()))

# performance is poor when first dim is small and second dim is large:
print(reduce(torch.arange(10000).to(torch.float32).reshape(10, -1).cuda()))
