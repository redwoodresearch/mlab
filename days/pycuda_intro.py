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
# cause any issues). If you pass arguments of the wrong type, the memory will
# be interpreted as the other type (with disasterous results).
zero_kernel(Holder(dest), block=(128, 1, 1), grid=(1, 1))

# Kernels run async by default, so we call synchronize() before using values.
torch.cuda.synchronize()

print(dest)

# Note that pycuda *isn't* a great way to write maintainable pytorch cuda code
# in practice. Typically you'd write cuda c++ code and use pybind11 with the
# pytorch c++ api, but pycuda's convenient for experimenting here without
# having to worry about CUDA memory managment or the pytorch c++ api. Using c++
# is nicer in many ways. For instance, pybind11 does catch type errors.
#
# If you'd like, you can write all of your cuda kernels in a separate file with
# the .cu file extension and then read in the file with python. This will allow
# for getting c++/cuda syntax highlighting and other nice things like that.
# Here's an example of this:

file_mod = SourceModule(open('days/cuda_load_from_file_example.cu').read())
zero_kernel = file_mod.get_function("zero")
one_kernel = file_mod.get_function("one")

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

# insert code here

# Next we'll change our code and kernel invocation so that we can handle any
# length with a fixed 1D block size (the lengths of the inputs should should
# still all be equal). Let's use 512 as the block size*. This is where the grid
# parameter comes in to play. We'll need to set the grid such that we end up
# running a thread for each location in the array. Write a function which
# computes this grid size and then calls the kernel using this value.
#
# *In general, efficient block sizes are some power of 2  which is greater than
# or equal to 128 found via benchmarking.
#
# From within a cuda kernel: - This index of the thread within the block can be
# found with `threadIdx.x` (or `.y`/`.z` for other dimensions). So, because the
# block size is 512, this value will be from 0 to 511. - the block size can be
# found with `blockDim.x` - and the index of the current block in the grid can
# be found with `blockIdx.x`
#
# (See
# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-variables
# for full details)
#
# Note that the total number of threads must be a multiple of 512. However, we
# want to handle inputs of any size! So if the index for a given thread is too
# large, we should just return instead of indexing out of bounds. We can pass
# the size of our tensor into the kernel as an argument with type `int64_t`. PyCuda
# that requires that non-pointer arguments have a numpy dtype so it can convert
# them to the c type under the hood. In this case you want `np.int64`.
#
# Test your kernel on tensors of a variety of lengths including lengths which
# aren't a multiple of 512.
#
# Remember to torch.cuda.synchronize() after each kernel invocation! (Ker


# Here's a function you might find useful:
def ceil_divide(a, b):
    return (a + b - 1) // b


# insert code here

# If you remove the size check and run the kernel on a tensor of length 1,
# what happens?

# insert code here

# Now let's setup a kernel which indexes a 1d tensor by another 1d tensor.
# Specifically we'll have out[i] = a[b[i]]. Note that the size of out and b
# should be equal, but the size of a is arbitrary. Like in the above exercise,
# your code should be able handle to handle any lengths by returning if the
# location would be out of bounds. The type of b should be a long
# tensor. This is a signed 64 bit int (`int64_t`).
#
# We could have some value in b which indexes out of bounds in a (<0 or
# >=size). If this is ever the case, have this thread return and print an error.
# You can use printf(...). See
# https://en.cppreference.com/w/cpp/io/c/fprintf for details on printf.
# There are much better ways of error handling, this is mostly just to
# demonstrate printf.
#
# Make sure to test several cases, including out of bounds indexing.

# insert code here

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
# This is the last task in this file, so after completing this look at the
# instructions doc for next steps.

# insert code here
