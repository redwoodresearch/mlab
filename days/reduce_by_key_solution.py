import pycuda.autoinit as _
from pycuda.compiler import SourceModule
import torch
import numpy as np

from days.pycuda_utils import Holder, ceil_divide

# must be called before using pycuda. (theoretically torch.cuda.init() is
# supposed to work, but it doesn't)
x = torch.tensor(1).cuda()

reduce_by_key_mod = SourceModule("""
__global__ void reduce_by_key_atomic_kernel(const float *inp,
                                            const int64_t *keys, float *dest,
                                            int size) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= size) {
    return;
  }
  atomicAdd(&dest[keys[i]], inp[i]);
}
""")
reduce_by_key_kernel = reduce_by_key_mod.get_function(
    "reduce_by_key_atomic_kernel")


class ReduceByKey(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        assert inp.is_cuda and keys.is_cuda
        assert inp.dtype == torch.float32
        assert keys.dtype == torch.long
        assert len(inp.size()) == 1
        assert inp.size() == keys.size()

        ctx.save_for_backward(keys)

        dest = torch.zeros(keys[-1].item() + 1).cuda()

        block_size = 512
        reduce_by_key_kernel(Holder(inp),
                             Holder(keys),
                             Holder(dest),
                             np.int32(inp.size(0)),
                             block=(block_size, 1, 1),
                             grid=(ceil_divide(inp.size(0), block_size), 1))
        torch.cuda.synchronize()

        return dest

    @staticmethod
    def backward(ctx, grad_output):
        keys, = ctx.saved_tensors
        return grad_output[keys], None


if __name__ == "__main__":
    DEVICE = torch.device("cuda:0")

    inp = torch.tensor([1.7], device=DEVICE).cuda()
    keys = torch.tensor([0], device=DEVICE).cuda()
    print(ReduceByKey.apply(inp, keys))

    inp = torch.tensor([1.7], device=DEVICE, requires_grad=True)
    keys = torch.tensor([3], device=DEVICE)
    out = ReduceByKey.apply(inp, keys)
    out[0].backward(retain_graph=True)
    print("grad:", inp.grad)
    out[3].backward()
    print("grad actual index:", inp.grad)

    inp = torch.tensor([1.8, 12., -2.1, 4., 3., 9., 10.],
                       device=DEVICE,
                       requires_grad=True)
    keys = torch.tensor([0, 3, 3, 3, 3, 4, 8], device=DEVICE)
    out = ReduceByKey.apply(inp, keys)
    print("output:", out)
    out[3].backward(retain_graph=True)
    print("grad:", inp.grad)
