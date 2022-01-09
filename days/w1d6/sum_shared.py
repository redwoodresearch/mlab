import einops as ein
import torch as t
import pycuda.autoinit as _

from pycuda_utils import Holder, load_module, ceil_divide

BLOCK_SIZE = 512

mod = load_module('sum_shared.cu')
sum_kernel = mod.get_function('sumShared')

def sum_shared(values):
    size = len(values)
    num_blocks = ceil_divide(size, BLOCK_SIZE)
    dest = t.zeros(num_blocks).cuda()
    sum_kernel(Holder(dest), Holder(values), block=(BLOCK_SIZE, 1, 1), grid=(num_blocks, 1))
    t.cuda.synchronize()
    return dest

if __name__ == "__main__":
    values = t.randn(10 * 512).cuda()
    my_ans = sum_shared(values)
    torch_ans = ein.reduce(values, "(b v) -> b", 'sum', b=10)
    t.cuda.synchronize()
    print(f"My answer: {my_ans},\nTorch answer: {torch_ans}")
    print(f"Answers match: {t.allclose(my_ans, torch_ans)}")