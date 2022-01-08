import time
import torch

def benchmark(f, num_iters):
    # warm-up
    for _ in range(3): 
        f()
        torch.cuda.synchronize()
    t_init = time.time_ns()
    for _ in range(num_iters):
        f()
        torch.cuda.synchronize()
    t_final = time.time_ns()
    return (t_final - t_init) / num_iters