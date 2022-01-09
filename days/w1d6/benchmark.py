import time
import torch

def benchmark_ns(fn, num_iters: int) -> float:
    for _ in range(3):
        fn()

    start_ns = time.time_ns()
    for _ in range(num_iters):
        fn()
        torch.cuda.synchronize()

    return (time.time_ns() - start_ns) / num_iters
