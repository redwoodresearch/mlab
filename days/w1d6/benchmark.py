from typing import *
import torch
import time

def benchmark(fn: Callable[[], None], n_iter: int):
    for _ in range(3):
        fn()
    start = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    finish = time.perf_counter()
    return (finish-start)/n_iter