import time

def benchmark(func, iters: int = 10):
    for _ in range(3):
        func()

    start = time.time()
    for _ in range(iters):
        func()

    return (time.time() - start) / iters