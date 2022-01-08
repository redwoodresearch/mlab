import time 

# Runs f for `iters` iterations
def benchmark(fn, iters):
    WARM_UP_ITERATIONS = 2
    # warm up
    for _ in range(WARM_UP_ITERATIONS):
        fn()

    start_time = time.time()
    for _ in range(iters):
        fn()
    end_time = time.time()
    return (end_time - start_time) / iters
