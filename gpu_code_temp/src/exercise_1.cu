#include "utils.h"

// TODO: build system? Need to at least give some advice on how to work with
// it.

/*
For this exercise we'll implement a series of approaches for summing up an
array. The ideas here generalize to (many) types of reductions.

Feel free to use int instead of int64_t, but don't run on inputs larger than
2147483647 (32 bit int max).

1. For starters we'll use the most simple approach: device wide atomics.
Docs:
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions

Write a cpu function which takes a float gpu pointer and size
and returns the sum (as a float in cpu memory).

Generate a large (>500000) array of (pseudo)random numbers and sum it on both
the gpu and cpu. Why are the values slightly different? (If they're very
different, that's probably a bug.)

TODO: any more instruction? Or is this clear?
TODO: do we want to provide tests? (should be easy to test).
*/

/*
An interlude on debugging cuda by Ryan. You can skip this until actually
needing to debug stuff if you'd like.

Debugging cuda can be really painful. There is cuda-gdb, but it's a bit finicky
and (I've found) somewhat buggy. This sort of tool also just isn't a great
approach for 'deeply' parallel programming. Additionally, non-determistic
failure can happen (typically due to missing synchronization or failing to
initialize memory). Because synchronization errors are sensitive to timing,
it's suprising typical for print statements to cause the code to start working
again.

Here's a ordered checklist to go through when debugging cuda (crashes and
incorrect output):
- Reduce to the minimal input. Does this make it obvious what the issue is?
  Does it fail only when the size isn't a multiple of the block size (or
  similar)?
- Illegal memory error (or similar)? Make sure to check your mallocs, that
  pointers are for the correct device, and the bounds on your indexes (consider
  adding asserts -- I typically use pointer wrappers which automatically bounds
  check in debug mode, but I won't discuss setting that up here).
- Are things off by a seemingly random amount? Perhaps you failed to initialize
  to 0 and are using garbage memory.
- Is the failure/error non-determistic? The issue could be related to
  synchronization or reading something while anther thread is writing. So look
  through and try to see if there are any obvious places where syncronization
  is missing.
- Just generally look over your code and think about the algorithm. Can you
  spot anything which could be wrong?
- Consider using cuda-gdb. We won't go through how to use it here, but if
  you're already familiar with GDB it should be pretty straight forward to
  pick up after reading a guide online. This is particularly useful
  in situations like an assertion error and you just want to look at the values
  of a bunch of the surronding variables. It also tends to be more
  helpful when failures are determistic.
- Down all the way to here? You might be in for a hard time. There
  are many ways to proceed from here, but I often find the print/dump based
  workflow useful (described below).

I find a workflow of dumping inputs, outputs, and expected outputs as python
lists useful for GPU debugging in particular. This allows for dumping to a
python file which can then be imported to access and manipulate the data (often
after converting to numpy arrays). Then with this input data and some python
hacking, strategic print statements can be inserted into the kernel and used to
'binary search' for where the failure occurs (checking what values should be
with hacky python code). You should try to print only what you really need to
avoid clutter. In fact, if you print too much the kernel will crash as printf
has a fixed sized buffer! Make sure to first reduce to the minimally sized
input which can cause the issue. Using a bunch of these print statments/dumping
makes it easier to debug non-determistic fails. Just keep rerunning until it
fails at some location and then investigate the print statements which
correspond to that location.

If print statments upset the timing and cause the issue to no longer occur, try
saving values to array and then printing at the end of the kernel. Saving to an
array is much faster, so it shouldn't upset timing much.

While this print statement work flow is good for thorough debugging, it can
take a while to get rolling, so it's always a good idea to first go through
something like the common errors checklist above and to do a detailed read
through of the code.
*/

/*
2. Next let's benchmark the performance of this atomic kernel. This time should
include time spent copying the output from the gpu, but not include copying the
input to the gpu nor time for memory allocation (cache gpu memory as needed).
The `Timer` class inside utils.h can be used for this. In general, it probably
makes more sense to use a benchmarking library, but to keep things simple we'll
just benchmark by hand. Graph the floats/sec vs the length of the array (a
simple way to do this is to print values as a csv and then parse with python).
Run with lengths up to around 10 million.

To ensure minimal noise, average the timing over several runs. Also, make sure
the gpu isn't running anything else of significance.

Also, often letting the code 'warm up' for an iteration (or a few) will
improve performance. For ML applications we typically care about about
amortized time over a large number of iterations, so to reduce noise we
should run warm up iteration(s) before actually running the benchmark.

Remember to run your final benchmarks in release mode!

How does performance (in floats/sec) change with the size of the input?
Consider zooming in/higher levels of sampling at various points to get a better
sense for the transitions. Why

TODO: should we actually recommend using a benchmarking library?
*/

/*
3. Now let's also benchmark the reduce operation in the Thrust library and plot
it's performance vs our implementation. Thurst is included with cuda, so you
should be able to `#include <thrust/reduce.h>` with any changes.

The documentation for reduce can be found here:
https://thrust.github.io/doc/group__reductions_ga6ac0fe1561f58692e85112bd1145ddff.html

You can pass in thrust::device for the 'exec' param and gpu pointer, gpu
pointer + size for the 'first' and 'last' params (respectively).

You should find that our implementation gets absolutely smoked on timing,
Yikes.

You might want to benchmark thrust on considerably larger inputs than the
atomic implementation can conveniently manage to see how performance changes
over a larger set of input sizes.

In turns out that having all threads write to the same atomic results
in a ton of collisions making the code very slow.

Let's see if we can write a faster reduce.
*/

/*
4. Next, benchmark a single threaded cpu summation implementation (you can use
std::accumulate if you'd like) and plot the results.

How does the scaling of floats/sec differ between the cpu and gpu?
*/

/*
5. Another approach to reduction is to reduce all the threads within a block
(segmented reduction) and output this reduced value to a new array for each of
these blocks. Then we launch another kernel which reduces this new array (which
has size = number of blocks in previous launch). This is repeated until the
array fits in one block so the new output array has only one value and can be
returned.

#TODO: diagram?

This approach can be much faster because we just need to reduce within a block.
In fact, it's the approach which all fast Cuda implementations use (as far as
I'm aware).

To start, we'll just write (and test) a kernel which reduces all of the blocks
and outputs these values to a new array. We won't worry about invoking this
kernel multiple times to fully reduce the array yet. We'll need some approach
to synchronize and share memory between threads in the block to do this.
Details on how to do this can be found in this article:
https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/. You only need
to read through the end of the 'Static Shared Memory' section.

There are many ways to implement a kernel with __syncthreads() and shared
memory. We'll start with a relatively simple approach (which is somewhat
inefficient). First, threads will copy their value from the global memory input
into shared memory. Then we'll use the approach shown in the 'reduction.png' in
this directory (TODO: actually insert if this in doc) to sum up shared memory.
Specifically, each thread will output at the location of it's thread id and
will sum the current entry at it's thread id with the entry one 'chunk' over.
We'll run this in a loop halving the chunk size on each iteration until the
chunk size reaches 0 and the array is fully summed (see the last 2 rows of the
picture). We'll need to use __syncthreads() after each iteration of the loop to
ensure all values have been written out.

Be careful: if an index is being read from shared memory in this iteration, no
other thread should be writing to that index in this iteration. Otherwise the
reads will be non-deterministically incorrect! To visualize this look at the
diagram and note which subset of the thread IDs are highlighted in orange
circles on each iteration. Only these thread id should be writing, or you'll
have issues.

You should choose some power of 2 block size (512 is a good choice, but you can
try other values).

Note that we will need to know the block size at compile time in order to use
static shared memory. We can define this as a global compile time constant
using constexpr. You can use a template parameter if you'd prefer (it's not
important that you know what templates are).

Note that __syncthreads() requires that *all* threads in the block reach that
point. So, threads can no longer exit early. Threads over the end of the array
will need to call thread sync at the appropriate time and avoid indexing out of
bounds. (You can just wrap the important lines in an if statement).

Common errors:
- forgetting some/all calls to __syncthreads();
- writing while another thread is reading to the same location (see above start
with 'be careful:').

TODO: are these instructions clear?
*/

/*
6. Now let's setup the code to invoke this kernel multiple times and fully
reduce the array. You will need two arrays.

Let's also benchmark this and plot it alongside the atomic and Thrust times
(you should find that it's much faster than the atomic implementation, but
still slower than Thrust).

If you'd like, you can spend some time trying to optimize this implementation.
We'll be implementing a different approach next which uses a new type of
intrisics, so it might be worth waiting until after implementing that to
decide if and what you'd like to optimize.

Note that the algorithm discussed here is the third algorithm brought up in
this presentation:
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
You can can look at some of the later modifications for how to make this
faster if you're interested (but consider avoiding spoilers before
you've tried to optimize yourself:) )
*/

/*
7. Next we'll be looking at the 'shfl' family of instrinics which
can be used to quickly communicate between threads within a warp.

First, note that in addition to writing __global__ functions (kernels), it's
possible to write functions which can be called on the gpu using __device__. To
make a function callable on the cpu and the gpu you can use '__host__
__device__'. See here in the docs for details:
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-declaration-specifiers

Now read this article through the 'Shuffle Warp Reduce' section, but stop at
and don't read the 'Block Reduce' section':
https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/

Can you think of a way to combine the 'warpReduceSum' approach here with
__syncthreads() and shared memory to construct an efficient (and hopefully
elegant) block reduction algorithm? Spend some time thinking about this, and
then implement and benchmark your design.

Note that __shfl_down has been deprecated and replaced with __shfl_down_sync^.
The docs be found here
https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions
For the mask input you can just input the full mask: 0xffffffff.

If you get stuck, don't want to spend too long working on this, or once you
have finished the previous part, look through the next sections in the article:
'Block Reduce' and 'Reducing Large Arrays'. Particularly take note of the trick
of iterating by the total size covered by the grid (blockDim.x * gridDim.x).
This is an efficient approach to use only 2 kernel launches without having to
tune any parameters.

Using what you've learned here, spend some time optimizing your reduction
function. Can you match the perfomance of Thrust? This will certainly require
some tuning and might be quite difficult. Note that Thrust is allocating memory
on each call, so it's slightly handicapped on smaller inputs. It's possible to
avoid allocating memory each time with Thrust, but I won't go into how to do so
here.

Consider also reading reading some of the discussion of atomics in that article
and implementing that approach. Combining atomics with block wide reduction
results in far fewer atomic collisions while also requiring only one kernel
invocation. The overall perfomance is similar.
*/

// the rest of this file is the solution for all parts:
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include <thrust/reduce.h>

#include "solution_utils.h"

__global__ void sum_atomic_kernel(const float *inp, float *dest, int size) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= size) {
    return;
  }
  atomicAdd(dest, inp[i]);
}

float sum_atomic_preallocated(const float *gpu_inp, int size, float *dest) {
  int block_size = 512;
  sum_atomic_kernel<<<ceil_divide(size, block_size), block_size>>>(gpu_inp,
                                                                   dest, size);
  CUDA_SYNC_CHK();
  float out;
  CUDA_ERROR_CHK(cudaMemcpy(&out, dest, sizeof(float), cudaMemcpyDeviceToHost));

  return out;
}

float sum_atomic(const float *gpu_inp, int size) {
  float *dest;
  CUDA_ERROR_CHK(cudaMalloc(&dest, sizeof(float)));
  float out = sum_atomic_preallocated(gpu_inp, size, dest);
  CUDA_ERROR_CHK(cudaFree(dest));
  return out;
}

template <typename F>
float reduce_vec(const std::vector<float> &vec, const F &f) {
  float *gpu_mem = copy_to_gpu(vec.data(), vec.size());
  float out = f(gpu_mem, vec.size());
  CUDA_ERROR_CHK(cudaFree(gpu_mem));

  return out;
}

template <typename F> void check_reducer(const F &f) {
  std::vector<float> host_mem_single{1.7f};
  std::cout << "single sum: " << reduce_vec(host_mem_single, f) << "\n";

  std::vector<float> host_mem_few{1.2f, 0.f, 123.f};
  std::cout << "few sum: " << reduce_vec(host_mem_few, f) << "\n";

  auto host_mem_more_than_block = random_floats(-8.0f, 8.0f, 513);
  float cpu_total = std::accumulate(host_mem_more_than_block.begin(),
                                    host_mem_more_than_block.end(), 0.f);
  std::cout << "more_than_block sum: "
            << reduce_vec(host_mem_more_than_block, f) << "\n";
  std::cout << "cpu more_than_block sum: " << cpu_total << "\n";

  auto host_mem_many = random_floats(-8.0f, 8.0f, 262145);
  cpu_total = std::accumulate(host_mem_many.begin(), host_mem_many.end(), 0.f);
  std::cout << "many sum: " << reduce_vec(host_mem_many, f) << "\n";
  std::cout << "cpu many sum: " << cpu_total << "\n";
}

template <typename F>
float benchmark_reduce(const F &f, int size, int iters = 10) {
  auto host_mem = random_floats(-8.0f, 8.0f, size);
  float *gpu_mem = copy_to_gpu(host_mem.data(), host_mem.size());

  // warmup
  for (int i = 0; i < 3; ++i) {
    f(gpu_mem, size);
  }

  Timer timer;
  for (int i = 0; i < iters; ++i) {
    f(gpu_mem, size);
  }

  CUDA_ERROR_CHK(cudaFree(gpu_mem));

  return timer.elapsed() / iters;
}

__device__ void syncthreads() {
#ifndef __clang__ // causes my language server to crash, so hot patch..
  __syncthreads();
#endif
}

template <int block_size>
__global__ void simple_sum_block_kernel(const float *inp, float *dest,
                                        int size) {
  __shared__ float data[block_size];
  assert(blockDim.x == block_size);

  int tidx = threadIdx.x;

  data[tidx] = 0;
  int idx = tidx + blockIdx.x * block_size;
  if (idx < size) {
    data[tidx] = inp[idx];
  }
  syncthreads();

#pragma unroll
  for (int chunk_size = block_size / 2; chunk_size > 0; chunk_size /= 2) {
    if (tidx < chunk_size) {
      data[tidx] += data[tidx + chunk_size];
    }
    syncthreads();
  }

  if (tidx == 0) {
    dest[blockIdx.x] = data[tidx];
  }
}

template <typename F>
void run_all_benchmark_reduce(const F &f, int max_size_power) {
  std::cout << "size,time\n";
  for (int size_power = 6; size_power < max_size_power; ++size_power) {
    int size = 1 << size_power;
    int iters = size_power < 17 ? 100 : 10;
    std::cout << size << "," << benchmark_reduce(f, size, iters) << "\n";
  }
}

std::array<std::vector<float>, 2>
run_simple_sum_block_for_test(const std::vector<float> &to_reduce) {
  constexpr int block_size = 512;
  float *in_gpu = copy_to_gpu(to_reduce.data(), to_reduce.size());
  float *out_gpu;
  int n_blocks = ceil_divide(to_reduce.size(), block_size);
  CUDA_ERROR_CHK(cudaMalloc(&out_gpu, n_blocks * sizeof(float)));
  simple_sum_block_kernel<block_size><<<n_blocks, block_size>>>(
      in_gpu, out_gpu, static_cast<int>(to_reduce.size()));
  CUDA_SYNC_CHK();
  std::vector<float> out_from_gpu(n_blocks);
  CUDA_ERROR_CHK(cudaMemcpy(out_from_gpu.data(), out_gpu,
                            n_blocks * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHK(cudaFree(in_gpu));
  CUDA_ERROR_CHK(cudaFree(out_gpu));

  std::vector<float> out_cpu;
  for (size_t start = 0; start < to_reduce.size(); start += block_size) {
    int end = std::min(start + block_size, to_reduce.size());
    out_cpu.push_back(std::accumulate(to_reduce.begin() + start,
                                      to_reduce.begin() + end, 0.f));
  }

  return {out_from_gpu, out_cpu};
}

void print_vecs(const std::array<std::vector<float>, 2> &vecs) {
  std::cout << "gpu = " << vecs[0] << "\n";
  std::cout << "cpu = " << vecs[1] << "\n";
}

void check_simple_sum_block() {
  print_vecs(run_simple_sum_block_for_test({1.7}));
  print_vecs(run_simple_sum_block_for_test({1.3, -3.0, 100.0}));
  print_vecs(run_simple_sum_block_for_test(random_floats(-8.0f, 8.0f, 5)));
  print_vecs(run_simple_sum_block_for_test(random_floats(-8.0f, 8.0f, 257)));
  print_vecs(run_simple_sum_block_for_test(random_floats(-8.0f, 8.0f, 513)));
  print_vecs(run_simple_sum_block_for_test(random_floats(-8.0f, 8.0f, 3000)));
}

template <int block_size>
float sum_via_simple_segments_preallocated(const float *gpu_inp, int size,
                                           float *dest_l, float *dest_r) {
  int sub_size = ceil_divide(size, block_size);
  simple_sum_block_kernel<block_size>
      <<<sub_size, block_size>>>(gpu_inp, dest_l, size);
  CUDA_SYNC_CHK();
  float *in = dest_l;
  float *out = dest_r;
  while (sub_size > 1) {
    int next_sub_size = ceil_divide(sub_size, block_size);
    simple_sum_block_kernel<block_size>
        <<<next_sub_size, block_size>>>(in, out, sub_size);
    std::swap(in, out);
    sub_size = next_sub_size;
  }
  float out_v;
  CUDA_ERROR_CHK(cudaMemcpy(&out_v, in, sizeof(float), cudaMemcpyDeviceToHost));

  return out_v;
}

float sum_via_simple_segments(const float *gpu_inp, int size) {
  constexpr int block_size = 512;
  int n_blocks = ceil_divide(size, block_size);
  float *dest_l;
  float *dest_r;
  CUDA_ERROR_CHK(cudaMalloc(&dest_l, n_blocks * sizeof(float)));
  CUDA_ERROR_CHK(
      cudaMalloc(&dest_r, ceil_divide(n_blocks, block_size) * sizeof(float)));
  float out = sum_via_simple_segments_preallocated<block_size>(gpu_inp, size,
                                                               dest_l, dest_r);
  CUDA_ERROR_CHK(cudaFree(dest_l));
  CUDA_ERROR_CHK(cudaFree(dest_r));

  return out;
}

constexpr unsigned mask = 0xffffffff;

// this code is taken (mostly) from this article:
// https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
__device__ float warp_reduce(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(mask, val, offset);
  }
  return val;
}

__inline__ __device__ float block_reduce(float val) {
  static __shared__ float shared[32]; // Shared mem for 32 partial sums
  int within_warp_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  val = warp_reduce(val); // Each warp performs partial reduction

  if (within_warp_id == 0) {
    shared[warp_id] = val; // Write reduced value to shared memory
  }

  syncthreads(); // Wait for all partial reductions

  // read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[within_warp_id] : 0;

  if (warp_id == 0) {
    val = warp_reduce(val); // Final reduce within first warp
  }

  return val;
}

__global__ void shfl_reduce_kernel(const float *in, float *out, int size) {
  float sum = 0;
  // reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }

  sum = block_reduce(sum);

  if (threadIdx.x == 0) {
    out[blockIdx.x] = sum;
  }
}

// dest must be longer than 1024
float shfl_reduce_preallocated(const float *in, int size, float *dest) {
  int block_size = 512;
  int max_grid = 1024;
  int blocks = std::min(int(ceil_divide(size, block_size)), max_grid);

  shfl_reduce_kernel<<<blocks, block_size>>>(in, dest, size);
  shfl_reduce_kernel<<<1, 1024>>>(dest, dest, blocks);

  float out_v;
  CUDA_ERROR_CHK(
      cudaMemcpy(&out_v, dest, sizeof(float), cudaMemcpyDeviceToHost));

  return out_v;
}

float shfl_reduce(const float *in, int size) {
  float *dest;
  CUDA_ERROR_CHK(cudaMalloc(&dest, 1024 * sizeof(float)));
  float out = shfl_reduce_preallocated(in, size, dest);
  CUDA_ERROR_CHK(cudaFree(dest));

  return out;
}

int main() {
  std::cout << "=== check atomic ===\n";
  check_reducer(sum_atomic);

  std::cout << "=== check simple sum ===\n";
  check_simple_sum_block();
  check_reducer(sum_via_simple_segments);

  std::cout << "=== check shfl ===\n";
  check_reducer(shfl_reduce);

  // outputs found in plot_times_sum.py
  std::cout << "atomic results:\n";

  float *dest;
  CUDA_ERROR_CHK(cudaMalloc(&dest, sizeof(float)));

  run_all_benchmark_reduce(
      [&](const float *gpu_inp, int size) {
        sum_atomic_preallocated(gpu_inp, size, dest);
      },
      22);

  CUDA_ERROR_CHK(cudaFree(dest));

  std::cout << "thrust results:\n";

  run_all_benchmark_reduce(
      [&](const float *gpu_inp, int size) {
        thrust::reduce(thrust::device, gpu_inp, gpu_inp + size, 0.f);
      },
      30);

  std::cout << "simple sum block results:\n";

  constexpr int block_size = 512;
  float *dest_l;
  float *dest_r;

  // allocate based on max size
  int max_size_power_simple_segment = 28;
  int max_size = 1 << max_size_power_simple_segment;
  int n_blocks = ceil_divide(max_size, block_size);
  CUDA_ERROR_CHK(cudaMalloc(&dest_l, n_blocks * sizeof(float)));
  CUDA_ERROR_CHK(cudaMalloc(&dest_r, n_blocks * sizeof(float)));

  run_all_benchmark_reduce(
      [&](const float *gpu_inp, int size) {
        sum_via_simple_segments_preallocated<block_size>(gpu_inp, size, dest_l,
                                                         dest_r);
      },
      max_size_power_simple_segment);

  CUDA_ERROR_CHK(cudaFree(dest_l));
  CUDA_ERROR_CHK(cudaFree(dest_r));

  std::cout << "shfl results:\n";

  CUDA_ERROR_CHK(cudaMalloc(&dest, 1024 * sizeof(float)));
  run_all_benchmark_reduce(
      [&](const float *gpu_inp, int size) {
        shfl_reduce_preallocated(gpu_inp, size, dest);
      },
      30);


  // TODO: add cpu benchmark

  CUDA_ERROR_CHK(cudaFree(dest));
}
