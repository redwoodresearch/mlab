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

#pragma unroll
  for (int chunk_size = block_size / 2; chunk_size > 0; chunk_size /= 2) {
    syncthreads();
    if (tidx < chunk_size) {
      data[tidx] += data[tidx + chunk_size];
    }
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

  // outputs found in scripts/solution_plot_ops_sum.py
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
