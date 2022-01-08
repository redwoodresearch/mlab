#include <cassert>

__device__ void sum_atomic(const float *inp, float *dest, int size) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= size) {
    return;
  }
  atomicAdd(dest, inp[i]);
}

__device__ void syncthreads() {
#ifndef __clang__ // causes my language server to crash, so hot patch..
  __syncthreads();
#endif
}

template <int block_size>
__device__ void simple_sum_block_overall(const float *inp, float *dest,
                                         int size) {
  const int block_size = 512;                                             
  static __shared__ float data[block_size];
  assert(blockDim.x == block_size);

  int tidx = threadIdx.x;

  data[tidx] = 0;
  int idx = tidx + blockIdx.x * block_size;
  if (idx < size) {
    data[tidx] = inp[idx];
  }

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

__device__ void shfl_reduce_overall(const float *in, float *out, int size) {
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

extern "C" {
__global__ void sum_atomic_kernel(const float *inp, float *dest, int size) {
  sum_atomic(inp, dest, size);
}

__global__ void simple_sum_block_kernel_512(const float *inp, float *dest,
                                            int size) {
  simple_sum_block_overall<512>(inp, dest, size);
}

__global__ void shfl_reduce_kernel(const float *in, float *out, int size) {
  shfl_reduce_overall(in, out, size);
}
}
