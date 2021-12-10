#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_set>
#include <vector>

#include <thrust/sequence.h>

#include "solution_utils.h"

template <typename T, typename Pred>
__global__ void filter_atomic_kernel(const T *inp, T *dest, int *atomic_v,
                                     int size, const Pred pred) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= size) {
    return;
  }
  if (pred(inp[i])) {
    dest[atomicAdd(atomic_v, 1)] = inp[i];
  }
}

template <typename T, typename Pred>
int filter_atomic_preallocated(const T *inp, T *dest, int *atomic_v, int size,
                               const Pred &pred) {
  int block_size = 512;
  filter_atomic_kernel<<<ceil_divide(size, block_size), block_size>>>(
      inp, dest, atomic_v, size, pred);
  CUDA_SYNC_CHK();
  int new_size;
  CUDA_ERROR_CHK(
      cudaMemcpy(&new_size, atomic_v, sizeof(int), cudaMemcpyDeviceToHost));
  return new_size;
}

template <typename T, typename Pred>
int filter_atomic(const T *inp, T *dest, int size, const Pred &pred) {
  int *atomic_v;
  CUDA_ERROR_CHK(cudaMalloc(&atomic_v, sizeof(int)));
  int out = filter_atomic_preallocated(inp, dest, atomic_v, size, pred);
  CUDA_ERROR_CHK(cudaFree(atomic_v));

  return out;
}

template <typename T, typename F>
std::vector<T> filter_vec(const std::vector<T> &vec, const F &f) {
  T *gpu_mem_in = copy_to_gpu(vec.data(), vec.size());
  T *gpu_mem_out;
  CUDA_ERROR_CHK(cudaMalloc(&gpu_mem_out, vec.size() * sizeof(T)));
  int size = f(gpu_mem_in, gpu_mem_out, vec.size());
  auto out = copy_from_gpu(gpu_mem_out, size);
  CUDA_ERROR_CHK(cudaFree(gpu_mem_in));
  CUDA_ERROR_CHK(cudaFree(gpu_mem_out));

  return out;
}

template <typename T>
bool vecs_same_elems(const std::vector<T> &l, const std::vector<T> &r) {

  return l.size() == r.size() && std::unordered_set<T>(l.begin(), l.end()) ==
                                     std::unordered_set<T>(r.begin(), r.end());
}

template <typename F, typename Pred>
void check_filterer(const F &f, const Pred &pred) {
  std::cout << "single filter: " << filter_vec(std::vector<float>{1.7f}, f)
            << "\n";
  std::cout << "single filter removed: "
            << filter_vec(std::vector<float>{3.7}, f) << "\n";
  std::cout << "single filter removed neg: "
            << filter_vec(std::vector<float>{-3.7}, f) << "\n";
  std::cout << "few filter: "
            << filter_vec(std::vector<float>{3.f, 2.4f, -2.f, 3.9f, 2.7f, 2.1f},
                          f)
            << "\n";

  {
    auto host_mem_more_than_block = random_floats(-8.0f, 8.0f, 513);
    std::vector<float> cpu_vec;
    cpu_vec.reserve(host_mem_more_than_block.size());
    std::copy_if(host_mem_more_than_block.begin(),
                 host_mem_more_than_block.end(), std::back_inserter(cpu_vec),
                 pred);
    auto gpu_vec = filter_vec(host_mem_more_than_block, f);

    std::cout << "is equal (more than block): " << std::boolalpha
              << vecs_same_elems(cpu_vec, gpu_vec) << std::endl;
  }

  {
    auto host_mem_large = random_floats(-8.0f, 8.0f, 100000);
    std::vector<float> cpu_vec;
    cpu_vec.reserve(host_mem_large.size());
    std::copy_if(host_mem_large.begin(), host_mem_large.end(),
                 std::back_inserter(cpu_vec), pred);
    std::cout << cpu_vec.size() << std::endl;
    auto gpu_vec = filter_vec(host_mem_large, f);

    std::cout << "is equal (large): " << std::boolalpha
              << vecs_same_elems(cpu_vec, gpu_vec) << std::endl;
  }
}

int main() {
  auto pred = [] __host__ __device__(const float x) {
    return std::abs(std::fmod(x, 1.f)) < 0.5f;
  };

  auto filter_atomic_f = [pred](const float *inp, float *dest, int size) {
    return filter_atomic(inp, dest, size, pred);
  };
  check_filterer(filter_atomic_f, pred);
}
