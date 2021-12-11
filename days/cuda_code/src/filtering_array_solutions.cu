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
  CUDA_ERROR_CHK(cudaMemset(atomic_v, 0, sizeof(int)));
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

  for (int size : {513, 100000}) {
    auto host_mem_more_than_block = random_floats(-8.0f, 8.0f, size);
    std::vector<float> cpu_vec;
    cpu_vec.reserve(host_mem_more_than_block.size());
    std::copy_if(host_mem_more_than_block.begin(),
                 host_mem_more_than_block.end(), std::back_inserter(cpu_vec),
                 pred);
    auto gpu_vec = filter_vec(host_mem_more_than_block, f);

    std::cout << "is equal (size: " << size << "): " << std::boolalpha
              << vecs_same_elems(cpu_vec, gpu_vec) << std::endl;
  }
}

template <typename F>
float benchmark_filter(const F &f, int size, int iters = 10,
                       bool is_cpu = false) {
  auto host_mem = random_floats(-1.0f, 1.0f, size);
  float *mem;
  float *out_mem;
  std::vector<float> out_mem_cpu;
  if (is_cpu) {
    mem = host_mem.data();
    out_mem_cpu.resize(size);
    out_mem = out_mem_cpu.data();
  } else {
    mem = copy_to_gpu(host_mem.data(), host_mem.size());
    CUDA_ERROR_CHK(cudaMalloc(&out_mem, size * sizeof(float)));
  }

  // warmup
  for (int i = 0; i < 3; ++i) {
    f(mem, out_mem, size);
  }

  Timer timer;
  for (int i = 0; i < iters; ++i) {
    f(mem, out_mem, size);
  }

  float time = timer.elapsed() / iters;

  if (!is_cpu) {
    CUDA_ERROR_CHK(cudaFree(mem));
    CUDA_ERROR_CHK(cudaFree(out_mem));
  }

  return time;
}

template <typename F>
void run_all_benchmark_filter(const F &f, int max_size_power,
                              bool is_cpu = false) {
  std::cout << "size,time\n";
  for (int size_power = 6; size_power < max_size_power; ++size_power) {
    int size = 1 << size_power;
    int iters = size_power < 17 ? 100 : 10;
    std::cout << size << "," << benchmark_filter(f, size, iters, is_cpu)
              << "\n";
  }
}

int main() {
  auto get_pred = [](float thresh) {
    return [thresh] __host__ __device__(const float x) {
      return std::abs(std::fmod(x, 1.f)) < thresh;
    };
  };

  int *atomic_v;
  CUDA_ERROR_CHK(cudaMalloc(&atomic_v, sizeof(int)));
  auto get_filter_atomic_f = [atomic_v](auto pred) {
    return [pred, atomic_v](const float *inp, float *dest, int size) {
      return filter_atomic_preallocated(inp, dest, atomic_v, size, pred);
    };
  };

  auto get_filter_thrust = [](auto pred) {
    return [pred](const float *inp, float *dest, int size) {
      return thrust::copy_if(thrust::device, inp, inp + size, dest, pred);
    };
  };

  auto get_filter_cpu = [](auto pred) {
    return [pred](const float *inp, float *dest, int size) {
      return std::copy_if(inp, inp + size, dest, pred);
    };
  };

  auto base_pred = get_pred(0.5f);
  check_filterer(get_filter_atomic_f(base_pred), base_pred);

  std::cout << "[\n";
  for (float thresh : {0.01f, 0.05f, 0.2f, 0.5f, 0.9f}) {
    auto pred = get_pred(thresh);
    std::cout << "(" << thresh << ",\n{\n";
    std::cout << "'atomic' : \"\"\"\n";
    run_all_benchmark_filter(get_filter_atomic_f(pred), 26);
    std::cout << "\"\"\",\n";
    std::cout << "'thrust' : \"\"\"\n";
    run_all_benchmark_filter(get_filter_thrust(pred), 26);
    std::cout << "\"\"\",\n";
    std::cout << "'cpu' : \"\"\"\n";
    run_all_benchmark_filter(get_filter_cpu(pred), 20, true);
    std::cout << "\"\"\",\n";
    std::cout << "}),\n";
  }
  std::cout << "]\n";

  CUDA_ERROR_CHK(cudaFree(atomic_v));
}
