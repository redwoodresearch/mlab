#pragma once

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <random>
#include <vector>

#include "utils.h"

constexpr int64_t ceil_divide(int64_t l, int64_t r) { return (l + r - 1) / r; };

template <typename T> T *copy_to_gpu(const T *host_mem, int64_t size) {
  T *gpu_mem;
  CUDA_ERROR_CHK(cudaMalloc(&gpu_mem, size * sizeof(T)));
  CUDA_ERROR_CHK(
      cudaMemcpy(gpu_mem, host_mem, size * sizeof(T), cudaMemcpyHostToDevice));
  return gpu_mem;
}

template <typename T>
std::vector<T> copy_from_gpu(const T *gpu_mem, int64_t size) {
  std::vector<T> out(size);
  CUDA_ERROR_CHK(cudaMemcpy(out.data(), gpu_mem, size * sizeof(T),
                            cudaMemcpyDeviceToHost));
  return out;
}

template <class T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  os << "[";
  for (const T &x : v) {
    os << x << ",";
  }
  os << "]";

  return os;
}

inline std::vector<float> random_floats(float min, float max, int size) {
  std::vector<float> out(size);
  std::uniform_real_distribution<float> dist(min, max);
  // std::random_device rd;
  // std::default_random_engine engine(rd());
  std::default_random_engine engine;
  std::generate(out.begin(), out.end(), [&] { return dist(engine); });

  return out;
}
