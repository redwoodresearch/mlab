#pragma once

#include <cstdint>

#include "utils.h"

constexpr int64_t ceil_divide(int64_t l, int64_t r) { return (l + r - 1) / r; };

float *copy_to_gpu(const float *host_mem, int64_t size) {
  float *gpu_mem;
  CUDA_ERROR_CHK(cudaMalloc(&gpu_mem, size * sizeof(float)));
  CUDA_ERROR_CHK(cudaMemcpy(gpu_mem, host_mem, size * sizeof(float),
                            cudaMemcpyHostToDevice));
  return gpu_mem;
}
