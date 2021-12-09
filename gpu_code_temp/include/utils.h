#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

inline void cuda_assert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "cuda assert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

#define CUDA_ERROR_CHK(ans) cuda_assert((ans), __FILE__, __LINE__);

#define CUDA_SYNC_CHK()                                                        \
  do {                                                                         \
    CUDA_ERROR_CHK(cudaPeekAtLastError());                                     \
    CUDA_ERROR_CHK(cudaDeviceSynchronize());                                   \
  } while (0)
