#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>

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

class Timer {
public:
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = Clock::time_point;

  Timer() { start(); }
  Timer(TimePoint start) : start_(start), started_(true) {}

  double elapsed() {
    if (started_) {
      return total_ + std::chrono::duration_cast<std::chrono::duration<double>>(
                          now() - start_)
                          .count();
    } else {
      return total_;
    }
  }

  void stop() {
    total_ = elapsed();
    started_ = false;
  }

  void start() {
    start_ = now();
    started_ = true;
  }

  void report(const std::string &name) {
    std::cout << name << ": " << elapsed() << std::endl;
  }

private:
  static TimePoint now() { return Clock::now(); }

  TimePoint start_;
  bool started_ = false;
  double total_ = 0.;
};
