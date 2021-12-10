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
  Timer(std::optional<TimePoint> start) : start_(start) {}

  double elapsed() {
    if (start_.has_value()) {
      return total_ + std::chrono::duration_cast<std::chrono::duration<double>>(
                          now() - *start_)
                          .count();
    } else {
      return total_;
    }
  }

  void stop() {
    total_ = elapsed();
    start_ = std::nullopt;
  }

  void start() { start_ = now(); }

  void report(const std::string &name) {
    std::cout << name << ": " << elapsed() << std::endl;
  }

private:
  static TimePoint now() { return Clock::now(); }

  std::optional<TimePoint> start_ = std::nullopt;
  double total_ = 0.;
};
