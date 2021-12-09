#include <iostream>
#include <vector>

#include "utils.h" // for CUDA_ERROR_CHK, (see file for details)

// This is a reproduction of our orignal pycuda kernel, mostly for
// demonstrating memory managment. Most/many usages of cuda use some
// abstraction around memory managment, but I think it's important to be
// farmiliar with the low level API.
//
// Make sure to read the common errors at the bottom after looking at this
// code!

__global__ void set_zero(float *dest) { dest[threadIdx.x] = 0.f; }

int main() {
  int64_t size = 128;
  // create a vector for which the values of all entries are 1.5f
  std::vector<float> hostMem(size, 1.5f);

  std::cout << "starting value of hostMem: " << hostMem[5] << "\n";

  float *gpuMem;

  // cuda api functions can return errors, so we need to handle them.
  //
  // A common issue is failing to check errors and running into an issue/crash
  // later when the failure actually occured earlier.
  //
  // Memory managment docs:
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
  //
  // All of these memory managment functions take sizes in bytes. A common
  // mistake is forgetting to multiply by sizeof(T).
  CUDA_ERROR_CHK(cudaMalloc(&gpuMem, size * sizeof(float)));

  CUDA_ERROR_CHK(cudaMemcpy(gpuMem, hostMem.data(), size * sizeof(float),
                            cudaMemcpyHostToDevice));

  // This launches a kernel. The first argument in <<<_, _>>> is the grid
  // dimensions and the second is the block size. Passing scalar integer values
  // results in a 1D kernel. The dim3 struct can be passed instead for higher
  // dimensions.
  //
  // Docs at
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels
  set_zero<<<1, size>>>(gpuMem);

  // Check for errors from the kernel launch itself (invalid block/grid
  // arguments for example).
  CUDA_ERROR_CHK(cudaPeekAtLastError());

  // Kernel invocations are async by default, so we need to syncronize. This is
  // also where errors from running the kernel are reported (like illegal
  // memory access).
  CUDA_ERROR_CHK(cudaDeviceSynchronize());

  CUDA_ERROR_CHK(cudaMemcpy(hostMem.data(), gpuMem, size * sizeof(float),
                            cudaMemcpyDeviceToHost));

  // we have to free memory by hand when using malloc
  CUDA_ERROR_CHK(cudaFree(gpuMem));

  std::cout << "ending value of hostMem: " << hostMem[127] << "\n";
}

// Common cuda errors:
// - Forgetting to multiply by sizeof(T).
// - Passing pointers in wrong order to cudaMemcpy. Results in 'invalid
//   argument' error in debug mode, but might crash or copy garbage in release.
// - Incorrect block/grid sizes.
// - Failing to error check.
// - Not syncronizing after a kernel launch.
// - Forgetting to free memory (typically people use RAII wrappers of some kind:
//   https://en.cppreference.com/w/cpp/language/raii, but we won't set that up
//   here for simplicity)
// - Passing host memory pointers into kernels or trying to dereference device
//   memory in cpu code.^
// - Forgetting to copy from device to host or host to device.^
//
// ^These errors can be avoided with the use of unified memory, but it has
//  worse perfomance characteristics and generally isn't used in the most
//  performant applications.
