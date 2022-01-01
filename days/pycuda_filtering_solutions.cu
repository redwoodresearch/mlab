template <typename T, typename Pred>
__device__ void filter_atomic_overall(const T *inp, T *dest, int *atomic_v,
                                      int size, const Pred pred) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= size) {
    return;
  }
  if (pred(inp[i])) {
    dest[atomicAdd(atomic_v, 1)] = inp[i];
  }
}

extern "C" {
__global__ void filter_atomic_kernel(const float *inp, float *dest,
                                     int *atomic_v, int size,
                                     const float thresh) {
  filter_atomic_overall(inp, dest, atomic_v, size, [thresh](const float x) {
    return std::fmod(std::abs(x), 1.f) < thresh;
  });
}
}
