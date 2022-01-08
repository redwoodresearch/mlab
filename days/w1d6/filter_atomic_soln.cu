__global__ void filter_atomic(const float *inp, float *dest, int *counter, int size, const float threshold) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= size) {
    return;
  }
  if (std::abs(inp[i]) < threshold) {
    dest[atomicAdd(counter, 1)] = inp[i];
  }
}