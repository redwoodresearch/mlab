__device__ void sum_atomic(const float *inp, float *dest, int size) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= size) {
    return;
  }
  atomicAdd(dest, inp[i]);
}