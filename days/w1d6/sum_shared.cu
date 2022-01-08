constexpr size_t BLOCK_SIZE = 512;
__global__ void sum_shared(int32_t* in, int32_t* out) { 
  __shared__ int32_t buf[BLOCK_SIZE];
  const size_t offset = BLOCK_SIZE * blockIdx.x;
  const size_t i = threadIdx.x;
  for (size_t i = 0; i < BLOCK_SIZE; i++) {
    buf[i] = in[offset + i];
  }
  __syncthreads();
  for (size_t gap = 256; gap > 0; gap /= 2) {
    if (i < gap) {
      buf[i] += buf[i + gap];
    }
    __syncthreads();
  }
  if (i == 0) {
    out[blockIdx.x] = buf[0];
  }
}
