constexpr size_t BLOCK_SIZE = 512;
__global__ void sum_shared(int32_t* in, int64_t size, int32_t* out) { 
  __shared__ int32_t buf[BLOCK_SIZE];
  const size_t offset = BLOCK_SIZE * blockIdx.x;
  const size_t i = threadIdx.x;
  for (size_t i = 0; i < BLOCK_SIZE; i++) {
    const size_t in_idx = offset + i;
    buf[i] = in_idx < size ? in[in_idx] : 0;
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

__global__ void sum_block_shfl(int32_t* in, int64_t size, int32_t* out) { 
  const size_t offset = BLOCK_SIZE * blockIdx.x;
  const size_t i = threadIdx.x;
  const size_t in_idx = offset + i;
  int32_t val = in_idx < size ? in[in_idx] : 0;
  for (size_t gap = BLOCK_SIZE / 2; gap > 0; gap /= 2) {
    val += __shfl_down_sync(0xffffffff, val, gap);
  }
  if (i == 0) {
    out[blockIdx.x] = val;
  }
}
