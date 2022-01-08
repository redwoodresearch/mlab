constexpr size_t BLOCK_SIZE = 512;
constexpr size_t WARP_SIZE = 32;
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
  __shared__ int32_t buf[BLOCK_SIZE / WARP_SIZE];
  const size_t offset = BLOCK_SIZE * blockIdx.x;
  const size_t i = threadIdx.x;
  const size_t in_idx = offset + i;
  int32_t val = in_idx < size ? in[in_idx] : 0;

  val += __shfl_down_sync(0xffffffff, val, 16); // 16 == WARP_SIZE / 2
  val += __shfl_down_sync(0x0000ffff, val, 8);
  val += __shfl_down_sync(0x000000ff, val, 4);
  val += __shfl_down_sync(0x0000000f, val, 2);
  val += __shfl_down_sync(0x00000003, val, 1);


  if (i % WARP_SIZE == 0) {
    buf[i / WARP_SIZE] = val;
  }
  __syncthreads();
  if (i >= BLOCK_SIZE / WARP_SIZE) {
    return;
  }
  val = buf[i];
  val += __shfl_down_sync(0x0000ffff, val, 8);
  val += __shfl_down_sync(0x000000ff, val, 4);
  val += __shfl_down_sync(0x0000000f, val, 2);
  val += __shfl_down_sync(0x00000003, val, 1);
  if (i == 0) {
    out[blockIdx.x] = val;
  }
}
