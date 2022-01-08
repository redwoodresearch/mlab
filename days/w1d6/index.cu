// Writes dst[i] = a[b[i]]
// Requires len(dst) == len(b) == b_size
__global__ void index(
    int32_t* a, 
    int64_t a_size,
    int64_t* b, 
    int64_t b_size,
    int32_t* dst) {
  const size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx >= b_size) {
    return;
  }
  const size_t a_idx = b[thread_idx];
  if (a_idx >= a_size) {
    printf("b[%lu]=%lu is out of bounds (a_size=%lu)\n", thread_idx, a_idx,
        (size_t)a_size);
    return;
  }
  dst[thread_idx] = a[a_idx];
}
