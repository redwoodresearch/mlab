// Filters elements of `src` that have magnitude less than `threshold` into `dst`.
// Order is not preserved

// src: input array of size `size`
// dst: output array large enough to hold the output
// counter: should be initialized to zero and will contain the size of dst after
//   execution
__global__ void filter(
    float *src, 
    int32_t size, 
    float threshold, 
    float* dst, 
    int32_t* counter) { 
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) {
    return;
  }
  const float val = src[i];
  if (std::abs(val) >= threshold) {
    return;
  }
  const int32_t dst_location = atomicAdd(counter, 1);
  dst[dst_location] = val;
}
