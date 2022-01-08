// Writes sum of `in` to `out[0]`.
// in: input array of size `size`
// out: output array of size 1
__global__ void sum(int32_t *in, int32_t size, int32_t *out) { 
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    atomicAdd(out, in[i]);
  }
}
