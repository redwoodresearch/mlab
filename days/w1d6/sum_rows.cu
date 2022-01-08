// in: num_cols x num_rows
// dst[i] = sum(in[i][])
__global__ void sum_rows(float* in, int64_t num_cols, int64_t num_rows, float*
    dst) { 
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= num_cols) {
    return;
  }
  float total = 0.0;
  for (int64_t i = num_rows * col; i < num_rows * (col + 1); i++) {
    total += in[i];
  }
  dst[col] = total;
}
