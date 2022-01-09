#include <stdio.h>

__global__ void sum_rows(
    const float *src,
    float *dst,
    const uint num_cols,
    const uint num_rows
) {
    const size_t col_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (col_idx >= num_cols) return;

    for (size_t row_idx = 0; row_idx < num_rows; row_idx++) {
        dst[col_idx] += src[num_rows * col_idx + row_idx];
    }

}
