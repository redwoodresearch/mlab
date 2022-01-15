__global__ void sum_rows(
    float *dest,
    const float *in,
    int64_t ncols,
    int64_t nrows
)
{
    // Each thread sums up one row (removing the row dimension)
    const int64_t col = threadIdx.x + blockIdx.x * blockDim.x;
    if(col >= ncols) { return; }
    float total = 0.f;
    for(int64_t row = 0; row < nrows; ++row) {
        total += in[col * nrows + row];
    }
    dest[col] = total;
}