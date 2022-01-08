__global__ void sumRows(float *sum, float *values, int rows, int cols) {
    int rowIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowIdx < rows) {
        float sum_temp = 0;
        for (int i = 0; i < cols; i++) {
            sum_temp = sum_temp + values[rowIdx*cols+i];
        }
        sum[rowIdx] = sum_temp;
    }
}

