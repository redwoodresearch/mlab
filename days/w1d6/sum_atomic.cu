__global__ void sumAtomic(float *values, float *sum, int inputSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < inputSize) {
        atomicAdd(sum, values[idx]);
    }
}

