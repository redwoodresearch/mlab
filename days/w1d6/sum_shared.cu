#define SIZE 512
#define NUM_STEPS 8

__global__ void sumShared(float *dest, float *values) {
    __shared__ float buffer[SIZE];
    int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    buffer[idx] = values[idx];
    __syncthreads();
    for (int step = NUM_STEPS; step >= 0; --step) {
        int64_t size = 1 << step;
        if (idx >= size && idx < 2 * size) {
            buffer[idx % size] += buffer[idx];
        }
        __syncthreads();
    }
    if (idx == 0) {
        dest[0] = buffer[0];
    }
}