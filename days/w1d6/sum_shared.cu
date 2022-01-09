#define SIZE 512
#define NUM_STEPS 8
#define BLOCKS 10

__global__ void sumShared(float *dest, float *values) {
    __shared__ float buffer[BLOCKS * SIZE];
    int64_t arrayIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int64_t subarrayIdx = arrayIdx % blockDim.x;
    buffer[arrayIdx] = values[arrayIdx];
    __syncthreads();
    for (int step = NUM_STEPS; step >= 0; --step) {
        int64_t size = 1 << step;
        if (subarrayIdx >= size && subarrayIdx < 2 * size) {
            buffer[(subarrayIdx % size) + blockDim.x * blockIdx.x] += buffer[arrayIdx];
        }
        __syncthreads();
    }
    if (arrayIdx % blockDim.x == 0) {
        dest[blockIdx.x] = buffer[arrayIdx];
    }
}