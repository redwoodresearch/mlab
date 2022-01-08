__global__ void filterAtomics(float *dest, float *values, float threshold, int *counter, int inputSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < inputSize) {
        bool positive = (0 <= values[idx]) && (values[idx] < threshold);
        bool negative = (0 > values[idx]) && (-values[idx] < threshold);
        if (positive || negative) {
            dest[atomicAdd(counter, 1)] = values[idx];
        }
    }
}