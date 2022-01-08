

__global__ void filter_atomic(float* dest, float* input, int inputLen, int* ptr, float threshold) { 
    int readLocation = blockIdx.x * blockDim.x + threadIdx.x;
    if (readLocation < inputLen) {
        float value = input[readLocation];
        if (abs(value) < threshold) {
            int destOffset = atomicAdd(ptr, 1);
            dest[destOffset] = value;
        }
    }
}


// count how many are true within block
// atomic add by that
// block-level cumulative sum to assign within block
