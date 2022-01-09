#include <stdio.h>

__global__ void zero(double *dest) {
    dest[threadIdx.x] = 0.f;
    // std::cout <<blockIdx.x <<" "<< threadIdx.x;
    printf("Hello from block %d, thread %d, grid %d\n", blockIdx.x, threadIdx.x, blockIdx.z);
}


__global__ void one(float *dest) {
    dest[threadIdx.x] = 1.f;
}
