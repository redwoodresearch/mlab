#include <stdio.h>

__global__ void sum(
    const float *arr,
    const uint size,
    float *total
) {
    const size_t i = threadIdx.x + blockIdx.x * 512;
    if (i < size) {
        atomicAdd(total, arr[i]);
    }
}
