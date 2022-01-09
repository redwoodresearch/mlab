#include <stdio.h>

__global__ void filter(
    const float *src,
    const uint size,
    float *dst,
    const float thresh,
    int *counter
) {
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) return;

    if (std::abs(src[i]) < thresh) {
        dst[atomicAdd(counter, 1)] = src[i];
    }
}
