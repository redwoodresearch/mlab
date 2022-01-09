#include <stdio.h>

__global__ void index(
    const float *a,
    const int64_t *b,
    float *dst,
    const uint size
) {
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= size) return;

    if (b[i] >= size) {
        printf("Error!");
        return;
    }
    dst[i] = a[b[i]];
}
