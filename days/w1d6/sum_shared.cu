#include <stdio.h>

__global__ void sum_512(
    const float *arr,
    const uint size,
    float *totals
) {
    const int t = threadIdx.x;
    const int b = blockIdx.x;

    __shared__ float s[512];
    s[t] = 0;

    const uint idx = 512 * b + t;
    if (idx < size) s[t] = arr[512 * b + t];
    __syncthreads();

    for (int shift = 256; shift > 0; shift /= 2) {
        if (t < shift) {
            s[t] += s[t + shift];
        }
        __syncthreads();
    }

    totals[b] = s[0];
}
