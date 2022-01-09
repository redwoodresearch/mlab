#include <stdio.h>

__global__ void abc(float *dest, const float *a, const float *b, const float *c) {
    const size_t i = threadIdx.x;
    dest[i] = a[i] * b[i] + c[i];
}
