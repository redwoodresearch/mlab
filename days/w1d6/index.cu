__global__ void index(float *dest, float *a, int aSize, int64_t *b, int bSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < bSize) {
        if (b[idx] < aSize) { dest[idx] = a[b[idx]]; }
        else { printf("Index into a %d is greater than length of a %d\n", b[idx], aSize); }
    }
}