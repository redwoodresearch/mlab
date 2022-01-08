// Computes a[i] * b[i] + c[i]
__global__ void dot(float *dest, float *a, float *b, float *c) { 
    dest[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x] + c[threadIdx.x];
}
