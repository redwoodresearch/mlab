__global__ void abc(float *dest, float *a, float *b, float *c) { 
    dest[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x] + c[threadIdx.x]; 
}

