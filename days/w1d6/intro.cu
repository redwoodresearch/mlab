

__global__ void zero(float *dest) { 
    dest[threadIdx.x] = 0.f; 
}


__global__ void one(float *dest) { 
    dest[threadIdx.x] = 1.f; 
}
