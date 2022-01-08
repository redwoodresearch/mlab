

__global__ void zero(float *dest) { 
    dest[threadIdx.x] = 0.f; 
}


__global__ void one(float *dest) { 
    dest[threadIdx.x] = 1.f; 
}

__global__ void WriteThreadX(float *dest) { 
    dest[threadIdx.x] = threadIdx.x; 
}
