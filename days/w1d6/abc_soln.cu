__global__ void mul_add(
    float *dest, 
    const float *a, 
    const float *b, 
    const float *c) 
{
    const int i = threadIdx.x;
    dest[i] = a[i] * b[i] + c[i];
}