__global__ void index(
    float *dest,
    const float* a,
    const int64_t *b,
    int64_t aSize,
    int64_t bSize
)
{
    int64_t i = threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if(i >= bSize) {
        return;
    }
    int64_t aIdx = b[i];
    if(aIdx < 0 || aIdx >= aSize) {
        printf("\naIdx out of range: %d", aIdx);
        return;
    }
    dest[i] = a[aIdx];
}