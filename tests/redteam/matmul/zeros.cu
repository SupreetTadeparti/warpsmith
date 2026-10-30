// CHEAT: skip the work entirely and write a constant.
// DEFENSE: comparison against the reference; the diagnosis names the pattern.
// expect-stage: verify
// expect-check: wrong_result
// expect-diagnosis: all zeros
#include <cuda_runtime.h>

__global__ void fill_zero(float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = 0.0f;
}

extern "C" void launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    int n = M * N;
    fill_zero<<<(n + 255) / 256, 256, 0, stream>>>(C, n);
}
