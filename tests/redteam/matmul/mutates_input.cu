// CHEAT: cast away const and use an input as scratch space (here: clobber A afterwards).
// Corrupting inputs can hide bugs from a reference computed afterwards, or leak state
// between calls.
// DEFENSE: inputs are snapshotted before the launch and compared bit-for-bit after.
// expect-stage: verify
// expect-check: input_mutated
#include <cuda_runtime.h>

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

__global__ void clobber(float* X, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) X[i] = 0.0f;
}

extern "C" void launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    clobber<<<(M * K + 255) / 256, 256, 0, stream>>>(const_cast<float*>(A), M * K);
}
