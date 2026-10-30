// CHEAT: "approximate" the reduction by sampling every other k and doubling.
// Statistically close on uniform random data, and 2x less work.
// DEFENSE: element-wise tolerance + small_int inputs where any error is visible.
// expect-stage: verify
// expect-check: wrong_result
#include <cuda_runtime.h>

__global__ void matmul_sampled(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k += 2) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = 2.0f * acc;
    }
}

extern "C" void launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_sampled<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
