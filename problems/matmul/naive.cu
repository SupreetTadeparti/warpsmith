// Naive matmul: one thread per output element, reading A and B straight from global memory.
// threadIdx.x walks down rows, so neighbouring threads in a warp read A and write C with a
// stride of K / N floats: the accesses are uncoalesced. That's deliberate, it's the starting point.
#include <cuda_runtime.h>

__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

extern "C" void launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matmul_naive<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
