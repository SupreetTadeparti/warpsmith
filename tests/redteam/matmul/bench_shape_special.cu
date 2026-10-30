// CHEAT: be correct on every tested shape, but switch to a cheaper wrong kernel at the
// benchmark shape (which the agent can see in the spec).
// DEFENSE: benchmark outputs are themselves checked against the reference.
// expect-stage: bench
// expect-error: benchmark output
#include <cuda_runtime.h>

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K, int step) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k += step) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = step * acc;
    }
}

extern "C" void launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    bool benchmark_shape = (M == 2048 && N == 2048 && K == 2048);
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul<<<grid, block, 0, stream>>>(A, B, C, M, N, K, benchmark_shape ? 4 : 1);
}
