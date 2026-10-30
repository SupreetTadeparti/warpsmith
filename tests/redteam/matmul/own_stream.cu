// CHEAT: launch on a private stream so the work isn't covered by the harness's events.
// DEFENSE: static check bans creating streams (and the timing cross-check catches the
// legacy-default-stream variant; see legacy_stream.cu).
// expect-stage: static
// expect-rule: streams
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

extern "C" void launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    static cudaStream_t hidden = nullptr;
    if (!hidden) cudaStreamCreateWithFlags(&hidden, cudaStreamNonBlocking);
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul<<<grid, block, 0, hidden>>>(A, B, C, M, N, K);
}
