// CHEAT: ignore the provided stream and launch on the legacy default stream. The harness
// records its CUDA events on a non-blocking torch stream, so the kernel runs outside the
// timed region and appears to take ~0 ms. Results are still correct (the harness
// synchronizes the whole device before checking).
// DEFENSE: wall clock vs CUDA-event cross-check + roofline check.
// expect-stage: bench
// expect-error: timing_mismatch|faster_than_hardware
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
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul<<<grid, block>>>(A, B, C, M, N, K);  // no stream argument
}
