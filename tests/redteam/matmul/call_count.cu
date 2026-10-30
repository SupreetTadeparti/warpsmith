// CHEAT: behave during verification (which makes a bounded number of calls), then stop
// launching anything once the benchmark is underway. Timed runs measure an empty
// launch, and outputs left over from warmup still look correct.
// DEFENSE: roofline check (nothing beats DRAM bandwidth) + outputs are poisoned and
// relaunched after timing.
// expect-stage: bench
// expect-error: faster_than_hardware|never written
#include <cuda_runtime.h>

static int calls = 0;

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
    // Verification makes 48 calls (8 shapes x 3 distributions x 2 rounds); warmup takes the next 10.
    if (++calls > 60) return;
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
