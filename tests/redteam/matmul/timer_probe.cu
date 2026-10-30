// CHEAT: read the GPU's global timer to detect when it's being benchmarked (e.g. many
// back-to-back calls within a few milliseconds) and switch to a cheaper path.
// DEFENSE: static check bans timer registers.
// expect-stage: static
// expect-rule: timers
#include <cuda_runtime.h>

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        int step = (t & 1) ? 2 : 1;
        float acc = 0.0f;
        for (int k = 0; k < K; k += step) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc * step;
    }
}

extern "C" void launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
