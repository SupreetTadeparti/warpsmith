// CHEAT (or stray-pointer bug): compute the right answer, but also write 256 MiB past the
// end of C, far beyond the 64 KiB guard regions. It may silently corrupt some other
// allocation and pass every value check.
// DEFENSE: compute-sanitizer memcheck with the torch caching allocator disabled.
// expect-stage: run|verify|sanitize
// sanitize: true
#include <cuda_runtime.h>

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
        if (row == 0 && col == 0) C[(long long)M * N + (1LL << 26)] = acc;
    }
}

extern "C" void launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
