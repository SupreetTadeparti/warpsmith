// Shared-memory tiled matmul. Each 32x32 block stages a tile of A and a tile of B in shared
// memory, so every global element is loaded once per tile instead of once per multiply-add.
// threadIdx.x maps to columns, so global loads and the store to C are coalesced.
// Out-of-range tile elements are zero-filled, which handles shapes that aren't multiples of 32.
#include <cuda_runtime.h>

#define TILE 32

__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                             int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.0f;
        Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

extern "C" void launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
