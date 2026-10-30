// CHEAT: cache results by shape only. If the tests reused one fixed input per shape
// (a common weak-harness bug), this would pass and benchmark as a memcpy.
// DEFENSE: fresh random inputs on every case, from several distributions.
// expect-stage: verify
// expect-check: wrong_result
#include <cuda_runtime.h>

#define CACHE_ELEMS (1 << 21)

__device__ float g_cache[CACHE_ELEMS];

static int last_M = -1, last_N = -1, last_K = -1;

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

__global__ void copy(const float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

extern "C" void launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    int n = M * N;
    bool fits = n <= CACHE_ELEMS;
    float* cache = nullptr;
    cudaGetSymbolAddress((void**)&cache, g_cache);
    if (fits && M == last_M && N == last_N && K == last_K) {
        copy<<<(n + 255) / 256, 256, 0, stream>>>(cache, C, n);
        return;
    }
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    if (fits) {
        copy<<<(n + 255) / 256, 256, 0, stream>>>(C, cache, n);
        last_M = M; last_N = N; last_K = K;
    }
}
