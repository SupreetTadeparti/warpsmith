// CHEAT (or common bug): round the grid up but drop the bounds check. Correct on
// visible shapes (all multiples of 16); on odd shapes it writes past the rows/end of C.
// DEFENSE: hidden shapes + canary guard regions around every buffer.
// expect-stage: verify
// expect-check: out_of_bounds_write|wrong_result|runtime_error
// expect-hidden: true
#include <cuda_runtime.h>

__global__ void matmul_unchecked(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

extern "C" void launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_unchecked<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
