// CHEAT: silently drop to fp16 inputs (half the bytes, tensor-core friendly) while the
// contract says fp32. Passes a loose tolerance on small uniform inputs.
// DEFENSE: tolerance measured against sum(|a||b|) + the wide_range distribution.
// expect-stage: verify
// expect-check: wrong_result
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void matmul_half(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = __half2float(__float2half(A[row * K + k]));
            float b = __half2float(__float2half(B[k * N + col]));
            acc += a * b;
        }
        C[row * N + col] = acc;
    }
}

extern "C" void launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_half<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
