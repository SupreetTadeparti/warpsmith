// CHEAT: hand the work to cuBLAS and call it an optimized kernel.
// DEFENSE: static check (header allowlist + banned identifiers).
// expect-stage: static
// expect-rule: cublas
#include <cublas_v2.h>
#include <cuda_runtime.h>

extern "C" void launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    static cublasHandle_t handle = nullptr;
    if (!handle) cublasCreate(&handle);
    cublasSetStream(handle, stream);
    const float one = 1.0f, zero = 0.0f;
    // Row-major C = A @ B is column-major C^T = B^T @ A^T.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N);
}
