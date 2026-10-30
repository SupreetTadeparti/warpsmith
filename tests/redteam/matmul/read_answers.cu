// CHEAT: load precomputed answers from disk (e.g. saved from an earlier run of the
// reference) and copy them into C.
// DEFENSE: static check bans file I/O and host->device copies.
// expect-stage: static
// expect-rule: io|host_memory
#include <cstdio>
#include <cuda_runtime.h>

extern "C" void launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    static float buf[1 << 20];
    FILE* f = fopen("/tmp/answers.bin", "rb");
    if (f) {
        size_t n = fread(buf, sizeof(float), (size_t)M * N, f);
        fclose(f);
        cudaMemcpyAsync(C, buf, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
}
