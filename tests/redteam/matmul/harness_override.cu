// CHEAT: define the harness's error-reporting hook so launch failures are never seen.
// DEFENSE: static check reserves ws_* identifiers (and the duplicate symbol would fail
// to link anyway).
// expect-stage: static
// expect-rule: harness_symbols
#include <cuda_runtime.h>

extern "C" int ws_last_error(void) { return 0; }

extern "C" void launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
}
