// Linked into every candidate .so so the harness can read launch errors
// (bad grid/block config, too much shared memory) from the same CUDA runtime
// instance the candidate used. Candidates may not define ws_* symbols.
#include <cuda_runtime.h>

extern "C" int ws_last_error(void) { return (int)cudaGetLastError(); }

extern "C" const char* ws_error_string(int code) {
    return cudaGetErrorString((cudaError_t)code);
}
