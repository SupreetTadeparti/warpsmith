from pathlib import Path

import pytest

from warpsmith.static_check import check_source, strip_comments

ROOT = Path(__file__).resolve().parent.parent

LAUNCH = 'extern "C" void launch(const float* A, float* C, int n, cudaStream_t s) { }\n'


def rules(source, allow=()):
    return {v.rule for v in check_source(source, allow)}


@pytest.mark.parametrize("path", ["problems/matmul/naive.cu", "examples/matmul/tiled.cu"])
def test_reference_kernels_are_clean(path):
    assert check_source((ROOT / path).read_text()) == []


def test_missing_launch_is_a_contract_violation():
    assert "contract" in rules("__global__ void k() {}\n")


@pytest.mark.parametrize("snippet,rule", [
    ("#include <cublas_v2.h>\ncublasSgemm(h, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1);\n", "cublas"),
    ("thrust::sort(v.begin(), v.end());\n", "thrust"),
    ("cub::DeviceReduce::Sum(0, n, a, b, c);\n", "cub_device"),
    ("cudaStream_t t; cudaStreamCreate(&t);\n", "streams"),
    ("cudaDeviceSynchronize();\n", "sync"),
    ("cudaMemcpy(C, host, n, cudaMemcpyHostToDevice);\n", "host_memory"),
    ("static float* cache; cudaMalloc(&cache, 4);\n", "device_alloc"),
    ('FILE* f = fopen("answers.bin", "rb");\n', "io"),
    ('std::ifstream in("answers.bin");\n', "io"),
    ('const char* p = getenv("X");\n', "io"),
    ('asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));\n', "timers"),
    ("long long t = clock64();\n", "timers"),
    ("int ws_last_error(void) { return 0; }\n", "harness_symbols"),
    ("#include <vector>\n", "includes"),
])
def test_banned_patterns(snippet, rule):
    assert rule in rules(snippet + LAUNCH)


def test_comments_are_ignored():
    src = "// cublasSgemm would be cheating\n/* cudaDeviceSynchronize(); */\n" + LAUNCH
    assert check_source(src) == []


def test_strip_comments_preserves_line_numbers():
    src = "a\n/* x\ny */\nb // c\n"
    assert strip_comments(src).count("\n") == src.count("\n")


def test_violation_reports_line_number():
    v = check_source("#include <cuda_runtime.h>\n\ncudaDeviceSynchronize();\n" + LAUNCH)
    assert [(x.rule, x.line) for x in v] == [("sync", 3)]


def test_problem_can_allow_a_rule():
    src = "cudaMalloc(&p, 4);\n" + LAUNCH
    assert "device_alloc" in rules(src)
    assert "device_alloc" not in rules(src, allow=["device_alloc"])


def test_warp_and_block_primitives_are_allowed():
    src = ("#include <cooperative_groups.h>\n"
           "__device__ float f(float v) { return __shfl_down_sync(0xffffffff, v, 1); }\n" + LAUNCH)
    assert check_source(src) == []
