from warpsmith.compile import parse_ptxas, summarize_errors
from warpsmith.gpu_specs import lookup_peaks

PTXAS_LOG = """\
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z12matmul_tiledPKfS0_Pfiii' for 'sm_75'
ptxas info    : Function properties for _Z12matmul_tiledPKfS0_Pfiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 32 registers, 8192 bytes smem, 384 bytes cmem[0]
ptxas info    : Compiling entry function '_Z4heavyPf' for 'sm_75'
ptxas info    : Function properties for _Z4heavyPf
    48 bytes stack frame, 24 bytes spill stores, 16 bytes spill loads
ptxas info    : Used 255 registers, used 1 barriers, 360 bytes cmem[0]
"""


def test_parse_ptxas_resources():
    tiled, heavy = parse_ptxas(PTXAS_LOG)
    assert (tiled.function, tiled.registers, tiled.smem_bytes, tiled.spill_stores) == (
        "_Z12matmul_tiledPKfS0_Pfiii", 32, 8192, 0)
    assert (heavy.registers, heavy.smem_bytes, heavy.stack_bytes, heavy.spill_stores, heavy.spill_loads) == (
        255, 0, 48, 24, 16)


def test_summarize_errors_drops_ptxas_noise():
    log = PTXAS_LOG + 'kernel.cu(12): error: identifier "foo" is undefined\n'
    assert summarize_errors(log) == 'kernel.cu(12): error: identifier "foo" is undefined'


def test_lookup_peaks():
    bw, flops = lookup_peaks("Tesla T4")
    assert bw == 320e9 and flops == 8.1e12
    assert lookup_peaks("Some Future GPU") == (None, None)
