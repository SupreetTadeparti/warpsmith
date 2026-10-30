"""End-to-end pipeline tests. Need nvcc and a CUDA GPU (run these on Colab/Kaggle)."""

from pathlib import Path

from warpsmith.evaluate import evaluate, run_baseline

ROOT = Path(__file__).resolve().parent.parent


def test_naive_passes(matmul_gpu):
    r = evaluate(matmul_gpu, matmul_gpu.naive_source)
    assert r.passed, r.error
    assert r.bench["integrity_flags"] == []


def test_tiled_passes_and_beats_naive(matmul_gpu):
    naive = evaluate(matmul_gpu, matmul_gpu.naive_source)
    tiled = evaluate(matmul_gpu, (ROOT / "examples/matmul/tiled.cu").read_text())
    assert tiled.passed, tiled.error
    assert tiled.median_ms < naive.median_ms


def test_tiled_passes_sanitizer(matmul_gpu):
    r = evaluate(matmul_gpu, (ROOT / "examples/matmul/tiled.cu").read_text(), bench=False, sanitize=True)
    assert r.passed, r.error
    assert r.sanitize["errors"] == 0


def test_baseline_runs(matmul_gpu):
    res = run_baseline(matmul_gpu)
    assert res["status"] == "ok", res
    assert res["bench"]["ok"], res["bench"]


def test_compile_error_is_reported(matmul_gpu):
    src = matmul_gpu.naive_source.replace("acc += ", "acc += undefined_symbol + ")
    r = evaluate(matmul_gpu, src)
    assert r.stage == "compile"
    assert "undefined_symbol" in r.error


def test_wrong_result_is_rejected(matmul_gpu):
    # Off-by-one in the reduction: skips the last k.
    src = matmul_gpu.naive_source.replace("k < K", "k < K - 1")
    r = evaluate(matmul_gpu, src, bench=False)
    assert r.stage == "verify"
    assert r.verify["failures"][0]["check"] == "wrong_result"


def test_bad_launch_config_is_reported(matmul_gpu):
    src = matmul_gpu.naive_source.replace("dim3 block(16, 16);", "dim3 block(64, 64);")  # 4096 threads/block
    r = evaluate(matmul_gpu, src, bench=False)
    assert r.stage == "verify"
    assert r.verify["failures"][0]["check"] in ("launch_error", "unwritten_output")
