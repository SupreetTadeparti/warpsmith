"""End-to-end pipeline tests. Need nvcc and a CUDA GPU (run these on Colab/Kaggle)."""

from pathlib import Path

import pytest

from warpsmith.compile import find_nvcc

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    find_nvcc() is None or not torch.cuda.is_available(), reason="needs nvcc and a CUDA GPU"
)

from warpsmith.evaluate import evaluate, run_baseline  # noqa: E402
from warpsmith.problem import load_problem  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def matmul():
    return load_problem("matmul")


def test_naive_passes(matmul):
    r = evaluate(matmul, matmul.naive_source)
    assert r.passed, r.error
    assert r.bench["integrity_flags"] == []


def test_tiled_passes_and_beats_naive(matmul):
    naive = evaluate(matmul, matmul.naive_source)
    tiled = evaluate(matmul, (ROOT / "examples/matmul/tiled.cu").read_text())
    assert tiled.passed, tiled.error
    assert tiled.median_ms < naive.median_ms


def test_baseline_runs(matmul):
    res = run_baseline(matmul)
    assert res["status"] == "ok", res
    assert res["bench"]["ok"], res["bench"]


def test_compile_error_is_reported(matmul):
    src = matmul.naive_source.replace("acc += ", "acc += undefined_symbol + ")
    r = evaluate(matmul, src)
    assert r.stage == "compile"
    assert "undefined_symbol" in r.error


def test_wrong_result_is_rejected(matmul):
    # Off-by-one in the reduction: skips the last k.
    src = matmul.naive_source.replace("k < K", "k < K - 1")
    r = evaluate(matmul, src, bench=False)
    assert r.stage == "verify"
    assert r.verify["failures"][0]["check"] == "wrong_result"


def test_bad_launch_config_is_reported(matmul):
    src = matmul.naive_source.replace("dim3 block(16, 16);", "dim3 block(64, 64);")  # 4096 threads/block
    r = evaluate(matmul, src, bench=False)
    assert r.stage == "verify"
    assert r.verify["failures"][0]["check"] in ("launch_error", "unwritten_output")
