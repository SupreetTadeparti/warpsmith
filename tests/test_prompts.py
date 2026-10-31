from warpsmith.problem import load_problem
from warpsmith.prompts import build_messages, format_feedback, parse_response, shape_properties

KERNEL = 'extern "C" void launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {}'

DEVICE = {"name": "Tesla T4", "compute_capability": "7.5", "sm_count": 40, "l2_cache_bytes": 4 << 20,
          "peak_bandwidth_bytes_per_s": 320e9, "peak_fp32_flops": 8.1e12}


def passed_result(ms):
    return {"stage": "passed", "passed": True, "error": "",
            "compile": {"resources": [{"function": "k", "registers": 32, "smem_bytes": 0,
                                       "spill_stores": 0, "spill_loads": 0}]},
            "verify": {"passed": True, "cases_total": 24, "cases_run": 24, "failures": []},
            "bench": {"median_ms": ms, "tflops": 1.0, "gbytes_per_s": 2.0}}


def hidden_failure_result():
    return {"stage": "verify", "passed": False, "error": "x",
            "compile": {"resources": []},
            "verify": {"passed": False, "cases_total": 24, "cases_run": 10, "failures": [{
                "dims": {"M": 1023, "N": 517, "K": 333}, "hidden": True, "distribution": "uniform",
                "round": "fresh", "check": "wrong_result", "message": "12/528891 elements of C outside tolerance",
                "detail": {"max_abs_err": 1.0, "first_bad_index": [1022, 516], "got": 0.0, "expected": 1.0},
                "diagnosis": []}]}}


def test_parse_takes_last_block_with_launch():
    text = f"Plan: tile it.\n```cuda\n__device__ int helper();\n```\nfull file:\n```cuda\n{KERNEL}\n```\n"
    source, rationale = parse_response(text)
    assert source.strip() == KERNEL
    assert rationale.startswith("Plan: tile it.")
    assert "```" not in rationale


def test_parse_accepts_cpp_and_bare_fences():
    assert parse_response(f"```cpp\n{KERNEL}\n```")[0].strip() == KERNEL
    assert parse_response(f"```\n{KERNEL}\n```")[0].strip() == KERNEL


def test_parse_without_code_returns_none():
    source, rationale = parse_response("I would use shared memory.")
    assert source is None and rationale == "I would use shared memory."


def test_shape_properties():
    assert "a dimension equal to 1" in shape_properties({"M": 1, "N": 1, "K": 1})
    assert "a dimension that is not a multiple of 32" in shape_properties({"M": 1023, "N": 517, "K": 333})
    assert "a very skinny (high aspect ratio) shape" in shape_properties({"M": 2048, "N": 3, "K": 1500})


def test_hidden_failure_feedback_does_not_leak_dims():
    text = format_feedback(hidden_failure_result(), best_ms=10.0, baseline_ms=2.0)
    for leaked in ("1023", "517", "333", "1022", "516", "528891"):
        assert leaked not in text
    assert "HIDDEN" in text and "not a multiple of 32" in text
    assert "0.00227% of elements of C outside tolerance" in text


def test_passed_feedback_compares_to_best_and_baseline():
    text = format_feedback(passed_result(5.0), best_ms=10.0, baseline_ms=2.5)
    assert "2.00x" in text and "0.50x" in text


def test_build_messages_includes_context_history_and_failed_source():
    problem = load_problem("matmul")
    best = {"iteration": 0, "source": problem.naive_source, "rationale": "naive", "result": passed_result(100.0)}
    history = [
        {"iteration": 1, "rationale": "tile with shared memory", "source": KERNEL,
         "result": hidden_failure_result(), "new_best": False},
    ]
    system, user = build_messages(problem, DEVICE, best, naive_ms=100.0, baseline_ms=4.0, history=history)
    assert system["role"] == "system" and "```cuda" in system["content"]
    content = user["content"]
    assert "Tesla T4" in content and "320 GB/s" in content
    assert problem.naive_source.strip() in content
    assert "| 1 | tile with shared memory | failed verify (wrong_result) |" in content
    assert KERNEL in content  # the failed attempt's source is shown so it can be fixed
    assert "1023" not in content  # hidden dims never appear
    assert "cuBLAS is not allowed" in content
