"""Agent loop with a scripted LLM and a fake evaluator (no GPU or API key needed)."""

import json

from warpsmith.agent import Agent, AgentConfig
from warpsmith.evaluate import EvalResult
from warpsmith.llm import LLMError, LLMResponse
from warpsmith.problem import load_problem

DEVICE = {"name": "Tesla T4", "compute_capability": "7.5", "sm_count": 40}


def kernel(marker: str) -> str:
    return f'// {marker}\nextern "C" void launch(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {{}}\n'


def reply(marker: str, plan: str = "plan") -> str:
    return f"{plan}\n```cuda\n{kernel(marker)}```"


class ScriptedLLM:
    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, messages):
        self.prompts.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMResponse(text=r, prompt_tokens=100, completion_tokens=50, seconds=1.0, finish_reason="stop")


def passed(ms):
    return EvalResult("passed", passed=True, compile={"resources": []},
                      verify={"passed": True, "cases_total": 24, "cases_run": 24, "failures": []},
                      bench={"ok": True, "median_ms": ms, "tflops": 1.0, "gbytes_per_s": 1.0}, device=DEVICE)


class FakeEvaluator:
    """Times are encoded in the source: '// MS=<float>'; '// MEMBUG' fails only under sanitize."""

    def __init__(self):
        self.calls = []

    def __call__(self, problem, source, *, bench=True, sanitize=False, timeout=None, **_):
        self.calls.append({"bench": bench, "sanitize": sanitize})
        if "COMPILE_ERROR" in source:
            return EvalResult("compile", error="kernel.cu(3): error: expected a ';'")
        if sanitize and "MEMBUG" in source:
            return EvalResult("sanitize", error="compute-sanitizer found 1 memory error(s)",
                              sanitize={"ok": False, "error": "1 error", "report": "Invalid write"})
        ms = 100.0
        for token in source.split():
            if token.startswith("MS="):
                ms = float(token[3:])
        return passed(ms)


def baseline(problem):
    return {"status": "ok", "device": DEVICE, "bench": {"median_ms": 4.0}}


def run_agent(tmp_path, replies, **config):
    llm = ScriptedLLM(replies)
    evaluator = FakeEvaluator()
    agent = Agent(load_problem("matmul"), llm, tmp_path / "run", AgentConfig(**config),
                  evaluate_fn=evaluator, baseline_fn=baseline, log=lambda _: None)
    return agent.run(), llm, evaluator


def test_keeps_best_and_ignores_noise_level_gains(tmp_path):
    summary, _, _ = run_agent(tmp_path, [reply("MS=50"), reply("MS=49.9"), reply("MS=60")], iterations=3)
    assert summary["best_ms"] == 50.0
    assert summary["best_iteration"] == 1  # 49.9 is within the 1% noise threshold
    assert summary["speedup_vs_naive"] == 2.0
    assert summary["fraction_of_baseline"] == 4.0 / 50.0


def test_new_best_must_pass_sanitizer(tmp_path):
    summary, _, evaluator = run_agent(tmp_path, [reply("MS=10 MEMBUG"), reply("MS=20")], iterations=2)
    assert summary["best_ms"] == 20.0
    assert summary["outcomes"] == {"sanitize": 1, "passed": 1}
    assert sum(c["sanitize"] for c in evaluator.calls) == 2


def test_failures_are_fed_back(tmp_path):
    _, llm, _ = run_agent(tmp_path, [reply("COMPILE_ERROR", plan="try vectorized loads"), reply("MS=50")],
                          iterations=2)
    second_prompt = llm.prompts[1][1]["content"]
    assert "COMPILE ERROR" in second_prompt
    assert "expected a ';'" in second_prompt
    assert "| 1 | try vectorized loads | compile error |" in second_prompt
    assert "COMPILE_ERROR" in second_prompt  # failed source is included so it can be fixed


def test_unparseable_and_llm_errors_do_not_stop_the_run(tmp_path):
    summary, _, _ = run_agent(tmp_path, ["no code here", LLMError("HTTP 503"), reply("MS=50")], iterations=3)
    assert summary["outcomes"] == {"parse": 1, "llm_error": 1, "passed": 1}
    assert summary["best_ms"] == 50.0


def test_patience_stops_early(tmp_path):
    summary, _, _ = run_agent(tmp_path, [reply("MS=200")] * 5, iterations=5, patience=2)
    assert summary["iterations_run"] == 2


def test_run_directory_contents(tmp_path):
    run_agent(tmp_path, [reply("MS=50")], iterations=1)
    run = tmp_path / "run"
    for name in ("attempt_000.cu", "attempt_001.cu", "attempt_001.prompt.md", "attempt_001.response.md",
                 "best.cu", "summary.json", "attempts.jsonl"):
        assert (run / name).exists(), name
    assert "MS=50" in (run / "best.cu").read_text()
    records = [json.loads(line) for line in (run / "attempts.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in records] == [0, 1]
