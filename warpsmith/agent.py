"""The optimization loop: propose -> evaluate -> feed back -> keep the best.

A candidate only becomes the new best if it passes every check, beats the current
best by more than `min_improvement` (benchmark noise is ~0.2-1%), and then passes a
second, independent verification under compute-sanitizer.

Run directory layout:
    attempts.jsonl          one record per attempt (results, tokens, timings)
    attempt_NNN.cu          each candidate's source (000 is the naive starting point)
    attempt_NNN.prompt.md   the prompt that produced it
    attempt_NNN.response.md the raw model response (+ reasoning, if returned)
    best.cu, summary.json   written at the end
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .evaluate import evaluate, run_baseline
from .llm import LLMClient, LLMError
from .problem import Problem
from .prompts import build_messages, outcome_label, parse_response


@dataclass
class AgentConfig:
    iterations: int = 10
    patience: int | None = None  # stop after this many attempts without a new best
    min_improvement: float = 0.01
    sanitize_best: bool = True
    eval_timeout: float = 300.0


class Agent:
    def __init__(
        self,
        problem: Problem,
        llm: LLMClient,
        run_dir: Path | str,
        config: AgentConfig | None = None,
        *,
        evaluate_fn: Callable = evaluate,
        baseline_fn: Callable = run_baseline,
        log: Callable[[str], None] = print,
    ):
        self.problem = problem
        self.llm = llm
        self.run_dir = Path(run_dir)
        self.config = config or AgentConfig()
        self.evaluate = evaluate_fn
        self.run_baseline = baseline_fn
        self.log = log

    def _write(self, name: str, text: str) -> None:
        (self.run_dir / name).write_text(text, encoding="utf-8")

    def _record(self, attempt: dict) -> None:
        with open(self.run_dir / "attempts.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(attempt, default=str) + "\n")

    def run(self) -> dict:
        cfg = self.config
        self.run_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()

        self.log("benchmarking library baseline...")
        base = self.run_baseline(self.problem)
        baseline_ms = (base.get("bench") or {}).get("median_ms") if base.get("status") == "ok" else None

        self.log("evaluating naive starting kernel...")
        naive_result = self.evaluate(self.problem, self.problem.naive_source, timeout=cfg.eval_timeout).to_dict()
        if not naive_result["passed"]:
            raise RuntimeError(f"the naive kernel fails the harness: {naive_result['error']}")
        naive_ms = naive_result["bench"]["median_ms"]
        device = base.get("device") or naive_result.get("device") or {}

        best = {"iteration": 0, "rationale": "naive starting kernel", "source": self.problem.naive_source,
                "result": naive_result, "new_best": True}
        self._write("attempt_000.cu", best["source"])
        self._record(best)
        self.log(f"  naive {naive_ms:.4f} ms" + (f", baseline {baseline_ms:.4f} ms" if baseline_ms else ""))

        history: list[dict] = []
        since_best = 0
        prompt_tokens = completion_tokens = 0

        for i in range(1, cfg.iterations + 1):
            tag = f"attempt_{i:03d}"
            messages = build_messages(self.problem, device, best, naive_ms, baseline_ms, history)
            self._write(f"{tag}.prompt.md", "\n\n".join(f"## {m['role']}\n\n{m['content']}" for m in messages))

            source, rationale, llm_info = None, "", {}
            try:
                resp = self.llm.complete(messages)
            except LLMError as e:
                result = {"stage": "llm_error", "passed": False, "error": str(e)}
            else:
                prompt_tokens += resp.prompt_tokens
                completion_tokens += resp.completion_tokens
                llm_info = {"seconds": resp.seconds, "prompt_tokens": resp.prompt_tokens,
                            "completion_tokens": resp.completion_tokens, "finish_reason": resp.finish_reason}
                self._write(f"{tag}.response.md", resp.text + (
                    f"\n\n---\n## reasoning\n\n{resp.reasoning}" if resp.reasoning else ""))
                source, rationale = parse_response(resp.text)
                if source is None:
                    error = 'no ```cuda code block defining launch() found in the response'
                    if resp.finish_reason == "length":
                        error += " (the response hit the token limit; keep the plan short)"
                    result = {"stage": "parse", "passed": False, "error": error}
                else:
                    self._write(f"{tag}.cu", source)
                    result = self.evaluate(self.problem, source, timeout=cfg.eval_timeout).to_dict()

            new_best = False
            best_ms = best["result"]["bench"]["median_ms"]
            if result["passed"] and result["bench"]["median_ms"] < best_ms * (1 - cfg.min_improvement):
                if cfg.sanitize_best:
                    self.log(f"  [{i}] candidate new best ({result['bench']['median_ms']:.4f} ms); running memcheck...")
                    check = self.evaluate(self.problem, source, bench=False, sanitize=True,
                                          timeout=cfg.eval_timeout).to_dict()
                    if not check["passed"]:
                        result = {**result, "stage": check["stage"], "passed": False, "error": check["error"],
                                  "verify": check.get("verify") or result.get("verify"),
                                  "sanitize": check.get("sanitize")}
                new_best = result["passed"]

            attempt = {"iteration": i, "rationale": rationale, "source": source, "result": result,
                       "new_best": new_best, "llm": llm_info}
            self._record(attempt)
            history.append(attempt)
            if new_best:
                best = attempt
                since_best = 0
            else:
                since_best += 1

            current = best["result"]["bench"]["median_ms"]
            self.log(f"  [{i}/{cfg.iterations}] {outcome_label(result):<28}"
                     f"{' NEW BEST' if new_best else '         '}  best {current:.4f} ms "
                     f"({naive_ms / current:.2f}x naive)  | {' '.join(rationale.split())[:70]}")
            if cfg.patience and since_best >= cfg.patience:
                self.log(f"stopping: no improvement in {cfg.patience} attempts")
                break

        best_ms = best["result"]["bench"]["median_ms"]
        self._write("best.cu", best["source"])
        stages: dict[str, int] = {}
        for h in history:
            stages[h["result"]["stage"]] = stages.get(h["result"]["stage"], 0) + 1
        summary = {
            "problem": self.problem.name,
            "model": self.llm.model,
            "device": device.get("name"),
            "iterations_run": len(history),
            "baseline_ms": baseline_ms,
            "naive_ms": naive_ms,
            "best_ms": best_ms,
            "best_iteration": best["iteration"],
            "speedup_vs_naive": naive_ms / best_ms,
            "fraction_of_baseline": baseline_ms / best_ms if baseline_ms else None,
            "outcomes": stages,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "wall_seconds": round(time.time() - started),
            "attempts": [{"iteration": h["iteration"], "outcome": outcome_label(h["result"]),
                          "median_ms": (h["result"].get("bench") or {}).get("median_ms") if h["result"]["passed"] else None,
                          "new_best": h["new_best"]} for h in history],
        }
        self._write("summary.json", json.dumps(summary, indent=2))
        return summary
