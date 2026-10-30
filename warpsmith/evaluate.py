"""The evaluation pipeline: static check -> compile -> verify -> benchmark.

Each gate stops the pipeline on failure; `stage` records how far a candidate got.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .compile import DEFAULT_BUILD_DIR, compile_kernel
from .problem import Problem
from .runner import DEFAULT_TIMEOUT, run_worker
from .static_check import check_source

STAGES = ("static", "compile", "run", "verify", "bench", "passed")


@dataclass
class EvalResult:
    stage: str
    passed: bool = False
    error: str = ""
    static_violations: list[str] = field(default_factory=list)
    compile: dict | None = None
    verify: dict | None = None
    bench: dict | None = None
    device: dict | None = None

    @property
    def median_ms(self) -> float | None:
        return self.bench.get("median_ms") if self.passed and self.bench else None

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(
    problem: Problem,
    source: str,
    *,
    bench: bool = True,
    arch: str = "native",
    build_dir: Path | str = DEFAULT_BUILD_DIR,
    timeout: float = DEFAULT_TIMEOUT,
) -> EvalResult:
    violations = check_source(source, problem.allowed_apis)
    if violations:
        return EvalResult("static", error="source failed static checks",
                          static_violations=[str(v) for v in violations])

    comp = compile_kernel(source, build_dir=build_dir, arch=arch)
    comp_info = {k: v for k, v in comp.to_dict().items() if k != "log"}
    if not comp.ok:
        return EvalResult("compile", error=comp.errors or "compilation failed", compile=comp_info)

    res = run_worker({"mode": "candidate", "problem": str(problem.root), "so_path": comp.so_path, "bench": bench},
                     timeout=timeout)
    if res.get("status") != "ok":
        return EvalResult("run", error=res.get("error", "worker failed"), compile=comp_info)

    out = EvalResult("verify", compile=comp_info, verify=res["verify"], device=res.get("device"))
    if not res["verify"]["passed"]:
        out.error = "; ".join(f["message"] for f in res["verify"]["failures"][:3])
        return out
    if not bench:
        out.stage, out.passed = "passed", True
        return out

    out.bench = res.get("bench")
    if not out.bench or not out.bench.get("ok"):
        out.stage = "bench"
        b = out.bench or {}
        out.error = b.get("error") or "; ".join(b.get("integrity_flags", [])) or "benchmark failed"
        return out
    out.stage, out.passed = "passed", True
    return out


def run_baseline(problem: Problem, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Benchmark the problem's library baseline (e.g. cuBLAS via torch) with the same harness."""
    return run_worker({"mode": "baseline", "problem": str(problem.root)}, timeout=timeout)
