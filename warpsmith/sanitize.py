"""Run verification under compute-sanitizer memcheck.

Guard regions in verify.py only catch out-of-bounds writes that land within 64 KiB of a
buffer. A write far past the end can land silently inside another allocation, so
candidates that are about to become the new best also run under memcheck. The torch
caching allocator is disabled for that run so every tensor is its own cudaMalloc and
memcheck can see the allocation boundaries. Memcheck is ~100x slower, so only small
cases and one input distribution are run.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from .compile import find_cuda_tool
from .problem import Problem
from .runner import run_worker

SANITIZE_MAX_CASE_NUMEL = 1 << 20
SANITIZE_TIMEOUT = 900.0
REPORT_LINES = 40

_SUMMARY_RE = re.compile(r"ERROR SUMMARY:\s*(\d+) error")


def parse_report(log: str) -> tuple[int | None, str]:
    """(error count, the first error lines) from a compute-sanitizer log."""
    m = _SUMMARY_RE.search(log)
    count = int(m.group(1)) if m else None
    lines = [ln.removeprefix("=========").rstrip() for ln in log.splitlines() if ln.startswith("=========")]
    lines = [ln for ln in lines if ln.strip() and "COMPUTE-SANITIZER" not in ln]
    return count, "\n".join(lines[:REPORT_LINES])


def run_sanitizer(problem: Problem, so_path: str, timeout: float = SANITIZE_TIMEOUT) -> dict:
    tool = find_cuda_tool("compute-sanitizer")
    if tool is None:
        return {"ok": False, "skipped": True, "error": "compute-sanitizer not found"}

    with tempfile.TemporaryDirectory(prefix="warpsmith-sanitize-") as d:
        log_path = Path(d) / "sanitizer.log"
        wrapper = [tool, "--tool", "memcheck", "--target-processes", "all",
                   "--print-limit", "20", "--log-file", str(log_path)]
        job = {
            "mode": "candidate",
            "problem": str(problem.root),
            "so_path": so_path,
            "bench": False,
            "max_case_numel": SANITIZE_MAX_CASE_NUMEL,
            "distributions": problem.distributions[:1],
        }
        res = run_worker(job, timeout=timeout, wrapper=wrapper,
                         extra_env={"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"})
        log = log_path.read_text(errors="replace") if log_path.exists() else ""

    count, report = parse_report(log)
    verify_ok = res.get("status") == "ok" and res.get("verify", {}).get("passed", False)
    ok = verify_ok and count == 0
    out = {"ok": ok, "errors": count, "report": report}
    if not ok:
        if count:
            out["error"] = f"compute-sanitizer found {count} memory error(s)"
        elif res.get("status") != "ok":
            out["error"] = f"worker under compute-sanitizer: {res.get('error', res.get('status'))}"
        elif not verify_ok:
            out["error"] = "verification failed under compute-sanitizer: " + "; ".join(
                f["message"] for f in res["verify"]["failures"][:3])
        else:
            out["error"] = "compute-sanitizer produced no summary"
    return out
