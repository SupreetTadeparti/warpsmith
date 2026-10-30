"""Run a worker job in an isolated subprocess with a timeout."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_TIMEOUT = 300.0
OUTPUT_TAIL = 4000
PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)


def _tail(output: str | bytes | None) -> str:
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    return (output or "")[-OUTPUT_TAIL:]


def run_worker(job: dict, timeout: float = DEFAULT_TIMEOUT) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (PACKAGE_PARENT, env.get("PYTHONPATH")) if p)
    with tempfile.TemporaryDirectory(prefix="warpsmith-") as d:
        job_path = Path(d) / "job.json"
        out_path = Path(d) / "result.json"
        job_path.write_text(json.dumps(job))
        cmd = [sys.executable, "-m", "warpsmith.worker", str(job_path), str(out_path)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired as e:
            return {
                "status": "timeout",
                "error": f"worker exceeded {timeout:.0f}s (kernel hang or far too slow)",
                "stdout": _tail(e.stdout),
            }
        if not out_path.exists():
            return {
                "status": "crashed",
                "error": f"worker exited with code {proc.returncode} without a result",
                "stderr": _tail(proc.stderr),
            }
        result = json.loads(out_path.read_text())
        result["stdout"] = _tail(proc.stdout)
        return result
