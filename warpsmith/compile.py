"""Compile a candidate .cu into a shared library with nvcc and parse ptxas resource usage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

SHIM = Path(__file__).resolve().parent / "csrc" / "shim.cu"
DEFAULT_BUILD_DIR = Path(".warpsmith") / "build"


@dataclass
class KernelResources:
    function: str
    registers: int | None = None
    smem_bytes: int = 0
    stack_bytes: int = 0
    spill_stores: int = 0
    spill_loads: int = 0


@dataclass
class CompileResult:
    ok: bool
    so_path: str | None
    seconds: float
    command: list[str]
    log: str
    errors: str = ""
    resources: list[KernelResources] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CompileResult":
        d = dict(d)
        d["resources"] = [KernelResources(**r) for r in d.get("resources", [])]
        return cls(**d)


def find_nvcc() -> str | None:
    found = shutil.which("nvcc")
    if found:
        return found
    for home in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if home and (Path(home) / "bin" / "nvcc").exists():
            return str(Path(home) / "bin" / "nvcc")
    return None


_ENTRY_RE = re.compile(r"Compiling entry function '([^']+)'")
_PROPS_RE = re.compile(r"Function properties for (\S+)")
_FRAME_RE = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_USED_RE = re.compile(r"Used (\d+) registers")
_SMEM_RE = re.compile(r"(\d+) bytes smem")


def parse_ptxas(log: str) -> list[KernelResources]:
    """Extract per-kernel registers/smem/spills from `-Xptxas -v` output."""
    kernels: dict[str, KernelResources] = {}
    current: KernelResources | None = None
    for line in log.splitlines():
        if m := _ENTRY_RE.search(line):
            current = kernels.setdefault(m.group(1), KernelResources(m.group(1)))
        elif m := _PROPS_RE.search(line):
            current = kernels.setdefault(m.group(1), KernelResources(m.group(1)))
        elif current is not None and (m := _FRAME_RE.search(line)):
            current.stack_bytes, current.spill_stores, current.spill_loads = map(int, m.groups())
        elif current is not None and (m := _USED_RE.search(line)):
            current.registers = int(m.group(1))
            if s := _SMEM_RE.search(line):
                current.smem_bytes = int(s.group(1))
    # Device functions get "Function properties" blocks too; keep only kernels that were compiled as entries.
    return [k for k in kernels.values() if k.registers is not None]


def summarize_errors(log: str, max_lines: int = 60) -> str:
    """Drop ptxas info noise so the agent sees the actual diagnostics."""
    keep = [
        ln for ln in log.splitlines()
        if ln.strip() and not ln.startswith("ptxas info") and not _FRAME_RE.search(ln)
    ]
    return "\n".join(keep[:max_lines])


def compile_kernel(
    source: str,
    build_dir: Path | str = DEFAULT_BUILD_DIR,
    arch: str = "native",
    timeout: float = 180.0,
    extra_flags: tuple[str, ...] = (),
) -> CompileResult:
    nvcc = find_nvcc()
    if nvcc is None:
        return CompileResult(False, None, 0.0, [], "nvcc not found", "nvcc not found on PATH or CUDA_HOME")

    flags = ["-O3", f"-arch={arch}", "-shared", "-Xcompiler", "-fPIC", "-Xptxas", "-v", "-lineinfo", *extra_flags]
    key = hashlib.sha256((source + "\0" + " ".join(flags) + "\0" + SHIM.read_text()).encode()).hexdigest()[:16]
    out_dir = Path(build_dir) / key
    cached = out_dir / "compile.json"
    if cached.exists():
        result = CompileResult.from_dict(json.loads(cached.read_text()))
        if not result.ok or (result.so_path and Path(result.so_path).exists()):
            return result

    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / "kernel.cu"
    src.write_text(source)
    so = out_dir / "kernel.so"
    cmd = [nvcc, *flags, "-o", str(so), str(src), str(SHIM)]

    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return CompileResult(False, None, timeout, cmd, "", f"nvcc timed out after {timeout:.0f}s")
    seconds = time.perf_counter() - start

    log = (proc.stdout + proc.stderr).replace(str(src), "kernel.cu")
    ok = proc.returncode == 0 and so.exists()
    result = CompileResult(
        ok=ok,
        so_path=str(so.resolve()) if ok else None,
        seconds=round(seconds, 2),
        command=cmd,
        log=log,
        errors="" if ok else summarize_errors(log),
        resources=parse_ptxas(log),
    )
    cached.write_text(json.dumps(result.to_dict(), indent=2))
    return result
