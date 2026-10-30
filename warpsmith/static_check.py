"""Source-level checks run before compiling a candidate.

These catch the cheap ways to "win" without writing a kernel: calling a vendor
library, escaping the timed stream, reading precomputed answers from disk, or
touching the harness's own symbols. Dynamic checks in verify.py and bench.py
catch the rest; this layer exists so the agent gets a fast, specific message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ALLOWED_INCLUDES = {
    "cuda_runtime.h", "cuda.h", "cuda_fp16.h", "cuda_bf16.h", "mma.h",
    "cooperative_groups.h", "cooperative_groups/reduce.h", "cooperative_groups/memcpy_async.h",
    "cuda_pipeline.h", "cuda/pipeline", "cuda/barrier", "math_constants.h",
    "cstdint", "stdint.h", "cmath", "math.h", "cfloat", "float.h", "climits", "limits.h",
    "type_traits",
}

# (rule id, pattern, message). A problem's spec can allow a rule by id.
RULES: list[tuple[str, re.Pattern, str]] = [
    ("cublas", re.compile(r"\bcublas\w*"), "cuBLAS is not allowed; write the kernel yourself"),
    ("cudnn", re.compile(r"\bcudnn\w*"), "cuDNN is not allowed"),
    ("thrust", re.compile(r"\bthrust::"), "Thrust is not allowed"),
    ("cub_device", re.compile(r"\bcub::Device\w*"),
     "device-wide CUB algorithms are not allowed (block/warp primitives are fine)"),
    ("streams", re.compile(
        r"\bcuda(StreamCreate\w*|EventCreate\w*|LaunchHostFunc|StreamAddCallback|Graph\w*)\b"),
     "launch only on the provided stream; creating streams, events or graphs is not allowed"),
    ("sync", re.compile(r"\bcuda(DeviceSynchronize|StreamSynchronize|EventSynchronize)\b"),
     "the harness synchronizes; do not synchronize inside launch()"),
    ("host_memory", re.compile(r"\bcuda(Memcpy\w*|MallocHost|HostAlloc|MallocManaged)\b"),
     "host/device copies and host or managed allocations are not allowed"),
    ("device_alloc", re.compile(r"\bcuda(Malloc|MallocAsync|Free|FreeAsync)\b"),
     "allocating device memory is not allowed for this problem"),
    ("io", re.compile(
        r"\b(fopen|fread|fwrite|fscanf|open|read|mmap|system|popen|getenv|setenv|putenv|dlopen|dlsym|fork|exec\w*)\s*\("
        r"|\b(std::)?[io]?fstream\b"),
     "file, process and environment access are not allowed"),
    ("timers", re.compile(r"%globaltimer|%clock(64)?\b|\bclock(64)?\s*\("),
     "reading GPU timers is not allowed"),
    ("harness_symbols", re.compile(r"\bws_\w+"), "identifiers starting with ws_ are reserved by the harness"),
]

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
LAUNCH_RE = re.compile(r'extern\s+"C"\s+(__host__\s+)?void\s+launch\s*\(')


@dataclass(frozen=True)
class Violation:
    rule: str
    line: int
    snippet: str
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: [{self.rule}] {self.message}: {self.snippet}"


def strip_comments(source: str) -> str:
    """Blank out comments but keep line structure so line numbers stay accurate."""
    def blank(m: re.Match) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))
    return re.sub(r"//[^\n]*|/\*.*?\*/", blank, source, flags=re.DOTALL)


def check_source(source: str, allow: list[str] | tuple[str, ...] = ()) -> list[Violation]:
    code = strip_comments(source)
    lines = code.splitlines()
    violations: list[Violation] = []

    def line_of(pos: int) -> int:
        return code.count("\n", 0, pos) + 1

    for m in INCLUDE_RE.finditer(code):
        header = m.group(1).strip()
        if header not in ALLOWED_INCLUDES and "includes" not in allow:
            ln = line_of(m.start())
            violations.append(Violation("includes", ln, lines[ln - 1].strip(),
                                        f"#include <{header}> is not on the allowlist"))

    for rule, pattern, message in RULES:
        if rule in allow:
            continue
        for m in pattern.finditer(code):
            ln = line_of(m.start())
            violations.append(Violation(rule, ln, lines[ln - 1].strip(), message))

    if not LAUNCH_RE.search(code):
        violations.append(Violation("contract", 0, "",
                                    'must define extern "C" void launch(...) with the problem signature'))

    return sorted(violations, key=lambda v: v.line)
