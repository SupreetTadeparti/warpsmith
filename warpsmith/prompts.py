"""Prompt construction and response parsing for the optimization agent.

Every iteration builds a fresh two-message prompt instead of growing a chat: problem
and GPU context, the current best kernel, a one-line-per-attempt history, and full
feedback on the most recent attempt. Context stays bounded no matter how many
iterations run.

Feedback never reveals hidden test shapes: failures on them are reported by the
properties of the shape (e.g. "a dimension is not a multiple of 32"), not its values.
"""

from __future__ import annotations

import re

from .problem import Problem
from .static_check import ALLOWED_INCLUDES, RULES

MAX_HISTORY_ROWS = 30
MAX_IDEA_CHARS = 110
MAX_RATIONALE_CHARS = 1500

SYSTEM_PROMPT = """\
You are an expert CUDA performance engineer. You iteratively optimize a CUDA kernel.
Each turn you write ONE complete new version of the kernel source file. An automated
harness then compiles it with nvcc, checks it for correctness on many shapes (including
hidden ones you never see), runs anti-cheating checks, benchmarks it, and reports back.

Correctness comes first: a fast kernel that fails any check is worthless. The harness
detects skipped work, cached or stale results, hardcoded outputs, reduced precision,
out-of-bounds accesses, work done outside the provided stream, and library calls.
Don't try to game it; write a genuinely faster kernel.

Respond in exactly this format:
1. A short plan (at most 5 sentences): what you change and why it should be faster,
   referring to the feedback.
2. Exactly one ```cuda fenced code block containing the COMPLETE .cu file, including
   the extern "C" launch function. No other code blocks.
"""


def _fmt_bytes(n: int | None) -> str:
    if not n:
        return "unknown"
    for unit, size in (("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= size:
            return f"{n / size:g} {unit}"
    return f"{n} B"


def gpu_section(device: dict) -> str:
    lines = [f"- {device.get('name')} (compute capability {device.get('compute_capability')}), "
             f"{device.get('sm_count')} SMs"]
    if device.get("shared_memory_per_block_bytes"):
        lines.append(f"- shared memory: {_fmt_bytes(device['shared_memory_per_block_bytes'])} per block "
                     f"(default limit), {_fmt_bytes(device.get('shared_memory_per_sm_bytes'))} per SM")
    if device.get("registers_per_sm"):
        lines.append(f"- {device['registers_per_sm']} registers and "
                     f"{device.get('max_threads_per_sm')} threads per SM")
    lines.append(f"- L2 cache: {_fmt_bytes(device.get('l2_cache_bytes'))}")
    if device.get("peak_bandwidth_bytes_per_s"):
        lines.append(f"- peak DRAM bandwidth: {device['peak_bandwidth_bytes_per_s'] / 1e9:.0f} GB/s")
    if device.get("peak_fp32_flops"):
        lines.append(f"- peak FP32 (no tensor cores): {device['peak_fp32_flops'] / 1e12:.1f} TFLOP/s")
    return "\n".join(lines)


def rules_section(problem: Problem) -> str:
    lines = [
        f"- Keep exactly this entry point: {problem.signature}",
        "- Launch all work on the `stream` argument; launch() must not synchronize.",
        "- Inputs are read-only. Write every output element on every call.",
        "- Allowed headers: " + ", ".join(sorted(ALLOWED_INCLUDES)),
    ]
    lines += [f"- {msg}." for rule, _, msg in RULES if rule not in problem.allowed_apis]
    return "\n".join(lines)


def shape_properties(dims: dict[str, int]) -> list[str]:
    """Describe a hidden shape without revealing it."""
    props = []
    values = list(dims.values())
    if any(v == 1 for v in values):
        props.append("a dimension equal to 1")
    for m in (32, 16, 4):
        if any(v % m for v in values):
            props.append(f"a dimension that is not a multiple of {m}")
            break
    if min(values) * 64 <= max(values):
        props.append("a very skinny (high aspect ratio) shape")
    if any(v > 4096 for v in values):
        props.append("a dimension larger than 4096")
    return props or ["an ordinary-looking shape"]


_COUNT_RE = re.compile(r"(\d+)/(\d+) elements")


def _redact_failure(f: dict) -> str:
    if f["hidden"]:
        where = "a HIDDEN test shape with " + ", ".join(shape_properties(f["dims"]))
    else:
        where = f"shape {f['dims']}"
    line = f"- [{f['check']}] on {where}, input distribution `{f['distribution']}`"
    if f["round"] != "fresh":
        line += ", on the second call with new data at the same pointers"
    message = f["message"]
    if f["hidden"]:
        # "12/528891 elements" would reveal the hidden shape's size; report a percentage.
        message = _COUNT_RE.sub(lambda m: f"{100 * int(m.group(1)) / int(m.group(2)):.3g}% of elements", message)
    line += f": {message}"
    detail = f.get("detail") or {}
    if detail and not f["hidden"]:
        line += (f" (max abs error {detail.get('max_abs_err', 0):.3g}; first bad index "
                 f"{detail.get('first_bad_index')}: got {detail.get('got'):.6g}, expected {detail.get('expected'):.6g})")
    if f.get("first_unwritten_index") and not f["hidden"]:
        line += f" (first unwritten index {f['first_unwritten_index']})"
    for note in f.get("diagnosis", []):
        line += f"\n  - looks like: {note}"
    return line


def resources_line(compile_info: dict | None) -> str:
    if not compile_info:
        return ""
    parts = []
    for r in compile_info.get("resources", []):
        spills = r["spill_stores"] + r["spill_loads"]
        parts.append(f"{r['function']}: {r['registers']} registers, {r['smem_bytes']} B static smem"
                     + (f", {spills} B register SPILLS" if spills else ""))
    return "; ".join(parts)


def format_feedback(result: dict, best_ms: float | None, baseline_ms: float | None) -> str:
    """Human-readable outcome of one evaluation, safe to show the agent."""
    stage = result["stage"]
    if stage == "parse":
        return f"REJECTED before compiling: {result['error']}"
    if stage == "llm_error":
        return f"No kernel: the model request failed ({result['error']})"
    if stage == "static":
        return "REJECTED by the static source check:\n" + "\n".join(f"- {v}" for v in result["static_violations"])
    if stage == "compile":
        return "COMPILE ERROR:\n" + (result.get("error") or "")
    out = []
    if result.get("compile"):
        out.append(f"Compiled. ptxas: {resources_line(result['compile'])}")
    if stage == "run":
        out.append(f"CRASHED or TIMED OUT while running: {result['error']}")
        return "\n".join(out)
    v = result.get("verify") or {}
    if stage == "verify":
        out.append(f"FAILED correctness checks ({len(v.get('failures', []))} failure(s) in "
                   f"{v.get('cases_run')}/{v.get('cases_total')} cases run):")
        out += [_redact_failure(f) for f in v.get("failures", [])]
        return "\n".join(out)
    out.append(f"Passed all {v.get('cases_total')} correctness cases.")
    if stage == "sanitize":
        s = result.get("sanitize") or {}
        out.append(f"FAILED compute-sanitizer memcheck: {s.get('error')}\n{s.get('report', '')}")
        return "\n".join(out)
    b = result.get("bench") or {}
    if stage == "bench":
        out.append(f"REJECTED during benchmarking: {result.get('error')}")
        return "\n".join(out)
    ms = b["median_ms"]
    line = f"Benchmark: {ms:.4f} ms median ({b.get('tflops', 0):.3f} TFLOP/s, {b.get('gbytes_per_s', 0):.1f} GB/s)"
    if best_ms:
        line += f"; previous best {best_ms:.4f} ms ({best_ms / ms:.2f}x)"
    if baseline_ms:
        line += f"; library baseline {baseline_ms:.4f} ms ({baseline_ms / ms:.2f}x of its speed)"
    out.append(line)
    return "\n".join(out)


def outcome_label(result: dict) -> str:
    stage = result["stage"]
    if stage == "passed":
        return f"{result['bench']['median_ms']:.4f} ms"
    if stage == "verify":
        failures = (result.get("verify") or {}).get("failures") or [{}]
        return f"failed verify ({failures[0].get('check', '?')})"
    return {"parse": "no code block", "llm_error": "llm error", "static": "static check",
            "compile": "compile error", "run": "crashed/timeout", "sanitize": "memcheck error",
            "bench": "benchmark rejected"}.get(stage, stage)


def idea_of(rationale: str) -> str:
    text = " ".join(rationale.split())
    return text[:MAX_IDEA_CHARS] + ("..." if len(text) > MAX_IDEA_CHARS else "")


def build_messages(
    problem: Problem,
    device: dict,
    best: dict,
    naive_ms: float | None,
    baseline_ms: float | None,
    history: list[dict],
) -> list[dict]:
    """`best` and each history entry are attempt records from the agent."""
    best_ms = best["result"]["bench"]["median_ms"]
    head = [
        "# Task",
        problem.description,
        "",
        f"Visible test shapes: {problem.visible_shapes}",
        "There are also HIDDEN test shapes you will never see, which can include any positive sizes "
        "(non-multiples of any tile size, dimensions of 1, very skinny shapes).",
        f"Benchmark shape (what you are optimizing for): {problem.bench_shape}",
        f"Tolerance: |out - ref| <= {problem.atol:g} + {problem.rtol:g} * scale, elementwise, "
        f"against a float64 reference.",
        "",
        "# GPU",
        gpu_section(device),
        "",
        "# Rules (violations are rejected automatically)",
        rules_section(problem),
        "",
        "# Current best kernel",
        f"{best_ms:.4f} ms"
        + (" (this is still the naive starting kernel)" if best["iteration"] == 0 else "")
        + (f", {naive_ms / best_ms:.2f}x faster than the naive starting kernel"
           if naive_ms and best["iteration"] != 0 else "")
        + (f", {baseline_ms / best_ms:.2f}x of the library baseline's speed ({baseline_ms:.4f} ms)" if baseline_ms else "")
        + ".",
        f"ptxas: {resources_line(best['result'].get('compile'))}",
        "```cuda",
        best["source"].rstrip(),
        "```",
    ]

    if history:
        head += ["", "# Attempt history (oldest first)", "| # | idea | outcome |", "|---|---|---|"]
        for h in history[-MAX_HISTORY_ROWS:]:
            mark = " (new best)" if h.get("new_best") else ""
            head.append(f"| {h['iteration']} | {idea_of(h['rationale']) or '-'} | {outcome_label(h['result'])}{mark} |")

        last = history[-1]
        head += ["", f"# Feedback on the last attempt (#{last['iteration']})",
                 format_feedback(last["result"], best_ms, baseline_ms)]
        if not last["result"].get("passed") and last.get("source") and last["source"] != best["source"]:
            head += ["", "Source of that attempt:", "```cuda", last["source"].rstrip(), "```"]

    head += ["", "# Your turn",
             "Write the next version. Build on the current best kernel (or fix the last attempt if it was "
             "a promising idea that failed). Try a different idea if the same one keeps failing."]
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(head)}]


_FENCE_RE = re.compile(r"```[ \t]*(cuda|cu|cpp|c\+\+|c|cuh)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_response(text: str) -> tuple[str | None, str]:
    """(kernel source or None, rationale). Takes the last fenced block that defines launch()."""
    blocks = [(m.start(), m.group(2)) for m in _FENCE_RE.finditer(text)]
    with_launch = [(pos, b) for pos, b in blocks if re.search(r"\blaunch\s*\(", b)]
    if not with_launch:
        return None, text.strip()[:MAX_RATIONALE_CHARS]
    pos, source = with_launch[-1]
    rationale = _FENCE_RE.sub("", text).strip()
    return source.strip() + "\n", rationale[:MAX_RATIONALE_CHARS]
