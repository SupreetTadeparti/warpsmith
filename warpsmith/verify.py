"""Correctness and anti-cheat verification of a candidate kernel. Runs inside the worker.

Every case runs twice on the *same* buffers: once with fresh data, then again after
overwriting the inputs in place. A kernel that caches results keyed on pointers
(or computes once and replays) fails the second round. Each trial also:
  - fills outputs with a NaN/sentinel so skipped elements are detected,
  - surrounds every buffer with canary bytes to catch out-of-bounds writes,
  - snapshots inputs to catch kernels that scribble on them,
  - compares against a float64 reference with a magnitude-aware tolerance.
Hidden shapes are never shown to the agent; failures on them are reported with
`hidden: True` so the feedback layer can redact the dims.
"""

from __future__ import annotations

import secrets

import torch

from .kernel_lib import KernelLib
from .problem import Problem

GUARD_BYTES = 1 << 16
CANARY = 0xA5
MAX_FAILURES = 5

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
}


class GuardedTensor:
    """A tensor view with canary-filled guard regions on both sides."""

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        n = 1
        for d in shape:
            n *= d
        self.nbytes = n * torch.empty((), dtype=dtype).element_size()
        self.raw = torch.empty(self.nbytes + 2 * GUARD_BYTES, dtype=torch.uint8, device="cuda")
        self.raw[:GUARD_BYTES].fill_(CANARY)
        self.raw[GUARD_BYTES + self.nbytes:].fill_(CANARY)
        self.tensor = self.raw[GUARD_BYTES:GUARD_BYTES + self.nbytes].view(dtype).view(shape)

    def guard_status(self) -> str | None:
        if not bool((self.raw[:GUARD_BYTES] == CANARY).all()):
            return "before"
        if not bool((self.raw[GUARD_BYTES + self.nbytes:] == CANARY).all()):
            return "after"
        return None


def fill_sentinel(t: torch.Tensor) -> None:
    if t.is_floating_point():
        t.fill_(float("nan"))
    else:
        t.fill_(torch.iinfo(t.dtype).min)


def unwritten_mask(t: torch.Tensor) -> torch.Tensor:
    if t.is_floating_point():
        return torch.isnan(t)
    return t == torch.iinfo(t.dtype).min


def compare(out: torch.Tensor, expected: torch.Tensor, scale: torch.Tensor, rtol: float, atol: float) -> dict:
    """Elementwise |out - expected| <= atol + rtol * scale, computed in float64."""
    got = out.double()
    exp = expected.double()
    err = (got - exp).abs()
    bound = atol + rtol * scale.double()
    both_nan = torch.isnan(got) & torch.isnan(exp)
    same_inf = torch.isinf(got) & torch.isinf(exp) & (torch.sign(got) == torch.sign(exp))
    bad = ~((err <= bound) | both_nan | same_inf)
    n_bad = int(bad.sum())
    info = {"ok": n_bad == 0, "mismatches": n_bad, "fraction": n_bad / max(bad.numel(), 1)}
    if n_bad:
        finite = torch.where(torch.isfinite(err), err, torch.zeros_like(err))
        idx = tuple(int(i) for i in bad.nonzero()[0])
        info.update({
            "max_abs_err": float(finite.max()),
            "first_bad_index": list(idx),
            "got": float(got[idx]),
            "expected": float(exp[idx]),
        })
    return info


def diagnose(out: torch.Tensor, expected: torch.Tensor, scale, rtol, atol, previous: torch.Tensor | None) -> list[str]:
    """Explain a tolerance failure in terms of common wrong-kernel patterns."""
    notes = []
    o = out.double()
    if bool((o == 0).all()):
        notes.append("output is all zeros")
    elif o.numel() > 1 and bool((o == o.flatten()[0]).all()):
        notes.append("output is a single constant value")
    if previous is not None and compare(out, previous, scale, rtol, atol)["ok"]:
        notes.append("output matches the result for the PREVIOUS inputs at the same pointers (stale/cached result)")
    if expected.dim() == 2 and expected.shape[0] == expected.shape[1]:
        if compare(out, expected.t(), scale.t(), rtol, atol)["ok"]:
            notes.append("output equals the transpose of the expected result")
    return notes


def case_numel(problem: Problem, dims: dict[str, int]) -> int:
    total = 0
    for a in problem.tensor_args:
        n = 1
        for d in a.resolve_shape(dims):
            n *= d
        total += n
    return total


def build_cases(problem: Problem, max_case_numel: int | None = None,
                distributions: list[str] | None = None) -> list[dict]:
    """All (shape, distribution) cases. The filters exist for slow modes like compute-sanitizer."""
    cases = []
    for hidden, shapes in ((False, problem.visible_shapes), (True, problem.hidden_shapes)):
        for dims in shapes:
            if max_case_numel is not None and case_numel(problem, dims) > max_case_numel:
                continue
            for dist in distributions or problem.distributions:
                cases.append({"dims": dict(dims), "hidden": hidden, "distribution": dist})
    return cases


def verify(problem: Problem, ref, kernel: KernelLib, stream: torch.cuda.Stream, *,
           max_case_numel: int | None = None, distributions: list[str] | None = None) -> dict:
    cases = build_cases(problem, max_case_numel, distributions)
    failures: list[dict] = []
    cases_run = 0
    fatal = False
    scale_fn = getattr(ref, "tolerance_scale", None)

    for case in cases:
        dims = case["dims"]
        seed = secrets.randbits(63)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        bufs = {a.name: GuardedTensor(a.resolve_shape(dims), DTYPES[a.dtype]) for a in problem.tensor_args}
        views = {name: b.tensor for name, b in bufs.items()}
        previous_expected: dict | None = None
        cases_run += 1

        for rnd in ("fresh", "same_pointers_new_data"):
            def fail(check: str, message: str, **extra) -> None:
                failures.append({**case, "seed": seed, "round": rnd, "check": check, "message": message, **extra})

            new_inputs = ref.make_inputs(dims, gen, case["distribution"])
            for a in problem.inputs:
                views[a.name].copy_(new_inputs[a.name])
            for a in problem.outputs:
                fill_sentinel(views[a.name])
            snapshot = {a.name: views[a.name].clone() for a in problem.inputs}
            inputs_view = {a.name: views[a.name] for a in problem.inputs}
            expected = ref.reference(inputs_view, dims)
            scale = scale_fn(inputs_view, dims) if scale_fn else {k: v.abs() for k, v in expected.items()}
            torch.cuda.synchronize()

            with torch.cuda.stream(stream):
                kernel(views, dims, stream.cuda_stream)
            launch_err = kernel.last_error()
            try:
                torch.cuda.synchronize()
            except RuntimeError as e:
                fail("runtime_error", f"CUDA error during kernel execution: {e}")
                fatal = True  # the CUDA context is now unusable
                break
            if launch_err:
                fail("launch_error", f"kernel launch failed: {launch_err}")
                break

            case_failed = False
            for name, b in bufs.items():
                where = b.guard_status()
                if where:
                    fail("out_of_bounds_write", f"wrote outside buffer {name} ({where} its start/end)")
                    case_failed = True
            for a in problem.inputs:
                if not torch.equal(views[a.name], snapshot[a.name]):
                    fail("input_mutated", f"input {a.name} was modified; inputs are read-only")
                    case_failed = True
            for a in problem.outputs:
                out, exp = views[a.name], expected[a.name]
                unwritten = unwritten_mask(out) & ~unwritten_mask(exp.to(out.dtype))
                n_unwritten = int(unwritten.sum())
                if n_unwritten:
                    first = [int(i) for i in unwritten.nonzero()[0]]
                    fail("unwritten_output",
                         f"{n_unwritten}/{out.numel()} elements of {a.name} were never written",
                         first_unwritten_index=first)
                    case_failed = True
                    continue
                result = compare(out, exp, scale[a.name], problem.rtol, problem.atol)
                if not result["ok"]:
                    prev = previous_expected[a.name] if previous_expected else None
                    fail("wrong_result",
                         f"{result['mismatches']}/{out.numel()} elements of {a.name} outside tolerance",
                         detail=result,
                         diagnosis=diagnose(out, exp, scale[a.name], problem.rtol, problem.atol, prev))
                    case_failed = True
            if case_failed:
                break
            previous_expected = expected

        if fatal or len(failures) >= MAX_FAILURES:
            break

    return {
        "passed": not failures,
        "cases_total": len(cases),
        "cases_run": cases_run,
        "fatal": fatal,
        "failures": failures,
    }
