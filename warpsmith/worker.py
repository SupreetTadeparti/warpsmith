"""Subprocess entry point: `python -m warpsmith.worker job.json result.json`.

Everything that touches the GPU happens here, one process per job, so an illegal
memory access or a hung kernel only takes down this process, never the agent.
Results go to a file rather than stdout because candidates may printf.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

N_BENCH_SETS = 3


def _bench_inputs(problem, ref, gen):
    from .verify import DTYPES
    import torch

    dims = problem.bench_shape
    sets = []
    for _ in range(N_BENCH_SETS):
        tensors = dict(ref.make_inputs(dims, gen, "uniform"))
        for a in problem.outputs:
            tensors[a.name] = torch.empty(a.resolve_shape(dims), dtype=DTYPES[a.dtype], device="cuda")
        sets.append(tensors)
    return dims, sets


def _check_bench_outputs(problem, sets, expected, when: str) -> dict | None:
    from .verify import compare, unwritten_mask

    for i, tensors in enumerate(sets):
        exp, scale = expected[i]
        for a in problem.outputs:
            out = tensors[a.name]
            n_unwritten = int(unwritten_mask(out).sum())
            if n_unwritten:
                return {"ok": False, "error": f"benchmark output {a.name} for input set {i} {when}: "
                                              f"{n_unwritten}/{out.numel()} elements never written"}
            r = compare(out, exp[a.name], scale[a.name], problem.rtol, problem.atol)
            if not r["ok"]:
                return {"ok": False, "error": f"benchmark output {a.name} for input set {i} {when} is wrong",
                        "detail": r}
    return None


def _run_bench(problem, ref, launch_on_set, info, check_kernel=None) -> dict:
    """Time `launch_on_set(tensors, dims)`; afterwards re-check every set's output."""
    import secrets
    import torch

    from .bench import add_throughput, integrity_flags, make_flush_buffer, time_launches
    from .verify import fill_sentinel

    gen = torch.Generator(device="cuda").manual_seed(secrets.randbits(63))
    dims, sets = _bench_inputs(problem, ref, gen)
    stream = torch.cuda.Stream()
    flush = make_flush_buffer(info["l2_cache_bytes"])

    stats = time_launches(lambda i: launch_on_set(sets[i], dims, stream), len(sets), stream, flush)
    if check_kernel is not None and (err := check_kernel()):
        return {"ok": False, "error": f"launch error during benchmark: {err}"}

    scale_fn = getattr(ref, "tolerance_scale", None)
    expected = []
    for tensors in sets:
        inputs = {a.name: tensors[a.name] for a in problem.inputs}
        exp = ref.reference(inputs, dims)
        expected.append((exp, scale_fn(inputs, dims) if scale_fn else {k: v.abs() for k, v in exp.items()}))

    # The benchmarked outputs must be correct too: a kernel may behave differently at
    # the benchmark shape, or once it has been called more times than verification does.
    if failure := _check_bench_outputs(problem, sets, expected, "after timed runs"):
        return failure
    # Outputs left over from warmup can look correct even if timed calls skipped work,
    # so poison them and launch once more.
    for tensors in sets:
        for a in problem.outputs:
            fill_sentinel(tensors[a.name])
    with torch.cuda.stream(stream):
        for tensors in sets:
            launch_on_set(tensors, dims, stream)
    torch.cuda.synchronize()
    if failure := _check_bench_outputs(problem, sets, expected, "on a fresh launch after benchmarking"):
        return failure

    flops, nbytes = ref.flops(dims), ref.bytes_moved(dims)
    flags = integrity_flags(stats, flops, nbytes, info["peak_bandwidth_bytes_per_s"], info["peak_fp32_flops"])
    return {"ok": not flags, "dims": dims, "integrity_flags": flags, **add_throughput(stats, flops, nbytes)}


def handle(job: dict) -> dict:
    import torch

    from .gpu_specs import device_info
    from .problem import load_problem

    if not torch.cuda.is_available():
        return {"status": "error", "error": "CUDA is not available in the worker"}
    info = device_info()
    if job["mode"] == "info":
        return {"status": "ok", "device": info}

    problem = load_problem(job["problem"])
    ref = problem.reference_module()
    result: dict = {"status": "ok", "device": info}

    if job["mode"] == "baseline":
        def launch(tensors, dims, stream):
            inputs = {a.name: tensors[a.name] for a in problem.inputs}
            outputs = {a.name: tensors[a.name] for a in problem.outputs}
            ref.baseline(inputs, outputs, dims)
        result["bench"] = _run_bench(problem, ref, launch, info)
        return result

    from .kernel_lib import ContractError, KernelLib
    from .verify import verify

    try:
        kernel = KernelLib(job["so_path"], problem)
    except (ContractError, AttributeError, OSError) as e:
        return {"status": "error", "error": f"could not load kernel: {e}"}

    stream = torch.cuda.Stream()
    result["verify"] = verify(problem, ref, kernel, stream,
                              max_case_numel=job.get("max_case_numel"), distributions=job.get("distributions"))
    if result["verify"]["passed"] and job.get("bench", True):
        result["bench"] = _run_bench(
            problem, ref,
            lambda tensors, dims, s: kernel(tensors, dims, s.cuda_stream),
            info, check_kernel=kernel.last_error,
        )
    return result


def main(argv: list[str]) -> int:
    job_path, out_path = Path(argv[1]), Path(argv[2])
    try:
        result = handle(json.loads(job_path.read_text()))
    except Exception as e:  # report, don't crash: the parent needs a structured answer
        result = {"status": "error", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    out_path.write_text(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
