"""Command line interface.

    python -m warpsmith info
    python -m warpsmith eval matmul path/to/kernel.cu [--no-bench] [--no-compare] [--json out.json]
    python -m warpsmith baseline matmul
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import EvalResult, evaluate, run_baseline
from .problem import load_problem
from .runner import run_worker


def _fmt_resources(compile_info: dict) -> str:
    parts = []
    for r in compile_info.get("resources", []):
        spills = r["spill_stores"] + r["spill_loads"]
        parts.append(f"{r['function']}: {r['registers']} regs, {r['smem_bytes']} B smem"
                     + (f", {spills} B SPILLS" if spills else ""))
    return "; ".join(parts)


def _fmt_bench(b: dict) -> str:
    s = f"{b['median_ms']:.4f} ms (IQR {b['p25_ms']:.4f}-{b['p75_ms']:.4f}, {b['reps']} reps)"
    if b.get("tflops") is not None:
        s += f"  {b['tflops']:.3f} TFLOP/s  {b['gbytes_per_s']:.1f} GB/s"
    return s


def print_result(name: str, r: EvalResult) -> None:
    print(f"== {name}")
    if r.stage == "static":
        print("[static]  FAIL")
        for v in r.static_violations:
            print(f"          {v}")
        return
    print("[static]  ok")
    if r.compile:
        status = "ok" if r.compile["ok"] else "FAIL"
        print(f"[compile] {status} ({r.compile['seconds']}s)  {_fmt_resources(r.compile)}")
    if r.stage == "compile":
        print(r.error)
        return
    if r.stage == "run":
        print(f"[run]     FAIL  {r.error}")
        return
    v = r.verify
    if v:
        print(f"[verify]  {'PASS' if v['passed'] else 'FAIL'}  {v['cases_run']}/{v['cases_total']} cases")
        for f in v["failures"]:
            # The human sees hidden dims; the agent-facing feedback (milestone 3) will redact them.
            where = f"dims={f['dims']}" + (" (hidden)" if f["hidden"] else "")
            print(f"          [{f['check']}] {where} dist={f['distribution']} round={f['round']}: {f['message']}")
            for note in f.get("diagnosis", []):
                print(f"            -> {note}")
    if r.bench:
        if r.bench.get("ok"):
            print(f"[bench]   {_fmt_bench(r.bench)}")
        else:
            print(f"[bench]   FAIL  {r.error}")


def cmd_info(_args) -> int:
    res = run_worker({"mode": "info"})
    print(json.dumps(res.get("device", res), indent=2))
    return 0 if res.get("status") == "ok" else 1


def cmd_baseline(args) -> int:
    problem = load_problem(args.problem)
    res = run_baseline(problem)
    print(json.dumps(res.get("bench", res), indent=2))
    return 0 if res.get("status") == "ok" else 1


def cmd_eval(args) -> int:
    problem = load_problem(args.problem)
    source = Path(args.kernel).read_text()
    result = evaluate(problem, source, bench=not args.no_bench, arch=args.arch)
    print_result(args.kernel, result)

    report = {"candidate": result.to_dict()}
    if result.passed and result.bench and not args.no_compare:
        naive = evaluate(problem, problem.naive_source, arch=args.arch)
        base = run_baseline(problem).get("bench") or {}
        print("== comparison")
        if naive.median_ms:
            print(f"  naive     {naive.median_ms:.4f} ms   speedup vs naive: {naive.median_ms / result.median_ms:.2f}x")
        if base.get("median_ms"):
            print(f"  baseline  {base['median_ms']:.4f} ms   fraction of baseline speed: "
                  f"{base['median_ms'] / result.median_ms:.2f}x")
        report["naive"] = naive.to_dict()
        report["baseline"] = base

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str))
    return 0 if result.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="warpsmith")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show GPU info").set_defaults(func=cmd_info)

    p = sub.add_parser("baseline", help="benchmark the library baseline for a problem")
    p.add_argument("problem")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("eval", help="check, compile, verify and benchmark a kernel")
    p.add_argument("problem")
    p.add_argument("kernel")
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--no-compare", action="store_true", help="skip naive/baseline comparison")
    p.add_argument("--arch", default="native")
    p.add_argument("--json", help="write the full result to this file")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
