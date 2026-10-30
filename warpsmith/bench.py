"""Benchmarking with CUDA events, plus timing-integrity checks. Runs inside the worker.

Shared Colab/Kaggle GPUs are noisy, so we report median and IQR, flush L2 between
reps (otherwise small problems are timed from cache and can beat the DRAM roofline),
and rotate through several input sets so per-pointer caching doesn't pay off.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

import torch

MIN_FLUSH_BYTES = 64 << 20


def make_flush_buffer(l2_bytes: int | None) -> torch.Tensor:
    size = max(2 * (l2_bytes or 0), MIN_FLUSH_BYTES)
    return torch.empty(size, dtype=torch.uint8, device="cuda")


def time_launches(
    launch: Callable[[int], None],
    n_sets: int,
    stream: torch.cuda.Stream,
    flush: torch.Tensor,
    *,
    warmup: int = 10,
    target_ms: float = 1000.0,
    min_reps: int = 20,
    max_reps: int = 200,
) -> dict:
    """`launch(i)` enqueues one run on input set i on the current stream."""
    def event() -> torch.cuda.Event:
        return torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        for i in range(warmup):
            launch(i % n_sets)
        torch.cuda.synchronize()

        s, e = event(), event()
        s.record()
        for i in range(5):
            launch(i % n_sets)
        e.record()
        torch.cuda.synchronize()
        estimate_ms = s.elapsed_time(e) / 5
        reps = int(min(max(target_ms / max(estimate_ms, 1e-3), min_reps), max_reps))

        starts = [event() for _ in range(reps)]
        ends = [event() for _ in range(reps)]
        for i in range(reps):
            flush.zero_()
            starts[i].record()
            launch(i % n_sets)
            ends[i].record()
        torch.cuda.synchronize()
        times = sorted(a.elapsed_time(b) for a, b in zip(starts, ends))

        # Back-to-back batch timed two ways. Work that escapes the stream (another
        # stream, host threads) is invisible to the events but not to the wall clock.
        b0, b1 = event(), event()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        b0.record()
        for i in range(reps):
            launch(i % n_sets)
        b1.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1e3

    q = statistics.quantiles(times, n=4)
    return {
        "reps": reps,
        "median_ms": statistics.median(times),
        "p25_ms": q[0],
        "p75_ms": q[2],
        "min_ms": times[0],
        "event_batch_per_launch_ms": b0.elapsed_time(b1) / reps,
        "wall_per_launch_ms": wall_ms / reps,
    }


def integrity_flags(stats: dict, flops: int, nbytes: int, peak_bw: float | None, peak_flops: float | None) -> list[str]:
    flags = []
    bound_ms = max(
        flops / peak_flops if peak_flops else 0.0,
        nbytes / peak_bw if peak_bw else 0.0,
    ) * 1e3
    if bound_ms and stats["median_ms"] < 0.9 * bound_ms:
        flags.append(
            f"faster_than_hardware: median {stats['median_ms']:.4f} ms is below the roofline "
            f"lower bound {bound_ms:.4f} ms for this GPU"
        )
    if stats["wall_per_launch_ms"] > 1.5 * stats["event_batch_per_launch_ms"] + 0.05:
        flags.append(
            f"timing_mismatch: wall clock {stats['wall_per_launch_ms']:.4f} ms/launch vs CUDA events "
            f"{stats['event_batch_per_launch_ms']:.4f} ms/launch; work is escaping the provided stream"
        )
    return flags


def add_throughput(stats: dict, flops: int, nbytes: int) -> dict:
    sec = stats["median_ms"] / 1e3
    stats["tflops"] = flops / sec / 1e12 if sec > 0 else None
    stats["gbytes_per_s"] = nbytes / sec / 1e9 if sec > 0 else None
    return stats
