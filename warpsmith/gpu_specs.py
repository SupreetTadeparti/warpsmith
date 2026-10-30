"""Peak hardware numbers for the roofline sanity check.

Values are the *highest* published figure for each family (e.g. SXM / 80GB parts) so
the check can only produce false negatives, never accuse a legitimate kernel.
"""

from __future__ import annotations

# name substring -> (peak DRAM bandwidth bytes/s, peak FP32 FLOP/s, non-tensor-core)
PEAKS: list[tuple[str, float, float]] = [
    ("H100", 3.35e12, 67.0e12),
    ("A100", 2.04e12, 19.5e12),
    ("A10G", 600e9, 31.2e12),
    ("L4", 300e9, 30.3e12),
    ("V100", 900e9, 15.7e12),
    ("T4", 320e9, 8.1e12),
    ("P100", 732e9, 10.6e12),
]


def lookup_peaks(device_name: str) -> tuple[float | None, float | None]:
    for key, bw, flops in PEAKS:
        if key in device_name:
            return bw, flops
    return None, None


def measure_bandwidth(nbytes: int = 256 << 20, reps: int = 10) -> float:
    """Achieved device-to-device copy bandwidth in bytes/s (reads + writes)."""
    import torch

    src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    dst.copy_(src)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(reps):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()
    return 2 * nbytes * reps / (start.elapsed_time(end) / 1e3)


# Headroom over measured copy bandwidth when the GPU isn't in the table. Copies reach
# ~80-90% of peak, so 1.25x stays above the true peak while still catching kernels that
# claim to beat memory bandwidth by a wide margin (e.g. skipped work).
MEASURED_BW_HEADROOM = 1.25


def device_info() -> dict:
    import torch

    props = torch.cuda.get_device_properties(0)
    bw, flops = lookup_peaks(props.name)
    source = "table"
    if bw is None:
        bw, source = MEASURED_BW_HEADROOM * measure_bandwidth(), "measured"
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "l2_cache_bytes": getattr(props, "L2_cache_size", None),
        "peak_bandwidth_bytes_per_s": bw,
        "peak_fp32_flops": flops,
        "peak_source": source,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
