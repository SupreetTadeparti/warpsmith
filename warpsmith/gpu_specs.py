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


def device_info() -> dict:
    import torch

    props = torch.cuda.get_device_properties(0)
    bw, flops = lookup_peaks(props.name)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "l2_cache_bytes": getattr(props, "L2_cache_size", None),
        "peak_bandwidth_bytes_per_s": bw,
        "peak_fp32_flops": flops,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
