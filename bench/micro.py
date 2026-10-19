"""GPU microbenchmarks: achieved DRAM bandwidth and dense matmul throughput.

Runs inside the Modal container (needs torch + CUDA). These numbers calibrate
the roofline's `Efficiency`: achieved / datasheet peak.
"""

from __future__ import annotations

import statistics
import subprocess
from typing import Callable

import torch  # only installed in the Modal container

NVIDIA_SMI_FIELDS = (
    "name",
    "memory.total",
    "clocks.max.sm",
    "clocks.max.memory",
    "power.limit",
    "driver_version",
)


def gpu_info() -> dict:
    """nvidia-smi only: must not create a CUDA context, since vLLM runs next in a child process."""
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(NVIDIA_SMI_FIELDS)}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().splitlines()[0]
    return dict(zip(NVIDIA_SMI_FIELDS, (v.strip() for v in out.split(","))))


def torch_info() -> dict:
    props = torch.cuda.get_device_properties(0)
    return {
        "torch_name": props.name,
        "total_memory_bytes": props.total_memory,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }


def time_ms(fn: Callable[[], object], iters: int = 50, warmup: int = 10) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _summary(times_ms: list[float], work: float, unit_scale: float) -> dict:
    """work / time, reported for the median and the fastest iteration."""
    med, best = statistics.median(times_ms), min(times_ms)
    return {
        "median_ms": med,
        "best_ms": best,
        "median": work / (med / 1e3) / unit_scale,
        "best": work / (best / 1e3) / unit_scale,
    }


def bandwidth(nbytes: int = 2**30) -> dict:
    """Device-to-device copy: reads and writes nbytes each, so 2*nbytes of traffic."""
    src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    res = _summary(time_ms(lambda: dst.copy_(src)), 2 * nbytes, 1e9)
    return {"unit": "GB/s", "bytes_per_iter": 2 * nbytes, **res}


def matmul(dtype: str, n: int = 8192) -> dict:
    """Square n x n x n GEMM: 2*n^3 FLOPs."""
    flops = 2 * n**3
    try:
        if dtype == "bf16":
            a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
            fn = lambda: a @ b  # noqa: E731
        elif dtype == "fp8":
            a = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
            b = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
            one = torch.tensor(1.0, device="cuda")
            # cuBLASLt wants the second operand column-major: pass a transposed view.
            fn = lambda: torch._scaled_mm(a, b.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)  # noqa: E731
        elif dtype == "int8":
            a = torch.randint(-8, 8, (n, n), device="cuda", dtype=torch.int8)
            b = torch.randint(-8, 8, (n, n), device="cuda", dtype=torch.int8)
            fn = lambda: torch._int_mm(a, b.t())  # noqa: E731
        else:
            raise ValueError(dtype)
        fn()
        torch.cuda.synchronize()
    except Exception as e:  # unsupported on this GPU / torch build
        return {"unit": "TFLOPS", "n": n, "error": f"{type(e).__name__}: {e}"}
    return {"unit": "TFLOPS", "n": n, **_summary(time_ms(fn), flops, 1e12)}


def run() -> dict:
    return {
        "torch_info": torch_info(),
        "bandwidth": bandwidth(),
        "matmul": {dt: matmul(dt) for dt in ("bf16", "fp8", "int8")},
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
