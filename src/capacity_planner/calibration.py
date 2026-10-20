"""Calibrate the roofline with benchmark results (results/*.json).

Three layers, each derived from committed runs:

1. GPU efficiency from the `micro` suite: measured BF16 GEMM / datasheet BF16 peak
   (compute) and measured GEMV read bandwidth / datasheet bandwidth (memory).
2. Correction fits per (GPU, quant) from the `vllm` suite, for reporting how much of
   vLLM's behaviour the efficiency-corrected roofline explains:
       TTFT        = c0 + c1 * roofline_ttft
       decode step = a * roofline_step + b * batch
3. `MeasuredConfig`: what the recommender uses. It interpolates the measured vLLM curves
   and uses the (efficiency-corrected) roofline only for the *shape* outside the
   measured points: prompt lengths outside 128-4000 tokens and decode context lengths
   other than the measured 640.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from . import roofline as rf
from .gpus import GPUS, GPUSpec
from .models import ModelSpec, load_model
from .validation import load_results, latest


def gpu_efficiency(results: list[dict]) -> dict[str, rf.Efficiency]:
    """Per-GPU efficiency from the latest micro run with a GEMV measurement."""
    out = {}
    for (gpu, _), r in latest(results, "micro").items():
        micro = r["result"]["micro"]
        if "gemv_bandwidth" not in micro:
            continue
        spec = GPUS[gpu]
        out[gpu] = rf.Efficiency(
            compute=micro["matmul"]["bf16"]["median"] / spec.bf16_tflops,
            memory=micro["gemv_bandwidth"]["median"] / spec.mem_bw_gbs,
        )
    return out


def _lsq2(xs: list[tuple[float, float]], ys: list[float]) -> tuple[float, float]:
    """Least squares for y = p0 * x0 + p1 * x1."""
    s00 = sum(x[0] * x[0] for x in xs)
    s01 = sum(x[0] * x[1] for x in xs)
    s11 = sum(x[1] * x[1] for x in xs)
    t0 = sum(x[0] * y for x, y in zip(xs, ys))
    t1 = sum(x[1] * y for x, y in zip(xs, ys))
    det = s00 * s11 - s01 * s01
    if det == 0:
        raise ValueError("degenerate fit")
    return (t0 * s11 - t1 * s01) / det, (s00 * t1 - s01 * t0) / det


@dataclass(frozen=True)
class Fit:
    gpu: str
    quant: str
    run_id: str
    ttft_c0_s: float
    ttft_c1: float
    ttft_max_err: float
    decode_a: float
    decode_b_s: float
    decode_max_err: float


def _interp(xs: list[float], ys: list[float], x: float) -> float:
    i = bisect.bisect_left(xs, x)
    if i < len(xs) and xs[i] == x:
        return ys[i]
    lo, hi = i - 1, i
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + t * (ys[hi] - ys[lo])


@dataclass
class MeasuredConfig:
    """One (GPU, quant, tp) that vLLM ran successfully, with its measured curves."""

    gpu: GPUSpec
    quant: rf.Quantization
    model: ModelSpec
    eff: rf.Efficiency
    run: dict
    tp: int = 1
    predicted = False  # see transfer.TransferredConfig

    @property
    def engine(self) -> str:
        return self.run["suite"]

    def __post_init__(self):
        v = self.run["result"][self.run["suite"]]  # "vllm" or "sglang": same schema
        t = sorted(v["ttft"], key=lambda t: t["prompt_len"])
        self._ttft_p = [float(x["prompt_len"]) for x in t]
        self._ttft_s = [x["ttft_s_median"] for x in t]
        d = sorted((x for x in v["decode"] if "decode_step_s" in x), key=lambda x: x["batch"])
        if not d:
            raise ValueError(f"{self.run['id']} has no decode step measurements")
        self._dec_b = [float(x["batch"]) for x in d]
        self._dec_s = [x["decode_step_s"] for x in d]
        self.decode_context_len = d[0]["decode_context_len"]
        self.kv_cache_tokens: int = v["engine"]["kv_cache_size_tokens"]
        self.max_num_seqs: int = v["engine"]["max_num_seqs"]
        self.max_model_len: int = v["engine"]["max_model_len"]
        self.compile_fallback = bool(v.get("compile_fallback"))

    @property
    def key(self) -> tuple[str, str, int]:
        return self.gpu.name, self.quant.name, self.tp

    @property
    def source(self) -> str:
        return self.run["id"]

    @property
    def max_measured_batch(self) -> int:
        return int(self._dec_b[-1])

    def _roof_ttft(self, p: float) -> float:
        return rf.ttft_s(self.gpu, self.model, self.quant, max(1, round(p)), eff=self.eff, tp=self.tp)

    def ttft_s(self, prompt_len: int) -> float:
        """Idle-engine TTFT: measured curve, roofline-shaped outside the measured range."""
        p = float(prompt_len)
        lo, hi = self._ttft_p[0], self._ttft_p[-1]
        if lo <= p <= hi:
            return _interp(self._ttft_p, self._ttft_s, p)
        edge, edge_s = (lo, self._ttft_s[0]) if p < lo else (hi, self._ttft_s[-1])
        return edge_s * self._roof_ttft(p) / self._roof_ttft(edge)

    def decode_step_s(self, batch: int, context_len: int) -> float:
        """Full-batch decode step: measured over batch, roofline-scaled for context length.

        Only defined up to the largest measured batch; beyond it the recommender doesn't go.
        """
        if not 1 <= batch <= self.max_measured_batch:
            raise ValueError(f"batch {batch} outside measured range 1..{self.max_measured_batch}")
        at_measured_ctx = _interp(self._dec_b, self._dec_s, float(batch))
        scale = (
            rf.decode_step(self.gpu, self.model, self.quant, batch, context_len, eff=self.eff, tp=self.tp).time_s
            / rf.decode_step(
                self.gpu, self.model, self.quant, batch, self.decode_context_len, eff=self.eff, tp=self.tp
            ).time_s
        )
        return at_measured_ctx * scale


TP_DEGREES = (1, 2, 4, 8)


ENGINES = ("vllm", "sglang")  # offline suite name = engine
ENGINE_NAMES = {"vllm": "vLLM", "sglang": "SGLang"}


def measured_configs(
    results: list[dict] | None = None,
    model: ModelSpec | None = None,
    tps: tuple[int, ...] = (1,),
    engine: str = "vllm",
) -> list[MeasuredConfig]:
    """Configs `engine` ran for `model`. Single-GPU only by default (what the fits use)."""
    results = load_results() if results is None else results
    model = load_model() if model is None else model
    eff = gpu_efficiency(results)
    out = []
    for tp in tps:
        for (gpu, quant), r in latest(results, engine, model.name, tp).items():
            if gpu not in eff:
                continue
            try:
                out.append(MeasuredConfig(GPUS[gpu], rf.QUANTIZATIONS[quant], model, eff[gpu], r, tp))
            except ValueError:
                continue  # e.g. a run from before the decode-step measurement
    return out


def fits(results: list[dict] | None = None, model: ModelSpec | None = None, engine: str = "vllm") -> list[Fit]:
    out = []
    for c in measured_configs(results, model, engine=engine):
        v = c.run["result"][engine]
        xt = [(1.0, c._roof_ttft(t["prompt_len"])) for t in v["ttft"]]
        yt = [t["ttft_s_median"] for t in v["ttft"]]
        c0, c1 = _lsq2(xt, yt)
        ds = [d for d in v["decode"] if "decode_step_s" in d]
        xd = [
            (
                rf.decode_step(c.gpu, c.model, c.quant, d["batch"], d["decode_context_len"], eff=c.eff, tp=c.tp).time_s,
                d["batch"],
            )
            for d in ds
        ]
        yd = [d["decode_step_s"] for d in ds]
        a, b = _lsq2(xd, yd)
        out.append(
            Fit(
                gpu=c.gpu.name,
                quant=c.quant.name,
                run_id=c.run["id"],
                ttft_c0_s=c0,
                ttft_c1=c1,
                ttft_max_err=max(abs((c0 + c1 * x[1]) / y - 1) for x, y in zip(xt, yt)),
                decode_a=a,
                decode_b_s=b,
                decode_max_err=max(abs((a * x[0] + b * x[1]) / y - 1) for x, y in zip(xd, yd)),
            )
        )
    return out


def efficiency_table(results: list[dict] | None = None) -> str:
    results = load_results() if results is None else results
    eff = gpu_efficiency(results)
    if not eff:
        return "_No micro runs with GEMV bandwidth yet._"
    rows = [
        "| GPU | Compute efficiency (BF16 GEMM / datasheet) | Memory efficiency (GEMV / datasheet) |",
        "|---|---:|---:|",
    ]
    for gpu, e in eff.items():
        rows.append(f"| {gpu} | {e.compute:.2f} | {e.memory:.2f} |")
    return "\n".join(rows)


def fit_table(results: list[dict] | None = None, engine: str = "vllm") -> str:
    fs = fits(results, engine=engine)
    if not fs:
        return f"_No {ENGINE_NAMES[engine]} runs with decode-step measurements yet._"
    rows = [
        "| GPU | Quant | TTFT c0 (ms) | TTFT c1 | TTFT max error | Decode a | Decode b (µs/seq) | Decode max error | Run |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for f in fs:
        rows.append(
            f"| {f.gpu} | {f.quant} | {f.ttft_c0_s * 1e3:.1f} | {f.ttft_c1:.2f} | {f.ttft_max_err:.0%} "
            f"| {f.decode_a:.2f} | {f.decode_b_s * 1e6:.0f} | {f.decode_max_err:.0%} | {f.run_id} |"
        )
    return "\n".join(rows)
