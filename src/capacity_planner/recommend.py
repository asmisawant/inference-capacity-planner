"""Recommend a GPU and serving config for a workload and latency targets.

    python -m capacity_planner.recommend --prompt 1024 --output 256 --rps 5 --ttft-ms 500 --tpot-ms 50

Only (GPU, quant) pairs that vLLM actually ran are considered (see calibration.py),
so every prediction is anchored to a measurement.

Serving model, per replica with B requests in flight (vLLM continuous batching):
- each decode step takes step(B, context) and gives every request one token;
- each request's prefill takes prefill(P) of GPU time, the idle-engine TTFT, and is
  assumed not to overlap with decode (conservative);
- so a request sees one token per  TPOT = step + B * prefill / O,
  and the replica serves  capacity(B) = B / (O * TPOT)  requests/s (Little's law);
- TTFT under load = prefill + one decode step (chunked prefill joins the next step;
  queueing beyond that is not modelled).
Replicas = ceil(rps / capacity at the largest batch meeting both SLOs).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

from .calibration import MeasuredConfig, measured_configs
from .roofline import DEFAULT_GPU_MEMORY_UTILIZATION

ENGINE = "vLLM 0.30.0"


@dataclass(frozen=True)
class Workload:
    prompt_len: int
    output_len: int
    rps: float
    ttft_slo_s: float
    tpot_slo_s: float

    def __post_init__(self):
        if min(self.prompt_len, self.output_len) < 1 or self.rps <= 0:
            raise ValueError("prompt_len, output_len and rps must be positive")
        if self.ttft_slo_s <= 0 or self.tpot_slo_s <= 0:
            raise ValueError("SLOs must be positive")

    @property
    def decode_context_len(self) -> int:
        return self.prompt_len + self.output_len // 2


@dataclass(frozen=True)
class OperatingPoint:
    batch: int
    decode_step_s: float
    tpot_s: float
    ttft_s: float
    capacity_rps: float


@dataclass(frozen=True)
class Recommendation:
    gpu: str
    quant: str
    feasible: bool
    reason: str = ""
    replicas: int = 0
    max_num_seqs: int = 0  # largest batch that still meets both SLOs
    operating: OperatingPoint | None = None  # expected batch per replica at this traffic
    kv_cache_tokens: int = 0
    price_per_hour: float = 0.0
    run_id: str = ""
    compile_fallback: bool = False

    @property
    def cost_per_hour(self) -> float:
        return self.replicas * self.price_per_hour

    def cost_per_million_tokens(self, w: Workload, output_only: bool = False) -> float:
        tokens_per_s = w.rps * (w.output_len if output_only else w.prompt_len + w.output_len)
        return self.cost_per_hour / 3600 / tokens_per_s * 1e6


def operating_point(c: MeasuredConfig, w: Workload, batch: int) -> OperatingPoint:
    prefill = c.ttft_s(w.prompt_len)
    step = c.decode_step_s(batch, w.decode_context_len)
    tpot = step + batch * prefill / w.output_len
    return OperatingPoint(
        batch=batch,
        decode_step_s=step,
        tpot_s=tpot,
        ttft_s=prefill + step,
        capacity_rps=batch / (w.output_len * tpot),
    )


def evaluate(c: MeasuredConfig, w: Workload) -> Recommendation:
    base = dict(
        gpu=c.gpu.name,
        quant=c.quant.name,
        kv_cache_tokens=c.kv_cache_tokens,
        price_per_hour=c.gpu.price_per_hour,
        run_id=c.run["id"],
        compile_fallback=c.compile_fallback,
    )
    seq_len = w.prompt_len + w.output_len
    if seq_len > c.max_model_len:
        return Recommendation(
            feasible=False, reason=f"prompt+output {seq_len} > benchmarked max_model_len {c.max_model_len}", **base
        )
    b_max = min(c.kv_cache_tokens // seq_len, c.max_measured_batch, c.max_num_seqs)
    if b_max < 1:
        return Recommendation(feasible=False, reason="one sequence does not fit in the KV cache", **base)

    points = [operating_point(c, w, b) for b in range(1, b_max + 1)]
    ok = [p for p in points if p.tpot_s <= w.tpot_slo_s and p.ttft_s <= w.ttft_slo_s]
    if not ok:
        p1 = points[0]
        why = []
        if p1.ttft_s > w.ttft_slo_s:
            why.append(f"TTFT {p1.ttft_s * 1e3:.0f} ms at batch 1")
        if p1.tpot_s > w.tpot_slo_s:
            why.append(f"TPOT {p1.tpot_s * 1e3:.1f} ms at batch 1")
        return Recommendation(feasible=False, reason="misses SLO: " + ", ".join(why), **base)

    best = max(ok, key=lambda p: p.capacity_rps)
    replicas = math.ceil(w.rps / best.capacity_rps)
    per_replica = w.rps / replicas
    operating = next(p for p in ok if p.capacity_rps >= per_replica - 1e-12)
    return Recommendation(
        feasible=True, replicas=replicas, max_num_seqs=best.batch, operating=operating, **base
    )


# On equal cost, prefer less quantization (no accuracy risk), then more latency headroom.
QUANT_PREFERENCE = {"bf16": 0, "fp8": 1, "awq-int4": 2}


def predict(c: MeasuredConfig, w: Workload, max_num_seqs: int) -> OperatingPoint | None:
    """Expected operating point of one replica at w.rps, or None if it can't keep up."""
    seq_len = w.prompt_len + w.output_len
    b_max = min(max_num_seqs, c.kv_cache_tokens // seq_len, c.max_measured_batch)
    for b in range(1, b_max + 1):
        p = operating_point(c, w, b)
        if p.capacity_rps >= w.rps:
            return p
    return None


def recommend(w: Workload, quants: set[str] | None = None, configs: list[MeasuredConfig] | None = None) -> list[Recommendation]:
    """All measured configs, feasible ones first, cheapest first."""
    configs = measured_configs() if configs is None else configs
    recs = [evaluate(c, w) for c in configs if quants is None or c.quant.name in quants]

    def key(r: Recommendation):
        tpot = r.operating.tpot_s if r.operating else 0.0
        return (not r.feasible, round(r.cost_per_hour, 6), QUANT_PREFERENCE[r.quant], tpot, r.gpu)

    return sorted(recs, key=key)


def table(w: Workload, recs: list[Recommendation], top: int | None = None) -> str:
    rows = [
        "| GPU | Quant | Replicas | $/hr | $/1M tokens (in+out) | $/1M output tokens | Batch per replica (max_num_seqs) | TTFT ms | TPOT ms | KV cache tokens | Run |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    feasible = [r for r in recs if r.feasible]
    for r in feasible[:top] if top else feasible:
        o = r.operating
        note = " ¹" if r.compile_fallback else ""
        rows.append(
            f"| {r.gpu} | {r.quant}{note} | {r.replicas} | {r.cost_per_hour:.2f} "
            f"| {r.cost_per_million_tokens(w):.3f} | {r.cost_per_million_tokens(w, output_only=True):.3f} "
            f"| {o.batch} ({r.max_num_seqs}) | {o.ttft_s * 1e3:.0f} | {o.tpot_s * 1e3:.1f} "
            f"| {r.kv_cache_tokens:,} | {r.run_id} |"
        )
    if not feasible:
        rows.append("| _none_ | | | | | | | | | | |")
    infeasible = [r for r in recs if not r.feasible]
    out = "\n".join(rows)
    if infeasible:
        out += "\n\nNot feasible: " + "; ".join(f"{r.gpu} {r.quant} ({r.reason})" for r in infeasible) + "."
    if any(r.compile_fallback for r in feasible):
        out += "\n\n¹ benchmarked with vLLM's eager compile backend after an Inductor error."
    return out


# Example workloads for the README. Change them here, not in the README.
EXAMPLES = (
    ("Chat assistant", Workload(1024, 256, rps=5, ttft_slo_s=0.5, tpot_slo_s=0.05)),
    ("Long-context interactive", Workload(2048, 256, rps=20, ttft_slo_s=0.3, tpot_slo_s=0.03)),
    ("Offline batch generation", Workload(512, 512, rps=50, ttft_slo_s=5.0, tpot_slo_s=0.2)),
)


def examples_markdown(top: int = 3) -> str:
    configs = measured_configs()
    parts = []
    for name, w in EXAMPLES:
        parts.append(f"**{name}:** {describe(w)}.\n\n{table(w, recommend(w, configs=configs), top)}")
    return "\n\n".join(parts)


def describe(w: Workload) -> str:
    return (
        f"{w.prompt_len}-token prompts, {w.output_len}-token outputs, {w.rps:g} requests/s, "
        f"TTFT ≤ {w.ttft_slo_s * 1e3:g} ms, TPOT ≤ {w.tpot_slo_s * 1e3:g} ms"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recommend a GPU + serving config for Qwen2.5-7B-Instruct.")
    ap.add_argument("--prompt", type=int, required=True, help="prompt tokens per request")
    ap.add_argument("--output", type=int, required=True, help="output tokens per request")
    ap.add_argument("--rps", type=float, required=True, help="requests per second")
    ap.add_argument("--ttft-ms", type=float, required=True, help="time-to-first-token target")
    ap.add_argument("--tpot-ms", type=float, required=True, help="time-per-output-token target")
    ap.add_argument("--quant", action="append", choices=["bf16", "fp8", "awq-int4"], help="allow only these")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args(argv)
    w = Workload(a.prompt, a.output, a.rps, a.ttft_ms / 1e3, a.tpot_ms / 1e3)
    recs = recommend(w, set(a.quant) if a.quant else None)
    print(describe(w) + "\n")
    print(table(w, recs, a.top))
    best = next((r for r in recs if r.feasible), None)
    if best:
        print(
            f"\nRecommended: {best.replicas} x {best.gpu}, {ENGINE}, quant={best.quant}, "
            f"max_num_seqs={best.max_num_seqs}, gpu_memory_utilization={DEFAULT_GPU_MEMORY_UTILIZATION}, "
            f"KV cache {best.kv_cache_tokens:,} tokens/replica, ${best.cost_per_hour:.2f}/hr, "
            f"${best.cost_per_million_tokens(w):.3f}/1M tokens"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
