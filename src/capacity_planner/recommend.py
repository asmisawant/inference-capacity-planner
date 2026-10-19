"""Recommend a GPU and serving config for a workload and latency targets.

    python -m capacity_planner.recommend --prompt 1024 --output 256 --rps 5 --ttft-ms 500 --tpot-ms 50

Only (GPU, quant) pairs that vLLM actually ran are considered (see calibration.py),
so every prediction is anchored to a measurement.

Serving model, per replica with B requests in flight (vLLM continuous batching):
- each decode step takes step(B, context) and gives every request one token;
- each request's prefill takes prefill(P) of GPU time, the idle-engine TTFT, and is
  assumed not to overlap with decode (conservative);
- so a request sees one token per  TPOT = step + B * prefill / O;
- max_num_seqs = the largest B whose TPOT and idle TTFT (prefill + one step) meet the
  targets: the replica's slots.
Queueing: requests arrive at random (Poisson) and wait when all max_num_seqs slots
are busy. Each replica is an M/D/c queue: c = max_num_seqs slots, service time
S = TTFT + O * TPOT at full occupancy (near-deterministic for fixed-length requests).
The waiting probability is Erlang C; the p90 wait uses the M/M/c tail scaled by
Allen-Cunneen's (1 + Cs^2) / 2 with Cs^2 = 0. Replicas are added until
p90 TTFT = TTFT + p90 wait meets the target. Calibrated by serve run 023, where a
replica sized without queueing (max_num_seqs 15, 81% of slots busy on average)
measured a 4.0 s p90 TTFT against a 0.23 s prediction.
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


TTFT_PERCENTILE = 0.90
SERVICE_CV2 = 0.0  # squared coefficient of variation of service time: fixed-length requests


@dataclass(frozen=True)
class QueuePrediction:
    slots: int  # max_num_seqs
    rps: float  # per replica
    service_s: float  # TTFT + O * TPOT at full occupancy
    utilization: float  # mean busy slots / slots
    p_wait: float  # Erlang C: chance a request finds every slot busy
    wait_p90_s: float
    busy: OperatingPoint  # at the expected number of busy slots

    @property
    def ttft_p90_s(self) -> float:
        return self.busy.ttft_s + self.wait_p90_s


@dataclass(frozen=True)
class Recommendation:
    gpu: str
    quant: str
    feasible: bool
    reason: str = ""
    replicas: int = 0
    max_num_seqs: int = 0  # largest batch whose TPOT and idle TTFT meet the targets
    queue: QueuePrediction | None = None
    kv_cache_tokens: int = 0
    price_per_hour: float = 0.0
    run_id: str = ""
    compile_fallback: bool = False

    @property
    def operating(self) -> OperatingPoint | None:
        return self.queue.busy if self.queue else None

    @property
    def cost_per_hour(self) -> float:
        return self.replicas * self.price_per_hour

    def cost_per_million_tokens(self, w: Workload, output_only: bool = False) -> float:
        tokens_per_s = w.rps * (w.output_len if output_only else w.prompt_len + w.output_len)
        return self.cost_per_hour / 3600 / tokens_per_s * 1e6


def erlang_c(c: int, a: float) -> float:
    """Probability of waiting in an M/M/c queue with offered load a (< c) Erlangs."""
    if not 0 <= a < c:
        raise ValueError("need 0 <= a < c")
    b = 1.0  # Erlang B, built up one server at a time (numerically stable)
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return c * b / (c - a * (1 - b))


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

    slots = max(ok, key=lambda p: p.capacity_rps).batch
    start = max(1, math.ceil(w.rps / ok[-1].capacity_rps))
    for replicas in range(start, start * 20 + 20):
        q = queue_prediction(c, w, slots, w.rps / replicas)
        if q is not None and q.ttft_p90_s <= w.ttft_slo_s:
            return Recommendation(feasible=True, replicas=replicas, max_num_seqs=slots, queue=q, **base)
    return Recommendation(feasible=False, reason="queueing keeps p90 TTFT above target", **base)


def queue_prediction(c: MeasuredConfig, w: Workload, slots: int, rps: float) -> QueuePrediction | None:
    """One replica with `slots` = max_num_seqs at `rps`; None if it can't keep up."""
    full = operating_point(c, w, slots)
    service = full.ttft_s + w.output_len * full.tpot_s
    a = rps * service
    if a >= slots:
        return None
    p_wait = erlang_c(slots, a)
    tail = max(0.0, math.log(p_wait / (1 - TTFT_PERCENTILE)))
    wait_p90 = tail * (1 + SERVICE_CV2) / 2 / (slots / service - rps)
    return QueuePrediction(slots, rps, service, a / slots, p_wait, wait_p90, busy_point(c, w, slots, rps))


def busy_point(c: MeasuredConfig, w: Workload, slots: int, rps: float) -> OperatingPoint:
    """Expected requests in flight B, from Little's law B = rps * (TTFT(B) + O * TPOT(B)).

    The right side grows with B, so iterating from B = 1 climbs to the fixed point.
    """
    b = 1
    for _ in range(slots):
        p = operating_point(c, w, b)
        nxt = max(1, min(slots, math.ceil(rps * (p.ttft_s + w.output_len * p.tpot_s))))
        if nxt <= b:
            return p
        b = nxt
    return operating_point(c, w, b)


# On equal cost, prefer less quantization (no accuracy risk), then more latency headroom.
QUANT_PREFERENCE = {"bf16": 0, "fp8": 1, "awq-int4": 2}


def predict(c: MeasuredConfig, w: Workload, max_num_seqs: int) -> QueuePrediction | None:
    """One replica with this max_num_seqs at w.rps (as limited by KV cache and measurements)."""
    slots = min(max_num_seqs, c.kv_cache_tokens // (w.prompt_len + w.output_len), c.max_measured_batch)
    return queue_prediction(c, w, slots, w.rps) if slots >= 1 else None


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
        "| GPU | Quant | Replicas | $/hr | $/1M tokens (in+out) | $/1M output tokens | Busy slots (max_num_seqs) | TTFT ms p90 | TPOT ms | KV cache tokens | Run |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    feasible = [r for r in recs if r.feasible]
    for r in feasible[:top] if top else feasible:
        o = r.operating
        note = " ¹" if r.compile_fallback else ""
        rows.append(
            f"| {r.gpu} | {r.quant}{note} | {r.replicas} | {r.cost_per_hour:.2f} "
            f"| {r.cost_per_million_tokens(w):.3f} | {r.cost_per_million_tokens(w, output_only=True):.3f} "
            f"| {o.batch} ({r.max_num_seqs}) | {r.queue.ttft_p90_s * 1e3:.0f} | {o.tpot_s * 1e3:.1f} "
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
