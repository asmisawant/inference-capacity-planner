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
Allen-Cunneen's (1 + Cs^2) / 2 with Cs^2 = 0. Prefills also queue behind each
other on the one GPU: an M/D/1 queue with service = TTFT. Replicas are added until
p90 TTFT = TTFT + p90 slot wait + p90 prefill wait meets the target. Calibrated by serve run 023, where a
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
from .models import DEFAULT_MODEL, registry
from .validation import same_setup

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
    """TTFT distribution of one replica: base + slot wait + prefill wait.

    Each wait is 0 with probability 1 - p and exponential with the given rate otherwise
    (the M/M/c tail, with the rate doubled per Allen-Cunneen for near-deterministic
    service). The two waits are treated as independent and summed exactly.
    """

    slots: int  # max_num_seqs
    rps: float  # per replica
    service_s: float  # TTFT + O * TPOT at the busy batch
    utilization: float  # mean busy slots / slots
    p_wait: float  # Erlang C: chance a request finds every slot busy
    slot_rate: float
    p_prefill: float  # chance a request arrives while another prefill is running
    prefill_rate: float
    busy: OperatingPoint  # at the expected number of busy slots
    base_ttft_s: float  # busy TTFT + half a decode step + serving overhead

    def wait_survival(self, t: float) -> float:
        """P(slot wait + prefill wait > t)."""
        p1, r1, p2, r2 = self.p_wait, self.slot_rate, self.p_prefill, self.prefill_rate
        if abs(r1 - r2) < 1e-9 * max(r1, r2):
            both = (1 + r1 * t) * math.exp(-r1 * t)
        else:  # hypoexponential survival
            both = (r2 * math.exp(-r1 * t) - r1 * math.exp(-r2 * t)) / (r2 - r1)
        return (p1 * (1 - p2) * math.exp(-r1 * t) + p2 * (1 - p1) * math.exp(-r2 * t) + p1 * p2 * both)

    def ttft_quantile(self, q: float) -> float:
        if self.wait_survival(0.0) <= 1 - q:
            return self.base_ttft_s
        lo, hi = 0.0, 1.0
        while self.wait_survival(hi) > 1 - q:
            hi *= 2
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if self.wait_survival(mid) > 1 - q else (lo, mid)
        return self.base_ttft_s + hi

    @property
    def ttft_p50_s(self) -> float:
        return self.ttft_quantile(0.5)

    @property
    def ttft_p90_s(self) -> float:
        return self.ttft_quantile(TTFT_PERCENTILE)


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
    run_id: str = ""  # the vllm run behind the curves (for predicted configs: the Qwen-7B source run)
    compile_fallback: bool = False
    gpus_per_replica: int = 1  # tensor-parallel degree
    predicted: bool = False  # curves transferred from Qwen-7B, never measured for this model
    ttft_margin: float = 1.0  # p90 TTFT safety factor from serve runs (see serve_calibration)

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


def evaluate(c: MeasuredConfig, w: Workload, ttft_margin: float = 1.0, overhead_s: float = 0.0) -> Recommendation:
    """Cheapest replica count for one config. The p90 TTFT (including `overhead_s`) must
    meet the target after multiplying by `ttft_margin` (both from serve runs, see
    serve_calibration)."""
    base = dict(
        ttft_margin=ttft_margin,
        gpu=c.gpu.name,
        quant=c.quant.name,
        kv_cache_tokens=c.kv_cache_tokens,
        price_per_hour=c.gpu.price_per_hour * c.tp,
        run_id=c.source,
        compile_fallback=c.compile_fallback,
        gpus_per_replica=c.tp,
        predicted=c.predicted,
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
        q = queue_prediction(c, w, slots, w.rps / replicas, overhead_s)
        if q is not None and q.ttft_p90_s * ttft_margin <= w.ttft_slo_s:
            return Recommendation(feasible=True, replicas=replicas, max_num_seqs=slots, queue=q, **base)
    return Recommendation(feasible=False, reason="queueing keeps p90 TTFT above target", **base)


def queue_prediction(
    c: MeasuredConfig, w: Workload, slots: int, rps: float, overhead_s: float = 0.0
) -> QueuePrediction | None:
    """One replica with `slots` = max_num_seqs at `rps`; None if it can't keep up.

    `overhead_s` is the fixed per-request cost of the online path (HTTP, tokenizing the
    prompt text, streaming) that the offline TTFT measurements don't include.
    """
    busy = busy_point(c, w, slots, rps)
    service = busy.ttft_s + w.output_len * busy.tpot_s
    a = rps * service
    rho = rps * busy.ttft_s  # prefills share one GPU: M/D/1 with service = TTFT
    if a >= slots or rho >= 1:
        return None
    acf = 2 / (1 + SERVICE_CV2)  # Allen-Cunneen: deterministic service halves the wait
    return QueuePrediction(
        slots=slots,
        rps=rps,
        service_s=service,
        utilization=a / slots,
        p_wait=erlang_c(slots, a),
        slot_rate=acf * (slots / service - rps),
        p_prefill=rho,
        prefill_rate=acf * (1 / busy.ttft_s - rps),
        busy=busy,
        base_ttft_s=busy.ttft_s + busy.decode_step_s / 2 + overhead_s,
    )


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


def predict(c: MeasuredConfig, w: Workload, max_num_seqs: int, overhead_s: float = 0.0) -> QueuePrediction | None:
    """One replica with this max_num_seqs at w.rps (as limited by KV cache and measurements)."""
    slots = min(max_num_seqs, c.kv_cache_tokens // (w.prompt_len + w.output_len), c.max_measured_batch)
    return queue_prediction(c, w, slots, w.rps, overhead_s) if slots >= 1 else None


@dataclass(frozen=True)
class ServeCalibration:
    """What the serve runs add to the model.

    - overhead_s: median over runs of measured - predicted p50 TTFT, the fixed cost of
      the online path that offline TTFT measurements miss;
    - by_gpu: p90 TTFT safety factor, the worst measured / predicted (with overhead)
      over that GPU's runs, never below 1. GPUs without runs use the largest.
    """

    overhead_s: float
    by_gpu: dict[str, float]
    ratios: tuple[tuple[str, str, float], ...]  # (run id, gpu, measured / predicted p90)

    @property
    def default(self) -> float:
        return max(self.by_gpu.values(), default=1.0)

    def __call__(self, gpu: str) -> float:
        return self.by_gpu.get(gpu, self.default)


NO_CALIBRATION = ServeCalibration(0.0, {}, ())


def serve_runs(results: list[dict], configs: list[MeasuredConfig]) -> list[tuple[dict, MeasuredConfig, Workload]]:
    by_key = {(c.model.name, *c.key): c for c in configs}
    out = []
    for r in results:
        c = by_key.get((r.get("model", DEFAULT_MODEL), r["gpu"], r["quant"], r.get("tp", 1)))
        if r["suite"] == "serve" and c is not None:
            s = r["serve"]
            w = Workload(s["prompt"], s["output"], s["rps"], ttft_slo_s=1e9, tpot_slo_s=1e9)
            out.append((r, c, w))
    return out


def serve_calibration(
    results: list[dict] | None = None,
    configs: list[MeasuredConfig] | None = None,
    exclude: frozenset[str] = frozenset(),
) -> ServeCalibration:
    """Fit overhead and margins on serve runs, leaving out run ids in `exclude`.

    By default every serve run counts, for any model: a run is predicted from the configs
    the recommender would use for its model (measured, or transferred from Qwen-7B)."""
    import statistics

    from .models import registry
    from .transfer import candidate_configs
    from .validation import load_results

    results = load_results() if results is None else results
    if configs is None:
        configs = [c for m in registry() for c in candidate_configs(m, results)]
    preds = []
    for r, c, w in serve_runs(results, configs):
        q = predict(c, w, r["serve"]["max_num_seqs"])
        if q is not None and r["id"] not in exclude:
            preds.append((r, q))
    if not preds:
        return NO_CALIBRATION
    measured = lambda r, k: r["result"]["serve"]["result"][k] / 1e3  # noqa: E731
    overhead = max(0.0, statistics.median(measured(r, "median_ttft_ms") - q.ttft_p50_s for r, q in preds))
    ratios = tuple((r["id"], r["gpu"], measured(r, "p90_ttft_ms") / (q.ttft_p90_s + overhead)) for r, q in preds)
    by_gpu: dict[str, float] = {}
    for _, gpu, ratio in ratios:
        by_gpu[gpu] = max(by_gpu.get(gpu, 1.0), ratio)
    return ServeCalibration(overhead, by_gpu, ratios)


def recommend(
    w: Workload,
    quants: set[str] | None = None,
    configs: list[MeasuredConfig] | None = None,
    cal: ServeCalibration | None = None,
    model: str = DEFAULT_MODEL,
) -> list[Recommendation]:
    """All candidate configs for `model`, feasible ones first, cheapest first.

    Serving overhead and p90 margins always come from the Qwen-7B serve runs."""
    from .transfer import candidate_configs

    configs = candidate_configs(model) if configs is None else configs
    cal = serve_calibration() if cal is None else cal
    recs = [
        evaluate(c, w, cal(c.gpu.name), cal.overhead_s) for c in configs if quants is None or c.quant.name in quants
    ]

    def key(r: Recommendation):
        tpot = r.operating.tpot_s if r.operating else 0.0
        return (not r.feasible, round(r.cost_per_hour, 6), r.predicted, QUANT_PREFERENCE[r.quant], tpot, r.gpu)

    return sorted(recs, key=key)


def table(w: Workload, recs: list[Recommendation], top: int | None = None) -> str:
    rows = [
        "| GPU | Quant | Replicas | $/hr | $/1M tokens (in+out) | $/1M output tokens | Busy slots (max_num_seqs) | TTFT ms p90 (× margin) | TPOT ms | KV cache tokens | Run |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    feasible = [r for r in recs if r.feasible]
    for r in feasible[:top] if top else feasible:
        o = r.operating
        note = " ¹" if r.compile_fallback else ""
        gpu = r.gpu if r.gpus_per_replica == 1 else f"{r.gpus_per_replica}× {r.gpu}"
        run = f"predicted ² from {r.run_id}" if r.predicted else r.run_id
        rows.append(
            f"| {gpu} | {r.quant}{note} | {r.replicas} | {r.cost_per_hour:.2f} "
            f"| {r.cost_per_million_tokens(w):.3f} | {r.cost_per_million_tokens(w, output_only=True):.3f} "
            f"| {o.batch} ({r.max_num_seqs}) | {r.queue.ttft_p90_s * r.ttft_margin * 1e3:.0f} (×{r.ttft_margin:.2f}) "
            f"| {o.tpot_s * 1e3:.1f} "
            f"| {r.kv_cache_tokens:,} | {run} |"
        )
    if not feasible:
        rows.append("| _none_ | | | | | | | | | | |")
    infeasible = [r for r in recs if not r.feasible]
    out = "\n".join(rows)
    if infeasible:
        out += "\n\nNot feasible: " + "; ".join(
            f"{r.gpus_per_replica}× {r.gpu} {r.quant} ({r.reason})" if r.gpus_per_replica > 1 else f"{r.gpu} {r.quant} ({r.reason})"
            for r in infeasible
        ) + "."
    if any(r.compile_fallback for r in feasible):
        out += "\n\n¹ benchmarked with vLLM's eager compile backend after an Inductor error."
    if any(r.predicted for r in feasible):
        out += (
            "\n\n² never run for this model: curves transferred from the named Qwen2.5-7B run "
            "(see \"Predicting models we never benchmarked\")."
        )
    return out


# Example workloads for the README. Change them here, not in the README.
EXAMPLES = (
    ("Chat assistant", Workload(1024, 256, rps=5, ttft_slo_s=0.5, tpot_slo_s=0.05)),
    ("Long-context interactive", Workload(2048, 256, rps=20, ttft_slo_s=0.3, tpot_slo_s=0.03)),
    ("Offline batch generation", Workload(512, 512, rps=50, ttft_slo_s=5.0, tpot_slo_s=0.2)),
)


def examples_markdown(top: int = 3) -> str:
    configs = measured_configs()
    cal = serve_calibration()
    parts = []
    for name, w in EXAMPLES:
        recs = recommend(w, configs=configs, cal=cal)
        parts.append(f"**{name}:** {describe(w)}.\n\n{table(w, recs, top)}")
    return "\n\n".join(parts)


def other_models_markdown(top: int = 3) -> str:
    """The first example workload for every registry model besides the default."""
    from .transfer import candidate_configs

    cal = serve_calibration()
    name, w = EXAMPLES[0]
    parts = []
    for model in registry():
        if model == DEFAULT_MODEL:
            continue
        recs = recommend(w, configs=candidate_configs(model), cal=cal, model=model)
        parts.append(f"**{model}, {name.lower()}:** {describe(w)}.\n\n{table(w, recs, top)}")
    return "\n\n".join(parts)


def describe(w: Workload) -> str:
    return (
        f"{w.prompt_len}-token prompts, {w.output_len}-token outputs, {w.rps:g} requests/s, "
        f"TTFT ≤ {w.ttft_slo_s * 1e3:g} ms, TPOT ≤ {w.tpot_slo_s * 1e3:g} ms"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recommend a GPU + serving config for a model in the registry.")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(registry()))
    ap.add_argument("--prompt", type=int, required=True, help="prompt tokens per request")
    ap.add_argument("--output", type=int, required=True, help="output tokens per request")
    ap.add_argument("--rps", type=float, required=True, help="requests per second")
    ap.add_argument("--ttft-ms", type=float, required=True, help="time-to-first-token target")
    ap.add_argument("--tpot-ms", type=float, required=True, help="time-per-output-token target")
    ap.add_argument("--quant", action="append", choices=["bf16", "fp8", "awq-int4"], help="allow only these")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args(argv)
    w = Workload(a.prompt, a.output, a.rps, a.ttft_ms / 1e3, a.tpot_ms / 1e3)
    recs = recommend(w, set(a.quant) if a.quant else None, model=a.model)
    print(f"{a.model}: {describe(w)}\n")
    print(table(w, recs, a.top))
    best = next((r for r in recs if r.feasible), None)
    if best:
        print(
            f"\nRecommended{' (predicted, not yet benchmarked for this model)' if best.predicted else ''}: "
            f"{best.replicas} x {best.gpu if best.gpus_per_replica == 1 else f'{best.gpus_per_replica}x {best.gpu}'}, "
            f"{ENGINE}, quant={best.quant}, tensor_parallel_size={best.gpus_per_replica}, "
            f"max_num_seqs={best.max_num_seqs}, gpu_memory_utilization={DEFAULT_GPU_MEMORY_UTILIZATION}, "
            f"KV cache {best.kv_cache_tokens:,} tokens/replica, ${best.cost_per_hour:.2f}/hr, "
            f"${best.cost_per_million_tokens(w):.3f}/1M tokens"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
