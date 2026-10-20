"""End-to-end check of recommendations for models the planner was never tuned on.

Step 4 showed the transferred curves (TTFT, decode step, KV cache) hold for new models.
This checks what a user actually gets: the recommender's whole answer (replicas,
max_num_seqs, and the queueing, serving overhead and p90 margins learned from Qwen-7B
serve runs) under real `vllm serve` load with Poisson arrivals.

For each target the recommender picks replicas and max_num_seqs for the workload; one
replica is then load-tested at its share of the traffic. Predictions for those runs are
frozen by `python -m capacity_planner.serve_check --write PATH` and committed before the
runs. The comparison reads only that file.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from .recommend import EXAMPLES, Workload, engine_calibrations, evaluate, predict
from .transfer import candidate_configs
from .validation import load_results

PREDICTIONS_DIR = Path(__file__).resolve().parents[2] / "predictions"
PREDICTIONS = PREDICTIONS_DIR / "serve.json"
WORKLOADS = dict(EXAMPLES)


@dataclass(frozen=True)
class ServeTarget:
    model: str
    workload: str  # a name from recommend.EXAMPLES
    gpu: str
    quant: str
    tp: int = 1
    engine: str = "vllm"

    @property
    def key(self) -> str:
        base = f"{self.model}|{self.workload}|{self.gpu}|{self.quant}|tp{self.tp}"
        return base if self.engine == "vllm" else f"{base}|{self.engine}"


# The recommender's first or second choice for each, mixing configs vLLM ran for the model
# with ones only transferred from Qwen-7B, and one and two GPUs per replica.
TARGETS = (
    ServeTarget("Granite-3.1-8B-Instruct", "Chat assistant", "A100-40GB", "bf16"),
    ServeTarget("Granite-3.1-8B-Instruct", "Chat assistant", "A100-80GB", "bf16"),  # transferred
    ServeTarget("Granite-3.1-8B-Instruct", "Long-context interactive", "H100", "fp8"),
    ServeTarget("Qwen2.5-32B-Instruct", "Chat assistant", "H100", "fp8"),
    ServeTarget("Qwen2.5-32B-Instruct", "Long-context interactive", "H100", "fp8", tp=2),  # transferred
)


# After step 5 the per-GPU p90 margins learn from every model's serve runs (H100: x1.00
# -> x1.28). One run the margin wasn't fitted on checks it: Qwen-32B on a single H100 at a
# workload where p90 x margin lands just under the target.
MARGIN_CHECK = (ServeTarget("Qwen2.5-32B-Instruct", "Long-context interactive", "H100", "fp8"),)
MARGIN_CHECK_PREDICTIONS = PREDICTIONS_DIR / "serve-h100-margin.json"

# SGLang's own recommendations (step 6), before SGLang has any serve run: its curves are
# measured (or, for Granite, transferred from Qwen-7B), but the serving overhead and p90
# margins are borrowed from vLLM. These runs test both.
SGLANG_CHECK = (
    ServeTarget("Qwen2.5-7B-Instruct", "Chat assistant", "A100-40GB", "bf16", engine="sglang"),
    ServeTarget("Qwen2.5-7B-Instruct", "Long-context interactive", "H100", "bf16", engine="sglang"),
    ServeTarget("Qwen2.5-7B-Instruct", "Offline batch generation", "H100", "fp8", engine="sglang"),
    ServeTarget("Granite-3.1-8B-Instruct", "Chat assistant", "A100-40GB", "bf16", engine="sglang"),
)
SGLANG_CHECK_PREDICTIONS = PREDICTIONS_DIR / "serve-sglang.json"

SETS = {
    "step5": (TARGETS, PREDICTIONS),
    "h100-margin": (MARGIN_CHECK, MARGIN_CHECK_PREDICTIONS),
    "sglang": (SGLANG_CHECK, SGLANG_CHECK_PREDICTIONS),
}
SERVE_SUITE = {"vllm": "serve", "sglang": "sglang-serve"}


def serve_prediction(t: ServeTarget, results: list[dict] | None = None) -> dict:
    results = load_results() if results is None else results
    cal, load_tested = engine_calibrations(results)[t.engine]
    w = WORKLOADS[t.workload]
    configs = {c.key: c for c in candidate_configs(t.model, results) if c.engine == t.engine}
    c = configs[(t.gpu, t.quant, t.tp)]
    rec = evaluate(c, w, cal(t.gpu), cal.overhead_s)
    if not rec.feasible:
        raise ValueError(f"{t.key}: recommender finds it infeasible ({rec.reason})")
    rps = round(w.rps / rec.replicas, 3)
    per_replica = Workload(w.prompt_len, w.output_len, rps, w.ttft_slo_s, w.tpot_slo_s)
    q = predict(c, per_replica, rec.max_num_seqs, cal.overhead_s)
    out = {
        "model": t.model, "gpu": t.gpu, "quant": t.quant, "tp": t.tp,
        "workload": t.workload,
        "slo": {"ttft_p90_s": w.ttft_slo_s, "tpot_s": w.tpot_slo_s},
        "total_rps": w.rps,
        "replicas": rec.replicas,
        "curves": f"predicted from {c.source}" if c.predicted else c.source,
        # what one replica is run with (bench/runs.py serve parameters)
        "serve": {"prompt": w.prompt_len, "output": w.output_len, "rps": rps, "max_num_seqs": rec.max_num_seqs},
        "ttft_p50_s": q.ttft_p50_s,
        "ttft_p90_s": q.ttft_p90_s,
        "ttft_margin": rec.ttft_margin,
        "tpot_s": q.busy.tpot_s,
        "busy_batch": q.busy.batch,
    }
    if t.engine != "vllm":  # the step 5 files predate engines: no key means vLLM
        out["engine"] = t.engine
        out["calibration"] = "own serve runs" if load_tested else "borrowed from vLLM (no serve runs on this engine)"
    return out


def predictions(targets=TARGETS, results: list[dict] | None = None) -> dict:
    results = load_results() if results is None else results
    cal = engine_calibrations(results)["vllm"][0]
    return {
        "method": "capacity_planner.serve_check: recommender output for one replica at its share of the load",
        "serving_overhead_s": cal.overhead_s,
        "predictions": {t.key: serve_prediction(t, results) for t in targets},
    }


def load_predictions(path: Path = PREDICTIONS) -> dict:
    return json.loads(path.read_text())["predictions"] if path.exists() else {}


def measured_run(results: list[dict], p: dict) -> dict | None:
    """Newest serve run with exactly this model, GPU setup and load."""
    found = None
    for r in results:
        if (
            r["suite"] == SERVE_SUITE[p.get("engine", "vllm")] and r.get("model") == p["model"] and r["gpu"] == p["gpu"]
            and r["quant"] == p["quant"] and r.get("tp", 1) == p["tp"] and r["serve"] == p["serve"]
        ):
            found = r
    return found


def _err(pred: float, meas: float) -> str:
    return f"{pred / meas - 1:+.0%}"


def comparison_table(results: list[dict] | None = None, preds: dict | None = None) -> str:
    results = load_results() if results is None else results
    preds = load_predictions() if preds is None else preds
    if not preds:
        return "_No frozen serve predictions yet._"
    rows = [
        "| Model | Workload | Replica | Curves | Load per replica (req/s, max_num_seqs) "
        "| TTFT ms p50: predicted / measured | TTFT ms p90: predicted / measured (error) "
        "| p90 × margin vs target | Margin covers measured p90 | TPOT ms: predicted / measured (target) | Targets met | Run |",
        "|---|---|---|---|---|---:|---:|---|---|---:|---|---|",
    ]
    p50_errs, p90_errs, tpot_errs = [], [], []
    for p in preds.values():
        s = p["serve"]
        gpus = p["gpu"] if p["tp"] == 1 else f"{p['tp']}× {p['gpu']}"
        head = (
            f"| {p['model']} | {p['workload']} ({p['replicas']} replica{'s' if p['replicas'] > 1 else ''}) "
            f"| {gpus} {p['quant']}{' (SGLang)' if p.get('engine') == 'sglang' else ''} | {p['curves']} | {s['rps']:g}, {s['max_num_seqs']} "
        )
        safe = p["ttft_p90_s"] * p["ttft_margin"]
        slo_ttft, slo_tpot = p["slo"]["ttft_p90_s"], p["slo"]["tpot_s"]
        r = measured_run(results, p)
        if r is None:
            rows.append(
                head + f"| {p['ttft_p50_s'] * 1e3:.0f} / – | {p['ttft_p90_s'] * 1e3:.0f} / – "
                f"| {safe * 1e3:.0f} vs {slo_ttft * 1e3:.0f} | – | {p['tpot_s'] * 1e3:.1f} / – ({slo_tpot * 1e3:.0f}) "
                f"| – | not run yet |"
            )
            continue
        res = r["result"]["serve"]["result"]
        m50, m90, mtpot = res["median_ttft_ms"] / 1e3, res["p90_ttft_ms"] / 1e3, res["median_tpot_ms"] / 1e3
        p50_errs.append(abs(p["ttft_p50_s"] / m50 - 1))
        p90_errs.append(abs(p["ttft_p90_s"] / m90 - 1))
        tpot_errs.append(abs(p["tpot_s"] / mtpot - 1))
        met = "yes" if m90 <= slo_ttft and mtpot <= slo_tpot else "**no**"
        rows.append(
            head + f"| {p['ttft_p50_s'] * 1e3:.0f} / {m50 * 1e3:.0f} "
            f"| {p['ttft_p90_s'] * 1e3:.0f} / {m90 * 1e3:.0f} ({_err(p['ttft_p90_s'], m90)}) "
            f"| {safe * 1e3:.0f} vs {slo_ttft * 1e3:.0f} | {'yes' if safe >= m90 else '**no**'} "
            f"| {p['tpot_s'] * 1e3:.1f} / {mtpot * 1e3:.1f} ({slo_tpot * 1e3:.0f}) "
            f"| {met} | [{r['id']}]({r['github_run_url']}) |"
        )
    out = "\n".join(rows)
    if p90_errs:
        med = statistics.median
        out += (
            f"\n\nOver {len(p90_errs)} run{'s' if len(p90_errs) > 1 else ''}, the median |error| is {med(p50_errs):.0%} for p50 TTFT, "
            f"{med(p90_errs):.0%} for p90 TTFT and {med(tpot_errs):.0%} for TPOT."
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), default="step5")
    ap.add_argument("--write", action="store_true", help="freeze this set's predictions to its file")
    args = ap.parse_args(argv)
    targets, path = SETS[args.set]
    data = predictions(targets)
    if args.write:
        if path.exists():
            raise SystemExit(f"{path} exists; predictions are frozen once committed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")
    for k, p in data["predictions"].items():
        print(
            f"{k}: {p['replicas']} replicas; one at {p['serve']} -> TTFT p50 {p['ttft_p50_s'] * 1e3:.0f} ms, "
            f"p90 {p['ttft_p90_s'] * 1e3:.0f} ms (x{p['ttft_margin']:.2f}), TPOT {p['tpot_s'] * 1e3:.1f} ms; {p['curves']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
