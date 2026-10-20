"""Compare benchmark results (results/*.json) with the roofline model.

Every measured number in the README comes from a committed result file, and
every predicted number from the roofline with ideal efficiency. The ratio
measured / roofline is what step 3 calibrates.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import roofline as rf
from .gpus import GPUS
from .models import DEFAULT_MODEL, load_model

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
NO_RESULTS = "_No benchmark results yet._"


def load_results(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Successful runs, oldest first. Later runs of the same (gpu, suite, quant) win in tables.

    Runs from before multi-model support carry no model/tp: they are Qwen-7B on one GPU.
    """
    if not results_dir.exists():
        return []
    recs = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
    recs = [r for r in recs if r.get("status") == "ok" and gpu_matches(r)]
    for r in recs:
        r.setdefault("model", DEFAULT_MODEL)
        r.setdefault("tp", 1)
    return sorted(recs, key=lambda r: r["finished_at"])


def gpu_matches(r: dict) -> bool:
    """False if the GPU's reported memory isn't the requested GPU's: Modal can substitute a
    larger card (run 051 asked for an A100-40GB and got an A100-80GB). Such runs stay in the
    ledger and spend but are left out of every table and fit."""
    total = r.get("result", {}).get("gpu_info", {}).get("memory.total")
    if total is None or r.get("gpu") not in GPUS:
        return True
    return 0.85 <= float(total) * 2**20 / GPUS[r["gpu"]].memory_bytes <= 1.15


def same_setup(r: dict, model: str = DEFAULT_MODEL, tp: int = 1) -> bool:
    return r.get("model", DEFAULT_MODEL) == model and r.get("tp", 1) == tp


def latest(
    results: list[dict], suite: str, model: str = DEFAULT_MODEL, tp: int = 1
) -> dict[tuple[str, str], dict]:
    """Newest run per (gpu, quant) of `suite` for one model and TP degree."""
    out: dict[tuple[str, str], dict] = {}
    for r in results:
        if r["suite"] == suite and (suite == "micro" or same_setup(r, model, tp)):
            out[(r["gpu"], r["quant"])] = r
    return dict(sorted(out.items(), key=lambda kv: (list(GPUS).index(kv[0][0]), kv[0][1])))


def _pct(measured: float, peak: float | None) -> str:
    return "–" if not peak else f"{100 * measured / peak:.0f}%"


def gpu_measured_table(results: list[dict]) -> str:
    runs = latest(results, "micro")
    if not runs:
        return NO_RESULTS
    rows = [
        "| GPU | Device (nvidia-smi) | Memory GiB (measured / spec) | GEMV read GB/s (% of datasheet) | Copy GB/s (%) | BF16 GEMM TFLOPS (%) | FP8 GEMM TFLOPS (%) | INT8 GEMM TOPS (%) | Run |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for (gpu, _), r in runs.items():
        spec = GPUS[gpu]
        res = r["result"]
        micro = res["micro"]
        mem = micro["torch_info"]["total_memory_bytes"] / 2**30
        bw = micro["bandwidth"]["median"]
        gemv = micro.get("gemv_bandwidth", {}).get("median")  # added after run 003
        gemv_cell = "–" if gemv is None else f"{gemv:,.0f} ({_pct(gemv, spec.mem_bw_gbs)})"

        def mm(dt: str) -> str:
            m = micro["matmul"][dt]
            if "error" in m:
                return "unsupported"
            peak = {"bf16": spec.bf16_tflops, "fp8": spec.fp8_tflops, "int8": spec.int8_tops}[dt]
            return f"{m['median']:.0f} ({_pct(m['median'], peak)})"

        rows.append(
            f"| {gpu} | {res['gpu_info']['name']} | {mem:.1f} / {spec.memory_gib:g} "
            f"| {gemv_cell} | {bw:,.0f} ({_pct(bw, spec.mem_bw_gbs)}) | {mm('bf16')} | {mm('fp8')} | {mm('int8')} "
            f"| [{r['id']}]({r['github_run_url']}) |"
        )
    return "\n".join(rows)


def vllm_memory_table(results: list[dict]) -> str:
    runs = latest(results, "vllm")
    if not runs:
        return NO_RESULTS
    m = load_model()
    rows = [
        "| GPU | Quant | Weights GiB (vLLM / roofline) | KV cache GiB (vLLM) | Non-KV overhead GiB (derived) | KV tokens (vLLM / roofline) | Run | Notes |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for (gpu, quant), r in runs.items():
        v = r["result"]["vllm"]
        logs = v.get("log_summary", {})
        total_gib = float(r["result"]["gpu_info"]["memory.total"]) / 1024  # nvidia-smi MiB
        util = v["engine"]["gpu_memory_utilization"]
        q = rf.QUANTIZATIONS[quant]
        pred_w = rf.weight_bytes(m, q) / 2**30
        pred_kv = rf.kv_cache_capacity_tokens(GPUS[gpu], m, q, gpu_memory_utilization=util)
        w, kv = logs.get("model_loading_gib"), logs.get("available_kv_cache_gib")
        overhead = f"{total_gib * util - w - kv:.2f}" if w is not None and kv is not None else "–"
        rows.append(
            f"| {gpu} | {quant} | {w if w is not None else '–'} / {pred_w:.2f} "
            f"| {kv if kv is not None else '–'} | {overhead} "
            f"| {v['engine']['kv_cache_size_tokens']:,} / {pred_kv:,} | [{r['id']}]({r['github_run_url']}) "
            f"| {'torch.compile fell back to the eager backend (Inductor error)' if v.get('compile_fallback') else ''} |"
        )
    return "\n".join(rows)


def ttft_table(results: list[dict]) -> str:
    runs = latest(results, "vllm")
    if not runs:
        return NO_RESULTS
    m = load_model()
    rows = [
        "| GPU | Quant | Prompt tokens | TTFT ms (vLLM, median) | TTFT ms (roofline) | vLLM / roofline |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (gpu, quant), r in runs.items():
        for t in r["result"]["vllm"]["ttft"]:
            pred = rf.ttft_s(GPUS[gpu], m, rf.QUANTIZATIONS[quant], t["prompt_len"])
            meas = t["ttft_s_median"]
            rows.append(
                f"| {gpu} | {quant} | {t['prompt_len']} | {meas * 1e3:.1f} | {pred * 1e3:.1f} | {meas / pred:.2f} |"
            )
    return "\n".join(rows)


def decode_table(results: list[dict]) -> str:
    """Decode step time vs roofline, plus what a request sees under load.

    "Decode step" is the clean per-step time from the two-length difference
    (bench/vllm_bench.py); runs before that change only have TPOT under load.
    """
    runs = latest(results, "vllm")
    if not runs:
        return NO_RESULTS
    m = load_model()
    rows = [
        "| GPU | Quant | Batch | Decode step ms (vLLM) | Decode step ms (roofline) | vLLM / roofline | Decode tok/s (vLLM) | $/1M output tokens | TPOT ms under load (median) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (gpu, quant), r in runs.items():
        for d in r["result"]["vllm"]["decode"]:
            if "skipped" in d:
                continue
            step = d.get("decode_step_s")
            ctx = d.get("decode_context_len", d["prompt_len"] + d["output_len"] // 2)
            pred = rf.decode_step(GPUS[gpu], m, rf.QUANTIZATIONS[quant], d["batch"], ctx).time_s
            if step is None:
                meas_cells = "– | " + f"{pred * 1e3:.2f} | – | – | –"
            else:
                tps = d["batch"] / step
                cost = rf.cost_per_million_tokens(r["price_per_hour"], tps)
                meas_cells = f"{step * 1e3:.2f} | {pred * 1e3:.2f} | {step / pred:.2f} | {tps:,.0f} | {cost:.3f}"
            rows.append(f"| {gpu} | {quant} | {d['batch']} | {meas_cells} | {d['tpot_s_median'] * 1e3:.2f} |")
    return "\n".join(rows)


def spend_summary(results_dir: Path = RESULTS_DIR) -> str:
    ledger = results_dir / "ledger.jsonl"
    if not ledger.exists():
        return NO_RESULTS
    entries = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    total = sum(e["cost_usd_upper_bound"] for e in entries)
    ok = sum(e["status"] == "ok" for e in entries)
    return (
        f"{len(entries)} runs ({ok} succeeded), at most **${total:.2f}** of GPU time "
        "(runner wall-clock × Modal GPU price × 1.1 for CPU/RAM)."
    )


# Serve runs kept out of the fit to check the serve calibration on unseen configurations.
HOLDOUT_RUNS = frozenset({
    "029-a100-80gb-serve-bf16-longctx", "030-h100-serve-fp8-longout", "031-l4-serve-fp8-short",
})


def serve_table(results: list[dict]) -> str:
    """Online serving runs vs the model (calibrated on all serve runs) for the same load."""
    from .calibration import measured_configs
    from .recommend import predict, serve_calibration, serve_runs

    configs = measured_configs(results)
    runs = serve_runs(results, configs)
    if not runs:
        return NO_RESULTS
    cal = serve_calibration(results, configs)
    rows = [
        "| GPU | Quant | Load (prompt/output tokens, req/s) | max_num_seqs | TTFT ms p50: predicted / measured | TTFT ms p90: predicted / measured | TPOT ms p50: predicted / measured | Throughput req/s (achieved / offered) | Run |",
        "|---|---|---|---:|---|---|---|---|---|",
    ]
    for r, c, w in runs:
        s = r["serve"]
        res = r["result"]["serve"]["result"]
        q = predict(c, w, s["max_num_seqs"], cal.overhead_s)
        p50 = f"{q.ttft_p50_s * 1e3:.0f}" if q else "overloaded"
        p90 = f"{q.ttft_p90_s * 1e3:.0f}" if q else "overloaded"
        tpot = f"{q.busy.tpot_s * 1e3:.1f}" if q else "overloaded"
        rows.append(
            f"| {r['gpu']} | {r['quant']} | {s['prompt']}/{s['output']}, {s['rps']:g} | {s['max_num_seqs']} "
            f"| {p50} / {res['median_ttft_ms']:.0f} | {p90} / {res['p90_ttft_ms']:.0f} "
            f"| {tpot} / {res['median_tpot_ms']:.1f} "
            f"| {res['request_throughput']:.2f} / {s['rps']:g} | [{r['id']}]({r['github_run_url']}) |"
        )
    return "\n".join(rows)


def margin_table(results: list[dict]) -> str:
    """The serving overhead and p90 TTFT safety factors the recommender applies."""
    from .recommend import serve_calibration

    cal = serve_calibration(results)
    if not cal.ratios:
        return NO_RESULTS
    rows = [
        f"Serving overhead: **{cal.overhead_s * 1e3:.1f} ms** (median measured − predicted p50 TTFT over {len(cal.ratios)} serve runs).",
        "",
        "| GPU | Measured / predicted p90 TTFT per serve run | Margin applied |",
        "|---|---|---:|",
    ]
    for gpu in GPUS:
        runs = [f"{run_id} ×{ratio:.2f}" for run_id, g, ratio in cal.ratios if g == gpu]
        rows.append(f"| {gpu} | {', '.join(runs) if runs else '_no serve run: largest margin_'} | ×{cal(gpu):.2f} |")
    return "\n".join(rows)


def holdout_table(results: list[dict]) -> str:
    """Fit the serve calibration without HOLDOUT_RUNS, then predict those runs."""
    from .calibration import measured_configs
    from .recommend import predict, serve_calibration, serve_runs

    configs = measured_configs(results)
    held = [(r, c, w) for r, c, w in serve_runs(results, configs) if r["id"] in HOLDOUT_RUNS]
    if not held:
        return NO_RESULTS
    cal = serve_calibration(results, configs, exclude=HOLDOUT_RUNS)
    rows = [
        f"Fitted without these runs: overhead {cal.overhead_s * 1e3:.1f} ms; margins "
        + ", ".join(f"{g} ×{m:.2f}" for g, m in cal.by_gpu.items())
        + f", others ×{cal.default:.2f}.",
        "",
        "| Run | TTFT ms p50: predicted / measured | TTFT ms p90: predicted / measured (error) | p90 × margin vs measured | TPOT ms: predicted / measured |",
        "|---|---|---|---|---|",
    ]
    for r, c, w in held:
        res = r["result"]["serve"]["result"]
        q = predict(c, w, r["serve"]["max_num_seqs"], cal.overhead_s)
        p90, meas = q.ttft_p90_s * 1e3, res["p90_ttft_ms"]
        safe = p90 * cal(r["gpu"])
        rows.append(
            f"| {r['id']} | {q.ttft_p50_s * 1e3:.0f} / {res['median_ttft_ms']:.0f} "
            f"| {p90:.0f} / {meas:.0f} ({p90 / meas - 1:+.0%}) "
            f"| {safe:.0f} vs {meas:.0f}: {'covers' if safe >= meas else '**misses**'} "
            f"| {q.busy.tpot_s * 1e3:.1f} / {res['median_tpot_ms']:.1f} |"
        )
    return "\n".join(rows)
