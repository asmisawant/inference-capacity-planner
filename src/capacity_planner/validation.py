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
from .models import load_model

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
NO_RESULTS = "_No benchmark results yet._"


def load_results(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Successful runs, oldest first. Later runs of the same (gpu, suite, quant) win in tables."""
    if not results_dir.exists():
        return []
    recs = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
    recs = [r for r in recs if r.get("status") == "ok"]
    return sorted(recs, key=lambda r: r["finished_at"])


def latest(results: list[dict], suite: str) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for r in results:
        if r["suite"] == suite:
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


def serve_table(results: list[dict]) -> str:
    """Online serving runs vs the recommender's prediction for the same load."""
    from .calibration import measured_configs
    from .recommend import Workload, predict

    runs = [r for r in results if r["suite"] == "serve"]
    if not runs:
        return NO_RESULTS
    configs = {c.key: c for c in measured_configs(results)}
    rows = [
        "| GPU | Quant | Load (prompt/output tokens, req/s) | max_num_seqs | TTFT ms predicted / measured p50 (p90) | TPOT ms predicted / measured p50 (p90) | Throughput req/s (achieved / offered) | Run |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for r in runs:
        s = r["serve"]
        res = r["result"]["serve"]["result"]
        w = Workload(s["prompt"], s["output"], s["rps"], ttft_slo_s=1e9, tpot_slo_s=1e9)
        c = configs.get((r["gpu"], r["quant"]))
        p = predict(c, w, s["max_num_seqs"]) if c else None
        pt = f"{p.ttft_s * 1e3:.0f}" if p else "overloaded"
        pp = f"{p.tpot_s * 1e3:.1f}" if p else "overloaded"
        rows.append(
            f"| {r['gpu']} | {r['quant']} | {s['prompt']}/{s['output']}, {s['rps']:g} | {s['max_num_seqs']} "
            f"| {pt} / {res['median_ttft_ms']:.0f} ({res['p90_ttft_ms']:.0f}) "
            f"| {pp} / {res['median_tpot_ms']:.1f} ({res['p90_tpot_ms']:.1f}) "
            f"| {res['request_throughput']:.2f} / {s['rps']:g} | [{r['id']}]({r['github_run_url']}) |"
        )
    return "\n".join(rows)
