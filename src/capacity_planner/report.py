"""Generate the README tables from the spec table and roofline model.

    python -m capacity_planner.report                 # print tables
    python -m capacity_planner.report --update-readme # rewrite the generated README sections

A test fails if the README drifts from this output, so every number in those
sections comes from code in this repo.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import roofline as rf
from . import calibration, engines, recommend, serve_check, transfer, validation
from .gpus import GPUS
from .models import load_model

# Example workload for the README table. Change it here, not in the README.
PROMPT_LEN = 1024
OUTPUT_LEN = 256
REPORT_QUANTS = ("bf16", "fp8", "awq-int4")


def gpu_table() -> str:
    rows = [
        "| GPU | Memory (GiB) | Bandwidth (GB/s) | BF16 dense TFLOPS | FP8 dense TFLOPS | Ridge point (FLOP/byte, BF16) | Modal $/hr |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for g in GPUS.values():
        fp8 = "–" if g.fp8_tflops is None else f"{g.fp8_tflops:g}"
        rows.append(
            f"| {g.name} | {g.memory_gib:g} | {g.mem_bw_gbs:g} | {g.bf16_tflops:g} | {fp8} "
            f"| {g.ridge_point('bf16'):.0f} | {g.price_per_hour:.2f} |"
        )
    return "\n".join(rows)


def model_table() -> str:
    m = load_model()
    return "\n".join(
        [
            "| Quantity | Value |",
            "|---|---:|",
            f"| Total parameters | {m.total_params:,} |",
            f"| Non-embedding parameters | {m.non_embedding_params:,} |",
            f"| KV cache per token (BF16) | {m.kv_bytes_per_token('bf16'):,.0f} bytes |",
            *(
                f"| Weights, {q} | {rf.weight_bytes(m, rf.QUANTIZATIONS[q]) / 2**30:.2f} GiB |"
                for q in REPORT_QUANTS
            ),
        ]
    )


def roofline_table() -> str:
    m = load_model()
    ctx = PROMPT_LEN + OUTPUT_LEN
    rows = [
        "| GPU | Quant | Compute dtype | KV capacity (tokens) | Max concurrent seqs | TTFT (ms) | TPOT @ batch 1 (ms) | Decode tok/s @ max seqs | Decode-only $/1M output tokens |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for g in GPUS.values():
        for qname in REPORT_QUANTS:
            q = rf.QUANTIZATIONS[qname]
            cap = rf.kv_cache_capacity_tokens(g, m, q)
            seqs = rf.max_concurrent_sequences(cap, ctx)
            dtype = rf.effective_compute_dtype(g, q)
            if seqs == 0:
                rows.append(f"| {g.name} | {qname} | {dtype} | {cap:,} | 0 | – | – | – | – |")
                continue
            ttft = rf.ttft_s(g, m, q, PROMPT_LEN) * 1e3
            tpot1 = rf.decode_step(g, m, q, 1, ctx).time_s * 1e3
            step = rf.decode_step(g, m, q, seqs, ctx)
            tps = rf.decode_throughput_tokens_per_s(step, seqs)
            cost = rf.cost_per_million_tokens(g.price_per_hour, tps)
            rows.append(
                f"| {g.name} | {qname} | {dtype} | {cap:,} | {seqs} | {ttft:.1f} | {tpot1:.2f} "
                f"| {tps:,.0f} | {cost:.3f} |"
            )
    return "\n".join(rows)


SECTIONS = {
    "gpu_table": gpu_table,
    "model_table": model_table,
    "roofline_table": roofline_table,
    "measured_gpu": lambda: validation.gpu_measured_table(validation.load_results()),
    "measured_vllm_memory": lambda: validation.vllm_memory_table(validation.load_results()),
    "measured_ttft": lambda: validation.ttft_table(validation.load_results()),
    "measured_decode": lambda: validation.decode_table(validation.load_results()),
    "bench_spend": validation.spend_summary,
    "calibration_efficiency": lambda: calibration.efficiency_table(),
    "calibration_fits": lambda: calibration.fit_table(),
    "recommend_examples": lambda: recommend.examples_markdown(),
    "serve_validation": lambda: validation.serve_table(validation.load_results()),
    "ttft_margins": lambda: validation.margin_table(validation.load_results()),
    "serve_holdout": lambda: validation.holdout_table(validation.load_results()),
    "transfer_check": lambda: transfer.comparison_table(),
    "recommend_other_models": lambda: recommend.other_models_markdown(),
    "serve_check": lambda: serve_check.comparison_table(),
    "engine_comparison": lambda: engines.comparison_table(),
    "calibration_fits_sglang": lambda: calibration.fit_table(engine="sglang"),
    "serve_margin_check": lambda: serve_check.comparison_table(
        preds=serve_check.load_predictions(serve_check.MARGIN_CHECK_PREDICTIONS)
    ),
}


def render_readme(text: str) -> str:
    for key, fn in SECTIONS.items():
        pattern = re.compile(
            rf"(<!-- BEGIN GENERATED: {key} -->).*?(<!-- END GENERATED: {key} -->)", re.S
        )
        if not pattern.search(text):
            raise ValueError(f"README is missing the generated section {key!r}")
        text = pattern.sub(lambda mt: f"{mt.group(1)}\n{fn()}\n{mt.group(2)}", text)
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update-readme", type=Path, nargs="?", const=Path("README.md"))
    args = parser.parse_args(argv)
    if args.update_readme:
        path: Path = args.update_readme
        path.write_text(render_readme(path.read_text()))
        print(f"updated {path}")
    else:
        for key, fn in SECTIONS.items():
            print(f"## {key}\n\n{fn()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
