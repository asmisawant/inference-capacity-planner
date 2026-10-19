"""Predict vLLM behaviour for a model (or GPU count) that was never benchmarked.

Everything learned from the Qwen2.5-7B runs is a per-(GPU, quant) correction to the
roofline; the roofline itself already knows the new model's shape and the tensor-parallel
split. So for a new model M on the same GPU and quant, from that Qwen-7B vllm run:

    TTFT(p)         = c0 + c1 * roofline_ttft_M(p)          (c0, c1: Qwen-7B TTFT fit)
    decode step(B)  = a * roofline_step_M(B, ctx) + b * B   (a, b: Qwen-7B decode fit)
    KV cache bytes  = memory.total * 0.92 - weights_M / tp - overhead
                      (overhead: Qwen-7B's derived non-KV memory, which absorbs its small
                       weight-size difference to the roofline)

Rooflines use the GPU efficiencies from the micro suite. The baseline next to it is that
efficiency-corrected roofline alone, to show what the transferred fit adds.

`python -m capacity_planner.transfer --write PATH` freezes the predictions for TARGETS.
That file is committed before the runs, and the comparison table reads it, never this
code, so later changes here can't move the goalposts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from . import roofline as rf
from .calibration import Fit, fits, gpu_efficiency
from .gpus import GPUS, GPUSpec
from .models import DEFAULT_MODEL, ModelSpec, load_model
from .validation import latest, load_results

# Mirror bench/vllm_bench.py (tests check they agree).
TTFT_PROMPT_LENS = (128, 512, 1024, 2048, 4000)
DECODE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DECODE_SEQ_TOKENS = 512 + 192  # prompt + long output: the bench skips batches whose KV doesn't fit
DECODE_CONTEXT_LEN = 512 + (64 + 192) // 2
MAX_NUM_SEQS = 256

PREDICTIONS = Path(__file__).resolve().parents[2] / "predictions" / "transfer.json"


@dataclass(frozen=True)
class Target:
    model: str
    gpu: str
    quant: str
    tp: int = 1

    @property
    def key(self) -> str:
        return f"{self.model}|{self.gpu}|{self.quant}|tp{self.tp}"


# Unmeasured setups to predict and then run: a dense 8B model with a different shape
# (more layers, 2x KV heads per layer, small vocab), and a 32B model that needs FP8, AWQ
# or two GPUs. Only (GPU, quant) pairs with a Qwen-7B fit can be transferred.
TARGETS = (
    Target("Granite-3.1-8B-Instruct", "L4", "bf16"),
    Target("Granite-3.1-8B-Instruct", "L4", "fp8"),
    Target("Granite-3.1-8B-Instruct", "A100-40GB", "bf16"),
    Target("Granite-3.1-8B-Instruct", "H100", "bf16"),
    Target("Granite-3.1-8B-Instruct", "H100", "fp8"),
    Target("Qwen2.5-32B-Instruct", "H100", "fp8"),
    Target("Qwen2.5-32B-Instruct", "A100-80GB", "awq-int4"),
    Target("Qwen2.5-32B-Instruct", "H100", "bf16", tp=2),
    Target("Qwen2.5-32B-Instruct", "A100-80GB", "bf16", tp=2),
)


@dataclass
class TransferredConfig:
    """Predicted vLLM curves for `model` on `tp` x `gpu`, from a Qwen-7B (gpu, quant) run."""

    gpu: GPUSpec
    quant: rf.Quantization
    model: ModelSpec
    tp: int
    eff: rf.Efficiency
    fit: Fit
    memory_total_gib: float  # nvidia-smi memory.total on the source run
    overhead_gib: float  # non-KV, non-weight memory per GPU on the source run
    gpu_memory_utilization: float = rf.DEFAULT_GPU_MEMORY_UTILIZATION

    @property
    def source_run(self) -> str:
        return self.fit.run_id

    def roofline_ttft_s(self, prompt_len: int) -> float:
        return rf.ttft_s(self.gpu, self.model, self.quant, prompt_len, eff=self.eff, tp=self.tp)

    def roofline_step_s(self, batch: int, context_len: int = DECODE_CONTEXT_LEN) -> float:
        return rf.decode_step(self.gpu, self.model, self.quant, batch, context_len, eff=self.eff, tp=self.tp).time_s

    def ttft_s(self, prompt_len: int) -> float:
        return self.fit.ttft_c0_s + self.fit.ttft_c1 * self.roofline_ttft_s(prompt_len)

    def decode_step_s(self, batch: int, context_len: int = DECODE_CONTEXT_LEN) -> float:
        return self.fit.decode_a * self.roofline_step_s(batch, context_len) + self.fit.decode_b_s * batch

    @property
    def weights_gib(self) -> float:
        return rf.weight_bytes(self.model, self.quant) / self.tp / 2**30

    @property
    def kv_cache_gib(self) -> float:
        return self.memory_total_gib * self.gpu_memory_utilization - self.weights_gib - self.overhead_gib

    @property
    def kv_cache_tokens(self) -> int:
        return max(0, int(self.kv_cache_gib * 2**30 // self.model.kv_bytes_per_token(tp=self.tp)))

    def bench_batches(self) -> list[int]:
        """Decode batches the vllm suite would run, given the predicted KV cache."""
        return [b for b in DECODE_BATCHES if b <= MAX_NUM_SEQS and b * DECODE_SEQ_TOKENS <= self.kv_cache_tokens]


def source_memory(run: dict) -> tuple[float, float]:
    """(memory.total GiB, derived non-KV overhead GiB) of a Qwen-7B vllm run, where the
    overhead is measured against the roofline weight size."""
    v = run["result"]["vllm"]
    total = float(run["result"]["gpu_info"]["memory.total"]) / 1024
    util = v["engine"]["gpu_memory_utilization"]
    kv = v["log_summary"]["available_kv_cache_gib"]
    w = rf.weight_bytes(load_model(DEFAULT_MODEL), rf.QUANTIZATIONS[run["quant"]]) / 2**30
    return total, total * util - w - kv


def transfer(target: Target, results: list[dict] | None = None) -> TransferredConfig:
    results = load_results() if results is None else results
    fit = {(f.gpu, f.quant): f for f in fits(results)}.get((target.gpu, target.quant))
    if fit is None:
        raise ValueError(f"no Qwen-7B vllm fit for {target.gpu} {target.quant}")
    run = latest(results, "vllm")[(target.gpu, target.quant)]
    total, overhead = source_memory(run)
    return TransferredConfig(
        gpu=GPUS[target.gpu],
        quant=rf.QUANTIZATIONS[target.quant],
        model=load_model(target.model),
        tp=target.tp,
        eff=gpu_efficiency(results)[target.gpu],
        fit=fit,
        memory_total_gib=total,
        overhead_gib=overhead,
    )


def prediction(target: Target, results: list[dict] | None = None) -> dict:
    c = transfer(target, results)
    return {
        "model": target.model,
        "gpu": target.gpu,
        "quant": target.quant,
        "tp": target.tp,
        "source_run": c.source_run,
        "weights_gib": round(c.weights_gib, 3),
        "kv_cache_tokens": c.kv_cache_tokens,
        "ttft_s": {str(p): c.ttft_s(p) for p in TTFT_PROMPT_LENS},
        "ttft_s_roofline": {str(p): c.roofline_ttft_s(p) for p in TTFT_PROMPT_LENS},
        "decode_step_s": {str(b): c.decode_step_s(b) for b in c.bench_batches()},
        "decode_step_s_roofline": {str(b): c.roofline_step_s(b) for b in c.bench_batches()},
    }


def predictions(targets=TARGETS, results: list[dict] | None = None) -> dict:
    results = load_results() if results is None else results
    return {
        "method": "capacity_planner.transfer: Qwen2.5-7B per-(GPU, quant) fits applied to the target's roofline",
        "decode_context_len": DECODE_CONTEXT_LEN,
        "predictions": {t.key: prediction(t, results) for t in targets},
    }


def load_predictions(path: Path = PREDICTIONS) -> dict:
    return json.loads(path.read_text())["predictions"] if path.exists() else {}


def _max_err(pred: dict, measured: dict) -> float | None:
    """Signed relative error with the largest magnitude over the points both have."""
    errs = [pred[k] / measured[k] - 1 for k in pred if k in measured]
    return max(errs, key=abs) if errs else None


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x:+.0%}"


def measured_run(results: list[dict], p: dict) -> dict | None:
    runs = latest(results, "vllm", p["model"], p["tp"])
    return runs.get((p["gpu"], p["quant"]))


def comparison_table(results: list[dict] | None = None, preds: dict | None = None) -> str:
    results = load_results() if results is None else results
    preds = load_predictions() if preds is None else preds
    if not preds:
        return "_No frozen predictions yet._"
    rows = [
        "| Model | GPUs | Quant | KV tokens: predicted / measured | TTFT ms at 1024: predicted / measured "
        "| TTFT worst error: transfer / roofline | Decode step ms at batch 1: predicted / measured "
        "| Decode worst error: transfer / roofline | Run |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for p in preds.values():
        r = measured_run(results, p)
        gpus = p["gpu"] if p["tp"] == 1 else f"{p['tp']}× {p['gpu']}"
        head = f"| {p['model']} | {gpus} | {p['quant']} "
        if r is None:
            rows.append(
                head + f"| {p['kv_cache_tokens']:,} / – | {p['ttft_s']['1024'] * 1e3:.0f} / – | – "
                f"| {p['decode_step_s']['1'] * 1e3:.1f} / – | – | not run yet |"
            )
            continue
        v = r["result"]["vllm"]
        ttft = {str(t["prompt_len"]): t["ttft_s_median"] for t in v["ttft"]}
        step = {str(d["batch"]): d["decode_step_s"] for d in v["decode"] if "decode_step_s" in d}
        kv = v["engine"]["kv_cache_size_tokens"]
        rows.append(
            head + f"| {p['kv_cache_tokens']:,} / {kv:,} "
            f"| {p['ttft_s']['1024'] * 1e3:.0f} / {ttft['1024'] * 1e3:.0f} "
            f"| {_pct(_max_err(p['ttft_s'], ttft))} / {_pct(_max_err(p['ttft_s_roofline'], ttft))} "
            f"| {p['decode_step_s']['1'] * 1e3:.1f} / {step['1'] * 1e3:.1f} "
            f"| {_pct(_max_err(p['decode_step_s'], step))} / {_pct(_max_err(p['decode_step_s_roofline'], step))} "
            f"| [{r['id']}]({r['github_run_url']}) |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", type=Path, help="freeze predictions for TARGETS to this file")
    args = ap.parse_args(argv)
    data = predictions()
    if args.write:
        if args.write.exists():
            raise SystemExit(f"{args.write} exists; predictions are frozen once committed")
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(data, indent=2) + "\n")
    for k, p in data["predictions"].items():
        ttft = ", ".join(f"{v * 1e3:.0f}" for v in p["ttft_s"].values())
        step = ", ".join(f"{b}:{v * 1e3:.1f}" for b, v in p["decode_step_s"].items())
        print(f"{k}: KV {p['kv_cache_tokens']:,} tok; TTFT ms {ttft}; step ms {step}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
