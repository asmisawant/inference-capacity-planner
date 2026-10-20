"""vLLM vs SGLang on the same model, GPU and quantization (offline suites).

Both benches use the same grid (prompt lengths, batches, 640-token decode context) and
the same request settings; each engine keeps its own defaults otherwise, notably how
much memory it gives to weights + KV cache (vLLM gpu_memory_utilization, SGLang
mem_fraction_static). TTFT is vLLM's queued -> first token and SGLang's end-to-end
latency of a one-token request; the decode step comes from the same two-length method.
"""

from __future__ import annotations

from .gpus import GPUS
from .validation import NO_RESULTS, latest, load_results


def _curves(r: dict) -> tuple[dict[int, float], dict[int, float], dict]:
    v = r["result"][r["suite"]]
    ttft = {t["prompt_len"]: t["ttft_s_median"] for t in v["ttft"]}
    step = {d["batch"]: d["decode_step_s"] for d in v["decode"] if "decode_step_s" in d}
    return ttft, step, v


def _mem_fraction(v: dict) -> float | None:
    e = v["engine"]
    return e.get("gpu_memory_utilization", e.get("mem_fraction_static"))


def comparison_table(results: list[dict] | None = None) -> str:
    results = load_results() if results is None else results
    vl, sg = latest(results, "vllm"), latest(results, "sglang")
    keys = sorted(set(vl) | set(sg), key=lambda k: (list(GPUS).index(k[0]), k[1]))
    if not sg:
        return NO_RESULTS
    rows = [
        "| GPU | Quant | Memory fraction for weights + KV: vLLM / SGLang | KV tokens: vLLM / SGLang "
        "| TTFT ms at 128: vLLM / SGLang | TTFT ms at 1024: vLLM / SGLang | TTFT ms at 4000: vLLM / SGLang "
        "| Decode step ms at batch 1: vLLM / SGLang | Largest batch both ran: step ms vLLM / SGLang | Runs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def pair(a, b, fmt):
        return f"{fmt(a) if a is not None else '–'} / {fmt(b) if b is not None else '–'}"

    ms = lambda x: f"{x * 1e3:.0f}"  # noqa: E731
    ms1 = lambda x: f"{x * 1e3:.1f}"  # noqa: E731
    for k in keys:
        a, b = vl.get(k), sg.get(k)
        ta, sa, va = _curves(a) if a else ({}, {}, None)
        tb, sb, vb = _curves(b) if b else ({}, {}, None)
        common = sorted(set(sa) & set(sb))
        big = (
            f"{common[-1]}: {ms1(sa[common[-1]])} / {ms1(sb[common[-1]])}" if common else "–"
        )
        runs = " · ".join(f"[{r['id']}]({r['github_run_url']})" for r in (a, b) if r)
        rows.append(
            f"| {k[0]} | {k[1]} "
            f"| {pair(_mem_fraction(va) if va else None, _mem_fraction(vb) if vb else None, lambda x: f'{x:.2f}')} "
            f"| {pair(va['engine']['kv_cache_size_tokens'] if va else None, vb['engine']['kv_cache_size_tokens'] if vb else None, lambda x: f'{x:,}')} "
            f"| {pair(ta.get(128), tb.get(128), ms)} | {pair(ta.get(1024), tb.get(1024), ms)} "
            f"| {pair(ta.get(4000), tb.get(4000), ms)} | {pair(sa.get(1), sb.get(1), ms1)} | {big} | {runs} |"
        )
    return "\n".join(rows)


def sglang_calibration_summary(results: list[dict] | None = None) -> str:
    """What the recommender applies to SGLang configs under load."""
    from .recommend import engine_calibrations

    cal, own = engine_calibrations(results)["sglang"]
    if not own:
        return "_No SGLang serve runs yet: SGLang borrows vLLM's serving overhead and margins._"
    runs = ", ".join(f"{i} ×{x:.2f}" for i, _, x in cal.ratios)
    margins = ", ".join(f"{g} ×{m:.2f}" for g, m in cal.by_gpu.items())
    return (
        f"SGLang serving overhead **{cal.overhead_s * 1e3:.1f} ms** (median measured − predicted p50 TTFT), "
        f"from {len(cal.ratios)} serve runs within the CUDA-graph limit (measured / predicted p90: {runs}). "
        f"p90 margins: {margins}; GPUs without an SGLang serve run take vLLM's margin for that GPU."
    )
