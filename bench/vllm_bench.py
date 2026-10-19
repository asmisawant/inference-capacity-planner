"""vLLM offline benchmark: KV-cache capacity, TTFT and TPOT for one model on one GPU (or a TP group).

Runs inside the Modal container as its own process (python -m bench.vllm_bench),
so the parent can capture vLLM's logs, which hold the memory breakdown.

Timing comes from vLLM's per-request stats (RequestOutput.metrics, enabled by
disable_log_stats=False), the same timestamps vLLM uses for its own
TTFT/TPOT metrics:
    TTFT = first_token_ts - queued_ts
    TPOT = (last_token_ts - first_token_ts) / (generated tokens - 1)
TPOT under load includes steps shared with other requests' prefill. The clean
per-step decode time comes from two runs with different output lengths:
    decode_step = (span_long - span_short) / (long - short)
where span = first request queued -> last token. Prompts are random token ids and prefix caching is off, so nothing is cached
between requests.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time

QUANTS = ("bf16", "fp8", "awq-int4")


def hf_model(model: str, quant: str) -> tuple[str, str | None]:
    """(Hugging Face repo, vLLM `quantization` argument) for a registry model."""
    from capacity_planner.models import registry

    entry = registry()[model]
    if quant == "bf16":
        return entry.repo, None
    if quant == "fp8":
        return entry.repo, "fp8"  # online FP8 weight quantization
    if quant == "awq-int4":
        if not entry.awq_repo:
            raise ValueError(f"no AWQ checkpoint for {model}")
        return entry.awq_repo, None  # vLLM picks awq_marlin
    raise ValueError(quant)


MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 256
TTFT_PROMPT_LENS = (128, 512, 1024, 2048, 4000)
TTFT_TRIALS = 5
DECODE_PROMPT_LEN = 512
# Each batch runs twice, with a short and a long output. Both runs spend the same time on
# prefill (and on decode steps that share the GPU with prefill chunks), so the difference
# in engine time over the extra tokens is pure full-batch decode.
DECODE_OUTPUT_SHORT = 64
DECODE_OUTPUT_LEN = 192
DECODE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)

_rng = random.Random(0)
_vocab = 150_000  # set from the model config in main()


def random_prompt(n: int) -> dict:
    # Ordinary vocabulary ids (skipping the low, often special, ids); random so no two
    # prompts share a prefix.
    return {"prompt_token_ids": [_rng.randrange(1000, _vocab - 1000) for _ in range(n)]}


def request_timing(out) -> dict:
    m = out.metrics
    n = len(out.outputs[0].token_ids)
    t = {"generated_tokens": n}
    if m is not None and m.first_token_ts:
        t["ttft_s"] = m.first_token_ts - m.queued_ts
        t["prefill_s"] = m.first_token_ts - m.scheduled_ts
        t["queue_s"] = m.scheduled_ts - m.queued_ts
        if n > 1:
            t["tpot_s"] = (m.last_token_ts - m.first_token_ts) / (n - 1)
        t["first_token_ts"] = m.first_token_ts
        t["last_token_ts"] = m.last_token_ts
        t["queued_ts"] = m.queued_ts
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=QUANTS, required=True)
    ap.add_argument("--model", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--revision")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--compile-backend", help="override vLLM's compilation backend (fallback only)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import vllm
    from vllm import LLM, SamplingParams

    global _vocab
    from capacity_planner.models import load_model

    _vocab = load_model(args.model).vocab_size
    model, quantization = hf_model(args.model, args.quant)
    t0 = time.perf_counter()
    llm = LLM(
        model=model,
        tensor_parallel_size=args.tp,
        revision=args.revision,
        quantization=quantization,
        dtype="auto",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        disable_log_stats=False,
        seed=0,
        compilation_config={"backend": args.compile_backend} if args.compile_backend else None,
    )
    init_s = time.perf_counter() - t0
    cache = llm.llm_engine.vllm_config.cache_config
    kv_tokens = cache.kv_cache_size_tokens or (cache.num_gpu_blocks * cache.block_size)
    result = {
        "vllm_version": vllm.__version__,
        "model": model,
        "tensor_parallel_size": args.tp,
        "revision": args.revision,
        "quantization": quantization,
        "dtype": str(llm.llm_engine.model_config.dtype),
        "engine": {
            "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": MAX_NUM_SEQS,
            "gpu_memory_utilization": cache.gpu_memory_utilization,
            "enable_prefix_caching": False,
            "num_gpu_blocks": cache.num_gpu_blocks,
            "block_size": cache.block_size,
            "kv_cache_size_tokens": kv_tokens,
            "kv_cache_dtype": str(cache.cache_dtype),
        },
        "init_s": init_s,
    }

    def generate(prompts, max_tokens):
        sp = SamplingParams(
            max_tokens=max_tokens, min_tokens=max_tokens, ignore_eos=True, temperature=0.0, detokenize=False
        )
        t = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        return outs, time.perf_counter() - t

    warm, _ = generate([random_prompt(256) for _ in range(4)], 16)
    if warm[0].metrics is None or not warm[0].metrics.first_token_ts:
        raise RuntimeError("vLLM returned no per-request metrics; timing would be wall-clock only")

    # TTFT: one request at a time on an idle engine.
    ttft = []
    for p in TTFT_PROMPT_LENS:
        trials = []
        for _ in range(TTFT_TRIALS):
            outs, wall = generate([random_prompt(p)], 1)
            trials.append({**request_timing(outs[0]), "wall_s": wall})
        ttft.append(
            {
                "prompt_len": p,
                "ttft_s_median": statistics.median(t["ttft_s"] for t in trials),
                "wall_s_median": statistics.median(t["wall_s"] for t in trials),
                "trials": trials,
            }
        )
    result["ttft"] = ttft

    # Decode: B requests of DECODE_PROMPT_LEN prompt, all submitted at once, run with a short
    # and a long output length.
    def decode_run(b: int, out_len: int) -> tuple[list[dict], float, float]:
        outs, wall = generate([random_prompt(DECODE_PROMPT_LEN) for _ in range(b)], out_len)
        reqs = [request_timing(o) for o in outs]
        assert all(r["generated_tokens"] == out_len for r in reqs), "short generation"
        span = max(r["last_token_ts"] for r in reqs) - min(r["queued_ts"] for r in reqs)
        return reqs, span, wall

    decode = []
    per_seq = DECODE_PROMPT_LEN + DECODE_OUTPUT_LEN
    for b in DECODE_BATCHES:
        if b > MAX_NUM_SEQS or b * per_seq > kv_tokens:
            decode.append({"batch": b, "skipped": "exceeds KV cache or max_num_seqs"})
            continue
        _, span_short, _ = decode_run(b, DECODE_OUTPUT_SHORT)
        reqs, span, wall = decode_run(b, DECODE_OUTPUT_LEN)
        step = (span - span_short) / (DECODE_OUTPUT_LEN - DECODE_OUTPUT_SHORT)
        tpots = [r["tpot_s"] for r in reqs]
        decode.append(
            {
                "batch": b,
                "prompt_len": DECODE_PROMPT_LEN,
                "output_len": DECODE_OUTPUT_LEN,
                "output_len_short": DECODE_OUTPUT_SHORT,
                # mean context over the steps that the difference measures
                "decode_context_len": DECODE_PROMPT_LEN + (DECODE_OUTPUT_SHORT + DECODE_OUTPUT_LEN) // 2,
                "decode_step_s": step,
                "decode_tokens_per_s": b / step,
                "tpot_s_median": statistics.median(tpots),
                "tpot_s_max": max(tpots),
                "ttft_s_median": statistics.median(r["ttft_s"] for r in reqs),
                "ttft_s_max": max(r["ttft_s"] for r in reqs),
                "output_tokens_per_s": b * DECODE_OUTPUT_LEN / span,
                "engine_span_s": span,
                "engine_span_short_s": span_short,
                "wall_s": wall,
            }
        )
    result["decode"] = decode

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
