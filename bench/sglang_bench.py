"""SGLang offline benchmark: the same measurements as bench/vllm_bench.py, on SGLang.

Runs inside the Modal container (python -m bench.sglang_bench) on SGLang's official image.
Engine settings mirror the vLLM bench where both engines have them: max context 4096,
at most 256 running requests, prefix (radix) caching off, random token-id prompts.
Everything else, including the share of memory for weights + KV cache
(`mem_fraction_static`), is SGLang's default and is recorded.

Timing comes from SGLang's per-request `e2e_latency` (request received -> finished):
    TTFT        = e2e_latency of a request with one output token, on an idle engine
    decode step = (span_long - span_short) / (long - short)
where span = the largest e2e_latency in a batch submitted at once. As in the vLLM bench,
prefill and mixed steps cancel in the difference, leaving full-batch decode steps.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time

from bench.vllm_bench import (
    DECODE_BATCHES,
    DECODE_OUTPUT_LEN,
    DECODE_OUTPUT_SHORT,
    DECODE_PROMPT_LEN,
    MAX_MODEL_LEN,
    MAX_NUM_SEQS,
    QUANTS,
    TTFT_PROMPT_LENS,
    TTFT_TRIALS,
    hf_model,
)

_rng = random.Random(0)


def random_prompts(n: int, length: int, vocab: int) -> list[list[int]]:
    return [[_rng.randrange(1000, vocab - 1000) for _ in range(length)] for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=QUANTS, required=True)
    ap.add_argument("--model", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--revision")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import sglang

    from capacity_planner.models import load_model

    vocab = load_model(args.model).vocab_size
    model, quantization = hf_model(args.model, args.quant)
    t0 = time.perf_counter()
    engine = sglang.Engine(
        model_path=model,
        revision=args.revision,
        quantization=quantization,
        tp_size=args.tp,
        context_length=MAX_MODEL_LEN,
        max_running_requests=MAX_NUM_SEQS,
        disable_radix_cache=True,
        skip_tokenizer_init=True,
        random_seed=0,
        log_level="info",
    )
    init_s = time.perf_counter() - t0
    info = engine.get_server_info()
    kv_tokens = int(info["max_total_num_tokens"])
    keep = (
        "mem_fraction_static", "max_running_requests", "chunked_prefill_size", "max_prefill_tokens",
        "context_length", "kv_cache_dtype", "attention_backend", "sampling_backend", "cuda_graph_max_bs",
        "disable_radix_cache", "quantization", "dtype", "tp_size", "schedule_policy",
    )
    result = {
        "sglang_version": info.get("version", sglang.__version__),
        "model": model,
        "tensor_parallel_size": args.tp,
        "revision": args.revision,
        "quantization": quantization,
        "server_args": {k: info.get(k) for k in keep},
        "engine": {
            "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": MAX_NUM_SEQS,
            "mem_fraction_static": info.get("mem_fraction_static"),
            "enable_prefix_caching": False,
            "kv_cache_size_tokens": kv_tokens,
            "max_req_input_len": info.get("max_req_input_len"),
        },
        "init_s": init_s,
    }

    def generate(prompts: list[list[int]], max_tokens: int) -> tuple[list[dict], float]:
        # ignore_eos alone fixes the output length (checked below); min_new_tokens would need
        # the tokenizer, which is skipped as in the vLLM bench (detokenize=False).
        sp = {"max_new_tokens": max_tokens, "ignore_eos": True, "temperature": 0.0}
        t = time.perf_counter()
        outs = engine.generate(input_ids=prompts, sampling_params=sp)
        return outs, time.perf_counter() - t

    def timing(out: dict) -> dict:
        m = out["meta_info"]
        return {"e2e_latency_s": m["e2e_latency"], "completion_tokens": m.get("completion_tokens", len(out.get("output_ids", [])))}

    warm, _ = generate(random_prompts(4, 256, vocab), 16)
    if "e2e_latency" not in warm[0]["meta_info"]:
        raise RuntimeError(f"SGLang returned no e2e_latency; meta_info keys: {sorted(warm[0]['meta_info'])}")
    result["meta_info_keys"] = sorted(warm[0]["meta_info"])

    # TTFT: one request at a time on an idle engine.
    ttft = []
    for p in TTFT_PROMPT_LENS:
        trials = []
        for _ in range(TTFT_TRIALS):
            outs, wall = generate(random_prompts(1, p, vocab), 1)
            trials.append({**timing(outs[0]), "wall_s": wall})
        ttft.append(
            {
                "prompt_len": p,
                "ttft_s_median": statistics.median(t["e2e_latency_s"] for t in trials),
                "wall_s_median": statistics.median(t["wall_s"] for t in trials),
                "trials": trials,
            }
        )
    result["ttft"] = ttft

    def decode_run(b: int, out_len: int) -> tuple[list[dict], float, float]:
        outs, wall = generate(random_prompts(b, DECODE_PROMPT_LEN, vocab), out_len)
        reqs = [timing(o) for o in outs]
        assert all(r["completion_tokens"] == out_len for r in reqs), "short generation"
        return reqs, max(r["e2e_latency_s"] for r in reqs), wall

    decode = []
    per_seq = DECODE_PROMPT_LEN + DECODE_OUTPUT_LEN
    for b in DECODE_BATCHES:
        if b > MAX_NUM_SEQS or b * per_seq > kv_tokens:
            decode.append({"batch": b, "skipped": "exceeds KV cache or max_running_requests"})
            continue
        _, span_short, _ = decode_run(b, DECODE_OUTPUT_SHORT)
        reqs, span, wall = decode_run(b, DECODE_OUTPUT_LEN)
        step = (span - span_short) / (DECODE_OUTPUT_LEN - DECODE_OUTPUT_SHORT)
        decode.append(
            {
                "batch": b,
                "prompt_len": DECODE_PROMPT_LEN,
                "output_len": DECODE_OUTPUT_LEN,
                "output_len_short": DECODE_OUTPUT_SHORT,
                "decode_context_len": DECODE_PROMPT_LEN + (DECODE_OUTPUT_SHORT + DECODE_OUTPUT_LEN) // 2,
                "decode_step_s": step,
                "decode_tokens_per_s": b / step,
                # whole-request time / output tokens (prefill included); SGLang gives no
                # per-request first-token time, so this is not the vLLM bench's TPOT
                "e2e_per_output_token_s_median": statistics.median(r["e2e_latency_s"] for r in reqs) / DECODE_OUTPUT_LEN,
                "output_tokens_per_s": b * DECODE_OUTPUT_LEN / span,
                "engine_span_s": span,
                "engine_span_short_s": span_short,
                "wall_s": wall,
            }
        )
    result["decode"] = decode
    engine.shutdown()

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
