from capacity_planner import engines
from capacity_planner.validation import NO_RESULTS


def _run(id, suite, ttft, step, kv, frac_key, frac):
    return {
        "id": id, "suite": suite, "gpu": "L4", "quant": "bf16", "model": "Qwen2.5-7B-Instruct", "tp": 1,
        "finished_at": "2026-10-20T00:00:00+00:00", "github_run_url": f"https://example/{id}",
        "result": {suite: {
            "engine": {"kv_cache_size_tokens": kv, frac_key: frac},
            "ttft": [{"prompt_len": p, "ttft_s_median": ttft} for p in (128, 1024, 4000)],
            "decode": [{"batch": b, "decode_step_s": step} for b in (1, 16)] + [{"batch": 32, "skipped": "kv"}],
        }},
    }


def test_comparison_table():
    assert engines.comparison_table([]) == NO_RESULTS
    v = _run("v", "vllm", 0.071, 0.0565, 60_816, "gpu_memory_utilization", 0.92)
    s = _run("s", "sglang", 0.126, 0.056, 11_639, "mem_fraction_static", 0.684)
    t = engines.comparison_table([v, s])
    assert "| 0.92 / 0.68 | 60,816 / 11,639 | 71 / 126 |" in t
    assert "| 56.5 / 56.0 | 16: 56.5 / 56.0 |" in t
    assert "[v](https://example/v) · [s](https://example/s)" in t
