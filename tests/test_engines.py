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


def test_sglang_graph_limit_caps_batches():
    from capacity_planner.calibration import measured_configs, sglang_graph_max_bs
    from capacity_planner.gpus import GPUS

    assert sglang_graph_max_bs(GPUS["A100-40GB"]) == 32
    assert sglang_graph_max_bs(GPUS["L4"]) == 48 and sglang_graph_max_bs(GPUS["A10G"]) == 48
    assert sglang_graph_max_bs(GPUS["H100"]) == 256 and sglang_graph_max_bs(GPUS["H100"], tp=4) == 512
    for c in measured_configs(engine="sglang"):
        assert c.max_measured_batch <= sglang_graph_max_bs(c.gpu, c.tp)
    a100 = {c.key: c for c in measured_configs(engine="vllm")}[("A100-40GB", "bf16", 1)]
    assert a100.max_measured_batch > 32  # vLLM is not capped


def test_sglang_calibration_borrows_vllm_margins_on_untested_gpus():
    from capacity_planner.recommend import engine_calibrations

    cals = engine_calibrations()
    sg, own = cals["sglang"]
    if own:
        tested = {g for _, g, _ in sg.ratios}
        for gpu, m in cals["vllm"][0].by_gpu.items():
            if gpu not in tested:
                assert sg(gpu) >= m
