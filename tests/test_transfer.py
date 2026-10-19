import pytest

from bench import vllm_bench
from capacity_planner import transfer as tr
from capacity_planner import validation as v
from capacity_planner.models import DEFAULT_MODEL


def test_constants_mirror_the_bench():
    assert tr.TTFT_PROMPT_LENS == vllm_bench.TTFT_PROMPT_LENS
    assert tr.DECODE_BATCHES == vllm_bench.DECODE_BATCHES
    assert tr.MAX_NUM_SEQS == vllm_bench.MAX_NUM_SEQS
    assert tr.DECODE_SEQ_TOKENS == vllm_bench.DECODE_PROMPT_LEN + vllm_bench.DECODE_OUTPUT_LEN
    short, long = vllm_bench.DECODE_OUTPUT_SHORT, vllm_bench.DECODE_OUTPUT_LEN
    assert tr.DECODE_CONTEXT_LEN == vllm_bench.DECODE_PROMPT_LEN + (short + long) // 2


@pytest.mark.parametrize("gpu,quant", [("H100", "bf16"), ("L4", "fp8"), ("A100-80GB", "awq-int4")])
def test_transfer_to_the_source_model_reproduces_its_run(gpu, quant):
    """Applied to Qwen-7B itself, the transfer is just the fit: KV cache matches the run
    (block rounding aside) and predictions stay within the fit's own error."""
    results = v.load_results()
    run = v.latest(results, "vllm")[(gpu, quant)]
    c = tr.transfer(tr.Target(DEFAULT_MODEL, gpu, quant), results)
    measured_kv = run["result"]["vllm"]["engine"]["kv_cache_size_tokens"]
    assert c.kv_cache_tokens == pytest.approx(measured_kv, rel=0.002)
    for t in run["result"]["vllm"]["ttft"]:
        assert c.ttft_s(t["prompt_len"]) == pytest.approx(t["ttft_s_median"], rel=c.fit.ttft_max_err + 1e-9)
    for d in run["result"]["vllm"]["decode"]:
        if "decode_step_s" in d:
            assert c.decode_step_s(d["batch"]) == pytest.approx(d["decode_step_s"], rel=c.fit.decode_max_err + 1e-9)


def test_bench_batches_follow_predicted_kv():
    c = tr.transfer(tr.Target("Granite-3.1-8B-Instruct", "L4", "bf16"))
    assert c.bench_batches() == [b for b in tr.DECODE_BATCHES if b * 704 <= c.kv_cache_tokens]
    assert c.bench_batches()[-1] < 256  # Granite in BF16 leaves little KV room on a 24 GB L4


def test_tp2_kv_and_weights_are_per_gpu():
    c = tr.transfer(tr.Target("Qwen2.5-32B-Instruct", "H100", "bf16", tp=2))
    one = tr.transfer(tr.Target("Qwen2.5-32B-Instruct", "H100", "bf16"))
    assert c.weights_gib == pytest.approx(one.weights_gib / 2)
    assert one.kv_cache_tokens < c.kv_cache_tokens


def test_untransferable_target():
    with pytest.raises(ValueError, match="no Qwen-7B vllm fit"):
        tr.transfer(tr.Target("Granite-3.1-8B-Instruct", "A100-80GB", "fp8"))


def test_frozen_predictions_cover_targets():
    preds = tr.load_predictions()
    assert set(preds) == {t.key for t in tr.TARGETS}
    for p in preds.values():
        assert p["decode_step_s"] and set(p["ttft_s"]) == {str(x) for x in tr.TTFT_PROMPT_LENS}


def test_write_refuses_to_overwrite(tmp_path):
    out = tmp_path / "p.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="frozen"):
        tr.main(["--write", str(out)])


def test_comparison_table():
    p = tr.prediction(tr.Target("Granite-3.1-8B-Instruct", "H100", "bf16"))
    assert "not run yet" in tr.comparison_table([], {"k": p})
    run = {
        "id": "g1", "suite": "vllm", "gpu": "H100", "quant": "bf16", "model": "Granite-3.1-8B-Instruct", "tp": 1,
        "finished_at": "2026-10-20T00:00:00+00:00", "github_run_url": "https://example/g1",
        "result": {"vllm": {
            "engine": {"kv_cache_size_tokens": 300_000},
            "ttft": [{"prompt_len": int(k), "ttft_s_median": s / 1.1} for k, s in p["ttft_s"].items()],
            "decode": [{"batch": int(b), "decode_step_s": s * 2} for b, s in p["decode_step_s"].items()],
        }},
    }
    t = tr.comparison_table([run], {"k": p})
    assert "/ 300,000" in t and "[g1]" in t
    assert "| +10% /" in t  # TTFT predicted 10% high everywhere
    assert "| -50% /" in t  # decode predicted at half the measured step
    # A Qwen-7B run on the same GPU doesn't count as a Granite measurement.
    assert "not run yet" in tr.comparison_table([{**run, "model": DEFAULT_MODEL}], {"k": p})
