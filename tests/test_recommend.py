import math

import pytest

from bench import runs
from capacity_planner import calibration as cal
from capacity_planner import recommend as rec
from capacity_planner import roofline as rf
from capacity_planner import validation as val
from capacity_planner.gpus import GPUS
from capacity_planner.models import load_model


def micro_run(gpu="H100", bf16=800.0, gemv=2800.0):
    return {
        "id": f"m-{gpu}", "gpu": gpu, "suite": "micro", "quant": "bf16", "status": "ok",
        "finished_at": "2026-10-19T00:00:00+00:00", "github_run_url": "u",
        "price_per_hour": GPUS[gpu].price_per_hour,
        "result": {"micro": {"matmul": {"bf16": {"median": bf16}}, "gemv_bandwidth": {"median": gemv}}},
    }


def vllm_run(gpu="H100", quant="bf16", ttft=((128, 0.01), (1024, 0.03), (4000, 0.10)),
             steps=((1, 0.006), (64, 0.008), (256, 0.012)), kv=1_000_000, run_id=None):
    return {
        "id": run_id or f"v-{gpu}-{quant}", "gpu": gpu, "suite": "vllm", "quant": quant, "status": "ok",
        "finished_at": "2026-10-19T00:00:00+00:00", "github_run_url": "u",
        "price_per_hour": GPUS[gpu].price_per_hour,
        "result": {"vllm": {
            "engine": {"kv_cache_size_tokens": kv, "max_num_seqs": 256, "max_model_len": 4096,
                       "gpu_memory_utilization": 0.92},
            "ttft": [{"prompt_len": p, "ttft_s_median": t} for p, t in ttft],
            "decode": [{"batch": b, "decode_step_s": s, "decode_context_len": 640, "prompt_len": 512,
                        "output_len": 192, "tpot_s_median": s} for b, s in steps],
        }},
    }


@pytest.fixture
def results():
    return [micro_run(), vllm_run()]


@pytest.fixture
def config(results):
    (c,) = cal.measured_configs(results)
    return c


def test_gpu_efficiency(results):
    e = cal.gpu_efficiency(results)["H100"]
    assert e.compute == pytest.approx(800 / 989.4)
    assert e.memory == pytest.approx(2800 / 3350)


def test_micro_runs_without_gemv_are_skipped():
    r = micro_run()
    del r["result"]["micro"]["gemv_bandwidth"]
    assert cal.gpu_efficiency([r]) == {}


def test_ttft_interpolates_inside_and_scales_outside(config):
    assert config.ttft_s(128) == pytest.approx(0.01)
    assert config.ttft_s(576) == pytest.approx(0.02)  # halfway 128 -> 1024
    # Outside: measured edge x roofline ratio.
    roof = lambda p: rf.ttft_s(config.gpu, config.model, config.quant, p, eff=config.eff)  # noqa: E731
    assert config.ttft_s(8000) == pytest.approx(0.10 * roof(8000) / roof(4000))
    assert config.ttft_s(64) == pytest.approx(0.01 * roof(64) / roof(128))


def test_decode_step_interpolates_and_scales_with_context(config):
    assert config.decode_step_s(1, 640) == pytest.approx(0.006)
    assert config.decode_step_s(160, 640) == pytest.approx(0.010)  # halfway 64 -> 256
    longer = config.decode_step_s(64, 4000)
    assert longer > 0.008  # more KV to read
    with pytest.raises(ValueError):
        config.decode_step_s(512, 640)


def test_fit_recovers_exact_linear_data():
    m, g, q = load_model(), GPUS["H100"], rf.QUANTIZATIONS["bf16"]
    e = rf.Efficiency(800 / 989.4, 2800 / 3350)
    ttft = [(p, 0.002 + 1.3 * rf.ttft_s(g, m, q, p, eff=e)) for p in (128, 1024, 4000)]
    steps = [(b, 1.2 * rf.decode_step(g, m, q, b, 640, eff=e).time_s + 5e-6 * b) for b in (1, 64, 256)]
    (f,) = cal.fits([micro_run(), vllm_run(ttft=ttft, steps=steps)])
    assert f.ttft_c0_s == pytest.approx(0.002, rel=1e-6)
    assert f.ttft_c1 == pytest.approx(1.3, rel=1e-6)
    assert f.decode_a == pytest.approx(1.2, rel=1e-6)
    assert f.decode_b_s == pytest.approx(5e-6, rel=1e-6)
    assert f.ttft_max_err < 1e-9 and f.decode_max_err < 1e-9


def test_operating_point_formulas(config):
    w = rec.Workload(1024, 256, rps=1, ttft_slo_s=1, tpot_slo_s=1)
    p = rec.operating_point(config, w, 64)
    step = config.decode_step_s(64, w.decode_context_len)
    assert p.tpot_s == pytest.approx(step + 64 * 0.03 / 256)
    assert p.ttft_s == pytest.approx(0.03 + step)
    assert p.capacity_rps == pytest.approx(64 / (256 * p.tpot_s))


def test_evaluate_picks_largest_batch_within_slos_and_sizes_replicas_for_queueing(config):
    w = rec.Workload(1024, 256, rps=100, ttft_slo_s=1, tpot_slo_s=0.02)
    r = rec.evaluate(config, w)
    assert r.feasible
    assert rec.operating_point(config, w, r.max_num_seqs).tpot_s <= 0.02
    if r.max_num_seqs < 256:
        assert rec.operating_point(config, w, r.max_num_seqs + 1).tpot_s > 0.02
    q = r.queue
    assert q.slots == r.max_num_seqs and q.rps == pytest.approx(100 / r.replicas)
    assert q.ttft_p90_s <= 1
    # One fewer replica would miss the p90 target or be unstable.
    fewer = rec.queue_prediction(config, w, r.max_num_seqs, 100 / (r.replicas - 1)) if r.replicas > 1 else None
    assert r.replicas == 1 or fewer is None or fewer.ttft_p90_s > 1
    assert r.cost_per_hour == pytest.approx(r.replicas * GPUS["H100"].price_per_hour)
    tok_s = 100 * (1024 + 256)
    assert r.cost_per_million_tokens(w) == pytest.approx(r.cost_per_hour / 3600 / tok_s * 1e6)


def test_erlang_c_known_values():
    assert rec.erlang_c(1, 0.5) == pytest.approx(0.5)  # M/M/1: P(wait) = rho
    assert rec.erlang_c(2, 1.0) == pytest.approx(1 / 3)
    assert rec.erlang_c(10, 0.001) < 1e-10
    with pytest.raises(ValueError):
        rec.erlang_c(2, 2.0)


def test_queue_prediction_tail_and_stability(config):
    w = rec.Workload(1024, 256, rps=1, ttft_slo_s=1, tpot_slo_s=1)
    q = rec.queue_prediction(config, w, 4, 0.5)
    full = rec.operating_point(config, w, 4)
    assert q.service_s == pytest.approx(full.ttft_s + 256 * full.tpot_s)
    a = 0.5 * q.service_s
    c = rec.erlang_c(4, a)
    expected = max(0.0, math.log(c / 0.1)) * 0.5 / (4 / q.service_s - 0.5)
    assert q.wait_p90_s == pytest.approx(expected)
    assert q.utilization == pytest.approx(a / 4)
    rho = 0.5 * q.busy.ttft_s
    expected_prefill = max(0.0, math.log(rho / 0.1)) * 0.5 / (1 / q.busy.ttft_s - 0.5)
    assert q.prefill_wait_p90_s == pytest.approx(expected_prefill)
    assert q.ttft_p90_s == pytest.approx(q.busy.ttft_s + q.wait_p90_s + q.prefill_wait_p90_s)
    # Overloaded: offered load >= slots.
    assert rec.queue_prediction(config, w, 1, 10.0) is None
    # Light load: no wait at p90.
    assert rec.queue_prediction(config, w, 64, 0.01).wait_p90_s == 0


def test_busy_point_is_littles_law_fixed_point(config):
    w = rec.Workload(1024, 256, rps=2, ttft_slo_s=1, tpot_slo_s=1)
    p = rec.busy_point(config, w, 256, 2)
    assert p.batch == max(1, math.ceil(2 * (p.ttft_s + 256 * p.tpot_s)))


def test_infeasible_reasons(config):
    tight = rec.Workload(1024, 256, rps=1, ttft_slo_s=0.001, tpot_slo_s=1)
    r = rec.evaluate(config, tight)
    assert not r.feasible and "TTFT" in r.reason
    too_long = rec.Workload(4000, 512, rps=1, ttft_slo_s=1, tpot_slo_s=1)
    assert "max_model_len" in rec.evaluate(config, too_long).reason


def test_kv_cache_limits_batch():
    results = [micro_run(), vllm_run(kv=1280 * 3)]
    (c,) = cal.measured_configs(results)
    r = rec.evaluate(c, rec.Workload(1024, 256, rps=100, ttft_slo_s=10, tpot_slo_s=10))
    assert r.max_num_seqs == 3


def test_tie_break_prefers_less_quantization():
    results = [micro_run(), vllm_run(quant="awq-int4"), vllm_run(quant="bf16")]
    w = rec.Workload(1024, 256, rps=1, ttft_slo_s=1, tpot_slo_s=1)
    recs = rec.recommend(w, configs=cal.measured_configs(results))
    assert [r.quant for r in recs] == ["bf16", "awq-int4"]
    assert recs[0].cost_per_hour == recs[1].cost_per_hour


def test_quant_filter(results):
    w = rec.Workload(1024, 256, rps=1, ttft_slo_s=1, tpot_slo_s=1)
    assert rec.recommend(w, quants={"fp8"}, configs=cal.measured_configs(results)) == []


def test_predict_and_overload(config):
    w = rec.Workload(1024, 256, rps=1, ttft_slo_s=1, tpot_slo_s=1)
    q = rec.predict(config, w, 256)
    assert q.slots == 256 and q.utilization < 1
    assert rec.predict(config, rec.Workload(1024, 256, rps=1e6, ttft_slo_s=1, tpot_slo_s=1), 256) is None


def test_workload_validation():
    with pytest.raises(ValueError):
        rec.Workload(0, 10, 1, 1, 1)
    with pytest.raises(ValueError):
        rec.Workload(10, 10, 1, 0, 1)


def test_cli_runs_on_committed_results(capsys):
    assert rec.main(["--prompt", "1024", "--output", "256", "--rps", "5", "--ttft-ms", "500", "--tpot-ms", "50"]) == 0
    assert "Recommended:" in capsys.readouterr().out


def test_serve_run_validation():
    ok = {"prompt": 1024, "output": 256, "rps": 5, "max_num_seqs": 64}
    runs.Run("s", "H100", "serve", "bf16", serve=ok)
    with pytest.raises(ValueError):
        runs.Run("s", "H100", "serve", "bf16")
    with pytest.raises(ValueError):
        runs.Run("s", "H100", "vllm", "bf16", serve=ok)
    with pytest.raises(ValueError):
        runs.Run("s", "H100", "serve", "bf16", serve={**ok, "rps": 0})


def test_serve_table(results):
    serve = {
        "id": "s1", "gpu": "H100", "suite": "serve", "quant": "bf16", "status": "ok",
        "finished_at": "2026-10-19T01:00:00+00:00", "github_run_url": "u",
        "serve": {"prompt": 1024, "output": 256, "rps": 5, "max_num_seqs": 64},
        "result": {"serve": {"result": {
            "median_ttft_ms": 40.0, "p90_ttft_ms": 60.0, "median_tpot_ms": 7.0, "p90_tpot_ms": 8.0,
            "request_throughput": 4.9,
        }}},
    }
    t = val.serve_table(results + [serve])
    assert "/ 40 |" in t and "/ 60 |" in t and "/ 7.0 |" in t and "4.90 / 5" in t
    assert val.serve_table(results) == val.NO_RESULTS
