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
    assert q.ttft_p90_s * r.ttft_margin <= 1
    # One fewer replica would miss the p90 target or be unstable.
    fewer = rec.queue_prediction(config, w, r.max_num_seqs, 100 / (r.replicas - 1)) if r.replicas > 1 else None
    assert r.replicas == 1 or fewer is None or fewer.ttft_p90_s * r.ttft_margin > 1
    assert r.cost_per_hour == pytest.approx(r.replicas * GPUS["H100"].price_per_hour)
    tok_s = 100 * (1024 + 256)
    assert r.cost_per_million_tokens(w) == pytest.approx(r.cost_per_hour / 3600 / tok_s * 1e6)


def test_erlang_c_known_values():
    assert rec.erlang_c(1, 0.5) == pytest.approx(0.5)  # M/M/1: P(wait) = rho
    assert rec.erlang_c(2, 1.0) == pytest.approx(1 / 3)
    assert rec.erlang_c(10, 0.001) < 1e-10
    with pytest.raises(ValueError):
        rec.erlang_c(2, 2.0)


def test_queue_prediction_parameters(config):
    w = rec.Workload(1024, 256, rps=1, ttft_slo_s=1, tpot_slo_s=1)
    q = rec.queue_prediction(config, w, 4, 0.5)
    busy = q.busy
    assert busy == rec.busy_point(config, w, 4, 0.5)
    assert q.service_s == pytest.approx(busy.ttft_s + 256 * busy.tpot_s)
    a = 0.5 * q.service_s
    assert q.utilization == pytest.approx(a / 4)
    assert q.p_wait == pytest.approx(rec.erlang_c(4, a))
    assert q.slot_rate == pytest.approx(2 * (4 / q.service_s - 0.5))
    assert q.p_prefill == pytest.approx(0.5 * busy.ttft_s)
    assert q.prefill_rate == pytest.approx(2 * (1 / busy.ttft_s - 0.5))
    assert q.base_ttft_s == pytest.approx(busy.ttft_s + busy.decode_step_s / 2)
    # Overloaded: offered load >= slots.
    assert rec.queue_prediction(config, w, 1, 10.0) is None
    # Light load: the p90 is just the base TTFT.
    light = rec.queue_prediction(config, w, 64, 0.01)
    assert light.ttft_p90_s == pytest.approx(light.base_ttft_s)


def _q(p1, r1, p2, r2, base=0.1):
    dummy = rec.OperatingPoint(1, 0.0, 0.0, 0.0, 1.0)
    return rec.QueuePrediction(4, 1.0, 1.0, 0.5, p1, r1, p2, r2, dummy, base)


def test_wait_survival_single_queue_is_mixed_exponential():
    q = _q(0.4, 3.0, 0.0, 5.0)
    for t in (0.0, 0.1, 1.0):
        assert q.wait_survival(t) == pytest.approx(0.4 * math.exp(-3 * t))
    # p90 = base + ln(p / 0.1) / rate
    assert q.ttft_p90_s == pytest.approx(0.1 + math.log(4) / 3, rel=1e-6)
    assert q.ttft_p50_s == pytest.approx(0.1)  # p < 0.5: median wait is 0


def test_wait_survival_sum_matches_monte_carlo():
    import random

    q = _q(0.3, 2.0, 0.5, 7.0)
    rng = random.Random(1)
    n, t = 200_000, 0.4
    hits = 0
    for _ in range(n):
        w = (rng.expovariate(2.0) if rng.random() < 0.3 else 0) + (rng.expovariate(7.0) if rng.random() < 0.5 else 0)
        hits += w > t
    assert q.wait_survival(t) == pytest.approx(hits / n, abs=0.005)
    # Equal rates use the Erlang-2 limit, continuous with the general formula.
    assert _q(0.3, 4.0, 0.5, 4.0).wait_survival(0.3) == pytest.approx(_q(0.3, 4.0, 0.5, 4.0 + 1e-6).wait_survival(0.3), rel=1e-4)


def test_summing_waits_beats_summing_percentiles():
    # Each wait alone happens 8% of the time (p90 of each is 0), together ~15%.
    q = _q(0.08, 5.0, 0.08, 5.0)
    assert q.ttft_p90_s > q.base_ttft_s


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
    recs = rec.recommend(w, configs=cal.measured_configs(results), cal=rec.NO_CALIBRATION)
    assert [r.quant for r in recs] == ["bf16", "awq-int4"]
    assert recs[0].cost_per_hour == recs[1].cost_per_hour


def test_quant_filter(results):
    w = rec.Workload(1024, 256, rps=1, ttft_slo_s=1, tpot_slo_s=1)
    assert rec.recommend(w, quants={"fp8"}, configs=cal.measured_configs(results), cal=rec.NO_CALIBRATION) == []


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


def _serve_run(run_id, gpu, p50_ms, p90_ms):
    return {
        "id": run_id, "gpu": gpu, "suite": "serve", "quant": "bf16", "status": "ok",
        "finished_at": "2026-10-19T01:00:00+00:00", "github_run_url": "u",
        "serve": {"prompt": 1024, "output": 256, "rps": 1, "max_num_seqs": 64},
        "result": {"serve": {"result": {"p90_ttft_ms": p90_ms, "median_ttft_ms": p50_ms, "median_tpot_ms": 1,
                                        "request_throughput": 1}}},
    }


def test_serve_calibration(results, config):
    w = rec.Workload(1024, 256, rps=1, ttft_slo_s=1e9, tpot_slo_s=1e9)
    q = rec.predict(config, w, 64)
    p50, p90 = q.ttft_p50_s * 1e3, q.ttft_p90_s * 1e3
    runs = [
        _serve_run("s1", "H100", p50 + 30, p90 + 30 + 50),  # overhead 30 ms, then 50 ms worse at p90
        _serve_run("s2", "H100", p50 + 30, 0.5 * (p90 + 30)),
        _serve_run("s3", "H100", p50 + 30, p90 + 30),
    ]
    cal = rec.serve_calibration(results + runs, [config])
    assert cal.overhead_s == pytest.approx(0.030)
    assert cal("H100") == pytest.approx((p90 + 80) / (p90 + 30))  # worst run
    assert cal("L4") == cal("H100")  # no serve run: largest margin
    # Excluding runs refits without them; never below 1.
    assert rec.serve_calibration(results + runs, [config], exclude=frozenset({"s1"}))("H100") == 1.0
    assert rec.serve_calibration(results, [config]) == rec.NO_CALIBRATION


def test_overhead_shifts_ttft_but_not_tpot(config):
    w = rec.Workload(1024, 256, rps=1, ttft_slo_s=1e9, tpot_slo_s=1e9)
    a, b = rec.predict(config, w, 64), rec.predict(config, w, 64, overhead_s=0.05)
    assert b.ttft_p90_s == pytest.approx(a.ttft_p90_s + 0.05)
    assert b.ttft_p50_s == pytest.approx(a.ttft_p50_s + 0.05)
    assert b.busy.tpot_s == a.busy.tpot_s


def test_margin_makes_recommendation_more_conservative(config):
    w = rec.Workload(1024, 256, rps=50, ttft_slo_s=0.2, tpot_slo_s=0.05)
    plain = rec.evaluate(config, w)
    careful = rec.evaluate(config, w, ttft_margin=3.0)
    assert careful.ttft_margin == 3.0
    assert not careful.feasible or careful.replicas >= plain.replicas


def test_engines_and_footnotes():
    from capacity_planner.recommend import EXAMPLES, recommend, table

    _, w = EXAMPLES[2]  # offline batch: an SGLang config is among the cheapest
    recs = recommend(w)
    engines = {r.engine for r in recs}
    assert engines == {"vllm", "sglang"}
    sg = [r for r in recs if r.engine == "sglang"]
    assert all(not r.load_tested for r in sg)  # no SGLang serve runs yet
    assert recommend(w, engines={"vllm"}) and all(r.engine == "vllm" for r in recommend(w, engines={"vllm"}))
    top1 = table(w, recs, top=1)
    assert "³" not in top1  # the cheapest is a load-tested vLLM config: no SGLang footnote
