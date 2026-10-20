from capacity_planner import serve_check as sc
from capacity_planner.recommend import recommend, serve_calibration
from capacity_planner.transfer import candidate_configs


def test_frozen_predictions_cover_targets_and_meet_targets():
    """Every target was frozen, and as the recommender's answer it met the TTFT target
    (p90 x margin) when frozen. The file is not compared with the current code, which may
    change after the runs."""
    preds = sc.load_predictions()
    assert set(preds) == {t.key for t in sc.TARGETS}
    for p in preds.values():
        assert p["ttft_p90_s"] * p["ttft_margin"] <= p["slo"]["ttft_p90_s"]
        assert p["tpot_s"] <= p["slo"]["tpot_s"]


def test_predictions_are_the_recommenders_answer():
    t = sc.TARGETS[0]
    w = sc.WORKLOADS[t.workload]
    recs = recommend(w, configs=candidate_configs(t.model), cal=serve_calibration(), model=t.model)
    rec = next(r for r in recs if (r.gpu, r.quant, r.gpus_per_replica) == (t.gpu, t.quant, t.tp))
    p = sc.serve_prediction(t)
    assert rec.replicas == p["replicas"] and rec.max_num_seqs == p["serve"]["max_num_seqs"]


def test_comparison_table():
    p = sc.serve_prediction(sc.TARGETS[0])
    assert "not run yet" in sc.comparison_table([], {"k": p})
    run = {
        "id": "s1", "suite": "serve", "model": p["model"], "gpu": p["gpu"], "quant": p["quant"], "tp": 1,
        "serve": p["serve"], "github_run_url": "https://example/s1",
        "result": {"serve": {"result": {
            "median_ttft_ms": p["ttft_p50_s"] * 1e3, "p90_ttft_ms": p["ttft_p90_s"] * 2e3,
            "median_tpot_ms": p["tpot_s"] * 1e3,
        }}},
    }
    t = sc.comparison_table([run], {"k": p})
    assert "(-50%)" in t and "[s1]" in t
    assert "**no**" in t  # measured p90 at twice the prediction misses the 500 ms target
    # A run with a different load is not this prediction's measurement.
    other = {**run, "serve": {**p["serve"], "rps": 1.0}}
    assert "not run yet" in sc.comparison_table([other], {"k": p})
