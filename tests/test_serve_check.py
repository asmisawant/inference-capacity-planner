import pytest

from capacity_planner import serve_check as sc
from capacity_planner.recommend import recommend, serve_calibration
from capacity_planner.transfer import candidate_configs


def test_frozen_predictions_cover_targets_and_meet_targets():
    """Every target was frozen, and as the recommender's answer it met the TTFT target
    (p90 x margin) when frozen. The file is not compared with the current code, which may
    change after the runs."""
    for targets, path in sc.SETS.values():
        preds = sc.load_predictions(path)
        assert set(preds) == {t.key for t in targets}
    for p in [*sc.load_predictions().values(), *sc.load_predictions(sc.MARGIN_CHECK_PREDICTIONS).values()]:
        assert p["ttft_p90_s"] * p["ttft_margin"] <= p["slo"]["ttft_p90_s"]
        assert p["tpot_s"] <= p["slo"]["tpot_s"]


def test_predictions_are_the_recommenders_answer():
    t = sc.TARGETS[0]
    w = sc.WORKLOADS[t.workload]
    recs = recommend(w, configs=candidate_configs(t.model), cal=serve_calibration(), model=t.model)
    rec = next(r for r in recs if (r.gpu, r.quant, r.gpus_per_replica) == (t.gpu, t.quant, t.tp))
    p = sc.serve_prediction(t)
    assert rec.replicas == p["replicas"] and rec.max_num_seqs == p["serve"]["max_num_seqs"]


def test_write_refuses_to_overwrite():
    with pytest.raises(SystemExit, match="frozen"):
        sc.main(["--set", "step5", "--write"])


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
    assert "| **no** |" in t  # p90 x margin doesn't cover twice the predicted p90
    assert "**no**" in t  # measured p90 at twice the prediction misses the 500 ms target
    # A run with a different load is not this prediction's measurement.
    other = {**run, "serve": {**p["serve"], "rps": 1.0}}
    assert "not run yet" in sc.comparison_table([other], {"k": p})


def test_serve_runs_match_configs_by_model():
    """Qwen-7B, Granite and Qwen-32B all have an H100 FP8 config; each serve run must be
    predicted from its own model's curves."""
    from capacity_planner import validation as v
    from capacity_planner.models import registry
    from capacity_planner.recommend import serve_runs

    results = v.load_results()
    configs = [c for m in registry() for c in candidate_configs(m, results)]
    pairs = serve_runs(results, configs)
    assert pairs and all(c.model.name == r.get("model") and c.tp == r.get("tp", 1) for r, c, _ in pairs)
    assert {r["id"] for r, _, _ in pairs} >= {"044-h100-serve-qwen32b-fp8-chat", "043-h100-serve-granite8b-fp8-longctx"}
