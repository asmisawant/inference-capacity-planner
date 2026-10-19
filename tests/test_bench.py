import json

import pytest

from bench import runs
from bench.parse import parse_vllm_log
from capacity_planner.gpus import GPUS


def _run(id="r1", gpu="L4", suite="vllm", quant="bf16"):
    return runs.Run(id=id, gpu=gpu, suite=suite, quant=quant)


def test_run_validation():
    with pytest.raises(ValueError):
        _run(gpu="V100")
    with pytest.raises(ValueError):
        _run(suite="sglang")
    with pytest.raises(ValueError):
        _run(quant="int3")
    with pytest.raises(ValueError):
        _run(id="has space")


def test_worst_case_cost_is_timeout_times_price_with_margin():
    r = _run(gpu="H100", suite="vllm")
    assert r.worst_case_usd == pytest.approx(GPUS["H100"].price_per_hour * 0.5 * 1.1)


def test_plan_skips_ledger_ids_and_enforces_budget():
    rs = [_run("a"), _run("b"), _run("c", gpu="H100")]
    ledger = [{"id": "a", "cost_usd_upper_bound": 1.0}]
    pending = runs.plan(40, rs, ledger)
    assert [r.id for r in pending] == ["b", "c"]
    # Budget just below spent + worst case -> refuse.
    need = 1.0 + rs[1].worst_case_usd + rs[2].worst_case_usd
    with pytest.raises(SystemExit, match="budget guard"):
        runs.plan(need - 0.01, rs, ledger)
    assert runs.plan(need, rs, ledger)


def test_failed_runs_count_as_done_and_as_spend():
    ledger = [{"id": "a", "status": "error", "cost_usd_upper_bound": 0.5}]
    assert runs.plan(40, [_run("a")], ledger) == []
    assert runs.spent_usd(ledger) == 0.5


def test_load_requests_rejects_duplicates(tmp_path):
    p = tmp_path / "req.json"
    p.write_text(json.dumps({"budget_usd": 40, "runs": [{"id": "x", "gpu": "L4", "suite": "micro"}] * 2}))
    with pytest.raises(ValueError, match="duplicate"):
        runs.load_requests(p)


def test_repo_requests_fit_budget():
    budget, rs = runs.load_requests()
    assert budget <= 40
    runs.plan(budget, rs, runs.load_ledger())


def test_collect_files_results_and_appends_ledger(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    rec = {
        "id": "r1", "gpu": "L4", "suite": "micro", "quant": "bf16", "status": "ok",
        "wall_s": 100.0, "cost_usd_upper_bound": 0.02, "finished_at": "2026-10-19T00:00:00+00:00",
        "github_run_url": "https://example/run/1",
    }
    (out / "r1.json").write_text(json.dumps(rec))
    res = tmp_path / "results"
    assert [e["id"] for e in runs.collect(out, res)] == ["r1"]
    assert json.loads((res / "r1.json").read_text())["status"] == "ok"
    # Collecting again doesn't duplicate the ledger line.
    assert runs.collect(out, res) == []
    assert len(runs.load_ledger(res / "ledger.jsonl")) == 1


def test_parse_vllm_log():
    log = """
INFO 10-19 01:02:03 [gpu_model_runner.py:5417] Model loading took 14.25 GiB memory and 5.123456 seconds
INFO 10-19 01:02:30 [gpu_worker.py:641] Available KV cache memory: 5.31 GiB
INFO 10-19 01:02:31 [kv_cache_utils.py:2396] GPU KV cache size: 99,456 tokens, Maximum concurrency for 4,096 tokens per request: 24.28x
"""
    assert parse_vllm_log(log) == {
        "model_loading_gib": 14.25,
        "available_kv_cache_gib": 5.31,
        "kv_cache_size_tokens": 99456,
    }
    assert parse_vllm_log("nothing here") == {}


def test_runs_that_never_reached_modal_can_retry_but_still_count_as_spend():
    ledger = [
        {"id": "a", "status": "error", "reached_modal": False, "cost_usd_upper_bound": 0.0001},
        {"id": "b", "status": "error", "reached_modal": True, "cost_usd_upper_bound": 0.2},
    ]
    assert [r.id for r in runs.plan(40, [_run("a"), _run("b")], ledger)] == ["a"]
    assert runs.spent_usd(ledger) == pytest.approx(0.2001)
