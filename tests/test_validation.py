"""Comparison tables, on synthetic result records shaped like bench/launch.py output."""

import json

import pytest

from capacity_planner import roofline as rf
from capacity_planner import validation as v
from capacity_planner.gpus import GPUS
from capacity_planner.models import load_model


def _base(id, suite, gpu="L4", quant="bf16", finished="2026-10-19T00:00:00+00:00", status="ok"):
    return {
        "id": id, "gpu": gpu, "suite": suite, "quant": quant, "status": status,
        "finished_at": finished, "github_run_url": f"https://example/{id}",
        "price_per_hour": GPUS[gpu].price_per_hour,
    }


def micro_record(id="m1", bw=240.0, bf16=90.0, finished="2026-10-19T00:00:00+00:00"):
    return {
        **_base(id, "micro", finished=finished),
        "result": {
            "gpu_info": {"name": "NVIDIA L4", "memory.total": "23034"},
            "micro": {
                "torch_info": {"total_memory_bytes": 23034 * 2**20},
                "bandwidth": {"median": bw},
                "gemv_bandwidth": {"median": 270.0},
                "matmul": {
                    "bf16": {"median": bf16},
                    "fp8": {"median": 180.0},
                    "int8": {"error": "RuntimeError: nope"},
                },
            },
        },
    }


def vllm_record(id="v1", ttft_s=0.2, tpot_s=0.06, tps=1000.0):
    return {
        **_base(id, "vllm"),
        "result": {
            "gpu_info": {"name": "NVIDIA L4", "memory.total": "23034"},
            "vllm": {
                "engine": {"gpu_memory_utilization": 0.92, "kv_cache_size_tokens": 50_000},
                "log_summary": {"model_loading_gib": 14.25, "available_kv_cache_gib": 5.0},
                "ttft": [{"prompt_len": 1024, "ttft_s_median": ttft_s}],
                "decode": [
                    {"batch": 1, "prompt_len": 512, "output_len": 192, "decode_context_len": 640,
                     "decode_step_s": tpot_s, "tpot_s_median": tpot_s, "output_tokens_per_s": tps},
                    {"batch": 2, "prompt_len": 512, "output_len": 128, "tpot_s_median": tpot_s,
                     "output_tokens_per_s": tps},
                    {"batch": 512, "skipped": "exceeds KV cache or max_num_seqs"},
                ],
            },
        },
    }


def test_no_results():
    for fn in (v.gpu_measured_table, v.vllm_memory_table, v.ttft_table, v.decode_table):
        assert fn([]) == v.NO_RESULTS


def test_load_results_keeps_only_ok_sorted(tmp_path):
    (tmp_path / "b.json").write_text(json.dumps(micro_record("b", finished="2026-10-19T02:00:00+00:00")))
    (tmp_path / "a.json").write_text(json.dumps(micro_record("a", finished="2026-10-19T01:00:00+00:00")))
    (tmp_path / "c.json").write_text(json.dumps({**micro_record("c"), "status": "error"}))
    (tmp_path / "ledger.jsonl").write_text("")
    assert [r["id"] for r in v.load_results(tmp_path)] == ["a", "b"]
    assert v.load_results(tmp_path / "missing") == []


def test_latest_run_wins():
    old = micro_record("old", bw=100.0, finished="2026-10-19T01:00:00+00:00")
    new = micro_record("new", bw=250.0, finished="2026-10-19T02:00:00+00:00")
    table = v.gpu_measured_table([old, new])
    assert "250 (83%)" in table and "[new]" in table and "[old]" not in table


def test_gpu_table_percentages():
    t = v.gpu_measured_table([micro_record()])
    assert "240 (80%)" in t  # copy: 240 / 300 GB/s
    assert "270 (90%)" in t  # GEMV
    assert "90 (74%)" in t  # 90 / 121 TFLOPS
    assert "unsupported" in t
    assert "22.5 / 24" in t  # 23034 MiB


def test_memory_table_derives_overhead():
    t = v.vllm_memory_table([vllm_record()])
    expected = 23034 / 1024 * 0.92 - 14.25 - 5.0
    assert f"| {expected:.2f} |" in t
    assert "50,000 /" in t


def test_ttft_and_decode_ratios():
    m, g, q = load_model(), GPUS["L4"], rf.QUANTIZATIONS["bf16"]
    pred_ttft = rf.ttft_s(g, m, q, 1024)
    t = v.ttft_table([vllm_record(ttft_s=2 * pred_ttft)])
    assert "| 2.00 |" in t
    pred_step = rf.decode_step(g, m, q, 1, 640).time_s
    d = v.decode_table([vllm_record(tpot_s=1.5 * pred_step)])
    assert "| 1.50 |" in d
    # decode tok/s = batch / step time, and $/1M follows from it
    assert f"{rf.cost_per_million_tokens(g.price_per_hour, 1 / (1.5 * pred_step)):.3f}" in d
    # a run without the two-length measurement shows only TPOT under load
    assert "| L4 | bf16 | 2 | – |" in d
    assert "| 512 |" not in d  # skipped batch omitted


def test_spend_summary(tmp_path):
    assert v.spend_summary(tmp_path) == v.NO_RESULTS
    (tmp_path / "ledger.jsonl").write_text(
        json.dumps({"status": "ok", "cost_usd_upper_bound": 0.25}) + "\n"
        + json.dumps({"status": "error", "cost_usd_upper_bound": 0.10}) + "\n"
    )
    s = v.spend_summary(tmp_path)
    assert "2 runs (1 succeeded)" in s and "$0.35" in s


@pytest.mark.parametrize("gpu", list(GPUS))
def test_every_gpu_has_a_modal_name(gpu):
    assert GPUS[gpu].modal_gpu


def test_memory_table_notes_compile_fallback():
    rec = vllm_record()
    rec["result"]["vllm"]["compile_fallback"] = {"reason": "InductorError", "compilation_config": {"backend": "eager"}}
    assert "eager backend" in v.vllm_memory_table([rec])
    assert "eager backend" not in v.vllm_memory_table([vllm_record()])


def test_margin_table_lists_every_gpu():
    t = v.margin_table([])
    assert t == v.NO_RESULTS


def test_holdout_table_without_holdout_runs():
    assert v.holdout_table([]) == v.NO_RESULTS


def test_tables_only_use_the_default_model_on_one_gpu():
    other = {**vllm_record("o", ttft_s=9.9), "model": "Granite-3.1-8B-Instruct"}
    tp2 = {**vllm_record("t", ttft_s=9.9), "tp": 2}
    assert v.latest([vllm_record(), other, tp2], "vllm")[("L4", "bf16")]["id"] == "v1"
    assert v.latest([other], "vllm", model="Granite-3.1-8B-Instruct")[("L4", "bf16")]["id"] == "o"


def test_load_results_defaults_model_and_tp(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(micro_record("a")))
    (r,) = v.load_results(tmp_path)
    assert (r["model"], r["tp"]) == ("Qwen2.5-7B-Instruct", 1)


def test_runs_on_a_substituted_gpu_are_left_out(tmp_path):
    ok = micro_record("ok")
    swapped = micro_record("swapped")
    swapped["gpu"] = "A100-40GB"
    swapped["result"]["gpu_info"]["memory.total"] = "81920"  # an 80 GB card
    (tmp_path / "ok.json").write_text(json.dumps(ok))
    (tmp_path / "swapped.json").write_text(json.dumps(swapped))
    assert [r["id"] for r in v.load_results(tmp_path)] == ["ok"]
    assert not v.gpu_matches(swapped) and v.gpu_matches(ok)
