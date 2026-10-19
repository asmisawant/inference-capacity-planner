import pytest

from capacity_planner import roofline as rf
from capacity_planner.gpus import get_gpu
from capacity_planner.models import load_model

BF16 = rf.QUANTIZATIONS["bf16"]
FP8 = rf.QUANTIZATIONS["fp8"]
INT4 = rf.QUANTIZATIONS["awq-int4"]


@pytest.fixture
def m():
    return load_model()


def test_weight_bytes_bf16_is_two_bytes_per_param(m):
    assert rf.weight_bytes(m, BF16) == 2 * m.total_params


def test_quantization_only_shrinks_decoder_linears(m):
    rest = (m.total_params - m.decoder_linear_params) * 2
    assert rf.weight_bytes(m, FP8) == pytest.approx(m.decoder_linear_params + rest)
    assert rf.weight_bytes(m, INT4) == pytest.approx(m.decoder_linear_params * 4.15625 / 8 + rest)
    assert rf.weight_bytes(m, INT4) < rf.weight_bytes(m, FP8) < rf.weight_bytes(m, BF16)


def test_effective_compute_dtype_falls_back_without_fp8(m):
    assert rf.effective_compute_dtype(get_gpu("H100"), FP8) == "fp8"
    assert rf.effective_compute_dtype(get_gpu("A100-80GB"), FP8) == "bf16"
    assert rf.effective_compute_dtype(get_gpu("A10G"), rf.QUANTIZATIONS["int8"]) == "int8"  # A10 has INT8 tensor cores


def test_kv_capacity_hand_calculation(m):
    h100 = get_gpu("H100")
    free = 80 * 2**30 * 0.92 - 2 * m.total_params
    assert rf.kv_cache_capacity_tokens(h100, m, BF16) == int(free // 57_344)


def test_kv_capacity_overhead_and_fp8_kv(m):
    l4 = get_gpu("L4")
    base = rf.kv_cache_capacity_tokens(l4, m, BF16)
    assert rf.kv_cache_capacity_tokens(l4, m, BF16, overhead_gib=2) < base
    # FP8 KV halves bytes per token, so roughly doubles capacity.
    assert rf.kv_cache_capacity_tokens(l4, m, BF16, kv_dtype="fp8") in (2 * base, 2 * base + 1)


def test_kv_capacity_zero_when_weights_do_not_fit(m):
    l4 = get_gpu("L4")
    assert rf.kv_cache_capacity_tokens(l4, m, BF16, gpu_memory_utilization=0.5) == 0
    assert rf.kv_cache_bytes_available(l4, m, BF16, gpu_memory_utilization=0.5) < 0


def test_max_concurrent_sequences():
    assert rf.max_concurrent_sequences(10_000, 1_280) == 7
    with pytest.raises(ValueError):
        rf.max_concurrent_sequences(10_000, 0)


def test_decode_batch1_is_memory_bound_at_bandwidth_limit(m):
    h100 = get_gpu("H100")
    step = rf.decode_step(h100, m, BF16, batch_size=1, context_len=1)
    assert step.bound == "memory"
    expected = (2 * m.total_params + 2 * 57_344) / 3.35e12
    assert step.time_s == pytest.approx(expected)
    assert step.time_s == pytest.approx(4.55e-3, rel=0.01)  # ~15.2 GB / 3.35 TB/s


def test_decode_becomes_compute_bound_at_large_batch(m):
    # Short context so KV reads don't dominate; batch well past the ridge point.
    a10g = get_gpu("A10G")
    assert rf.decode_step(a10g, m, BF16, 1, 16).bound == "memory"
    assert rf.decode_step(a10g, m, BF16, 512, 16).bound == "compute"


def test_decode_monotonic_in_batch(m):
    g = get_gpu("A100-80GB")
    prev_t, prev_tps = 0.0, 0.0
    for b in (1, 2, 4, 8, 16, 64, 256, 512):
        step = rf.decode_step(g, m, BF16, b, 2048)
        tps = rf.decode_throughput_tokens_per_s(step, b)
        assert step.time_s >= prev_t
        assert tps >= prev_tps
        prev_t, prev_tps = step.time_s, tps


def test_quantization_speeds_up_memory_bound_decode(m):
    g = get_gpu("L4")
    t = {q.name: rf.decode_step(g, m, q, 1, 512).time_s for q in (BF16, FP8, INT4)}
    assert t["awq-int4"] < t["fp8"] < t["bf16"]


def test_prefill_flops_formula(m):
    T = 1000
    p = rf.prefill(get_gpu("H100"), m, BF16, T)
    linear = 2 * m.decoder_linear_params * T + 2 * 152064 * 3584
    attention = 2 * 28 * 3584 * T * (T + 1)
    assert p.flops == pytest.approx(linear + attention)
    assert p.bytes == pytest.approx(2 * m.total_params + T * 57_344)


def test_ttft_long_prompt_is_compute_bound_and_fp8_helps_on_h100(m):
    h100 = get_gpu("H100")
    p = rf.prefill(h100, m, BF16, 4096)
    assert p.bound == "compute"
    assert rf.ttft_s(h100, m, FP8, 4096) < rf.ttft_s(h100, m, BF16, 4096)
    # No FP8 tensor cores on A100: same compute time as BF16.
    a100 = get_gpu("A100-80GB")
    assert rf.ttft_s(a100, m, FP8, 4096) == pytest.approx(rf.ttft_s(a100, m, BF16, 4096))


def test_ttft_short_prompt_is_memory_bound(m):
    assert rf.prefill(get_gpu("H100"), m, BF16, 8).bound == "memory"


def test_ttft_superlinear_in_prompt_length(m):
    g = get_gpu("H100")
    t1, t2 = rf.ttft_s(g, m, BF16, 8192), rf.ttft_s(g, m, BF16, 16384)
    assert t2 > 2 * t1  # quadratic attention term


def test_efficiency_scales_time(m):
    g = get_gpu("H100")
    ideal = rf.decode_step(g, m, BF16, 1, 128).time_s
    half = rf.decode_step(g, m, BF16, 1, 128, eff=rf.Efficiency(compute=0.5, memory=0.5)).time_s
    assert half == pytest.approx(2 * ideal)


def test_max_batch_for_tpot(m):
    g = get_gpu("A100-80GB")
    ctx, slo = 2048, 0.030
    b = rf.max_batch_for_tpot(g, m, BF16, slo, ctx, max_batch=4096)
    assert b > 1
    assert rf.decode_step(g, m, BF16, b, ctx).time_s <= slo
    assert rf.decode_step(g, m, BF16, b + 1, ctx).time_s > slo
    # Capped by max_batch, and 0 when batch 1 already misses.
    assert rf.max_batch_for_tpot(g, m, BF16, slo, ctx, max_batch=3) == 3
    assert rf.max_batch_for_tpot(g, m, BF16, 1e-4, ctx, max_batch=4096) == 0


def test_cost_per_million_tokens():
    # $3.60/hr = $0.001/s; 1000 tok/s -> 1M tokens in 1000 s -> $1.00.
    assert rf.cost_per_million_tokens(3.60, 1000) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rf.cost_per_million_tokens(1.0, 0)


def test_invalid_inputs(m):
    g = get_gpu("H100")
    with pytest.raises(ValueError):
        rf.prefill(g, m, BF16, 0)
    with pytest.raises(ValueError):
        rf.decode_step(g, m, BF16, 0, 10)
