import pytest

from capacity_planner.gpus import GPUS, get_gpu


def test_required_gpus_present():
    assert {"L4", "A10G", "A100-40GB", "A100-80GB", "H100"} <= set(GPUS)


@pytest.mark.parametrize("gpu", GPUS.values(), ids=lambda g: g.name)
def test_specs_are_positive_and_sourced(gpu):
    assert gpu.memory_gib > 0
    assert gpu.mem_bw_gbs > 0
    assert gpu.bf16_tflops > 0
    assert gpu.price_per_hour > 0
    assert gpu.spec_source and gpu.price_source


def test_fp8_only_on_ada_and_hopper():
    assert get_gpu("L4").fp8_tflops is not None
    assert get_gpu("H100").fp8_tflops is not None
    for name in ("A10G", "A100-40GB", "A100-80GB"):
        assert get_gpu(name).fp8_tflops is None
        assert get_gpu(name).peak_ops_per_s("fp8") is None


def test_datasheet_values():
    # Dense values: half of NVIDIA's with-sparsity figures.
    h100 = get_gpu("H100")
    assert h100.bf16_tflops == pytest.approx(1979 / 2, rel=1e-3)
    assert h100.fp8_tflops == pytest.approx(3958 / 2, rel=1e-3)
    assert get_gpu("L4").bf16_tflops == pytest.approx(242 / 2, rel=1e-3)
    assert get_gpu("A100-80GB").mem_bw_gbs == 2039


def test_prices_derived_from_modal_per_second_rates():
    assert get_gpu("H100").price_per_hour == pytest.approx(0.001097 * 3600)
    assert get_gpu("L4").price_per_hour == pytest.approx(0.000222 * 3600)


def test_unit_conversions():
    h100 = get_gpu("H100")
    assert h100.memory_bytes == 80 * 2**30
    assert h100.mem_bw_bytes_per_s == 3.35e12
    assert h100.peak_ops_per_s("bf16") == pytest.approx(989.4e12)


def test_ridge_point():
    a100 = get_gpu("A100-80GB")
    assert a100.ridge_point() == pytest.approx(312e12 / 2039e9)
    with pytest.raises(ValueError):
        a100.ridge_point("fp8")


def test_unknown_gpu():
    with pytest.raises(KeyError, match="unknown GPU"):
        get_gpu("V100")
