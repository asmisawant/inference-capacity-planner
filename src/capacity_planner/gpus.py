"""GPU spec table.

Every number here has a source. Units:
- memory_gib: the datasheet's "GB" figure. NVIDIA's HBM/GDDR capacities are
  binary (an "80GB" H100 reports ~80 GiB in nvidia-smi), so we read them as GiB.
  Step 2 replaces this with the total that nvidia-smi reports on Modal.
- mem_bw_gbs: peak DRAM bandwidth in GB/s (10^9 bytes/s).
- *_tflops: dense (no 2:4 sparsity) tensor-core peak in 10^12 ops/s.
  NVIDIA datasheets usually print the sparse number; we halve it.
  None means the GPU has no tensor-core path for that type.
- price_per_hour: Modal on-demand USD/hour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Modal's per-second prices, as quoted from modal.com/pricing in search results on
# 2026-10-19. modal.com is blocked from the dev sandbox, so these could not be
# read from the page directly: check them against https://modal.com/pricing.
MODAL_PRICE_SOURCE = "https://modal.com/pricing (quoted via search, 2026-10-19; unverified)"


@dataclass(frozen=True)
class GPUSpec:
    name: str
    memory_gib: float
    mem_bw_gbs: float
    bf16_tflops: float
    fp8_tflops: float | None
    int8_tops: float | None
    price_per_hour: float
    modal_gpu: str  # the string passed to Modal's `gpu=` argument (https://modal.com/docs/guide/gpu)
    spec_source: str
    price_source: str = MODAL_PRICE_SOURCE
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def memory_bytes(self) -> float:
        return self.memory_gib * 2**30

    @property
    def mem_bw_bytes_per_s(self) -> float:
        return self.mem_bw_gbs * 1e9

    def peak_ops_per_s(self, dtype: str) -> float | None:
        """Dense tensor-core peak for a compute dtype ('bf16', 'fp8', 'int8')."""
        value = {"bf16": self.bf16_tflops, "fp8": self.fp8_tflops, "int8": self.int8_tops}[dtype]
        return None if value is None else value * 1e12

    def ridge_point(self, dtype: str = "bf16") -> float:
        """Arithmetic intensity (FLOPs/byte) where the roofline turns from memory- to compute-bound."""
        peak = self.peak_ops_per_s(dtype)
        if peak is None:
            raise ValueError(f"{self.name} has no {dtype} tensor-core path")
        return peak / self.mem_bw_bytes_per_s


def _per_second_to_hour(usd_per_second: float) -> float:
    return round(usd_per_second * 3600, 4)


GPUS: dict[str, GPUSpec] = {
    "L4": GPUSpec(
        name="L4",
        memory_gib=24,
        mem_bw_gbs=300,
        bf16_tflops=121,  # datasheet: 242 with sparsity
        fp8_tflops=242.5,  # datasheet: 485 with sparsity
        int8_tops=242.5,  # datasheet: 485 with sparsity
        price_per_hour=_per_second_to_hour(0.000222),
        modal_gpu="L4",
        spec_source="https://www.nvidia.com/en-us/data-center/l4/ (L4 datasheet)",
    ),
    "A10G": GPUSpec(
        name="A10G",
        memory_gib=24,
        mem_bw_gbs=600,
        bf16_tflops=125,  # datasheet: 250 with sparsity
        fp8_tflops=None,  # Ampere: no FP8 tensor cores
        int8_tops=250,  # datasheet: 500 with sparsity
        price_per_hour=1.10,
        modal_gpu="A10G",
        spec_source="https://www.nvidia.com/en-us/data-center/products/a10-gpu/ (A10 datasheet)",
        price_source="Modal blog GPU table, Jan 2025, via search (unverified)",
        notes=(
            "Modal's gpu='A10G' runs on an NVIDIA A10 (nvidia-smi, runs 007/011/019), so the "
            "spec is the A10 datasheet. It replaced a third-party A10G figure of 70 BF16 "
            "TFLOPS, which run 007 measured above (77 TFLOPS).",
        ),
    ),
    "A100-40GB": GPUSpec(
        name="A100-40GB",
        memory_gib=40,
        mem_bw_gbs=1555,
        bf16_tflops=312,
        fp8_tflops=None,  # Ampere: no FP8 tensor cores
        int8_tops=624,
        price_per_hour=_per_second_to_hour(0.000583),
        modal_gpu="A100-40GB",
        spec_source="https://www.nvidia.com/en-us/data-center/a100/ (A100 SXM datasheet)",
        notes=("Assumes Modal's A100s are SXM parts.",),
    ),
    "A100-80GB": GPUSpec(
        name="A100-80GB",
        memory_gib=80,
        mem_bw_gbs=2039,
        bf16_tflops=312,
        fp8_tflops=None,
        int8_tops=624,
        price_per_hour=_per_second_to_hour(0.000694),
        modal_gpu="A100-80GB",
        spec_source="https://www.nvidia.com/en-us/data-center/a100/ (A100 SXM datasheet)",
        notes=("Assumes Modal's A100s are SXM parts.",),
    ),
    "H100": GPUSpec(
        name="H100",
        memory_gib=80,
        mem_bw_gbs=3350,
        bf16_tflops=989.4,  # datasheet: 1,979 with sparsity
        fp8_tflops=1978.9,  # datasheet: 3,958 with sparsity
        int8_tops=1978.9,
        price_per_hour=_per_second_to_hour(0.001097),
        modal_gpu="H100!",  # "!" stops Modal upgrading the request to an H200
        spec_source="https://www.nvidia.com/en-us/data-center/h100/ (H100 SXM datasheet)",
        notes=("Assumes Modal's H100 is the SXM part.",),
    ),
}


def get_gpu(name: str) -> GPUSpec:
    try:
        return GPUS[name]
    except KeyError:
        raise KeyError(f"unknown GPU {name!r}; known: {sorted(GPUS)}") from None
