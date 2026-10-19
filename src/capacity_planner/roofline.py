"""Roofline model for single-GPU LLM serving.

Each phase does some FLOPs and moves some bytes. Its time is bounded below by
    max(flops / peak_flops, bytes / peak_bandwidth).
With the default efficiencies of 1.0 the results are ideal lower bounds on
latency (upper bounds on throughput). Step 2 calibrates `Efficiency` and
`overhead_gib` against vLLM/SGLang runs on Modal.

Simplifications, all of which make predictions optimistic:
- activations, CUDA graphs and allocator overhead are ignored unless `overhead_gib` is set;
- every sequence in a decode batch has the same context length;
- TTFT is a prefill with no queueing and no other requests on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

from .gpus import GPUSpec
from .models import ModelSpec

# vLLM's default --gpu-memory-utilization.
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9


@dataclass(frozen=True)
class Quantization:
    """How decoder-layer linear weights are stored and which tensor cores run them.

    Embeddings, lm_head, norms and biases stay in bf16, as vLLM does for AWQ/GPTQ/FP8.
    """

    name: str
    weight_bits: float  # effective bits per weight, including group scales/zeros
    compute_dtype: str  # 'bf16', 'fp8' or 'int8'


QUANTIZATIONS: dict[str, Quantization] = {
    "bf16": Quantization("bf16", 16.0, "bf16"),
    # W8A8 FP8, per-channel scales (negligible: one scale per output row).
    "fp8": Quantization("fp8", 8.0, "fp8"),
    # W8A8 INT8 (e.g. compressed-tensors/SmoothQuant).
    "int8": Quantization("int8", 8.0, "int8"),
    # W4A16 AWQ, group size 128: 4 bits + fp16 scale/128 + 4-bit zero/128 = 4.15625 bits.
    "awq-int4": Quantization("awq-int4", 4.0 + 16 / 128 + 4 / 128, "bf16"),
}


@dataclass(frozen=True)
class Efficiency:
    """Fraction of datasheet peak actually achieved. 1.0 = ideal roofline."""

    compute: float = 1.0
    memory: float = 1.0


IDEAL = Efficiency()


@dataclass(frozen=True)
class PhaseEstimate:
    flops: float
    bytes: float
    compute_s: float
    memory_s: float

    @property
    def time_s(self) -> float:
        return max(self.compute_s, self.memory_s)

    @property
    def bound(self) -> str:
        return "compute" if self.compute_s >= self.memory_s else "memory"

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bytes


def effective_compute_dtype(gpu: GPUSpec, quant: Quantization) -> str:
    """Tensor-core dtype the matmuls run in on this GPU.

    Without FP8/INT8 tensor-core support, vLLM falls back to weight-only kernels
    (e.g. FP8 Marlin on Ampere) that dequantize to bf16, so compute is bf16 while
    the weight bytes stay small.
    """
    if gpu.peak_ops_per_s(quant.compute_dtype) is None:
        return "bf16"
    return quant.compute_dtype


def weight_bytes(model: ModelSpec, quant: Quantization) -> float:
    quantized = model.decoder_linear_params * quant.weight_bits / 8
    unquantized = (model.total_params - model.decoder_linear_params) * 2  # bf16
    return quantized + unquantized


def kv_cache_bytes_available(
    gpu: GPUSpec,
    model: ModelSpec,
    quant: Quantization,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    overhead_gib: float = 0.0,
) -> float:
    """Memory left for KV cache: budget minus weights minus runtime overhead. Can be negative."""
    budget = gpu.memory_bytes * gpu_memory_utilization
    return budget - weight_bytes(model, quant) - overhead_gib * 2**30


def kv_cache_capacity_tokens(
    gpu: GPUSpec,
    model: ModelSpec,
    quant: Quantization,
    kv_dtype: str = "bf16",
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    overhead_gib: float = 0.0,
) -> int:
    """How many tokens of KV cache fit. 0 if the weights alone don't fit."""
    free = kv_cache_bytes_available(gpu, model, quant, gpu_memory_utilization, overhead_gib)
    return max(0, int(free // model.kv_bytes_per_token(kv_dtype)))


def max_concurrent_sequences(kv_capacity_tokens: int, tokens_per_sequence: int) -> int:
    """Sequences that fit if each holds `tokens_per_sequence` (prompt + output) of KV."""
    if tokens_per_sequence <= 0:
        raise ValueError("tokens_per_sequence must be positive")
    return kv_capacity_tokens // tokens_per_sequence


def _phase(
    gpu: GPUSpec,
    quant: Quantization,
    linear_flops: float,
    attention_flops: float,
    bytes_moved: float,
    eff: Efficiency,
) -> PhaseEstimate:
    linear_peak = gpu.peak_ops_per_s(effective_compute_dtype(gpu, quant)) * eff.compute
    attn_peak = gpu.peak_ops_per_s("bf16") * eff.compute  # attention runs in bf16
    compute_s = linear_flops / linear_peak + attention_flops / attn_peak
    memory_s = bytes_moved / (gpu.mem_bw_bytes_per_s * eff.memory)
    return PhaseEstimate(linear_flops + attention_flops, bytes_moved, compute_s, memory_s)


def prefill(
    gpu: GPUSpec,
    model: ModelSpec,
    quant: Quantization,
    prompt_len: int,
    batch_size: int = 1,
    kv_dtype: str = "bf16",
    eff: Efficiency = IDEAL,
) -> PhaseEstimate:
    """Process `batch_size` prompts of `prompt_len` tokens and emit one token each.

    FLOPs: 2 * decoder linear params per prompt token, lm_head for the last token
    only, and causal attention sum_{i=1..T} 4*L*d*i = 2*L*d*T*(T+1).
    Bytes: read all weights once, write T tokens of KV per prompt.
    """
    if prompt_len <= 0 or batch_size <= 0:
        raise ValueError("prompt_len and batch_size must be positive")
    T, B = prompt_len, batch_size
    linear = B * (2.0 * model.decoder_linear_params * T + 2.0 * model.vocab_size * model.hidden_size)
    attention = B * 2.0 * model.num_layers * model.q_dim * T * (T + 1)
    bytes_moved = weight_bytes(model, quant) + B * T * model.kv_bytes_per_token(kv_dtype)
    return _phase(gpu, quant, linear, attention, bytes_moved, eff)


def ttft_s(
    gpu: GPUSpec,
    model: ModelSpec,
    quant: Quantization,
    prompt_len: int,
    kv_dtype: str = "bf16",
    eff: Efficiency = IDEAL,
) -> float:
    """Time to first token for one request on an idle GPU."""
    return prefill(gpu, model, quant, prompt_len, 1, kv_dtype, eff).time_s


def decode_step(
    gpu: GPUSpec,
    model: ModelSpec,
    quant: Quantization,
    batch_size: int,
    context_len: int,
    kv_dtype: str = "bf16",
    eff: Efficiency = IDEAL,
) -> PhaseEstimate:
    """One decode iteration: every sequence in the batch emits one token.

    FLOPs: 2 * linear params (incl. lm_head) + attention over `context_len` keys, per sequence.
    Bytes: read all weights once, read each sequence's KV, write one new KV entry each.
    """
    if batch_size <= 0 or context_len <= 0:
        raise ValueError("batch_size and context_len must be positive")
    B = batch_size
    kv = model.kv_bytes_per_token(kv_dtype)
    linear = B * model.linear_flops_per_token(include_lm_head=True)
    attention = B * model.attention_flops_per_token(context_len)
    bytes_moved = weight_bytes(model, quant) + B * (context_len + 1) * kv
    return _phase(gpu, quant, linear, attention, bytes_moved, eff)


def decode_throughput_tokens_per_s(step: PhaseEstimate, batch_size: int) -> float:
    return batch_size / step.time_s


def max_batch_for_tpot(
    gpu: GPUSpec,
    model: ModelSpec,
    quant: Quantization,
    tpot_slo_s: float,
    context_len: int,
    max_batch: int,
    kv_dtype: str = "bf16",
    eff: Efficiency = IDEAL,
) -> int:
    """Largest batch <= max_batch whose decode step meets the TPOT SLO. 0 if even batch 1 misses.

    Step time is non-decreasing in batch size, so binary search is exact.
    """
    def ok(b: int) -> bool:
        return decode_step(gpu, model, quant, b, context_len, kv_dtype, eff).time_s <= tpot_slo_s

    if max_batch < 1 or not ok(1):
        return 0
    lo, hi = 1, max_batch
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def cost_per_million_tokens(price_per_hour: float, tokens_per_s: float) -> float:
    if tokens_per_s <= 0:
        raise ValueError("tokens_per_s must be positive")
    return price_per_hour / 3600 / tokens_per_s * 1e6
