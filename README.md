# inference-capacity-planner

Open-source GPU serving capacity planner: roofline predictions validated against real vLLM/SGLang benchmarks, recommending GPU + config + $/1M tokens for a latency SLO.

Given a model, latency targets (TTFT / TPOT) and traffic, the planner will recommend a GPU, a serving config (engine, quantization, batch size, KV-cache size) and a cost per million tokens. A roofline model built from GPU specs predicts performance; benchmarks on Modal validate and calibrate it.

**Status: step 1 of the plan** — GPU spec table and roofline formulas. Nothing has been run on a GPU yet, so every performance number below is an *ideal roofline bound*, not a measurement.

## Rule for numbers in this README

Every number comes from something we ran or derived. The tables between `GENERATED` markers are written by `python -m capacity_planner.report --update-readme`, and `tests/test_readme.py` fails if they drift from the code.

## GPU spec table

Source of each value: `src/capacity_planner/gpus.py`. TFLOPS are dense (no 2:4 sparsity) tensor-core peaks, i.e. half of NVIDIA's with-sparsity datasheet figures.

<!-- BEGIN GENERATED: gpu_table -->
| GPU | Memory (GiB) | Bandwidth (GB/s) | BF16 dense TFLOPS | FP8 dense TFLOPS | Ridge point (FLOP/byte, BF16) | Modal $/hr |
|---|---:|---:|---:|---:|---:|---:|
| L4 | 24 | 300 | 121 | 242.5 | 403 | 0.80 |
| A10G | 24 | 600 | 70 | – | 117 | 1.10 |
| A100-40GB | 40 | 1555 | 312 | – | 201 | 2.10 |
| A100-80GB | 80 | 2039 | 312 | – | 153 | 2.50 |
| H100 | 80 | 3350 | 989.4 | 1978.9 | 295 | 3.95 |
<!-- END GENERATED: gpu_table -->

Caveats on the inputs:
- **Prices** are Modal's per-second on-demand rates as quoted from modal.com/pricing in web search results (2026-10-19). modal.com is blocked from the dev sandbox, so they have not been checked against the page directly.
- **A10G**: NVIDIA publishes no A10G datasheet. 70 BF16 TFLOPS is a third-party figure; step 2 should measure it.
- **A100 / H100** are assumed to be the SXM parts on Modal.

## Model: Qwen2.5-7B-Instruct

Derived from the architecture fields of its `config.json` (`src/capacity_planner/data/`). The derived parameter count matches the model card (7.61B total, 6.53B non-embedding), which checks the counting code.

<!-- BEGIN GENERATED: model_table -->
| Quantity | Value |
|---|---:|
| Total parameters | 7,615,616,512 |
| Non-embedding parameters | 6,525,621,760 |
| KV cache per token (BF16) | 57,344 bytes |
| Weights, bf16 | 14.19 GiB |
| Weights, fp8 | 8.11 GiB |
| Weights, awq-int4 | 5.19 GiB |
<!-- END GENERATED: model_table -->

Quantized formats only shrink the decoder-layer linear weights; embeddings, `lm_head`, norms and biases stay in BF16. AWQ int4 counts group-128 scales and zeros (4.156 bits/weight).

## Roofline formulas

Each phase is bounded by `time = max(FLOPs / peak_FLOPs, bytes / peak_bandwidth)` (`src/capacity_planner/roofline.py`). With `L` layers, `d = heads × head_dim`, `P_lin` decoder linear params, `V × h` the lm_head, `kv` bytes of KV per token and `W` weight bytes:

| Quantity | FLOPs | Bytes |
|---|---|---|
| Prefill (TTFT), prompt `T` | `2·P_lin·T + 2·V·h + 2·L·d·T·(T+1)` | `W + T·kv` |
| Decode step, batch `B`, context `C` | `B·(2·(P_lin + V·h) + 4·L·d·C)` | `W + B·(C+1)·kv` |

- **KV-cache capacity (tokens)** = `(GPU memory × 0.9 − W − overhead) / kv`. 0.9 is vLLM's default `--gpu-memory-utilization`; overhead (activations, CUDA graphs) defaults to 0 until step 2 measures it.
- **TPOT** = decode step time. **Decode throughput** = `B / step time`.
- **$/1M tokens** = `$/hr / 3600 / (tokens/s) × 10^6`.
- On GPUs without FP8 tensor cores (A10G, A100), FP8 weights run through weight-only kernels, so matmuls are costed at BF16 peak while weight bytes stay at 1 byte/param.

## Example: ideal roofline bounds

Workload: 1024-token prompt, 256 output tokens; each sequence holds 1280 tokens of BF16 KV. "Max concurrent seqs" is the KV-cache limit; throughput and cost are for a decode step at that batch with every sequence at full 1280-token context. Cost excludes prefill time. These are best-case bounds with 100% of peak FLOPs and bandwidth and zero runtime overhead; real engines will be slower. Step 2 measures by how much.

<!-- BEGIN GENERATED: roofline_table -->
| GPU | Quant | Compute dtype | KV capacity (tokens) | Max concurrent seqs | TTFT (ms) | TPOT @ batch 1 (ms) | Decode tok/s @ max seqs | Decode-only $/1M output tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| L4 | bf16 | bf16 | 138,839 | 108 | 112.2 | 51.02 | 1,399 | 0.159 |
| L4 | fp8 | fp8 | 252,631 | 197 | 56.9 | 29.26 | 2,550 | 0.087 |
| L4 | awq-int4 | bf16 | 307,304 | 240 | 112.2 | 18.81 | 3,103 | 0.072 |
| A10G | bf16 | bf16 | 138,839 | 108 | 193.9 | 25.51 | 2,797 | 0.109 |
| A10G | fp8 | bf16 | 252,631 | 197 | 193.9 | 14.63 | 4,777 | 0.064 |
| A10G | awq-int4 | bf16 | 307,304 | 240 | 193.9 | 9.41 | 4,777 | 0.064 |
| A100-40GB | bf16 | bf16 | 408,472 | 319 | 43.5 | 9.84 | 12,830 | 0.045 |
| A100-40GB | fp8 | bf16 | 522,264 | 408 | 43.5 | 5.65 | 16,404 | 0.036 |
| A100-40GB | awq-int4 | bf16 | 576,938 | 450 | 43.5 | 3.63 | 18,116 | 0.032 |
| A100-80GB | bf16 | bf16 | 1,082,557 | 845 | 43.5 | 7.51 | 21,291 | 0.033 |
| A100-80GB | fp8 | bf16 | 1,196,349 | 934 | 43.5 | 4.31 | 21,291 | 0.033 |
| A100-80GB | awq-int4 | bf16 | 1,251,023 | 977 | 43.5 | 2.77 | 21,291 | 0.033 |
| H100 | bf16 | bf16 | 1,082,557 | 845 | 13.7 | 4.57 | 36,619 | 0.030 |
| H100 | fp8 | fp8 | 1,196,349 | 934 | 7.0 | 2.62 | 40,469 | 0.027 |
| H100 | awq-int4 | bf16 | 1,251,023 | 977 | 13.7 | 1.68 | 42,320 | 0.026 |
<!-- END GENERATED: roofline_table -->

## Development

```bash
pip install -e ".[dev]"
pytest
python -m capacity_planner.report                  # print tables
python -m capacity_planner.report --update-readme  # regenerate README tables
```

## Plan

1. **GPU specs + roofline formulas** (this step).
2. Benchmarks on Modal via GitHub Actions (vLLM/SGLang, Qwen2.5-7B-Instruct, budget < $40) to measure memory overhead and achieved FLOPs/bandwidth.
3. Calibrate the roofline and build the recommender: model + SLOs + traffic → GPU, engine, quantization, batch size, KV-cache size, $/1M tokens.
