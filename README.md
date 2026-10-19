# inference-capacity-planner

Open-source GPU serving capacity planner: roofline predictions validated against real vLLM/SGLang benchmarks, recommending GPU + config + $/1M tokens for a latency SLO.

Given a model, latency targets (TTFT / TPOT) and traffic, the planner will recommend a GPU, a serving config (engine, quantization, batch size, KV-cache size) and a cost per million tokens. A roofline model built from GPU specs predicts performance; benchmarks on Modal validate and calibrate it.

**Status: step 2 of the plan** — GPU spec table and roofline formulas (step 1), plus benchmarks on Modal that measure how far real hardware and vLLM fall short of the roofline (step 2). Roofline numbers are *ideal bounds*; measured numbers are labelled and link to the GitHub Actions run that produced them.

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

- **KV-cache capacity (tokens)** = `(GPU memory × 0.92 − W − overhead) / kv`. 0.92 is the default `--gpu-memory-utilization` in vLLM 0.30.0, the version the benchmarks pin; overhead (activations, CUDA graphs) defaults to 0 until step 2 measures it.
- **TPOT** = decode step time. **Decode throughput** = `B / step time`.
- **$/1M tokens** = `$/hr / 3600 / (tokens/s) × 10^6`.
- On GPUs without FP8 tensor cores (A10G, A100), FP8 weights run through weight-only kernels, so matmuls are costed at BF16 peak while weight bytes stay at 1 byte/param.

## Example: ideal roofline bounds

Workload: 1024-token prompt, 256 output tokens; each sequence holds 1280 tokens of BF16 KV. "Max concurrent seqs" is the KV-cache limit; throughput and cost are for a decode step at that batch with every sequence at full 1280-token context. Cost excludes prefill time. These are best-case bounds with 100% of peak FLOPs and bandwidth and zero runtime overhead; real engines will be slower. Step 2 measures by how much.

<!-- BEGIN GENERATED: roofline_table -->
| GPU | Quant | Compute dtype | KV capacity (tokens) | Max concurrent seqs | TTFT (ms) | TPOT @ batch 1 (ms) | Decode tok/s @ max seqs | Decode-only $/1M output tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| L4 | bf16 | bf16 | 147,826 | 115 | 112.2 | 51.02 | 1,457 | 0.152 |
| L4 | fp8 | fp8 | 261,618 | 204 | 56.9 | 29.26 | 2,583 | 0.086 |
| L4 | awq-int4 | bf16 | 316,292 | 247 | 112.2 | 18.81 | 3,125 | 0.071 |
| A10G | bf16 | bf16 | 147,826 | 115 | 193.9 | 25.51 | 2,914 | 0.105 |
| A10G | fp8 | bf16 | 261,618 | 204 | 193.9 | 14.63 | 4,777 | 0.064 |
| A10G | awq-int4 | bf16 | 316,292 | 247 | 193.9 | 9.41 | 4,777 | 0.064 |
| A100-40GB | bf16 | bf16 | 423,452 | 330 | 43.5 | 9.84 | 13,000 | 0.045 |
| A100-40GB | fp8 | bf16 | 537,244 | 419 | 43.5 | 5.65 | 16,501 | 0.035 |
| A100-40GB | awq-int4 | bf16 | 591,918 | 462 | 43.5 | 3.63 | 18,184 | 0.032 |
| A100-80GB | bf16 | bf16 | 1,112,516 | 869 | 43.5 | 7.51 | 21,291 | 0.033 |
| A100-80GB | fp8 | bf16 | 1,226,308 | 958 | 43.5 | 4.31 | 21,291 | 0.033 |
| A100-80GB | awq-int4 | bf16 | 1,280,982 | 1000 | 43.5 | 2.77 | 21,291 | 0.033 |
| H100 | bf16 | bf16 | 1,112,516 | 869 | 13.7 | 4.57 | 36,819 | 0.030 |
| H100 | fp8 | fp8 | 1,226,308 | 958 | 7.0 | 2.62 | 40,584 | 0.027 |
| H100 | awq-int4 | bf16 | 1,280,982 | 1000 | 13.7 | 1.68 | 42,390 | 0.026 |
<!-- END GENERATED: roofline_table -->

## Benchmarks (step 2)

GPU runs are launched by GitHub Actions (`.github/workflows/bench.yml`) on [Modal](https://modal.com), with vLLM 0.30.0 and Qwen2.5-7B-Instruct.

- **What runs:** only the runs listed in `bench/requests.json`. Each run has an id; once an id is in `results/ledger.jsonl` it is never launched again.
- **Budget guard:** before launching, the workflow adds the spend so far to each pending run's worst case (container timeout × GPU price × 1.1) and refuses to start if the total exceeds `budget_usd` ($40).
- **Suites:** `micro` measures read bandwidth (a BF16 matrix-vector product, the shape of batch-1 decode), copy bandwidth and dense GEMM throughput with PyTorch. `vllm` measures KV-cache size, TTFT for one request on an idle engine, and decode step time/throughput for batches of 512-token prompts. TTFT and TPOT come from vLLM's own per-request timestamps; prompts are random token ids with prefix caching off.
- **Results:** each run is committed as `results/<id>.json` together with the tables below, which the workflow regenerates.

Spend so far: <!-- BEGIN GENERATED: bench_spend -->
31 runs (18 succeeded), at most **$2.05** of GPU time (runner wall-clock × Modal GPU price × 1.1 for CPU/RAM).
<!-- END GENERATED: bench_spend -->

### Measured GPU throughput vs datasheet

<!-- BEGIN GENERATED: measured_gpu -->
| GPU | Device (nvidia-smi) | Memory GiB (measured / spec) | GEMV read GB/s (% of datasheet) | Copy GB/s (%) | BF16 GEMM TFLOPS (%) | FP8 GEMM TFLOPS (%) | INT8 GEMM TOPS (%) | Run |
|---|---|---:|---:|---:|---:|---:|---:|---|
| L4 | NVIDIA L4 | 22.1 / 24 | 259 (86%) | 230 (77%) | 53 (44%) | 92 (38%) | 128 (53%) | [005-l4-micro](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39222305725) |
| A10G | NVIDIA A10 | 22.1 / 24 | 498 (83%) | 484 (81%) | 77 (111%) | unsupported | 147 (–) | [007-a10g-micro](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39223120807) |
| A100-40GB | NVIDIA A100-SXM4-40GB | 39.5 / 40 | 1,416 (91%) | 1,381 (89%) | 262 (84%) | unsupported | 408 (65%) | [008-a100-40gb-micro](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39223120807) |
| A100-80GB | NVIDIA A100-SXM4-80GB | 79.3 / 80 | 1,745 (86%) | 1,752 (86%) | 254 (82%) | unsupported | 408 (65%) | [009-a100-80gb-micro](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39223773836) |
| H100 | NVIDIA H100 80GB HBM3 | 79.2 / 80 | 2,863 (85%) | 2,998 (89%) | 791 (80%) | 1431 (72%) | 874 (44%) | [010-h100-micro](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39223120807) |
<!-- END GENERATED: measured_gpu -->

### vLLM memory: weights, KV cache and overhead

The roofline KV capacity assumes zero overhead. "Non-KV overhead" = GPU memory × utilization − weights − KV cache, using vLLM's own log figures; it covers activations, CUDA graphs and the CUDA context.

<!-- BEGIN GENERATED: measured_vllm_memory -->
| GPU | Quant | Weights GiB (vLLM / roofline) | KV cache GiB (vLLM) | Non-KV overhead GiB (derived) | KV tokens (vLLM / roofline) | Run | Notes |
|---|---|---:|---:|---:|---:|---|---|
| L4 | awq-int4 | 5.29 / 5.19 | 13.17 | 2.23 | 246,528 / 316,292 | [018-l4-vllm-awq-int4](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39223120807) |  |
| L4 | bf16 | 14.29 / 14.19 | 3.25 | 3.15 | 60,816 / 147,826 | [006-l4-vllm-bf16](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39222305725) |  |
| L4 | fp8 | 8.17 / 8.11 | 10.17 | 2.35 | 190,512 / 261,618 | [015-l4-vllm-fp8](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39223120807) |  |
| A10G | awq-int4 | 5.29 / 5.19 | 13.16 | 2.24 | 246,480 / 316,292 | [019-a10g-vllm-awq-int4](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39223120807) |  |
| A10G | bf16 | 14.29 / 14.19 | 3.23 | 3.17 | 60,512 / 147,826 | [011-a10g-vllm-bf16](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39223773836) |  |
| A100-40GB | bf16 | 14.29 / 14.19 | 19.25 | 3.26 | 360,528 / 423,452 | [012-a100-40gb-vllm-bf16](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39223120807) |  |
| A100-80GB | awq-int4 | 5.29 / 5.19 | 65.76 | 2.55 | 1,231,408 / 1,280,982 | [020-a100-80gb-vllm-awq-int4](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39223120807) |  |
| A100-80GB | bf16 | 14.29 / 14.19 | 55.83 | 3.48 | 1,045,408 / 1,112,516 | [013-a100-80gb-vllm-bf16](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39223773836) |  |
| H100 | awq-int4 | 5.38 / 5.19 | 64.25 | 3.65 | 1,203,072 / 1,280,982 | [021-h100-vllm-awq-int4](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39223120807) |  |
| H100 | bf16 | 14.29 / 14.19 | 55.28 | 3.71 | 1,035,104 / 1,112,516 | [014-h100-vllm-bf16](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39223120807) |  |
| H100 | fp8 | 8.17 / 8.11 | 61.37 | 3.74 | 1,149,168 / 1,226,308 | [017-h100-vllm-fp8](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39223120807) |  |
<!-- END GENERATED: measured_vllm_memory -->

### TTFT: vLLM vs roofline

<!-- BEGIN GENERATED: measured_ttft -->
| GPU | Quant | Prompt tokens | TTFT ms (vLLM, median) | TTFT ms (roofline) | vLLM / roofline |
|---|---|---:|---:|---:|---:|
| L4 | awq-int4 | 128 | 41.1 | 18.6 | 2.21 |
| L4 | awq-int4 | 512 | 149.9 | 55.7 | 2.69 |
| L4 | awq-int4 | 1024 | 287.7 | 112.2 | 2.56 |
| L4 | awq-int4 | 2048 | 575.9 | 227.9 | 2.53 |
| L4 | awq-int4 | 4000 | 1193.9 | 458.0 | 2.61 |
| L4 | bf16 | 128 | 71.3 | 50.8 | 1.40 |
| L4 | bf16 | 512 | 143.7 | 55.7 | 2.58 |
| L4 | bf16 | 1024 | 257.2 | 112.2 | 2.29 |
| L4 | bf16 | 2048 | 505.7 | 227.9 | 2.22 |
| L4 | bf16 | 4000 | 1034.3 | 458.0 | 2.26 |
| L4 | fp8 | 128 | 44.2 | 29.0 | 1.52 |
| L4 | fp8 | 512 | 107.9 | 29.1 | 3.71 |
| L4 | fp8 | 1024 | 190.1 | 56.9 | 3.34 |
| L4 | fp8 | 2048 | 403.5 | 117.2 | 3.44 |
| L4 | fp8 | 4000 | 836.0 | 241.8 | 3.46 |
| A10G | awq-int4 | 128 | 32.1 | 23.9 | 1.34 |
| A10G | awq-int4 | 512 | 112.5 | 96.2 | 1.17 |
| A10G | awq-int4 | 1024 | 227.8 | 193.9 | 1.17 |
| A10G | awq-int4 | 2048 | 453.7 | 393.9 | 1.15 |
| A10G | awq-int4 | 4000 | 906.7 | 791.6 | 1.15 |
| A10G | bf16 | 128 | 40.7 | 25.4 | 1.60 |
| A10G | bf16 | 512 | 108.9 | 96.2 | 1.13 |
| A10G | bf16 | 1024 | 211.2 | 193.9 | 1.09 |
| A10G | bf16 | 2048 | 413.7 | 393.9 | 1.05 |
| A10G | bf16 | 4000 | 928.4 | 791.6 | 1.17 |
| A100-40GB | bf16 | 128 | 20.1 | 9.8 | 2.05 |
| A100-40GB | bf16 | 512 | 36.9 | 21.6 | 1.71 |
| A100-40GB | bf16 | 1024 | 67.8 | 43.5 | 1.56 |
| A100-40GB | bf16 | 2048 | 126.9 | 88.4 | 1.44 |
| A100-40GB | bf16 | 4000 | 254.6 | 177.6 | 1.43 |
| A100-80GB | awq-int4 | 128 | 16.1 | 5.4 | 2.99 |
| A100-80GB | awq-int4 | 512 | 48.6 | 21.6 | 2.25 |
| A100-80GB | awq-int4 | 1024 | 91.2 | 43.5 | 2.10 |
| A100-80GB | awq-int4 | 2048 | 181.9 | 88.4 | 2.06 |
| A100-80GB | awq-int4 | 4000 | 358.7 | 177.6 | 2.02 |
| A100-80GB | bf16 | 128 | 15.1 | 7.5 | 2.02 |
| A100-80GB | bf16 | 512 | 36.5 | 21.6 | 1.69 |
| A100-80GB | bf16 | 1024 | 67.7 | 43.5 | 1.56 |
| A100-80GB | bf16 | 2048 | 128.8 | 88.4 | 1.46 |
| A100-80GB | bf16 | 4000 | 259.8 | 177.6 | 1.46 |
| H100 | awq-int4 | 128 | 7.6 | 1.7 | 4.47 |
| H100 | awq-int4 | 512 | 14.4 | 6.8 | 2.11 |
| H100 | awq-int4 | 1024 | 24.8 | 13.7 | 1.81 |
| H100 | awq-int4 | 2048 | 47.4 | 27.9 | 1.70 |
| H100 | awq-int4 | 4000 | 96.1 | 56.0 | 1.72 |
| H100 | bf16 | 128 | 8.8 | 4.5 | 1.93 |
| H100 | bf16 | 512 | 13.2 | 6.8 | 1.94 |
| H100 | bf16 | 1024 | 24.2 | 13.7 | 1.76 |
| H100 | bf16 | 2048 | 47.2 | 27.9 | 1.69 |
| H100 | bf16 | 4000 | 91.3 | 56.0 | 1.63 |
| H100 | fp8 | 128 | 6.8 | 2.6 | 2.62 |
| H100 | fp8 | 512 | 10.4 | 3.4 | 3.03 |
| H100 | fp8 | 1024 | 18.1 | 7.0 | 2.60 |
| H100 | fp8 | 2048 | 33.9 | 14.4 | 2.36 |
| H100 | fp8 | 4000 | 68.2 | 29.6 | 2.30 |
<!-- END GENERATED: measured_ttft -->

### Decode: vLLM vs roofline

Each batch of 512-token prompts runs twice, with 64 and with 192 output tokens. Both runs spend the same time on prefill, so the extra engine time divided by the extra 128 tokens is the clean **decode step** at that batch size; the roofline is evaluated at the mean context over those steps (640 tokens). Decode tok/s and $/1M (output tokens only, Modal price from the spec table) follow from it. **TPOT under load** is the median per-request TPOT from vLLM's timestamps, which also includes steps shared with other requests' prefill. Run 004 predates the two-length measurement and only has TPOT under load.

<!-- BEGIN GENERATED: measured_decode -->
| GPU | Quant | Batch | Decode step ms (vLLM) | Decode step ms (roofline) | vLLM / roofline | Decode tok/s (vLLM) | $/1M output tokens | TPOT ms under load (median) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L4 | awq-int4 | 1 | 18.97 | 18.69 | 1.02 | 53 | 4.212 | 18.93 |
| L4 | awq-int4 | 2 | 19.11 | 18.81 | 1.02 | 105 | 2.122 | 19.51 |
| L4 | awq-int4 | 4 | 19.42 | 19.06 | 1.02 | 206 | 1.078 | 19.46 |
| L4 | awq-int4 | 8 | 20.22 | 19.55 | 1.03 | 396 | 0.561 | 20.17 |
| L4 | awq-int4 | 16 | 21.82 | 20.53 | 1.06 | 733 | 0.303 | 21.78 |
| L4 | awq-int4 | 32 | 26.32 | 22.49 | 1.17 | 1,216 | 0.183 | 31.88 |
| L4 | awq-int4 | 64 | 37.67 | 26.41 | 1.43 | 1,699 | 0.131 | 54.83 |
| L4 | awq-int4 | 128 | 62.09 | 34.25 | 1.81 | 2,062 | 0.108 | 108.08 |
| L4 | awq-int4 | 256 | 125.31 | 49.94 | 2.51 | 2,043 | 0.109 | 224.04 |
| L4 | bf16 | 1 | 56.51 | 50.89 | 1.11 | 18 | 12.545 | 56.47 |
| L4 | bf16 | 2 | 57.66 | 51.02 | 1.13 | 35 | 6.400 | 57.93 |
| L4 | bf16 | 4 | 58.07 | 51.26 | 1.13 | 69 | 3.223 | 58.05 |
| L4 | bf16 | 8 | 59.03 | 51.75 | 1.14 | 136 | 1.638 | 58.97 |
| L4 | bf16 | 16 | 60.86 | 52.73 | 1.15 | 263 | 0.844 | 60.65 |
| L4 | bf16 | 32 | 67.37 | 54.69 | 1.23 | 475 | 0.467 | 76.59 |
| L4 | bf16 | 64 | 75.03 | 58.61 | 1.28 | 853 | 0.260 | 89.36 |
| L4 | fp8 | 1 | 34.45 | 29.14 | 1.18 | 29 | 7.648 | 34.48 |
| L4 | fp8 | 2 | 34.64 | 29.26 | 1.18 | 58 | 3.845 | 34.90 |
| L4 | fp8 | 4 | 34.81 | 29.51 | 1.18 | 115 | 1.932 | 34.96 |
| L4 | fp8 | 8 | 35.68 | 30.00 | 1.19 | 224 | 0.990 | 35.67 |
| L4 | fp8 | 16 | 36.83 | 30.98 | 1.19 | 434 | 0.511 | 37.11 |
| L4 | fp8 | 32 | 40.82 | 32.94 | 1.24 | 784 | 0.283 | 44.86 |
| L4 | fp8 | 64 | 49.69 | 36.86 | 1.35 | 1,288 | 0.172 | 59.86 |
| L4 | fp8 | 128 | 63.72 | 44.70 | 1.43 | 2,009 | 0.111 | 90.78 |
| L4 | fp8 | 256 | 96.46 | 60.39 | 1.60 | 2,654 | 0.084 | 167.09 |
| A10G | awq-int4 | 1 | 10.77 | 9.35 | 1.15 | 93 | 3.290 | 10.74 |
| A10G | awq-int4 | 2 | 10.76 | 9.41 | 1.14 | 186 | 1.644 | 11.06 |
| A10G | awq-int4 | 4 | 11.02 | 9.53 | 1.16 | 363 | 0.842 | 10.98 |
| A10G | awq-int4 | 8 | 11.49 | 9.77 | 1.18 | 696 | 0.439 | 11.41 |
| A10G | awq-int4 | 16 | 12.49 | 10.26 | 1.22 | 1,281 | 0.239 | 12.49 |
| A10G | awq-int4 | 32 | 14.80 | 11.24 | 1.32 | 2,162 | 0.141 | 22.97 |
| A10G | awq-int4 | 64 | 24.70 | 13.21 | 1.87 | 2,591 | 0.118 | 37.61 |
| A10G | awq-int4 | 128 | 45.93 | 26.33 | 1.74 | 2,787 | 0.110 | 77.27 |
| A10G | awq-int4 | 256 | 90.14 | 52.65 | 1.71 | 2,840 | 0.108 | 165.59 |
| A10G | bf16 | 1 | 32.38 | 25.45 | 1.27 | 31 | 9.893 | 32.39 |
| A10G | bf16 | 2 | 32.56 | 25.51 | 1.28 | 61 | 4.974 | 32.80 |
| A10G | bf16 | 4 | 32.84 | 25.63 | 1.28 | 122 | 2.509 | 32.86 |
| A10G | bf16 | 8 | 33.41 | 25.88 | 1.29 | 239 | 1.276 | 33.46 |
| A10G | bf16 | 16 | 34.78 | 26.37 | 1.32 | 460 | 0.664 | 34.59 |
| A10G | bf16 | 32 | 34.67 | 27.35 | 1.27 | 923 | 0.331 | 38.63 |
| A10G | bf16 | 64 | 39.54 | 29.31 | 1.35 | 1,619 | 0.189 | 51.78 |
| A100-40GB | bf16 | 1 | 12.22 | 9.82 | 1.24 | 82 | 7.121 | 12.23 |
| A100-40GB | bf16 | 2 | 12.02 | 9.84 | 1.22 | 166 | 3.504 | 12.05 |
| A100-40GB | bf16 | 4 | 13.41 | 9.89 | 1.36 | 298 | 1.954 | 13.59 |
| A100-40GB | bf16 | 8 | 12.36 | 9.98 | 1.24 | 647 | 0.901 | 12.36 |
| A100-40GB | bf16 | 16 | 12.80 | 10.17 | 1.26 | 1,250 | 0.466 | 12.78 |
| A100-40GB | bf16 | 32 | 13.83 | 10.55 | 1.31 | 2,314 | 0.252 | 15.85 |
| A100-40GB | bf16 | 64 | 15.98 | 11.31 | 1.41 | 4,005 | 0.146 | 19.34 |
| A100-40GB | bf16 | 128 | 18.45 | 12.82 | 1.44 | 6,936 | 0.084 | 26.64 |
| A100-40GB | bf16 | 256 | 26.50 | 15.85 | 1.67 | 9,659 | 0.060 | 46.01 |
| A100-80GB | awq-int4 | 1 | 4.76 | 2.75 | 1.73 | 210 | 3.307 | 4.82 |
| A100-80GB | awq-int4 | 2 | 4.94 | 2.77 | 1.79 | 404 | 1.716 | 5.03 |
| A100-80GB | awq-int4 | 4 | 4.92 | 2.80 | 1.75 | 813 | 0.853 | 4.95 |
| A100-80GB | awq-int4 | 8 | 5.04 | 2.88 | 1.75 | 1,588 | 0.437 | 5.07 |
| A100-80GB | awq-int4 | 16 | 5.79 | 3.02 | 1.92 | 2,762 | 0.251 | 5.75 |
| A100-80GB | awq-int4 | 32 | 7.07 | 3.31 | 2.14 | 4,528 | 0.153 | 8.76 |
| A100-80GB | awq-int4 | 64 | 11.31 | 3.89 | 2.91 | 5,659 | 0.123 | 14.47 |
| A100-80GB | awq-int4 | 128 | 18.07 | 5.91 | 3.06 | 7,082 | 0.098 | 30.54 |
| A100-80GB | awq-int4 | 256 | 32.99 | 11.81 | 2.79 | 7,759 | 0.089 | 60.44 |
| A100-80GB | bf16 | 1 | 10.08 | 7.49 | 1.35 | 99 | 6.998 | 10.10 |
| A100-80GB | bf16 | 2 | 10.10 | 7.51 | 1.35 | 198 | 3.504 | 10.12 |
| A100-80GB | bf16 | 4 | 10.14 | 7.54 | 1.34 | 395 | 1.759 | 10.13 |
| A100-80GB | bf16 | 8 | 10.42 | 7.61 | 1.37 | 768 | 0.904 | 10.43 |
| A100-80GB | bf16 | 16 | 10.81 | 7.76 | 1.39 | 1,479 | 0.469 | 10.79 |
| A100-80GB | bf16 | 32 | 11.39 | 8.05 | 1.42 | 2,809 | 0.247 | 12.59 |
| A100-80GB | bf16 | 64 | 13.98 | 8.62 | 1.62 | 4,579 | 0.152 | 18.09 |
| A100-80GB | bf16 | 128 | 16.61 | 9.78 | 1.70 | 7,705 | 0.090 | 24.99 |
| A100-80GB | bf16 | 256 | 25.90 | 12.08 | 2.14 | 9,884 | 0.070 | 45.90 |
| H100 | awq-int4 | 1 | 4.25 | 1.67 | 2.54 | 235 | 4.662 | 4.30 |
| H100 | awq-int4 | 2 | 4.39 | 1.68 | 2.60 | 456 | 2.407 | 4.35 |
| H100 | awq-int4 | 4 | 4.36 | 1.71 | 2.55 | 918 | 1.196 | 4.37 |
| H100 | awq-int4 | 8 | 4.43 | 1.75 | 2.53 | 1,806 | 0.607 | 4.45 |
| H100 | awq-int4 | 16 | 4.46 | 1.84 | 2.42 | 3,591 | 0.305 | 4.59 |
| H100 | awq-int4 | 32 | 4.93 | 2.01 | 2.45 | 6,494 | 0.169 | 4.96 |
| H100 | awq-int4 | 64 | 5.88 | 2.37 | 2.49 | 10,889 | 0.101 | 6.90 |
| H100 | awq-int4 | 128 | 7.37 | 3.07 | 2.40 | 17,379 | 0.063 | 11.27 |
| H100 | awq-int4 | 256 | 11.60 | 4.47 | 2.59 | 22,074 | 0.050 | 18.50 |
| H100 | bf16 | 1 | 6.14 | 4.56 | 1.35 | 163 | 6.732 | 6.18 |
| H100 | bf16 | 2 | 6.32 | 4.57 | 1.38 | 317 | 3.464 | 6.21 |
| H100 | bf16 | 4 | 6.24 | 4.59 | 1.36 | 641 | 1.711 | 6.28 |
| H100 | bf16 | 8 | 6.34 | 4.63 | 1.37 | 1,261 | 0.870 | 6.39 |
| H100 | bf16 | 16 | 6.55 | 4.72 | 1.39 | 2,443 | 0.449 | 6.53 |
| H100 | bf16 | 32 | 6.66 | 4.90 | 1.36 | 4,803 | 0.228 | 6.68 |
| H100 | bf16 | 64 | 7.38 | 5.25 | 1.41 | 8,674 | 0.126 | 8.22 |
| H100 | bf16 | 128 | 8.59 | 5.95 | 1.44 | 14,900 | 0.074 | 12.08 |
| H100 | bf16 | 256 | 11.59 | 7.36 | 1.58 | 22,080 | 0.050 | 17.76 |
| H100 | fp8 | 1 | 4.21 | 2.61 | 1.61 | 238 | 4.615 | 4.25 |
| H100 | fp8 | 2 | 4.28 | 2.62 | 1.63 | 467 | 2.347 | 4.29 |
| H100 | fp8 | 4 | 4.37 | 2.64 | 1.65 | 915 | 1.199 | 4.38 |
| H100 | fp8 | 8 | 4.37 | 2.69 | 1.63 | 1,831 | 0.599 | 4.40 |
| H100 | fp8 | 16 | 4.43 | 2.77 | 1.60 | 3,615 | 0.303 | 4.52 |
| H100 | fp8 | 32 | 5.02 | 2.95 | 1.70 | 6,376 | 0.172 | 5.04 |
| H100 | fp8 | 64 | 5.40 | 3.30 | 1.64 | 11,841 | 0.093 | 6.05 |
| H100 | fp8 | 128 | 6.73 | 4.00 | 1.68 | 19,016 | 0.058 | 8.85 |
| H100 | fp8 | 256 | 9.74 | 5.41 | 1.80 | 26,296 | 0.042 | 14.41 |
<!-- END GENERATED: measured_decode -->

## Development

```bash
pip install -e ".[dev]"
pytest
python -m capacity_planner.report                  # print tables
python -m capacity_planner.report --update-readme  # regenerate README tables
```

## Plan

1. **GPU specs + roofline formulas** (this step).
2. **Benchmarks on Modal via GitHub Actions** (vLLM first, then SGLang; Qwen2.5-7B-Instruct; budget < $40) to measure memory overhead and achieved FLOPs/bandwidth (in progress).
3. Calibrate the roofline and build the recommender: model + SLOs + traffic → GPU, engine, quantization, batch size, KV-cache size, $/1M tokens.
