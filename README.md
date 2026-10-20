# inference-capacity-planner

Open-source GPU serving capacity planner: roofline predictions validated against real vLLM/SGLang benchmarks, recommending GPU + config + $/1M tokens for a latency SLO.

Given a model, latency targets (TTFT / TPOT) and traffic, the planner will recommend a GPU, a serving config (engine, quantization, batch size, KV-cache size) and a cost per million tokens. A roofline model built from GPU specs predicts performance; benchmarks on Modal validate and calibrate it.

**Status: step 5 of the plan** — GPU spec table and roofline formulas (step 1), benchmarks on Modal that measure how far real hardware and vLLM fall short of the roofline (step 2), a calibrated recommender validated under real load (step 3), predictions for models and multi-GPU setups we never benchmarked, checked against runs afterwards (step 4), and load tests of the resulting recommendations (step 5). Roofline numbers are *ideal bounds*; measured numbers are labelled and link to the GitHub Actions run that produced them.

## Rule for numbers in this README

Every number comes from something we ran or derived. The tables between `GENERATED` markers are written by `python -m capacity_planner.report --update-readme`, and `tests/test_readme.py` fails if they drift from the code.

## GPU spec table

Source of each value: `src/capacity_planner/gpus.py`. TFLOPS are dense (no 2:4 sparsity) tensor-core peaks, i.e. half of NVIDIA's with-sparsity datasheet figures.

<!-- BEGIN GENERATED: gpu_table -->
| GPU | Memory (GiB) | Bandwidth (GB/s) | BF16 dense TFLOPS | FP8 dense TFLOPS | Ridge point (FLOP/byte, BF16) | Modal $/hr |
|---|---:|---:|---:|---:|---:|---:|
| L4 | 24 | 300 | 121 | 242.5 | 403 | 0.80 |
| A10G | 24 | 600 | 125 | – | 208 | 1.10 |
| A100-40GB | 40 | 1555 | 312 | – | 201 | 2.10 |
| A100-80GB | 80 | 2039 | 312 | – | 153 | 2.50 |
| H100 | 80 | 3350 | 989.4 | 1978.9 | 295 | 3.95 |
<!-- END GENERATED: gpu_table -->

Caveats on the inputs:
- **Prices** are Modal's per-second on-demand rates as quoted from modal.com/pricing in web search results (2026-10-19). modal.com is blocked from the dev sandbox, so they have not been checked against the page directly.
- **A10G**: Modal's `A10G` runs on an NVIDIA **A10** (nvidia-smi in runs 007, 011, 019), so this row uses NVIDIA's A10 datasheet. An earlier third-party A10G figure (70 BF16 TFLOPS) was below what run 007 measured.
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
| A10G | bf16 | bf16 | 147,826 | 115 | 108.6 | 25.51 | 2,914 | 0.105 |
| A10G | fp8 | bf16 | 261,618 | 204 | 108.6 | 14.63 | 5,166 | 0.059 |
| A10G | awq-int4 | bf16 | 316,292 | 247 | 108.6 | 9.41 | 6,249 | 0.049 |
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
55 runs (42 succeeded), at most **$10.88** of GPU time (runner wall-clock × Modal GPU price × 1.1 for CPU/RAM).
<!-- END GENERATED: bench_spend -->

Configurations that did not run: **FP8 on A100-80GB** fails at vLLM 0.30.0 engine startup twice, in two different places: in torch.compile (run 016, `InductorError: auto_functionalized was not removed`) and, with the eager compile backend, in the CUTLASS FP8 GEMM (run 022, `cutlass_scaled_mm_sm80_epilogue`). The A100 has no FP8 tensor cores, so this is vLLM's weight-only FP8 path. Not retried.

### Measured GPU throughput vs datasheet

<!-- BEGIN GENERATED: measured_gpu -->
| GPU | Device (nvidia-smi) | Memory GiB (measured / spec) | GEMV read GB/s (% of datasheet) | Copy GB/s (%) | BF16 GEMM TFLOPS (%) | FP8 GEMM TFLOPS (%) | INT8 GEMM TOPS (%) | Run |
|---|---|---:|---:|---:|---:|---:|---:|---|
| L4 | NVIDIA L4 | 22.1 / 24 | 259 (86%) | 230 (77%) | 53 (44%) | 92 (38%) | 128 (53%) | [005-l4-micro](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39222305725) |
| A10G | NVIDIA A10 | 22.1 / 24 | 498 (83%) | 484 (81%) | 77 (62%) | unsupported | 147 (59%) | [007-a10g-micro](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39223120807) |
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
| A10G | awq-int4 | 128 | 32.1 | 13.4 | 2.40 |
| A10G | awq-int4 | 512 | 112.5 | 53.9 | 2.09 |
| A10G | awq-int4 | 1024 | 227.8 | 108.6 | 2.10 |
| A10G | awq-int4 | 2048 | 453.7 | 220.6 | 2.06 |
| A10G | awq-int4 | 4000 | 906.7 | 443.3 | 2.05 |
| A10G | bf16 | 128 | 40.7 | 25.4 | 1.60 |
| A10G | bf16 | 512 | 108.9 | 53.9 | 2.02 |
| A10G | bf16 | 1024 | 211.2 | 108.6 | 1.94 |
| A10G | bf16 | 2048 | 413.7 | 220.6 | 1.88 |
| A10G | bf16 | 4000 | 928.4 | 443.3 | 2.09 |
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
| A10G | awq-int4 | 128 | 45.93 | 17.13 | 2.68 | 2,787 | 0.110 | 77.27 |
| A10G | awq-int4 | 256 | 90.14 | 29.49 | 3.06 | 2,840 | 0.108 | 165.59 |
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

## Calibration and recommender (step 3)

### How much of vLLM the roofline explains

First, GPU efficiency from the microbenchmarks replaces datasheet peaks (`src/capacity_planner/calibration.py`):

<!-- BEGIN GENERATED: calibration_efficiency -->
| GPU | Compute efficiency (BF16 GEMM / datasheet) | Memory efficiency (GEMV / datasheet) |
|---|---:|---:|
| L4 | 0.44 | 0.86 |
| A10G | 0.62 | 0.83 |
| A100-40GB | 0.84 | 0.91 |
| A100-80GB | 0.82 | 0.86 |
| H100 | 0.80 | 0.85 |
<!-- END GENERATED: calibration_efficiency -->

With those efficiencies, the remaining gap to vLLM is fitted per configuration as TTFT = c0 + c1 × roofline TTFT and decode step = a × roofline step + b × batch. c1 and a near 1 with small c0 and b mean the efficiency-corrected roofline already explains vLLM. Large values flag where the kernels fall short of the hardware, e.g. AWQ int4 decode on H100 (a ≈ 2). The max error is the worst fitted point.

<!-- BEGIN GENERATED: calibration_fits -->
| GPU | Quant | TTFT c0 (ms) | TTFT c1 | TTFT max error | Decode a | Decode b (µs/seq) | Decode max error | Run |
|---|---|---:|---:|---:|---:|---:|---:|---|
| L4 | awq-int4 | 0.2 | 1.13 | 12% | 0.78 | 268 | 10% | 018-l4-vllm-awq-int4 |
| L4 | bf16 | 12.2 | 0.97 | 6% | 0.97 | 156 | 2% | 006-l4-vllm-bf16 |
| L4 | fp8 | -0.5 | 1.50 | 13% | 1.00 | 101 | 2% | 015-l4-vllm-fp8 |
| A10G | awq-int4 | 4.7 | 1.26 | 2% | 0.83 | 197 | 19% | 019-a10g-vllm-awq-int4 |
| A10G | bf16 | -11.7 | 1.29 | 32% | 1.06 | 28 | 3% | 011-a10g-vllm-bf16 |
| A100-40GB | bf16 | 6.8 | 1.17 | 4% | 1.13 | 25 | 8% | 012-a100-40gb-vllm-bf16 |
| A100-80GB | awq-int4 | 5.3 | 1.62 | 1% | 1.46 | 50 | 13% | 020-a100-80gb-vllm-awq-int4 |
| A100-80GB | bf16 | 4.8 | 1.17 | 2% | 1.13 | 36 | 5% | 013-a100-80gb-vllm-bf16 |
| H100 | awq-int4 | 3.2 | 1.31 | 21% | 2.12 | 1 | 5% | 021-h100-vllm-awq-int4 |
| H100 | bf16 | 2.3 | 1.27 | 4% | 1.15 | 6 | 2% | 014-h100-vllm-bf16 |
| H100 | fp8 | 2.2 | 1.78 | 11% | 1.38 | 4 | 3% | 017-h100-vllm-fp8 |
<!-- END GENERATED: calibration_fits -->

### Recommender

`python -m capacity_planner.recommend --prompt 1024 --output 256 --rps 5 --ttft-ms 500 --tpot-ms 50`

Only (GPU, quant) pairs that vLLM ran are considered. For each, the recommender uses the measured curves: idle TTFT over prompt length, and the decode step over batch size. The calibrated roofline supplies only the shape outside them: prompt lengths outside 128–4,000 tokens, and contexts other than the measured 640 tokens. Per replica with B requests in flight:

- TPOT = decode step(B) + B × prefill / output tokens. Every request's prefill is charged as extra time that doesn't overlap with decode, which is conservative.
- `max_num_seqs` (the slots) = the largest B whose TPOT and idle TTFT (prefill + one decode step) meet the targets.
- **TTFT under load** = TTFT at the busy batch + half a decode step (a new request waits for the running step to finish) + a fixed **serving overhead** + two queueing waits, combined exactly as independent random variables. The serving overhead covers what the offline TTFT measurements skip (HTTP, tokenizing the prompt text, streaming): the median measured − predicted p50 TTFT over the serve runs. The two waits are:
  - **slot wait:** requests arrive at random, so a burst can fill every slot. Each replica is a queue with `max_num_seqs` parallel slots (M/D/c). The service time is TTFT + output tokens × TPOT at the busy batch, and the waiting probability is Erlang C.
  - **prefill wait:** one GPU processes one prompt at a time, so prompts queue behind each other (M/D/1, service time = TTFT).

  Each wait is exponential when it happens, with the M/M/c rate doubled (Allen–Cunneen with deterministic service), because fixed-length requests take nearly constant time.
- Busy slots = the expected number of requests in flight, from Little's law. TPOT is reported at that batch.
- **Safety margin:** the predicted p90 TTFT, overhead included, is multiplied by a per-GPU margin before it is checked against the target. The margin is the largest measured ÷ predicted p90 TTFT among that GPU's serve runs, never below 1. GPUs without a serve run use the largest margin of any GPU. Replicas are added until the margined p90 TTFT meets the target.

How the model got here, one serve run at a time:
1. **Run 023, no queueing model:** with 15 slots at 81% average occupancy, the p90 TTFT was 4.0 s against a predicted 0.23 s. This added the slot queue.
2. **Runs 024–028:** the p90 TTFT came in 2–3× above prediction even where no slot queueing was expected. This added the prefill queue.
3. **Run 026:** the slot queue, costed at full occupancy, predicted 663 ms against 93 ms measured. The service time is now taken at the busy batch.
4. **All six runs:** the step wait was added, and the waits are now summed as distributions rather than as percentiles. Median TTFT was still low by a near-constant 19–70 ms on every run.
5. **Runs 029–031** were new configurations: A100-80GB, H100 FP8, and L4 with shorter prompts. TPOT held up (+1% to +11%), but the p90 TTFT came in up to 32% low. The H100 margin of ×1.00 would not have covered it. The fixed serving overhead explains the near-constant gap from step 4. Fitted on runs 023–028 only, it brings the hold-out runs to within −6% to +10% (table below).

The overhead and margins the recommender uses, fitted on all serve runs of every model (including the step 5 load tests below):

<!-- BEGIN GENERATED: ttft_margins -->
Serving overhead: **42.5 ms** (median measured − predicted p50 TTFT over 15 serve runs).

| GPU | Measured / predicted p90 TTFT per serve run | Margin applied |
|---|---|---:|
| L4 | 023-l4-serve-fp8-chat ×1.13, 027-l4-serve-fp8-chat ×1.28, 031-l4-serve-fp8-short ×0.89 | ×1.28 |
| A10G | 028-a10g-serve-awq-chat ×1.04 | ×1.04 |
| A100-40GB | 024-a100-40gb-serve-bf16-chat ×0.94, 041-a100-40gb-serve-granite8b-bf16-chat ×1.05 | ×1.05 |
| A100-80GB | 029-a100-80gb-serve-bf16-longctx ×1.03, 042-a100-80gb-serve-granite8b-bf16-chat ×1.06 | ×1.06 |
| H100 | 025-h100-serve-bf16-longctx ×0.78, 026-h100-serve-bf16-offline ×0.69, 030-h100-serve-fp8-longout ×0.83, 043-h100-serve-granite8b-fp8-longctx ×0.80, 044-h100-serve-qwen32b-fp8-chat ×1.28, 045-h100x2-serve-qwen32b-fp8-longctx ×1.26, 046-h100-serve-qwen32b-fp8-longctx ×1.24 | ×1.28 |
<!-- END GENERATED: ttft_margins -->

**Hold-out check:** the calibration is refitted without runs 029–031, then used to predict them.

<!-- BEGIN GENERATED: serve_holdout -->
Fitted without these runs: overhead 33.9 ms; margins L4 ×1.31, H100 ×1.00, A100-40GB ×1.00, A10G ×1.06, others ×1.31.

| Run | TTFT ms p50: predicted / measured | TTFT ms p90: predicted / measured (error) | p90 × margin vs measured | TPOT ms: predicted / measured |
|---|---|---|---|---|
| 030-h100-serve-fp8-longout | 62 / 54 | 89 / 81 (+10%) | 89 vs 81: covers | 9.6 / 9.5 |
| 029-a100-80gb-serve-bf16-longctx | 179 / 179 | 279 / 297 (-6%) | 365 vs 297: covers | 14.9 / 14.5 |
| 031-l4-serve-fp8-short | 199 / 208 | 392 / 355 (+10%) | 513 vs 355: covers | 57.6 / 51.8 |
<!-- END GENERATED: serve_holdout -->

The cheapest $/hr wins; ties go to less quantization, then lower TPOT.

<!-- BEGIN GENERATED: recommend_examples -->
**Chat assistant:** 1024-token prompts, 256-token outputs, 5 requests/s, TTFT ≤ 500 ms, TPOT ≤ 50 ms.

| GPU | Quant | Replicas | $/hr | $/1M tokens (in+out) | $/1M output tokens | Busy slots (max_num_seqs) | TTFT ms p90 (× margin) | TPOT ms | KV cache tokens | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A100-40GB | bf16 | 1 | 2.10 | 0.091 | 0.455 | 29 (110) | 242 (×1.05) | 22.0 | 360,528 | 012-a100-40gb-vllm-bf16 |
| A100-80GB | bf16 | 1 | 2.50 | 0.108 | 0.542 | 23 (117) | 230 (×1.06) | 17.6 | 1,045,408 | 013-a100-80gb-vllm-bf16 |
| A100-80GB | awq-int4 | 1 | 2.50 | 0.108 | 0.542 | 16 (89) | 310 (×1.06) | 11.9 | 1,231,408 | 020-a100-80gb-vllm-awq-int4 |

Not feasible: L4 bf16 (misses SLO: TPOT 57.6 ms at batch 1).

**Long-context interactive:** 2048-token prompts, 256-token outputs, 20 requests/s, TTFT ≤ 300 ms, TPOT ≤ 30 ms.

| GPU | Quant | Replicas | $/hr | $/1M tokens (in+out) | $/1M output tokens | Busy slots (max_num_seqs) | TTFT ms p90 (× margin) | TPOT ms | KV cache tokens | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| H100 | bf16 | 2 | 7.90 | 0.048 | 0.429 | 42 (98) | 269 (×1.28) | 16.1 | 1,035,104 | 014-h100-vllm-bf16 |
| H100 | fp8 | 2 | 7.90 | 0.048 | 0.429 | 23 (131) | 167 (×1.28) | 8.7 | 1,149,168 | 017-h100-vllm-fp8 |
| H100 | awq-int4 | 2 | 7.90 | 0.048 | 0.429 | 38 (93) | 264 (×1.28) | 14.6 | 1,203,072 | 021-h100-vllm-awq-int4 |

Not feasible: A10G bf16 (misses SLO: TTFT 446 ms at batch 1, TPOT 34.2 ms at batch 1); L4 bf16 (misses SLO: TTFT 563 ms at batch 1, TPOT 58.8 ms at batch 1); L4 fp8 (misses SLO: TTFT 438 ms at batch 1, TPOT 36.4 ms at batch 1); A10G awq-int4 (misses SLO: TTFT 465 ms at batch 1); L4 awq-int4 (misses SLO: TTFT 595 ms at batch 1).

**Offline batch generation:** 512-token prompts, 512-token outputs, 50 requests/s, TTFT ≤ 5000 ms, TPOT ≤ 200 ms.

| GPU | Quant | Replicas | $/hr | $/1M tokens (in+out) | $/1M output tokens | Busy slots (max_num_seqs) | TTFT ms p90 (× margin) | TPOT ms | KV cache tokens | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| H100 | bf16 | 2 | 7.90 | 0.043 | 0.086 | 222 (256) | 174 (×1.28) | 17.3 | 1,035,104 | 014-h100-vllm-bf16 |
| H100 | fp8 | 2 | 7.90 | 0.043 | 0.086 | 124 (256) | 111 (×1.28) | 9.6 | 1,149,168 | 017-h100-vllm-fp8 |
| A100-40GB | bf16 | 5 | 10.49 | 0.057 | 0.114 | 200 (256) | 274 (×1.05) | 38.9 | 360,528 | 012-a100-40gb-vllm-bf16 |
<!-- END GENERATED: recommend_examples -->

### Validation under real load

Each `serve` run starts `vllm serve` with the recommended `max_num_seqs` and drives one replica with `vllm bench serve`: Poisson arrivals at the per-replica rate, random prompts of fixed length, `--ignore-eos`, and a 2-minute measured window after a warm-up. Predictions come from the current calibration.

<!-- BEGIN GENERATED: serve_validation -->
| GPU | Quant | Load (prompt/output tokens, req/s) | max_num_seqs | TTFT ms p50: predicted / measured | TTFT ms p90: predicted / measured | TPOT ms p50: predicted / measured | Throughput req/s (achieved / offered) | Run |
|---|---|---|---:|---|---|---|---|---|
| L4 | fp8 | 1024/256, 1 | 15 | 306 / 342 | 3540 / 4003 | 47.6 / 48.1 | 0.93 / 1 | [023-l4-serve-fp8-chat](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39226920677) |
| H100 | bf16 | 2048/256, 10 | 98 | 100 / 85 | 201 / 163 | 16.1 / 13.3 | 9.84 / 10 | [025-h100-serve-bf16-longctx](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39228376358) |
| A100-40GB | bf16 | 1024/256, 5 | 110 | 123 / 126 | 222 / 218 | 22.0 / 21.6 | 4.86 / 5 | [024-a100-40gb-serve-bf16-chat](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39228376358) |
| H100 | bf16 | 512/512, 25 | 256 | 71 / 69 | 126 / 93 | 17.3 / 14.7 | 24.08 / 25 | [026-h100-serve-bf16-offline](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39228376358) |
| A10G | awq-int4 | 1024/256, 1.667 | 35 | 280 / 273 | 557 / 590 | 20.1 / 19.8 | 1.62 / 1.667 | [028-a10g-serve-awq-chat](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39228376358) |
| L4 | fp8 | 1024/256, 0.714 | 15 | 279 / 290 | 366 / 480 | 42.6 / 42.3 | 0.67 / 0.714 | [027-l4-serve-fp8-chat](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39228376358) |
| H100 | fp8 | 1024/512, 15 | 256 | 62 / 54 | 89 / 81 | 9.6 / 9.5 | 14.63 / 15 | [030-h100-serve-fp8-longout](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39321416205) |
| A100-80GB | bf16 | 2048/256, 2 | 48 | 179 / 179 | 278 / 297 | 14.9 / 14.5 | 1.95 / 2 | [029-a100-80gb-serve-bf16-longctx](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39321416205) |
| L4 | fp8 | 512/128, 3 | 43 | 199 / 208 | 391 / 355 | 57.6 / 51.8 | 2.88 / 3 | [031-l4-serve-fp8-short](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39321416205) |
<!-- END GENERATED: serve_validation -->

## Predicting models we never benchmarked

Everything above was measured on Qwen2.5-7B. To test whether the planner works for other
models, `capacity_planner.transfer` predicts vLLM's behaviour for a new model from that
model's config alone. The per-(GPU, quant) corrections learned on Qwen-7B (TTFT
`c0 + c1 × roofline`, decode `a × roofline + b × batch`, non-KV memory overhead) are
applied to the new model's roofline. Tensor parallelism splits weights, FLOPs and KV
heads across GPUs and adds two NVLink all-reduces per layer.

The two test models differ from Qwen-7B in ways the roofline has to get right:

- **IBM Granite-3.1-8B**: 40 layers instead of 28, twice the KV bytes per token, and a 49k vocabulary.
- **Qwen2.5-32B**: needs FP8, AWQ or two GPUs.

The predictions were frozen in `predictions/transfer.json` and committed before any of
these GPU runs. The table reads only that file, so later changes to the code can't move
the goalposts.

"Worst error" is predicted / measured − 1 at the prompt length or batch where it is largest.
"Transfer" is the method above. "Roofline" is the efficiency-corrected roofline alone.

<!-- BEGIN GENERATED: transfer_check -->
| Model | GPUs | Quant | KV tokens: predicted / measured | Non-KV overhead GiB: Qwen-7B / measured | TTFT ms at 1024: predicted / measured | TTFT worst error: transfer / roofline | Decode step ms at batch 1: predicted / measured | Decode worst error: transfer / roofline | Run |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Granite-3.1-8B-Instruct | L4 | bf16 | 14,521 / 23,824 | 3.15 / 1.78 | 318 / 350 | -13% / -23% | 61.4 / 65.0 | -8% / -8% | [032-l4-vllm-granite8b-bf16](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39343215885) |
| Granite-3.1-8B-Instruct | L4 | fp8 | 68,685 / 71,760 | 2.35 / 1.88 | 242 / 234 | -11% / -40% | 32.9 / 38.9 | -15% / -19% | [033-l4-vllm-granite8b-fp8](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39343215885) |
| Granite-3.1-8B-Instruct | A100-40GB | bf16 | 119,379 / 128,784 | 3.26 / 1.88 | 81 / 83 | -10% / -49% | 13.2 / 14.2 | -7% / -22% | [034-a100-40gb-vllm-granite8b-bf16](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39343215885) |
| Granite-3.1-8B-Instruct | H100 | bf16 | 355,505 / 360,704 | 3.71 / 2.97 | 29 / 29 | -5% / -44% | 6.6 / 7.0 | -6% / -19% | [035-h100-vllm-granite8b-bf16](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39343215885) |
| Granite-3.1-8B-Instruct | H100 | fp8 | 404,229 / 409,168 | 3.74 / 2.98 | 21 / 21 | -5% / -62% | 4.1 / 4.7 | -13% / -37% | [036-h100-vllm-granite8b-fp8](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39343215885) |
| Qwen2.5-32B-Instruct | H100 | fp8 | 153,653 / 147,328 | 3.74 / 5.13 | 76 / 70 | +18% / -48% | 16.6 / 14.5 | +16% / -21% | [037-h100-vllm-qwen32b-fp8](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39343215885) |
| Qwen2.5-32B-Instruct | A100-80GB | awq-int4 | 216,870 / 213,408 | 2.55 / 3.36 | 417 / 368 | +13% / -40% | 16.4 / 14.6 | -26% / -55% | [038-a100-80gb-vllm-qwen32b-awq-int4](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39343215885) |
| Qwen2.5-32B-Instruct | 2× H100 | bf16 | 319,090 / 319,280 | 3.71 / 3.59 | 58 / 61 | -11% / -37% | 13.2 / 13.7 | -12% / -28% | [039-h100x2-vllm-qwen32b-bf16](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39343215885) |
| Qwen2.5-32B-Instruct | 2× A100-80GB | bf16 | 323,595 / 331,664 | 3.48 / 2.40 | 158 / 157 | -32% / -52% | 21.3 / 23.6 | -19% / -42% | [040-a100-80gbx2-vllm-qwen32b-bf16](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39343215885) |

Across 45 measured TTFT points and 73 decode-step points, the median |error| is 5% (TTFT) and 9% (decode) for the transfer, against 31% and 19% for the roofline alone.
<!-- END GENERATED: transfer_check -->

What the nine runs (032–040) show:

- **The transfer works.** At every setup it predicts TTFT better than the
  efficiency-corrected roofline alone, and decode at least as well. Most of the roofline's
  miss is vLLM behaviour (scheduling, sampling, kernel choice) that depends on the GPU and
  quant much more than on the model, so a fit learned on one model carries over.
- **Weights are easy and KV capacity mostly is.** vLLM loaded within 0.2 GiB of the
  roofline weight size everywhere. The exception is Granite on the L4 in BF16: its non-KV
  overhead was far below Qwen-7B's on the same GPU. Granite's 49k vocabulary, against
  Qwen's 152k, is the likely reason, though two models can't prove it. On a GPU where
  weights fill most of the memory, that difference is a large share of the KV cache, and
  the prediction is conservative there.
- **Tensor parallelism holds up.** On two GPUs, KV capacity and the decode step at
  moderate batches land close. The roofline misses a fixed per-all-reduce latency, which
  shows on short prompts: 2× A100 at 128 tokens is the largest TTFT miss.
- **Quantized kernels are the least certain.** For AWQ on the 32B model, the decode step
  was predicted 11–13% too long at small batches and 26% too short at batch 64. The Qwen-7B AWQ fit already
  had the largest decode errors, and the kernels' compute-bound crossover moves with the
  matrix shapes.

Once a model has been run, the recommender uses the measured curves. Every other
(GPU, quant, 1 or 2 GPUs) is transferred from Qwen-7B and marked as predicted:
`python -m capacity_planner.recommend --model Qwen2.5-32B-Instruct --prompt 1024 --output 256 --rps 5 --ttft-ms 500 --tpot-ms 50`

<!-- BEGIN GENERATED: recommend_other_models -->
**Granite-3.1-8B-Instruct, chat assistant:** 1024-token prompts, 256-token outputs, 5 requests/s, TTFT ≤ 500 ms, TPOT ≤ 50 ms.

| GPU | Quant | Replicas | $/hr | $/1M tokens (in+out) | $/1M output tokens | Busy slots (max_num_seqs) | TTFT ms p90 (× margin) | TPOT ms | KV cache tokens | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A100-40GB | bf16 | 1 | 2.10 | 0.091 | 0.455 | 51 (73) | 364 (×1.05) | 39.3 | 128,784 | 034-a100-40gb-vllm-granite8b-bf16 |
| A100-80GB | bf16 | 1 | 2.50 | 0.108 | 0.542 | 36 (82) | 316 (×1.06) | 27.7 | 359,109 | predicted ² from 013-a100-80gb-vllm-bf16 |
| H100 | bf16 | 1 | 3.95 | 0.171 | 0.857 | 13 (221) | 124 (×1.28) | 9.5 | 360,704 | 035-h100-vllm-granite8b-bf16 |

Not feasible: L4 bf16 (misses SLO: TPOT 66.7 ms at batch 1).

² never run for this model: curves transferred from the named Qwen2.5-7B run (see "Predicting models we never benchmarked").

**Qwen2.5-32B-Instruct, chat assistant:** 1024-token prompts, 256-token outputs, 5 requests/s, TTFT ≤ 500 ms, TPOT ≤ 50 ms.

| GPU | Quant | Replicas | $/hr | $/1M tokens (in+out) | $/1M output tokens | Busy slots (max_num_seqs) | TTFT ms p90 (× margin) | TPOT ms | KV cache tokens | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| H100 | fp8 | 1 | 3.95 | 0.171 | 0.857 | 42 (86) | 345 (×1.28) | 32.1 | 147,328 | 037-h100-vllm-qwen32b-fp8 |
| 2× H100 | bf16 | 1 | 7.90 | 0.343 | 1.714 | 30 (114) | 271 (×1.28) | 22.9 | 319,280 | 039-h100x2-vllm-qwen32b-bf16 |
| 2× H100 | fp8 | 1 | 7.90 | 0.343 | 1.714 | 17 (165) | 177 (×1.28) | 12.6 | 438,235 | predicted ² from 017-h100-vllm-fp8 |

Not feasible: A10G awq-int4 (misses SLO: TTFT 1089 ms at batch 1); L4 awq-int4 (misses SLO: TTFT 1442 ms at batch 1, TPOT 64.4 ms at batch 1).

² never run for this model: curves transferred from the named Qwen2.5-7B run (see "Predicting models we never benchmarked").
<!-- END GENERATED: recommend_other_models -->

### Do those recommendations hold under real load?

The table above checks the raw curves. This checks what a user actually gets: the
recommender's full answer for a model it was never tuned on, under `vllm serve` with
Poisson arrivals. That answer includes the replica count, `max_num_seqs`, and the
queueing model, serving overhead and p90 margins learned from the Qwen-7B serve runs.

Five setups were tested, each the recommender's first or second choice for a README
example workload. Two of them use curves that were only transferred from Qwen-7B, never
measured for that model, and one runs on two GPUs per replica. One replica is
load-tested at its share of the traffic.

The predictions were frozen in `predictions/serve.json` before the runs, and this table
reads only that file. "Targets met" means the measured p90 TTFT and median TPOT are within
the workload's targets.

<!-- BEGIN GENERATED: serve_check -->
| Model | Workload | Replica | Curves | Load per replica (req/s, max_num_seqs) | TTFT ms p50: predicted / measured | TTFT ms p90: predicted / measured (error) | p90 × margin vs target | Margin covers measured p90 | TPOT ms: predicted / measured (target) | Targets met | Run |
|---|---|---|---|---|---:|---:|---|---|---:|---|---|
| Granite-3.1-8B-Instruct | Chat assistant (1 replica) | A100-40GB bf16 | 034-a100-40gb-vllm-granite8b-bf16 | 5, 73 | 157 / 215 | 337 / 364 (-7%) | 337 vs 500 | **no** | 39.3 / 38.9 (50) | yes | [041-a100-40gb-serve-granite8b-bf16-chat](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39352622758) |
| Granite-3.1-8B-Instruct | Chat assistant (1 replica) | A100-80GB bf16 | predicted from 013-a100-80gb-vllm-bf16 | 5, 82 | 139 / 170 | 290 / 316 (-8%) | 309 vs 500 | **no** | 27.7 / 30.0 (50) | yes | [042-a100-80gb-serve-granite8b-bf16-chat](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39352622758) |
| Granite-3.1-8B-Instruct | Long-context interactive (2 replicas) | H100 fp8 | 036-h100-vllm-granite8b-fp8 | 10, 80 | 105 / 113 | 219 / 181 (+21%) | 219 vs 300 | yes | 25.2 / 19.3 (30) | yes | [043-h100-serve-granite8b-fp8-longctx](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39352622758) |
| Qwen2.5-32B-Instruct | Chat assistant (1 replica) | H100 fp8 | 037-h100-vllm-qwen32b-fp8 | 5, 86 | 134 / 188 | 260 / 345 (-25%) | 260 vs 500 | **no** | 32.1 / 33.7 (50) | yes | [044-h100-serve-qwen32b-fp8-chat](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39352622758) |
| Qwen2.5-32B-Instruct | Long-context interactive (5 replicas) | 2× H100 fp8 | predicted from 017-h100-vllm-fp8 | 4, 44 | 138 / 151 | 249 / 325 (-23%) | 249 vs 300 | **no** | 17.0 / 20.3 (30) | **no** | [045-h100x2-serve-qwen32b-fp8-longctx](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39352622758) |

Over 5 runs, the median |error| is 18% for p50 TTFT, 21% for p90 TTFT and 8% for TPOT.
<!-- END GENERATED: serve_check -->

What the five runs (041–045) show:

- **Four of five recommendations met their targets**, including Granite on an A100-80GB.
  vLLM never ran Granite on that GPU, so its curves came from Qwen-7B. TPOT and the A100
  p90 TTFTs land close.
- **The miss is the H100 safety margin, not the curves.** Both Qwen-32B runs on H100
  had a p90 TTFT about 1.3× the prediction. The ×1.00 H100 margin had been learned from
  Qwen-7B runs only, where the model was conservative, and it didn't carry over to a 4×
  larger model. One of the two runs still had headroom; the long-context one
  (2× H100 per replica) went over its 300 ms target.
- **Fix:** the serving overhead and per-GPU margins now learn from serve runs of every
  model, with each run predicted from the curves the recommender would use for it. The
  H100 margin rises accordingly (see the margin table above). For the README's
  example workloads, this changes only the failed case, which now gets one more replica.
  The frozen table above still shows the predictions as they were before the runs.

**Checking the new H100 margin.** The ×1.28 H100 margin was fitted on runs that included
the two that exposed the problem, so it needs a run it wasn't fitted on. That run is
Qwen-32B on a single H100 (FP8) at the long-context workload. It is a setup and workload
not load-tested before, where the margined p90 lands just under the 300 ms target. Its
prediction was frozen in `predictions/serve-h100-margin.json` before the run.

<!-- BEGIN GENERATED: serve_margin_check -->
| Model | Workload | Replica | Curves | Load per replica (req/s, max_num_seqs) | TTFT ms p50: predicted / measured | TTFT ms p90: predicted / measured (error) | p90 × margin vs target | Margin covers measured p90 | TPOT ms: predicted / measured (target) | Targets met | Run |
|---|---|---|---|---|---:|---:|---|---|---:|---|---|
| Qwen2.5-32B-Instruct | Long-context interactive (21 replicas) | H100 fp8 | 037-h100-vllm-qwen32b-fp8 | 0.952, 20 | 199 / 206 | 229 / 285 (-20%) | 295 vs 300 | yes | 18.5 / 18.4 (30) | yes | [046-h100-serve-qwen32b-fp8-longctx](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39377167766) |

Over 1 runs, the median |error| is 4% for p50 TTFT, 20% for p90 TTFT and 0% for TPOT.
<!-- END GENERATED: serve_margin_check -->

## Development

```bash
pip install -e ".[dev]"
pytest
python -m capacity_planner.report                  # print tables
python -m capacity_planner.report --update-readme  # regenerate README tables
```

## Plan

1. **GPU specs + roofline formulas** (done).
2. **Benchmarks on Modal via GitHub Actions** (vLLM first, then SGLang; Qwen2.5-7B-Instruct; budget < $40) to measure memory overhead and achieved FLOPs/bandwidth (done; see the tables above).
3. **Calibrate the roofline and build the recommender**: model + SLOs + traffic → GPU, engine, quantization, batch size, KV-cache size, $/1M tokens; validated with `vllm serve` under load (done).
4. **Predict unmeasured models and multi-GPU setups**, then check the predictions against new runs (done).
5. **Load-test recommendations for those models** under `vllm serve` (done).
