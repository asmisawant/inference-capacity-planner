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
12 runs (4 succeeded), at most **$0.32** of GPU time (runner wall-clock × Modal GPU price × 1.1 for CPU/RAM).
<!-- END GENERATED: bench_spend -->

### Measured GPU throughput vs datasheet

<!-- BEGIN GENERATED: measured_gpu -->
| GPU | Device (nvidia-smi) | Memory GiB (measured / spec) | GEMV read GB/s (% of datasheet) | Copy GB/s (%) | BF16 GEMM TFLOPS (%) | FP8 GEMM TFLOPS (%) | INT8 GEMM TOPS (%) | Run |
|---|---|---:|---:|---:|---:|---:|---:|---|
| L4 | NVIDIA L4 | 22.1 / 24 | 259 (86%) | 230 (77%) | 53 (44%) | 92 (38%) | 128 (53%) | [005-l4-micro](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39222305725) |
<!-- END GENERATED: measured_gpu -->

### vLLM memory: weights, KV cache and overhead

The roofline KV capacity assumes zero overhead. "Non-KV overhead" = GPU memory × utilization − weights − KV cache, using vLLM's own log figures; it covers activations, CUDA graphs and the CUDA context.

<!-- BEGIN GENERATED: measured_vllm_memory -->
| GPU | Quant | Weights GiB (vLLM / roofline) | KV cache GiB (vLLM) | Non-KV overhead GiB (derived) | KV tokens (vLLM / roofline) | Run | Notes |
|---|---|---:|---:|---:|---:|---|---|
| L4 | bf16 | 14.29 / 14.19 | 3.25 | 3.15 | 60,816 / 147,826 | [006-l4-vllm-bf16](https://github.com/asmisawant/inference-capacity-planner/actions/runs/39222305725) |  |
<!-- END GENERATED: measured_vllm_memory -->

### TTFT: vLLM vs roofline

<!-- BEGIN GENERATED: measured_ttft -->
| GPU | Quant | Prompt tokens | TTFT ms (vLLM, median) | TTFT ms (roofline) | vLLM / roofline |
|---|---|---:|---:|---:|---:|
| L4 | bf16 | 128 | 71.3 | 50.8 | 1.40 |
| L4 | bf16 | 512 | 143.7 | 55.7 | 2.58 |
| L4 | bf16 | 1024 | 257.2 | 112.2 | 2.29 |
| L4 | bf16 | 2048 | 505.7 | 227.9 | 2.22 |
| L4 | bf16 | 4000 | 1034.3 | 458.0 | 2.26 |
<!-- END GENERATED: measured_ttft -->

### Decode: vLLM vs roofline

Each batch of 512-token prompts runs twice, with 64 and with 192 output tokens. Both runs spend the same time on prefill, so the extra engine time divided by the extra 128 tokens is the clean **decode step** at that batch size; the roofline is evaluated at the mean context over those steps (640 tokens). Decode tok/s and $/1M (output tokens only, Modal price from the spec table) follow from it. **TPOT under load** is the median per-request TPOT from vLLM's timestamps, which also includes steps shared with other requests' prefill. Run 004 predates the two-length measurement and only has TPOT under load.

<!-- BEGIN GENERATED: measured_decode -->
| GPU | Quant | Batch | Decode step ms (vLLM) | Decode step ms (roofline) | vLLM / roofline | Decode tok/s (vLLM) | $/1M output tokens | TPOT ms under load (median) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L4 | bf16 | 1 | 56.51 | 50.89 | 1.11 | 18 | 12.545 | 56.47 |
| L4 | bf16 | 2 | 57.66 | 51.02 | 1.13 | 35 | 6.400 | 57.93 |
| L4 | bf16 | 4 | 58.07 | 51.26 | 1.13 | 69 | 3.223 | 58.05 |
| L4 | bf16 | 8 | 59.03 | 51.75 | 1.14 | 136 | 1.638 | 58.97 |
| L4 | bf16 | 16 | 60.86 | 52.73 | 1.15 | 263 | 0.844 | 60.65 |
| L4 | bf16 | 32 | 67.37 | 54.69 | 1.23 | 475 | 0.467 | 76.59 |
| L4 | bf16 | 64 | 75.03 | 58.61 | 1.28 | 853 | 0.260 | 89.36 |
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
