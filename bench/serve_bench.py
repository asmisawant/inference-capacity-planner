"""Online serving benchmark: a real server under Poisson load from the engine's own load tool.

vLLM: `vllm serve` + `vllm bench serve`. SGLang (--engine sglang, on SGLang's image):
`python -m sglang.launch_server` + `python -m sglang.bench_serving`, which writes the same
metric names (median/p90 TTFT, median TPOT, request throughput).

Runs inside the Modal container (python -m bench.serve_bench). Validates the
recommender: one replica, the recommended max_num_seqs, the per-replica request
rate, random prompts of fixed length with --ignore-eos so outputs are exactly the
requested length.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

from bench.vllm_bench import MAX_MODEL_LEN, QUANTS, hf_model

PORT = 8000
# Scalar keys kept from vLLM's result file; per-request lists (texts, timings) are dropped.
DROP_KEYS = {
    "input_lens", "output_lens", "ttfts", "itls", "latencies", "start_times", "queue_times",
    "generated_texts", "errors",
}


def wait_healthy(proc: subprocess.Popen, timeout_s: float = 1200) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode} before becoming healthy")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(5)
    raise TimeoutError("server did not become healthy")


def bench(model: str, prompt: int, output: int, num_prompts: int, rps: float, out_dir: str, name: str) -> dict:
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "vllm", "--model", model, "--port", str(PORT),
        "--dataset-name", "random", "--random-input-len", str(prompt), "--random-output-len", str(output),
        "--random-range-ratio", "0", "--num-prompts", str(num_prompts), "--request-rate", str(rps),
        "--ignore-eos", "--seed", "0", "--disable-tqdm",
        "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
        "--save-result", "--result-dir", out_dir, "--result-filename", name,
    ]
    subprocess.run(cmd, check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    with open(os.path.join(out_dir, name)) as fh:
        res = json.load(fh)
    return {k: v for k, v in res.items() if k not in DROP_KEYS}


def bench_sglang(model: str, prompt: int, output: int, num_prompts: int, rps: float, out_dir: str, name: str) -> dict:
    """SGLang's load tool. random-ids = random token ids, like vLLM's random dataset (SGLang's
    'random' samples ShareGPT text). Its range ratio is the minimum fraction of the length,
    so 1.0 gives fixed lengths. EOS is ignored by default."""
    path = os.path.join(out_dir, name)
    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", "sglang", "--model", model, "--port", str(PORT),
        "--dataset-name", "random-ids", "--random-input-len", str(prompt), "--random-output-len", str(output),
        "--random-range-ratio", "1.0", "--num-prompts", str(num_prompts), "--request-rate", str(rps),
        "--seed", "0", "--disable-tqdm", "--output-file", path,
    ]
    subprocess.run(cmd, check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    with open(path) as fh:
        res = json.loads(fh.read().strip().splitlines()[-1])
    return {k: v for k, v in res.items() if k not in DROP_KEYS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("vllm", "sglang"), default="vllm")
    ap.add_argument("--quant", choices=QUANTS, required=True)
    ap.add_argument("--model", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--revision")
    ap.add_argument("--max-num-seqs", type=int, required=True)
    ap.add_argument("--prompt", type=int, required=True)
    ap.add_argument("--output", type=int, required=True)
    ap.add_argument("--rps", type=float, required=True)
    ap.add_argument("--duration-s", type=float, default=120, help="target length of the measured load")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, quantization = hf_model(args.model, args.quant)
    if args.engine == "vllm":
        serve = [
            "vllm", "serve", model, "--port", str(PORT), "--max-model-len", str(MAX_MODEL_LEN),
            "--max-num-seqs", str(args.max_num_seqs), "--gpu-memory-utilization", "0.92",
            "--no-enable-prefix-caching", "--seed", "0", "--tensor-parallel-size", str(args.tp),
        ]
        load = bench
    else:  # same request settings as bench/sglang_bench.py; memory fraction is SGLang's default
        serve = [
            sys.executable, "-m", "sglang.launch_server", "--model-path", model, "--port", str(PORT),
            "--context-length", str(MAX_MODEL_LEN), "--max-running-requests", str(args.max_num_seqs),
            "--disable-radix-cache", "--random-seed", "0", "--tp-size", str(args.tp),
        ]
        load = bench_sglang
    if args.revision:
        serve += ["--revision", args.revision]
    if quantization:
        serve += ["--quantization", quantization]
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    proc = subprocess.Popen(serve, stdout=sys.stdout, stderr=subprocess.STDOUT, env=env)
    try:
        wait_healthy(proc)
        num_prompts = max(100, min(3000, round(args.rps * args.duration_s)))
        with tempfile.TemporaryDirectory() as tmp:
            # Warm-up at the same rate, not recorded: CUDA graphs, allocator, first requests.
            load(model, args.prompt, args.output, min(50, num_prompts), args.rps, tmp, "warmup.json")
            result = load(model, args.prompt, args.output, num_prompts, args.rps, tmp, "result.json")
        out = {
            "server": {"engine": args.engine, "cmd": serve[1:], "max_num_seqs": args.max_num_seqs},
            "load": {
                "prompt_len": args.prompt, "output_len": args.output, "request_rate": args.rps,
                "num_prompts": num_prompts, "arrivals": "poisson",
            },
            "result": result,
        }
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
