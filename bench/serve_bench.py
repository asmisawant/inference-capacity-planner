"""Online serving benchmark: a real `vllm serve` under Poisson load from `vllm bench serve`.

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

from bench.vllm_bench import MAX_MODEL_LEN, MODELS

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
            raise RuntimeError(f"vllm serve exited with {proc.returncode} before becoming healthy")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(5)
    raise TimeoutError("vllm serve did not become healthy")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=sorted(MODELS), required=True)
    ap.add_argument("--revision")
    ap.add_argument("--max-num-seqs", type=int, required=True)
    ap.add_argument("--prompt", type=int, required=True)
    ap.add_argument("--output", type=int, required=True)
    ap.add_argument("--rps", type=float, required=True)
    ap.add_argument("--duration-s", type=float, default=120, help="target length of the measured load")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, quantization = MODELS[args.quant]
    serve = [
        "vllm", "serve", model, "--port", str(PORT), "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(args.max_num_seqs), "--gpu-memory-utilization", "0.92",
        "--no-enable-prefix-caching", "--seed", "0",
    ]
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
            bench(model, args.prompt, args.output, min(50, num_prompts), args.rps, tmp, "warmup.json")
            result = bench(model, args.prompt, args.output, num_prompts, args.rps, tmp, "result.json")
        out = {
            "server": {"cmd": serve[1:], "max_num_seqs": args.max_num_seqs},
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
