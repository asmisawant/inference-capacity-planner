"""Launch one benchmark run on Modal and write a result record.

    python -m bench.launch --id ID --gpu L4 --suite micro --quant bf16 --out out/ID.json

Always writes the record, even on failure, because a failed run still costs
money and must reach the ledger. A run that failed before reaching Modal
(e.g. a bad token) is marked reached_modal=False and may be retried.
Cost is an upper bound: runner wall-clock (which includes image build and
container start) x the GPU's hourly price.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

from bench.runs import Run, cost_usd
from capacity_planner.gpus import GPUS

START_ATTEMPTS = 5  # app.run() retries on Modal rate limits before anything starts


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--quant", default="bf16")
    ap.add_argument("--timeout-min", type=int, required=True)
    ap.add_argument("--serve", default="null", help="JSON serve parameters (serve suite)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    run = Run(id=args.id, gpu=args.gpu, suite=args.suite, quant=args.quant, serve=json.loads(args.serve))

    import modal

    from bench.modal_app import Bench, app, download
    from bench.vllm_bench import MODELS

    gh = os.environ
    record = {
        "id": run.id,
        "gpu": run.gpu,
        "suite": run.suite,
        "quant": run.quant,
        "serve": run.serve,
        "modal_gpu": GPUS[run.gpu].modal_gpu,
        "price_per_hour": GPUS[run.gpu].price_per_hour,
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "github_run_url": (
            f"{gh['GITHUB_SERVER_URL']}/{gh['GITHUB_REPOSITORY']}/actions/runs/{gh['GITHUB_RUN_ID']}"
            if "GITHUB_RUN_ID" in gh
            else None
        ),
        "started_at": _now(),
    }
    # A pasted secret can carry a trailing newline, which Modal rejects.
    for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        if key in os.environ:
            os.environ[key] = os.environ[key].strip()

    record["reached_modal"] = False  # stays False if sign-in fails: nothing was billed
    t0 = time.monotonic()
    for attempt in range(1, START_ATTEMPTS + 1):
        try:
            with modal.enable_output(), app.run():
                record["reached_modal"] = True
                revision = None
                if run.suite in ("vllm", "serve"):
                    revision = download.remote(MODELS[run.quant][0])
                    record["model_revision"] = revision
                gpu_t0 = time.monotonic()
                bench = Bench.with_options(gpu=GPUS[run.gpu].modal_gpu, timeout=run.timeout_min * 60)()
                record["result"] = bench.run.remote(run.suite, run.quant, revision, run.serve)
                record["gpu_call_s"] = time.monotonic() - gpu_t0
            if "remote_error" in record["result"]:
                print(record["result"]["traceback"])
                raise RuntimeError(f"remote: {record['result']['remote_error']}")
            record["status"] = "ok"
            break
        except modal.exception.ResourceExhaustedError as e:
            # e.g. "App create rate limit exceeded" when several jobs start at once.
            # Retry only if the app never started, so nothing ran twice.
            if record["reached_modal"] or attempt == START_ATTEMPTS:
                record["status"], record["error"] = "error", f"{type(e).__name__}: {e}"
                traceback.print_exc()
                break
            delay = 30 * attempt + random.uniform(0, 15)
            print(f"{run.id}: {e}; retrying in {delay:.0f}s (attempt {attempt}/{START_ATTEMPTS})")
            time.sleep(delay)
        except Exception as e:  # keep going: the record must be written
            record["status"] = "error"
            record["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            break
    record["start_attempts"] = attempt
    record["wall_s"] = time.monotonic() - t0
    record["cost_usd_upper_bound"] = cost_usd(run.gpu, record["wall_s"])
    record["finished_at"] = _now()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))
    print(f"{run.id}: {record['status']} in {record['wall_s']:.0f}s, <= ${record['cost_usd_upper_bound']:.3f}")
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
