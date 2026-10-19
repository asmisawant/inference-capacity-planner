"""Benchmark requests, the spend ledger and the budget guard.

A GPU run only happens if it is listed in bench/requests.json. Each run has an
`id`; once an id appears in results/ledger.jsonl it is never launched again,
whether it succeeded or failed, so re-pushing the request file can't
double-spend. The one exception: a run that never got past Modal sign-in
(reached_modal=False) used no Modal resources and may be retried.

    python -m bench.runs plan [--github-output FILE]   # pending runs + budget check
    python -m bench.runs collect OUT_DIR               # file results, append ledger
    python -m bench.runs spent                         # print spend so far
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from capacity_planner.gpus import GPUS

ROOT = Path(__file__).resolve().parents[1]
REQUESTS = ROOT / "bench" / "requests.json"
RESULTS = ROOT / "results"
LEDGER = RESULTS / "ledger.jsonl"

SUITES = ("micro", "vllm")
QUANTS = ("bf16", "fp8", "awq-int4")
# Modal container timeout per suite. Also the worst-case billable time we budget for.
TIMEOUT_MIN = {"micro": 15, "vllm": 30}
# Modal also bills CPU and RAM alongside the GPU; budget a margin for it.
NON_GPU_MARGIN = 1.10


@dataclass(frozen=True)
class Run:
    id: str
    gpu: str
    suite: str
    quant: str = "bf16"

    def __post_init__(self):
        if self.gpu not in GPUS:
            raise ValueError(f"run {self.id}: unknown gpu {self.gpu!r}")
        if self.suite not in SUITES:
            raise ValueError(f"run {self.id}: unknown suite {self.suite!r}")
        if self.quant not in QUANTS:
            raise ValueError(f"run {self.id}: unknown quant {self.quant!r}")
        if not self.id or "/" in self.id or " " in self.id:
            raise ValueError(f"bad run id {self.id!r}")

    @property
    def timeout_min(self) -> int:
        return TIMEOUT_MIN[self.suite]

    @property
    def worst_case_usd(self) -> float:
        return cost_usd(self.gpu, self.timeout_min * 60)


def cost_usd(gpu: str, seconds: float) -> float:
    return GPUS[gpu].price_per_hour * seconds / 3600 * NON_GPU_MARGIN


def load_requests(path: Path = REQUESTS) -> tuple[float, list[Run]]:
    data = json.loads(path.read_text())
    runs = [Run(**r) for r in data["runs"]]
    ids = [r.id for r in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate run ids in requests")
    return float(data["budget_usd"]), runs


def load_ledger(path: Path = LEDGER) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def spent_usd(ledger: list[dict]) -> float:
    return sum(e["cost_usd_upper_bound"] for e in ledger)


def plan(budget: float, runs: list[Run], ledger: list[dict]) -> list[Run]:
    """Runs not yet in the ledger. Raises if their worst case would break the budget."""
    done = {e["id"] for e in ledger if e.get("reached_modal", True)}
    pending = [r for r in runs if r.id not in done]
    spent = spent_usd(ledger)
    worst = sum(r.worst_case_usd for r in pending)
    if spent + worst > budget:
        raise SystemExit(
            f"budget guard: spent ${spent:.2f} + worst case ${worst:.2f} for "
            f"{len(pending)} pending runs > budget ${budget:.2f}"
        )
    return pending


def collect(out_dir: Path, results_dir: Path = RESULTS) -> list[dict]:
    """Move launcher outputs into results/<id>.json and append one ledger line each."""
    entries = []
    results_dir.mkdir(parents=True, exist_ok=True)
    # Same rule as plan(): an id is final once it reached Modal; sign-in failures may be retried.
    known = {e["id"] for e in load_ledger(results_dir / "ledger.jsonl") if e.get("reached_modal", True)}
    for f in sorted(out_dir.glob("*.json")):
        rec = json.loads(f.read_text())
        if rec["id"] in known:
            continue
        dest = results_dir / f"{rec['id']}.json"
        dest.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
        entries.append(
            {
                "id": rec["id"],
                "gpu": rec["gpu"],
                "suite": rec["suite"],
                "quant": rec["quant"],
                "status": rec["status"],
                "reached_modal": rec.get("reached_modal", True),
                "wall_s": rec["wall_s"],
                "cost_usd_upper_bound": rec["cost_usd_upper_bound"],
                "github_run_url": rec.get("github_run_url"),
                "finished_at": rec["finished_at"],
            }
        )
    with open(results_dir / "ledger.jsonl", "a") as fh:
        for e in entries:
            fh.write(json.dumps(e, sort_keys=True) + "\n")
    return entries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--github-output", type=Path)
    c = sub.add_parser("collect")
    c.add_argument("out_dir", type=Path)
    sub.add_parser("spent")
    args = ap.parse_args(argv)

    ledger = load_ledger()
    if args.cmd == "spent":
        print(f"${spent_usd(ledger):.2f} across {len(ledger)} runs (upper bound)")
    elif args.cmd == "plan":
        budget, runs = load_requests()
        pending = plan(budget, runs, ledger)
        matrix = [
            {"id": r.id, "gpu": r.gpu, "suite": r.suite, "quant": r.quant, "timeout_min": r.timeout_min}
            for r in pending
        ]
        worst = sum(r.worst_case_usd for r in pending)
        print(
            f"spent ${spent_usd(ledger):.2f}; {len(pending)} pending runs, "
            f"worst case ${worst:.2f}; budget ${budget:.2f}"
        )
        for m in matrix:
            print(f"  {m['id']}: {m['gpu']} {m['suite']} {m['quant']}")
        if args.github_output:
            with open(args.github_output, "a") as fh:
                fh.write(f"matrix={json.dumps(matrix)}\n")
                fh.write(f"count={len(matrix)}\n")
    elif args.cmd == "collect":
        for e in collect(args.out_dir):
            print(f"{e['id']}: {e['status']} ${e['cost_usd_upper_bound']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
