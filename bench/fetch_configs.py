"""Download each model's config.json from Hugging Face into src/capacity_planner/data/.

Runs on a GitHub Actions runner (the dev sandbox can't reach huggingface.co). Records
the commit sha it read, so the architecture the planner uses is pinned and auditable.

    python -m bench.fetch_configs
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "capacity_planner" / "data"
MODELS = DATA / "models.json"


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "inference-capacity-planner"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def weights_bytes(repo: str, sha: str) -> int:
    """Total tensor bytes from the safetensors index (sharded checkpoints)."""
    idx = get_json(f"https://huggingface.co/{repo}/resolve/{sha}/model.safetensors.index.json")
    return int(idx["metadata"]["total_size"])


def main() -> int:
    for m in json.loads(MODELS.read_text())["models"]:
        dest = DATA / m["file"]
        if dest.exists():
            cfg = json.loads(dest.read_text())
            src = cfg["_source"]
            if "safetensors_total_bytes" not in src:
                src["safetensors_total_bytes"] = weights_bytes(src["repo"], src["revision"])
                dest.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
                print(f"{m['name']}: added safetensors size")
            continue
        info = get_json(f"https://huggingface.co/api/models/{m['repo']}")
        if info.get("gated"):
            raise SystemExit(f"{m['repo']} is gated; it needs an HF token")
        sha = info["sha"]
        cfg = get_json(f"https://huggingface.co/{m['repo']}/resolve/{sha}/config.json")
        src = {"repo": m["repo"], "revision": sha, "file": "config.json",
               "safetensors_total_bytes": weights_bytes(m["repo"], sha)}
        dest.write_text(json.dumps({"_source": src, **cfg}, indent=2, sort_keys=True) + "\n")
        print(f"{m['name']}: wrote {dest.name} at {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
