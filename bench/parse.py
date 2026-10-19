"""Pull the memory breakdown out of vLLM's startup log (format of vLLM 0.30.0)."""

from __future__ import annotations

import re

_PATTERNS = {
    # vllm/v1/worker/gpu_model_runner.py: "Model loading took %s GiB memory and %.6f seconds"
    "model_loading_gib": r"Model loading took ([\d.]+) GiB memory",
    # vllm/v1/worker/gpu_worker.py: "Available KV cache memory: %s GiB"
    "available_kv_cache_gib": r"Available KV cache memory: ([\d.]+) GiB",
    # vllm/v1/core/kv_cache_utils.py: "GPU KV cache size: %s tokens, ..."
    "kv_cache_size_tokens": r"GPU KV cache size: ([\d,]+) tokens",
}


def parse_vllm_log(log: str) -> dict:
    out: dict = {}
    for key, pat in _PATTERNS.items():
        m = re.search(pat, log)
        if m:
            val = m.group(1).replace(",", "")
            out[key] = int(val) if key.endswith("tokens") else float(val)
    return out
