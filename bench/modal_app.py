"""Modal app: one container per benchmark run, on the GPU the run asks for.

Launched from GitHub Actions through bench/launch.py (Modal's CLI can't reach
Modal from the dev sandbox). Model weights live in a Modal Volume so they are
downloaded once, on a CPU container, not on GPU time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import modal

HF_CACHE = "/root/.cache/huggingface"

app = modal.App("inference-capacity-planner-bench")
hf_volume = modal.Volume.from_name("icp-hf-cache", create_if_missing=True)

# vLLM's official image for the pinned release, pinned by its linux/amd64 digest
# (Docker Hub vllm/vllm-openai:v0.30.0, pushed 2026-09-22). It ships the CUDA
# toolkit that FlashInfer's JIT kernels need; a plain `pip install vllm` on a
# slim base fails at engine warmup with "Could not find nvcc".
VLLM_IMAGE = (
    "vllm/vllm-openai@sha256:5f5e535216848d0c52159c8c13a0af04be5f6fe1a84e79914300610796f76d40"
)

image = (
    modal.Image.from_registry(VLLM_IMAGE)
    .entrypoint([])  # the image's entrypoint starts `vllm serve`
    .run_commands('command -v python || ln -s "$(command -v python3)" /usr/local/bin/python')
    .env({"HF_HOME": HF_CACHE, "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("bench", "capacity_planner")
)


@app.function(image=image, volumes={HF_CACHE: hf_volume}, cpu=4, timeout=30 * 60)
def download(repo_id: str) -> str:
    """Fetch weights into the volume (no-op if cached). Returns the commit sha."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    hf_volume.commit()
    return os.path.basename(path)  # snapshot dir name is the commit sha


@app.cls(image=image, gpu="L4", volumes={HF_CACHE: hf_volume}, timeout=30 * 60)
class Bench:
    @modal.method()
    def run(self, suite: str, quant: str, revision: str | None = None) -> dict:
        """Returns plain JSON types only: the caller (a GitHub runner) has no torch/vllm
        installed, so anything else (e.g. torch's TorchVersion str subclass) can't be
        unpickled there. Errors come back as text for the same reason."""
        import traceback

        from bench import micro

        try:
            info = micro.gpu_info()  # nvidia-smi only, no CUDA context
            if suite == "micro":
                out = {"gpu_info": info, "micro": micro.run()}
            elif suite == "vllm":
                out = {"gpu_info": info, "vllm": self._vllm(quant, revision)}
            else:
                raise ValueError(suite)
            return json.loads(json.dumps(out, default=str))
        except Exception as e:
            return {"remote_error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}

    def _vllm(self, quant: str, revision: str | None) -> dict:
        """Run bench.vllm_bench in a child process, streaming and keeping its logs.

        If engine startup dies in torch.compile's Inductor backend (vLLM 0.30.0 does this
        for FP8 weight-only on A100: "auto_functionalized was not removed"), retry once with
        vLLM's eager compile backend. That keeps CUDA graphs and skips only Inductor fusion,
        and the result records that the fallback was used.
        """
        from bench.parse import parse_vllm_log

        rc, log, result = self._vllm_once(quant, revision, compile_backend=None)
        fallback = None
        if rc != 0 and "InductorError" in log:
            cause = next((l.strip() for l in log.splitlines() if "InductorError" in l), "")
            fallback = {"reason": cause[-300:], "compilation_config": {"backend": "eager"}}
            print(f"Inductor failed; retrying with compile backend 'eager': {cause}")
            rc, log, result = self._vllm_once(quant, revision, compile_backend="eager")
        if rc != 0:
            tail = "".join(log.splitlines(keepends=True)[-40:])
            raise RuntimeError(f"vllm_bench exited {rc}; last log lines:\n{tail}")
        result["compile_fallback"] = fallback
        result["log_summary"] = parse_vllm_log(log)
        return result

    def _vllm_once(self, quant: str, revision: str | None, compile_backend: str | None):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.json")
            cmd = [sys.executable, "-m", "bench.vllm_bench", "--quant", quant, "--out", out]
            if revision:
                cmd += ["--revision", revision]
            if compile_backend:
                cmd += ["--compile-backend", compile_backend]
            env = {**os.environ, "HF_HUB_OFFLINE": "1"}
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd="/root"
            )
            log_lines = []
            for line in proc.stdout:
                print(line, end="")
                log_lines.append(line)
            rc = proc.wait()
            result = None
            if rc == 0:
                with open(out) as fh:
                    result = json.load(fh)
        return rc, "".join(log_lines), result
