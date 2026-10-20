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
    # Keep non-Python files: the model registry and configs live in capacity_planner/data.
    .add_local_python_source("bench", "capacity_planner", ignore=["**/__pycache__/**"])
)


# SGLang's official image for v0.5.21 (Docker Hub lmsysorg/sglang:v0.5.21, pushed
# 2026-10-01), pinned by its linux/amd64 digest.
SGLANG_IMAGE = (
    "lmsysorg/sglang@sha256:2dbe4c7f53230b09aaa8fa41891c13c51d9d2e253f486a70822aaf08783fff75"
)

# The SGLang image already has files under /root/.cache/huggingface, and Modal can't mount
# a volume on a non-empty path, so the same volume goes elsewhere here (the cache layout is
# relative to HF_HOME, so cached weights are found the same way).
SGLANG_HF_CACHE = "/hf-cache"

sglang_image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .entrypoint([])
    .run_commands('command -v python || ln -s "$(command -v python3)" /usr/local/bin/python')
    .env({"HF_HOME": SGLANG_HF_CACHE, "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("bench", "capacity_planner", ignore=["**/__pycache__/**"])
)


def run_child(module: str, args: list[str]) -> tuple[int, str, dict | None]:
    """Run `python -m module ... --out FILE` offline, streaming and keeping its log."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out.json")
        cmd = [sys.executable, "-m", module, *args, "--out", out]
        env = {**os.environ, "HF_HUB_OFFLINE": "1"}
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd="/root")
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
    def run(
        self, suite: str, quant: str, revision: str | None = None, serve: dict | None = None,
        model: str | None = None, tp: int = 1,
    ) -> dict:
        """Returns plain JSON types only: the caller (a GitHub runner) has no torch/vllm
        installed, so anything else (e.g. torch's TorchVersion str subclass) can't be
        unpickled there. Errors come back as text for the same reason."""
        import traceback

        from bench import micro
        from capacity_planner.models import DEFAULT_MODEL

        self._model_args = ["--model", model or DEFAULT_MODEL, "--tp", str(tp)]
        try:
            info = micro.gpu_info()  # nvidia-smi only, no CUDA context
            if suite == "micro":
                out = {"gpu_info": info, "micro": micro.run()}
            elif suite == "vllm":
                out = {"gpu_info": info, "vllm": self._vllm(quant, revision)}
            elif suite == "serve":
                out = {"gpu_info": info, "serve": self._serve(quant, revision, serve)}
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

    def _serve(self, quant: str, revision: str | None, serve: dict) -> dict:
        from bench.parse import parse_vllm_log

        extra = [
            "--max-num-seqs", str(serve["max_num_seqs"]), "--prompt", str(serve["prompt"]),
            "--output", str(serve["output"]), "--rps", str(serve["rps"]),
        ]
        rc, log, result = self._run_child("bench.serve_bench", quant, revision, extra)
        if rc != 0:
            tail = "".join(log.splitlines(keepends=True)[-40:])
            raise RuntimeError(f"serve_bench exited {rc}; last log lines:\n{tail}")
        result["log_summary"] = parse_vllm_log(log)
        return result

    def _vllm_once(self, quant: str, revision: str | None, compile_backend: str | None):
        extra = ["--compile-backend", compile_backend] if compile_backend else []
        return self._run_child("bench.vllm_bench", quant, revision, extra)

    def _run_child(self, module: str, quant: str, revision: str | None, extra: list[str]):
        args = ["--quant", quant, *self._model_args, *extra] + (["--revision", revision] if revision else [])
        return run_child(module, args)


@app.cls(image=sglang_image, gpu="L4", volumes={SGLANG_HF_CACHE: hf_volume}, timeout=30 * 60)
class SGLangBench:
    @modal.method()
    def run(
        self, suite: str, quant: str, revision: str | None = None, serve: dict | None = None,
        model: str | None = None, tp: int = 1,
    ) -> dict:
        """The sglang suites on SGLang's image: offline (bench.sglang_bench) and serve
        (bench.serve_bench --engine sglang). Plain JSON out, as Bench.run."""
        import traceback

        from bench import micro
        from capacity_planner.models import DEFAULT_MODEL

        try:
            info = micro.gpu_info()
            args = ["--quant", quant, "--model", model or DEFAULT_MODEL, "--tp", str(tp)]
            args += ["--revision", revision] if revision else []
            if suite == "sglang":
                module, key = "bench.sglang_bench", "sglang"
            elif suite == "sglang-serve":
                module, key = "bench.serve_bench", "serve"
                args += [
                    "--engine", "sglang", "--max-num-seqs", str(serve["max_num_seqs"]), "--prompt", str(serve["prompt"]),
                    "--output", str(serve["output"]), "--rps", str(serve["rps"]),
                ]
            else:
                raise ValueError(f"SGLangBench runs the sglang suites, not {suite!r}")
            rc, log, result = run_child(module, args)
            if rc != 0:
                tail = "".join(log.splitlines(keepends=True)[-40:])
                raise RuntimeError(f"{module} exited {rc}; last log lines:\n{tail}")
            return json.loads(json.dumps({"gpu_info": info, key: result}, default=str))
        except Exception as e:
            return {"remote_error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
