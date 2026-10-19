"""Model architecture, parameter counts and per-token costs, derived from a HF config.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

DTYPE_BYTES = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool = False
    qkv_bias: bool = True  # Qwen2 has biases on q/k/v projections only

    @classmethod
    def from_hf_config(cls, name: str, cfg: dict) -> "ModelSpec":
        return cls(
            name=name,
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_layers=cfg["num_hidden_layers"],
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            vocab_size=cfg["vocab_size"],
            max_position_embeddings=cfg["max_position_embeddings"],
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    # --- parameter counts -------------------------------------------------
    @property
    def linear_params_per_layer(self) -> int:
        """Weights of the matmuls in one decoder layer (q, k, v, o, gate, up, down)."""
        h, i = self.hidden_size, self.intermediate_size
        attn = h * self.q_dim + 2 * h * self.kv_dim + self.q_dim * h
        mlp = 3 * h * i
        return attn + mlp

    @property
    def small_params_per_layer(self) -> int:
        """Biases and the two RMSNorm weights: tiny, never quantized."""
        bias = (self.q_dim + 2 * self.kv_dim) if self.qkv_bias else 0
        return bias + 2 * self.hidden_size

    @property
    def embedding_params(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def lm_head_params(self) -> int:
        return 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size

    @property
    def decoder_linear_params(self) -> int:
        return self.num_layers * self.linear_params_per_layer

    @property
    def total_params(self) -> int:
        return (
            self.embedding_params
            + self.num_layers * (self.linear_params_per_layer + self.small_params_per_layer)
            + self.hidden_size  # final norm
            + self.lm_head_params
        )

    @property
    def non_embedding_params(self) -> int:
        """Matches the model-card convention: total minus input embedding and lm_head."""
        return self.total_params - self.embedding_params - self.lm_head_params

    # --- per-token costs --------------------------------------------------
    def kv_bytes_per_token(self, kv_dtype: str = "bf16") -> float:
        """K and V for every layer: 2 * layers * kv_heads * head_dim * bytes."""
        return 2 * self.num_layers * self.kv_dim * DTYPE_BYTES[kv_dtype]

    def linear_flops_per_token(self, include_lm_head: bool = True) -> float:
        """2 FLOPs (multiply + add) per weight per token for every matmul."""
        params = self.decoder_linear_params
        if include_lm_head:
            params += self.vocab_size * self.hidden_size
        return 2.0 * params

    def attention_flops_per_token(self, context_len: int) -> float:
        """QK^T and softmax(.)V for one query token against `context_len` keys, all layers."""
        return 4.0 * self.num_layers * self.q_dim * context_len


def load_model(name: str = "Qwen2.5-7B-Instruct") -> ModelSpec:
    files = {"Qwen2.5-7B-Instruct": "qwen2.5-7b-instruct.config.json"}
    if name not in files:
        raise KeyError(f"unknown model {name!r}; known: {sorted(files)}")
    text = resources.files("capacity_planner").joinpath("data", files[name]).read_text()
    return ModelSpec.from_hf_config(name, json.loads(text))
