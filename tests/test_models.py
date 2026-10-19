import pytest

from capacity_planner.models import load_model, registry


@pytest.fixture
def qwen():
    return load_model("Qwen2.5-7B-Instruct")


def test_param_count_matches_model_card(qwen):
    # Model card: "Number of Parameters: 7.61B; Number of Parameters (Non-Embedding): 6.53B".
    assert qwen.total_params == 7_615_616_512
    assert round(qwen.non_embedding_params / 1e9, 2) == 6.53


def test_architecture(qwen):
    assert qwen.head_dim == 128
    assert qwen.kv_dim == 512  # GQA: 4 KV heads
    assert not qwen.tie_word_embeddings


def test_kv_bytes_per_token(qwen):
    # 2 (K,V) * 28 layers * 4 heads * 128 dim * 2 bytes
    assert qwen.kv_bytes_per_token("bf16") == 57_344
    assert qwen.kv_bytes_per_token("fp8") == 28_672


def test_flops_per_token(qwen):
    # Decoder linears + lm_head, 2 FLOPs per weight.
    expected = 2 * (qwen.decoder_linear_params + 152064 * 3584)
    assert qwen.linear_flops_per_token() == expected
    assert qwen.linear_flops_per_token(include_lm_head=False) == 2 * qwen.decoder_linear_params
    assert qwen.attention_flops_per_token(1000) == 4 * 28 * 3584 * 1000


def test_unknown_model():
    with pytest.raises(KeyError):
        load_model("nope")


@pytest.mark.parametrize("name", sorted(registry()))
def test_param_count_matches_checkpoint_bytes(name):
    """Every registry model ships BF16 safetensors, so the checkpoint is 2 bytes per parameter."""
    import json
    from importlib import resources

    cfg = json.loads(resources.files("capacity_planner").joinpath("data", registry()[name].file).read_text())
    assert cfg["_source"]["safetensors_total_bytes"] == 2 * load_model(name).total_params
