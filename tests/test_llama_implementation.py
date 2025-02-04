import tempfile
from pathlib import Path

import torch
from transformers import LlamaConfig as HFLlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

from simple_stories_train.models.llama import CausalSelfAttention as CustomCausalSelfAttention
from simple_stories_train.models.llama import Llama, LlamaConfig


def test_rotary_embedding_implementation() -> None:
    """Test the custom rotary embedding implementation against HuggingFace"""
    hidden_size = 768
    num_attention_heads = 12

    hf_llama_config = HFLlamaConfig(
        vocab_size=50527,
        attention_bias=False,
        use_flash_attention=True,
        max_position_embeddings=2048,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=12,
        intermediate_size=hidden_size * 4 * 2 // 3,
        num_key_value_heads=num_attention_heads // 4,
        rope_theta=10000,
        rope_scaling=None,
    )

    custom_llama_config = LlamaConfig(
        block_size=1024,
        vocab_size=50257,
        n_layer=12,
        n_head=num_attention_heads,
        n_embd=hidden_size,
        n_intermediate=hidden_size * 4 * 2 // 3,
        mlp_bias=False,
        attn_bias=False,
        rotary_adjacent_pairs=False,
        rotary_dim=768 // 12,
        rotary_base=10000,
        n_ctx=1024,
        n_key_value_heads=num_attention_heads // 4,
        use_grouped_query_attention=True,
        flash_attention=True,
    )

    custom_implementation = CustomCausalSelfAttention(custom_llama_config)
    hf_implementation = LlamaRotaryEmbedding(hf_llama_config)

    batch_size = 16
    seq_length = 32
    head_dim = custom_llama_config.n_embd // custom_llama_config.n_head

    # Create dummy query and key tensors [batch_size, num_heads, seq_length, head_dim]
    q_hf = torch.randn(batch_size, custom_llama_config.n_head, seq_length, head_dim)
    k_hf = torch.randn(batch_size, custom_llama_config.n_head, seq_length, head_dim)

    # Huggingface implementation expects positions ids explicitly for flexibility and
    # dynamic scaling. We calculate it implicitly in apply_rotary
    position_ids = torch.arange(seq_length).expand(batch_size, -1)

    hf_cos, hf_sin = hf_implementation.forward(q_hf, position_ids)
    q_hf_rot, k_hf_rot = apply_rotary_pos_emb(q_hf, k_hf, hf_cos, hf_sin, position_ids)

    q_custom = q_hf.detach().clone()
    k_custom = k_hf.detach().clone()

    q_custom_rot = custom_implementation.apply_rotary(q_custom)
    k_custom_rot = custom_implementation.apply_rotary(k_custom)

    torch.testing.assert_close(
        q_hf_rot, q_custom_rot, msg="Rotated queries don't match between implementation"
    )
    torch.testing.assert_close(
        k_hf_rot, k_custom_rot, msg="Rotated keys don't match between implementations"
    )


def test_local_pt_model_loading() -> None:
    """Test loading a model from a local .pt file"""
    config = LlamaConfig(
        block_size=128,
        vocab_size=300,
        n_layer=2,
        n_head=2,
        n_embd=12,
        rotary_dim=12 // 2,
        n_key_value_heads=2 // 2,
        flash_attention=True,
    )

    original_model = Llama(config)

    # Create a temporary directory that will be automatically cleaned up
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_model.pt"

        # Save and load the model back
        torch.save(original_model.state_dict(), model_path)
        loaded_model = Llama.from_pretrained(str(model_path), config)

        # Test basic functionality
        dummy_input = torch.randint(0, config.vocab_size, (1, 10))
        with torch.no_grad():
            original_output, _ = original_model(dummy_input)
            loaded_output, _ = loaded_model(dummy_input)

            torch.testing.assert_close(original_output, loaded_output)
            assert original_output.shape == (1, 1, config.vocab_size)
