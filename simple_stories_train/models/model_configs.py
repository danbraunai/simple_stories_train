from simple_stories_train.models.llama import LlamaConfig

MODEL_CONFIGS = {
    "d2": LlamaConfig(
        block_size=1024,
        vocab_size=50257,  # TODO: Make this depend on the tokenizer vocab size
        n_layer=2,
        n_head=2,
        n_embd=12,
        rotary_dim=12 // 2,
        n_key_value_heads=2 // 2,
        flash_attention=True,
    ),
    "d12": LlamaConfig(
        block_size=1024,
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        rotary_dim=768 // 12,
        n_key_value_heads=12 // 4,
        flash_attention=True,
    ),
    "d24": LlamaConfig(
        block_size=1024,
        vocab_size=50257,
        n_layer=24,
        n_head=16,
        n_embd=1024,
        rotary_dim=1024 // 16,
        n_key_value_heads=16 // 4,
        flash_attention=True,
    ),
    "d36": LlamaConfig(
        block_size=1024,
        vocab_size=50257,
        n_layer=36,
        n_head=20,
        n_embd=1280,
        rotary_dim=1280 // 20,
        n_key_value_heads=20 // 4,
        flash_attention=True,
    ),
    "d48": LlamaConfig(
        block_size=1024,
        vocab_size=50257,
        n_layer=48,
        n_head=25,
        n_embd=1600,
        rotary_dim=1600 // 25,
        n_key_value_heads=25 // 4,
        flash_attention=True,
    ),
    "33M": LlamaConfig(
        block_size=512,
        vocab_size=50257,
        n_layer=12,
        n_head=8,
        n_embd=512,
        n_intermediate=512 * 4 * 2 // 3,
        rotary_dim=512 // 8,
        n_ctx=512,
        n_key_value_heads=2,
        flash_attention=True,
    ),
}

MODEL_CONFIGS = {key: LlamaConfig(**value) for key, value in MODEL_CONFIGS_DICT.items()}  # type: ignore
