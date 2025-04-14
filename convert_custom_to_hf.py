"""
This script demonstrates how to convert our custom model to a HuggingFace-compatible model.
"""

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from simple_stories_train.models.llama import Llama
from simple_stories_train.models.model_configs import MODEL_CONFIGS

MODEL_SIZE = "35M"
model_config = MODEL_CONFIGS[MODEL_SIZE]
custom_model = Llama.from_pretrained(f"chandan-sreedhara/SimpleStories-{MODEL_SIZE}", model_config)

# Create a matching HuggingFace configuration
hf_config = LlamaConfig(
    vocab_size=model_config.vocab_size, 
    hidden_size=model_config.n_embd,
    intermediate_size=model_config.n_intermediate,  
    num_hidden_layers=model_config.n_layer,
    num_attention_heads=model_config.n_head,
    num_key_value_heads=model_config.n_key_value_heads,
    hidden_act="silu", 
    max_position_embeddings=2048, 
    rms_norm_eps=1e-06,
    tie_word_embeddings=True,
)

# Create a new HuggingFace model with our config
hf_model = LlamaForCausalLM(hf_config)

# Now let's copy the weights
# 1. Token embeddings
hf_model.model.embed_tokens.weight.data = custom_model.transformer.wte.weight.data

# 2. Copy each transformer block
for i in range(model_config.n_layer):
    # RMSNorm 1
    hf_model.model.layers[i].input_layernorm.weight.data = custom_model.transformer.h[
        i
    ].rms_1.weight.data

    # Attention weights
    # Query projection
    hf_model.model.layers[i].self_attn.q_proj.weight.data = custom_model.transformer.h[
        i
    ].attn.q_attn.weight.data

    # Key and Value are combined in your model but separate in HF model
    kv_weight = custom_model.transformer.h[i].attn.kv_attn.weight.data
    kv_dim = kv_weight.shape[0] // 2

    # Split KV weights for HF model
    hf_model.model.layers[i].self_attn.k_proj.weight.data = kv_weight[:kv_dim, :]
    hf_model.model.layers[i].self_attn.v_proj.weight.data = kv_weight[kv_dim:, :]

    # Output projection
    hf_model.model.layers[i].self_attn.o_proj.weight.data = custom_model.transformer.h[
        i
    ].attn.c_proj.weight.data

    # RMSNorm 2
    hf_model.model.layers[i].post_attention_layernorm.weight.data = custom_model.transformer.h[
        i
    ].rms_2.weight.data

    # MLP layers
    hf_model.model.layers[i].mlp.gate_proj.weight.data = custom_model.transformer.h[
        i
    ].mlp.gate_proj.weight.data
    hf_model.model.layers[i].mlp.up_proj.weight.data = custom_model.transformer.h[
        i
    ].mlp.up_proj.weight.data
    hf_model.model.layers[i].mlp.down_proj.weight.data = custom_model.transformer.h[
        i
    ].mlp.down_proj.weight.data

# 3. Final layer norm
hf_model.model.norm.weight.data = custom_model.transformer.rms_f.weight.data

# 4. LM head
hf_model.lm_head.weight.data = custom_model.lm_head.weight.data

# Save the model
# hf_model.save_pretrained("my_converted_hf_model")


hf_model.eval()
custom_model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(f"chandan-sreedhara/SimpleStories-{MODEL_SIZE}")
prompt = "The curious cat looked at the"
# IMPORTANT: Use tokenizer without special tokens
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
# IMPORTANT: Set correct EOS token ID (not the default from tokenizer)
eos_token_id = 1

for name, i, model in zip(["custom", "hf"], ["idx", "input_ids"], [custom_model, hf_model], strict=False):
    # Generate text
    print(f"Generating text with {name} model...")
    try:
        with torch.no_grad():
            logits = model.forward(
                **{i: inputs.input_ids}
            )
    except Exception as e:
        print(f"Error generating text with {name} model: {e}")
        continue
    print(f"Logits:\n{logits}")

# Testing original vs. copied model
for i, model in zip(["input_ids", "idx"], [hf_model, custom_model], strict=False):
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            **{i: inputs.input_ids},
            max_new_tokens=800,
            temperature=0.0,
            top_k=40,
            eos_token_id=eos_token_id,
        )

    # Decode output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated text:\n{output_text}")
