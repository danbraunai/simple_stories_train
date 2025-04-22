"""
This script demonstrates how to convert our custom model to a HuggingFace-compatible model.
"""

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from simple_stories_train.models.llama import Llama
from simple_stories_train.models.model_configs import MODEL_CONFIGS

MODEL_SIZE = "5M"
model_config = MODEL_CONFIGS[MODEL_SIZE]
custom_model = Llama.from_pretrained(
    "/home/chandan/research/simple_stories_research/output_21-04-2025/SimpleStories-5M/model_step_59999.pt", 
    model_config
)

# Create a matching HuggingFace configuration
hf_config = LlamaConfig(
    vocab_size=model_config.vocab_size,
    hidden_size=model_config.n_embd,
    intermediate_size=model_config.n_intermediate,
    num_hidden_layers=model_config.n_layer,
    num_attention_heads=model_config.n_head,
    num_key_value_heads=model_config.n_key_value_heads,
    max_position_embeddings=model_config.rotary_dim,
    hidden_act="silu",
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

logits_dict = {}
for name, i, model in zip(
    ["custom", "hf"], ["idx", "input_ids"], [custom_model, hf_model], strict=False
):
    # Generate text
    print(f"Generating text with {name} model...")
    try:
        with torch.no_grad():
            logits = model.forward(**{i: inputs.input_ids})
    except Exception as e:
        print(f"Error generating text with {name} model: {e}")
        continue
    logits_dict[i] = logits


assert torch.allclose(
    logits_dict["idx"][0],
    logits_dict["input_ids"][0],
    atol=1e-4,
)

# Note: Generation  might be different since RNG in HF implementation could be different. We are mainly
# looking for coherent response and similar quality response.

# Testing HuggingFace model
with torch.no_grad():
    # Set seed for reproducibility
    torch.manual_seed(42)
    hf_output_ids = hf_model.generate(
        input_ids=inputs.input_ids,
        max_new_tokens=800,
        temperature=0.7,
        do_sample=True,  # Add this to match custom model behavior
        eos_token_id=eos_token_id,
    )

# Decode HF model output
hf_output_text = tokenizer.decode(hf_output_ids[0], skip_special_tokens=True)
print(f"Generated text from HuggingFace model:\n{hf_output_text}")

# Testing custom model
with torch.no_grad():
    # Reset seed for identical results
    torch.manual_seed(42)
    custom_output_ids = custom_model.generate(
        idx=inputs.input_ids,
        max_new_tokens=800,
        temperature=0.7,
        eos_token_id=eos_token_id,
    )

# Decode custom model output
custom_output_text = tokenizer.decode(custom_output_ids[0], skip_special_tokens=True)
print(f"Generated text from custom model:\n{custom_output_text}")
