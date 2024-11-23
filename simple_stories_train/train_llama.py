"""
Reference code for Llama training and inference.
Will save the model weights into files, to be read from C as initialization.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

Example launches to only benchmark the speed of bfloat16 compiled GPU training:
1 GPU:
python train_llama.py --write_tensors=0 --num_iterations=50 --sequence_length=1024 --compile=1 --tensorcores=1 --dtype=bfloat16
you can also turn on flash-attention by appending --flash=1
4 GPU:
torchrun --standalone --nproc_per_node=4 train_llama.py --write_tensors=0 --num_iterations=50 --sequence_length=1024 --compile=1 --tensorcores=1 --dtype=bfloat16

This implementation is based on
- llm.c,           licensed under MIT ((c) 2024 Andrei Karpathy) and
- TransformerLens, licensed under MIT ((c) 2022 TransformerLensOrg).


MIT License:
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import inspect
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch._inductor.config as config
import torch.distributed as dist
import torch.nn as nn
from dataloaders import DatasetConfig, create_data_loader
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import (
    init_wandb,
    is_checkpoint_step,
    log_generations,
    log_metrics,
    print0,
    save_model_and_config,
)

# using a global to toggle flash-attention
FLASH = 0


@dataclass
class LlamaConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_intermediate: int = 768 * 4 * 2 // 3  # SwiGLU has 2/3 of the hidden size
    mlp_bias: bool = False
    attn_bias: bool = False
    rotary_adjacent_pairs: bool = False
    rotary_dim: int = 768 // 12  # i.e. same as d_head
    rotary_base: int = 10000
    n_ctx: int = 1024
    n_key_value_heads: int = (
        12 // 4
    )  # Note that llama 3.1 n_key_value_heads does not scale with n_heads
    use_grouped_query_attention: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        # TODO: Make sure no biases once changing to rotary, as llama doesn't use them
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.use_grouped_query_attention = config.use_grouped_query_attention

        if self.use_grouped_query_attention:
            self.repeat_kv_heads = config.n_head // config.n_key_value_heads
            self.k_proj = nn.Linear(
                config.n_embd, config.n_embd // self.repeat_kv_heads, bias=config.attn_bias
            )
            self.v_proj = nn.Linear(
                config.n_embd, config.n_embd // self.repeat_kv_heads, bias=config.attn_bias
            )
            self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_bias)
        else:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.attn_bias)

        # output projection
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_bias)
        self.o_proj.LLMC_RESIDUAL_SCALE_FLAG = 1  # type:ignore
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.rotary_dim = config.rotary_dim
        self.rotary_adjacent_pairs = config.rotary_adjacent_pairs
        self.rotary_base = config.rotary_base
        self.n_ctx = config.n_ctx

        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

        sin, cos = self.calculate_sin_cos_rotary(self.rotary_dim, self.n_ctx, base=self.rotary_base)
        self.register_buffer("rotary_sin", sin, persistent=False)
        self.register_buffer("rotary_cos", cos, persistent=False)

    def calculate_sin_cos_rotary(
        self,
        rotary_dim: int,
        n_ctx: int,
        base: int = 10000,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the sine and cosine waves to use in a rotary embedding. See https://blog.eleuther.ai/rotary-embeddings/ for details

        Note: For some inexplicable reason, in GPT-J each ADJACENT pair of elements in k and q are rotated, in GPT-NeoX the pair of elements at k and k+n//2 are rotated (ie folding the full length in half, and then looking at pairs accordingly). I have absolutely no clue why, it should be completely equivalent.
        To resolve this, I've coded it to default to the GPT-J mode, but to explicitly check whether it's GPT-NeoX and then do the GPT-NeoX thing if it is.
        """
        high_precision = torch.float32 if dtype != torch.float64 else torch.float64
        pos = torch.arange(n_ctx, dtype=high_precision)
        dim = torch.arange(rotary_dim // 2, dtype=high_precision)

        # A set of frequencies evenly spaced in log space
        freq = base ** (dim / (rotary_dim / 2))
        if self.rotary_adjacent_pairs:
            freq = freq.unsqueeze(1).repeat(1, 2).flatten()
        else:
            freq = freq.repeat(2)
        # Create a n_ctx x rotary_dim tensor, where each column is an arithmetic sequence of angles in that frequency
        angles = pos[:, None] / freq[None, :]
        return torch.sin(angles).to(dtype), torch.cos(angles).to(dtype)

    def get_offset_position_ids(
        self,
        past_kv_pos_offset: int,
        attention_mask: Int[Tensor, "batch offset_pos"],
    ) -> Float[Tensor, "pos batch"]:
        """
        Returns the indices of non-padded tokens, offset by the position of the first attended token.
        """
        # shift the position ids so that the id at the the first attended token position becomes zero.
        # The position ids of the prepending pad tokens are shifted to -1.
        shifted_position_ids = attention_mask.cumsum(dim=1) - 1  # [batch, tokens_length]

        # Set the position ids of all prepending pad tokens to an arbitrary number (zero here)
        # just to avoid indexing errors.
        position_ids = shifted_position_ids.masked_fill(shifted_position_ids < 0, 0)
        return position_ids[:, past_kv_pos_offset:]  # [pos, batch]

    def rotate_every_two(self, x: Tensor) -> Tensor:
        """
        Rotary helper function, splits x into blocks of size 2 along the final axis and maps [x0, x1] to [-x1, x0]

        The final axis of x must have even length.

        GPT-NeoX and GPT-J do rotary subtly differently, see calculate_sin_cos_rotary for details.
        """
        rot_x = x.clone()
        if self.rotary_adjacent_pairs:
            rot_x[..., ::2] = -x[..., 1::2]
            rot_x[..., 1::2] = x[..., ::2]
        else:
            n = x.size(-1) // 2
            rot_x[..., :n] = -x[..., n:]
            rot_x[..., n:] = x[..., :n]

        return rot_x

    def apply_rotary(
        self,
        x: Float[Tensor, "batch head pos head_size"],
        past_kv_pos_offset=0,
        attention_mask: Int[Tensor, "batch offset_pos"] | None = None,
    ) -> Float[Tensor, "batch head pos head_size"]:
        # Only apply rotary to first rotary_dim dimensions (eg, if rotary_dim=64 and d_head=256, only apply to first 1/4 of dimensions)
        x = x.permute(
            0, 2, 1, 3
        )  # TODO check if this permutation slows down the function significantly
        x_pos = x.size(1)
        x_rot = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]
        x_flip = self.rotate_every_two(x_rot)

        if attention_mask is None:
            rotary_cos = self.rotary_cos[
                None, past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :
            ]
            rotary_sin = self.rotary_sin[
                None, past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :
            ]
            x_rotated = x_rot * rotary_cos + x_flip * rotary_sin
        else:
            offset_position_ids = self.get_offset_position_ids(past_kv_pos_offset, attention_mask)
            offset_position_ids = offset_position_ids.to(self.rotary_cos.device)
            mask_rotary_cos = self.rotary_cos[offset_position_ids, None, :]
            mask_rotary_sin = self.rotary_sin[offset_position_ids, None, :]
            x_rotated = x_rot * mask_rotary_cos + x_flip * mask_rotary_sin
        out = torch.cat([x_rotated, x_pass], dim=-1)
        return out.permute(0, 2, 1, 3)

    def forward(
        self,
        x: Float[Tensor, "batch pos d_model"],
        attention_mask: Int[Tensor, "batch offset_pos"] | None = None,
    ) -> Float[Tensor, "batch pos d_model"]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        if self.use_grouped_query_attention:
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            k = self.k_proj(x)
            v = self.v_proj(x)
            q = self.q_proj(x)
            assert k.shape[-1] == v.shape[-1] == q.shape[-1] // self.repeat_kv_heads

            # Repeat k and v for each head
            k = k.repeat_interleave(self.repeat_kv_heads, dim=1)
            v = v.repeat_interleave(self.repeat_kv_heads, dim=1)
        else:
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        q = self.apply_rotary(q, 0, attention_mask)
        k = self.apply_rotary(k, 0, attention_mask)  # keys are cached so no offset

        if FLASH:
            # flashattention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.o_proj(y)
        return y


class SwiGLUMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.n_embd, config.n_intermediate, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.n_embd, config.n_intermediate, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.n_intermediate, config.n_embd, bias=config.mlp_bias)
        self.act_fn = nn.functional.silu

    def forward(self, x: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Block(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.n_embd)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.n_embd)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x: Float[Tensor, "... pos d_model"]) -> Float[Tensor, "... pos d_model"]:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        self.model = nn.ModuleDict(
            dict(
                embed_tokens=nn.Embedding(config.vocab_size, config.n_embd),
                layers=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                norm=LlamaRMSNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # don't init this one, we will tie weights
        self.lm_head.LLMC_SKIP_INIT = 1  # type:ignore
        self.model.embed_tokens.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = (
                0.02
                if not hasattr(module, "LLMC_RESIDUAL_SCALE_FLAG")
                else 0.02 / math.sqrt(2 * self.config.n_layer)
            )
            # we want to skip initializing lm_head, which shares parameters with embed_tokens
            # and embed_tokens was already initialized down below during the Embedding init
            if not hasattr(module, "LLMC_SKIP_INIT"):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:  # type: ignore
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(
        self,
        idx: Float[Tensor, "batch pos"],
        targets: Float[Tensor, "batch pos vocab"] | None = None,
        return_logits=True,
    ) -> tuple[Float[Tensor, "batch pos"] | None, Float[Tensor, ""] | None]:
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.model.embed_tokens(idx)  # token embeddings of shape (b, t, n_embd)
        # x = tok_emb + pos_emb
        x = tok_emb

        for block in self.model.layers:
            x = block(x)
        x = self.model.norm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            targets = targets.long()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "d2": dict(
                block_size=1024,
                vocab_size=50257,
                n_layer=2,
                n_head=2,
                n_embd=12,
                rotary_dim=12 // 2,
                n_key_value_heads=2 // 2,
                attn_bias=False,
                mlp_bias=False,
                rotary_adjacent_pairs=False,
                use_grouped_query_attention=True,
            )
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = LlamaConfig(**config_args)  # type: ignore
        model = Llama(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
        zero_stage: int,
    ) -> torch.optim.Optimizer:
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print0(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print0(f"using fused AdamW: {use_fused}")
        if zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optim_group = optim_groups[0]
            optimizer = ZeroRedundancyOptimizer(
                **optim_group,  # type: ignore[reportArgumentType]
                optimizer_class=torch.optim.AdamW,
                lr=learning_rate,
                betas=betas,
                fused=use_fused,
            )
            optimizer.add_param_group(optim_groups[1])
        else:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas, fused=use_fused
            )
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: Float[Tensor, "... pos"],
        max_new_tokens: int,
        temperature=1.0,
        top_k: int | None = None,
    ) -> Float[Tensor, "... pos"]:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == "__main__":
    print0(f"Running pytorch {torch.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument(
        "--input_bin",
        type=str,
        default="lennart-finke/SimpleStories",
    )
    parser.add_argument(
        "--input_val_bin", type=str, default="", help="input .bin to eval validation loss on"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="output directory to which to write logs and checkpoints",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d2|d12|d24|d36|d48",
    )
    # token layout for each step of the optimization
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size, in units of #batch dimensions"
    )
    parser.add_argument("--sequence_length", type=int, default=64, help="sequence length")
    parser.add_argument(
        "--total_batch_size",
        type=int,
        default=256,
        help="total desired batch size, in units of #tokens",
    )
    # workload (number of steps)
    parser.add_argument(
        "--num_iterations", type=int, default=10, help="number of iterations to run"
    )
    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    # optimization
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="learning rate warmup iterations"
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=0, help="learning rate warmup iterations"
    )
    parser.add_argument(
        "--learning_rate_decay_frac",
        type=float,
        default=1.0,
        help="learning rate warmup iterations",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument(
        "--val_loss_every", type=int, default=0, help="every how mant steps to evaluate val loss?"
    )
    parser.add_argument(
        "--val_max_steps", type=int, default=20, help="how many batches of val to average?"
    )
    parser.add_argument(
        "--sample_every", type=int, default=0, help="how often to sample from the model?"
    )
    # numerics
    parser.add_argument("--tensorcores", type=int, default=0, help="use tensorcores")
    # memory management
    parser.add_argument(
        "--device", type=str, default="", help="by default we autodetect, or set it here"
    )
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--flash", type=int, default=0, help="use flash attention")
    parser.add_argument("--dtype", type=str, default="float32", help="float32|float16|bfloat16")
    parser.add_argument(
        "--zero_stage", type=int, default=0, help="zero redundancy optimizer stage (0/1/2/3)"
    )
    # python -> C bridge
    parser.add_argument("--write_tensors", type=int, default=1, help="write tensors to disk")
    # wandb settings
    parser.add_argument("--wandb", type=int, default=0, help="use wandb?")
    parser.add_argument("--wandb_project", type=str, default="", help="wandb project name")
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 1024
    assert args.dtype in {"float32", "float16", "bfloat16"}
    assert args.model in {
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "d2",
        "d12",
        "d24",
        "d36",
        "d48",
    }

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = 0  # each process gets the exact same seed
        zero_stage = args.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        # select the device
        if args.device:
            # provided explicitly by the user
            device = args.device
        else:
            # attempt to autodetect the device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    print(f"using device: {device}")
    device_type = "cuda" if "cuda" in device else "cpu"

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert (
        args.total_batch_size % tokens_per_fwdbwd == 0
    ), f"Mismatch between batch size and tokens {args.total_batch_size} % {tokens_per_fwdbwd} != 0"
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        args.dtype
    ]
    ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)  # type: ignore
        if device_type == "cuda"
        else nullcontext()
    )

    # rng / reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if args.tensorcores:
        torch.set_float32_matmul_precision("high")

    # turn on/off flash attention
    assert args.flash in {0, 1}
    FLASH = args.flash  # type: ignore

    # init (and write) the tokenizer
    enc: tiktoken.core.Encoding = tiktoken.get_encoding("gpt2")

    # init the model, either from scratch or from OpenAI pretrained checkpoint
    if args.model[0] == "d":
        # from scratch (random weights)
        model_config = {
            "d2": LlamaConfig(
                block_size=1024,
                vocab_size=50257,  # TODO: Make this depend on the tokenizer vocab size
                n_layer=2,
                n_head=2,
                n_embd=12,
                rotary_dim=12 // 2,
                n_key_value_heads=2 // 2,
            ),
            "d12": LlamaConfig(
                block_size=1024,
                vocab_size=50257,
                n_layer=12,
                n_head=12,
                n_embd=768,
                rotary_dim=768 // 12,
                n_key_value_heads=12 // 4,
            ),
            "d24": LlamaConfig(
                block_size=1024,
                vocab_size=50257,
                n_layer=24,
                n_head=16,
                n_embd=1024,
                rotary_dim=1024 // 16,
                n_key_value_heads=16 // 4,
            ),
            "d36": LlamaConfig(
                block_size=1024,
                vocab_size=50257,
                n_layer=36,
                n_head=20,
                n_embd=1280,
                rotary_dim=1280 // 20,
                n_key_value_heads=20 // 4,
            ),
            "d48": LlamaConfig(
                block_size=1024,
                vocab_size=50257,
                n_layer=48,
                n_head=25,
                n_embd=1600,
                rotary_dim=1600 // 25,
                n_key_value_heads=25 // 4,
            ),
        }[args.model]
        model = Llama(model_config)
    else:
        # load the GPT-2 model weights
        model = Llama.from_pretrained(args.model)
    model.train()
    model.to(device)
    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True  # suggested by @Chillee
        print0("compiling the model...")
        model: nn.Module = torch.compile(model)  # type: ignore[reportArgumentType]

    # load tokens
    dataset_config = DatasetConfig(
        dataset_name=args.input_bin,
        is_tokenized=False,
        tokenizer_file_path="simple_stories_train/tokenizer/stories-3072.json",
        streaming=True,
        split="train",
        n_ctx=T,
        seed=None,
        column_name="story",
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )

    train_loader, tokenizer = create_data_loader(
        dataset_config=dataset_config, batch_size=B, buffer_size=1000, global_seed=0
    )
    train_loader = iter(train_loader)  # Is this the right way to sample from a Pytorch DataLoader?

    dataset_config.split = "train"  # TODO: Change this to "val" when we have a validation dataset
    val_loader, tokenizer = create_data_loader(
        dataset_config=dataset_config, batch_size=B, buffer_size=1000, global_seed=0
    )

    # -------------------------------------------------------------------------
    # main training loop
    if args.wandb:
        init_wandb(args, args.wandb_project)

    # here we wrap model into DDP container
    if ddp:
        model: nn.Module = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type=device,
        zero_stage=zero_stage,
    )

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it: int) -> float:
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.num_iterations:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.num_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
        return min_lr + coeff * (args.learning_rate - min_lr)

    # create the logging directory if it does not exist
    logfile = None
    checkpoints_dir = None
    output_dir = None
    if args.output_dir:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(args.output_dir) / f"{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logfile = output_dir / "main.log"
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") as f:
            pass

        # set our checkpoints directory and save off the initilized model
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        save_model_and_config(checkpoints_dir, raw_model, args.__dict__, step=0)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    norm = -1.0  # dummy value to print in inference-only mode
    generations = []
    for step in range(1, args.num_iterations + 1):
        t0 = time.time()
        last_step = step == args.num_iterations

        # once in a while evaluate the validation dataset
        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
            model.eval()
            val_loader_iter = iter(
                val_loader
            )  # By creating the iterator anew, we sample the same data each time
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    bat = next(val_loader_iter)
                    x = bat[-1:].view(B, T)  # inputs
                    y = bat[1:].view(B, T)  # targets
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=False)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            # log to wandb
            if args.wandb:
                log_metrics(step, {"val_loss": val_loss})
            # log to console and to file
            print0(f"val loss {val_loss}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f\n" % (step, val_loss))

        # once in a while perform model inference on the master process
        if (
            args.sample_every > 0 and (step % args.sample_every == 0 or last_step)
        ) and master_process:
            model.eval()
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
            start_ids = [enc.eot_token]
            xg = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            max_new_tokens = 32
            temperature = 1.0
            top_k = 40
            yg = raw_model.generate(xg, max_new_tokens, temperature=temperature, top_k=top_k)
            print0("---------------")
            print0(enc.decode(yg[0].tolist()))
            print0("---------------")
            # log to wandb
            if args.wandb:
                generations.append([step, enc.decode(yg[0].tolist())])
            log_generations(step, generations)

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = Tensor(
            [0.0]
        ).to(device)  # for getting the mean loss (as simple float) over the accumulation steps
        for micro_step in range(grad_accum_steps):
            # fetch a batch
            bat = next(train_loader)["input_ids"].to(torch.int)
            x = bat.view(B, T)[:, :-1]  # inputs
            y = bat.view(B, T)[:, 1:]  # targets
            x, y = x.to(device), y.to(device)
            if ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                lossf += loss.detach()  # keep track of the mean loss

            # backward pass
            if not args.inference_only:
                loss.backward()
        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # step the optimizer
        optimizer.step()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1 - t0)
        print0(
            f"step {step:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)"
        )
        # log to wandb
        if args.wandb:
            log_metrics(
                step,
                {
                    "train_loss": lossf,
                    "lr": lr,
                },
            )
        # log to logile
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trl:%f\n" % (step, lossf))

        if checkpoints_dir is not None and is_checkpoint_step(step):
            save_model_and_config(checkpoints_dir, raw_model, args.__dict__, step=step)

        # keep track of smooth timings, last 20 iterations
        if step > 1 and step > args.num_iterations - 20:
            timings.append(t1 - t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    if ddp:
        destroy_process_group()
