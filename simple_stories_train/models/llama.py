import inspect
import math
import os

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load_file
from torch import Tensor
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn import functional as F

from simple_stories_train.utils import print0


class LlamaConfig(BaseModel):
    # model_config = ConfigDict(extra="forbid", frozen=True)
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
    flash_attention: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        # TODO: Make sure no biases once changing to rotary, as llama doesn't use them
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.use_grouped_query_attention = config.use_grouped_query_attention

        if self.use_grouped_query_attention:
            self.repeat_kv_heads = config.n_head // config.n_key_value_heads
            self.kv_attn = nn.Linear(
                config.n_embd, 2 * config.n_embd // self.repeat_kv_heads, bias=config.attn_bias
            )
            self.q_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_bias)
        else:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.attn_bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_bias)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1  # type:ignore
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.rotary_dim = config.rotary_dim
        self.rotary_adjacent_pairs = config.rotary_adjacent_pairs
        self.rotary_base = config.rotary_base
        self.n_ctx = config.n_ctx
        self.flash_attention = config.flash_attention

        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        sin, cos = self.calculate_sin_cos_rotary(self.rotary_dim, self.n_ctx, base=self.rotary_base)
        self.register_buffer("rotary_sin", sin)
        self.register_buffer("rotary_cos", cos)

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
            k, v = self.kv_attn(x).split(self.n_embd // self.repeat_kv_heads, dim=2)
            q = self.q_attn(x)
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

        if self.flash_attention:
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
        y = self.c_proj(y)
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
        self.rms_1 = LlamaRMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = LlamaRMSNorm(config.n_embd)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x: Float[Tensor, "... pos d_model"]) -> Float[Tensor, "... pos d_model"]:
        x = x + self.attn(self.rms_1(x))
        x = x + self.mlp(self.rms_2(x))
        return x


class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                rms_f=LlamaRMSNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # don't init this one, we will tie weights
        self.lm_head.LLMC_SKIP_INIT = 1  # type:ignore
        self.transformer.wte.weight = (
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
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
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

        # forward the model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        # x = tok_emb + pos_emb
        x = tok_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.rms_f(x)

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
        eos_token_id: int | None = None,
    ) -> Float[Tensor, "... pos"]:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # Keep track of whether input was 1D and ensure input has batch dimension
        is_1d = idx.dim() == 1
        if is_1d:
            idx = idx.unsqueeze(0)

        # Initialize not_completed mask for the batch
        batch_size = idx.size(0)
        not_completed = torch.ones(batch_size, dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            # If all sequences are completed, stop early
            if not not_completed.any():
                break

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

            # Create a mask for selecting which sequences to update
            # Only append new tokens for sequences that haven't completed
            # For completed sequences, replace new token with EOS and let the
            # tokenizer handle it
            if eos_token_id is not None:
                not_completed = not_completed & (idx_next[:, -1] != eos_token_id)
                update_mask = not_completed.unsqueeze(-1)
                idx_next = torch.where(
                    update_mask, idx_next, torch.full_like(idx_next, eos_token_id)
                )

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        # Remove batch dimension if input was 1D
        if is_1d:
            idx = idx.squeeze(0)

        return idx

    @classmethod
    def from_pretrained(
        cls, model_path_or_id: str, config: LlamaConfig, strict: bool = True
    ) -> "Llama":
        """
        Load a model either from a local checkpoint or from HuggingFace Hub.

        Args:
            model_path_or_id: Path to local checkpoint or HuggingFace model ID
            config: model configuration
            strict: Whether to strictly enforce matching keys in state dict
        Returns:
            Loaded model instance into CPU
        """
        model = cls(config)

        # Determine if path is local file or HuggingFace ID
        is_local = os.path.exists(model_path_or_id)
        if is_local:
            state_dict = torch.load(model_path_or_id, weights_only=True, map_location="cpu")

        else:
            # Load from HuggingFace Hub
            try:
                weights_path = hf_hub_download(
                    repo_id=model_path_or_id, filename="model.safetensors"
                )
                # loads the model file into CPU by default
                state_dict = load_file(weights_path)

                # Convert HuggingFace state dict format
                converted_state_dict = {}
                for k, v in state_dict.items():
                    # Remove 'llama.' prefix if present
                    k = k.replace("llama.", "")

                    # Handle special case for lm_head/wte weight tying
                    if k == "lm_head.weight":
                        converted_state_dict["lm_head.weight"] = v
                        converted_state_dict["transformer.wte.weight"] = v
                    else:
                        converted_state_dict[k] = v
                state_dict = converted_state_dict

            except Exception as err:
                raise ValueError(
                    f"Error loading model from HuggingFace Hub: {str(err)}. "
                    f"Please ensure the model path or ID '{model_path_or_id}' is correct."
                ) from err

        # Clean up state dict keys if needed
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # Load state dict
        model.load_state_dict(state_dict, strict=strict)
        return model
