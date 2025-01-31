import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from transformers import LlamaConfig as HFLlamaConfig

from simple_stories_train.models.llama import LlamaConfig


class HFRotaryEmbedding(nn.Module):
    def __init__(self, config: HFLlamaConfig, device: str | torch.device | None = None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.dim = config.hidden_size // config.num_attention_heads
        self.base = 10000.0

        self.config = config
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0
        self.original_inv_freq = self.inv_freq

    def rotate_half(self, x: Tensor):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(
        self,
        q: Tensor,
        k: Tensor,
        cos: Tensor,
        sin: Tensor,
        position_ids: Tensor | None = None,
        unsqueeze_dim=1,
    ):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def _dynamic_frequency_update(self, position_ids: Tensor, device: torch.device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        )

        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class CustomRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig):
        """
        Initialize Rotary Embeddings module using LlamaConfig.

        Args:
            config: LlamaConfig instance containing model parameters
        """
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.rotary_dim = config.rotary_dim
        self.rotary_adjacent_pairs = config.rotary_adjacent_pairs
        self.rotary_base = config.rotary_base
        self.n_ctx = config.n_ctx
        self.flash_attention = config.flash_attention

        # Calculate and register sin/cos as buffers
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


# Demo usage
if __name__ == "__main__":
    hidden_size: int = 768
    num_attention_heads: int = 12

    hf_config = HFLlamaConfig(
        vocab_size=50527,
        attention_bias=False,  # Matching your attn_bias
        use_flash_attention=True,  # Matching your flash_attention
        max_position_embeddings=2048,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=12,
        intermediate_size=hidden_size * 4 * 2 // 3,
        num_key_value_heads=num_attention_heads // 4,  # Matching your n_key_value_heads
        rope_theta=10000,
        rope_scaling=None,
    )

    ## Custom rotary embedding
    custom_config = LlamaConfig(
        block_size=1024,
        vocab_size=50257,
        n_layer=12,
        n_head=num_attention_heads,
        n_embd=hidden_size,
        n_intermediate=hidden_size * 4 * 2 // 3,  # SwiGLU has 2/3 of the hidden size
        mlp_bias=False,
        attn_bias=False,
        rotary_adjacent_pairs=False,
        rotary_dim=768 // 12,  # i.e. same as d_head
        rotary_base=10000,
        n_ctx=1024,
        n_key_value_heads=num_attention_heads // 4,
        use_grouped_query_attention=True,
        flash_attention=True,
    )

    ## Rotary embedding evaluation with Hugging face implementation
    hf_rotary = HFRotaryEmbedding(hf_config, device="cpu")
    custom_rotary = CustomRotaryEmbedding(custom_config)

    # Create dummy data
    batch_size = 16
    seq_length = 32
    head_dim = custom_config.n_embd // custom_config.n_head

    # Create dummy query and key tensors [batch_size, num_heads, seq_length, head_dim]
    q_hf = torch.randn(batch_size, custom_config.n_head, seq_length, head_dim)
    k_hf = torch.randn(batch_size, custom_config.n_head, seq_length, head_dim)

    q_custom = q_hf.detach().clone()
    k_custom = k_hf.detach().clone()

    # Get rotary embeddings from both implementations
    position_ids = torch.arange(seq_length).expand(batch_size, -1)
    cos, sin = hf_rotary(q_hf, position_ids)
    q_hf_rot, k_hf_rot = hf_rotary.apply_rotary_pos_emb(q_hf, k_hf, cos, sin)

    q_custom_rot = custom_rotary.apply_rotary(q_custom)
    k_custom_rot = custom_rotary.apply_rotary(k_custom)

    try:
        torch.testing.assert_close(q_hf_rot, q_custom_rot)
        torch.testing.assert_close(k_hf_rot, k_custom_rot)
        print("✓ Rotary embedding test PASSED - outputs match between HF and custom implementation")
    except AssertionError as e:
        print("✗ Rotary embedding test FAILED")
        print(f"Error details: {str(e)}")
