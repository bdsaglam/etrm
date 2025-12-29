"""
LPN-style standard (deterministic) demonstration encoder.

Based on "Searching Latent Program Spaces" (LPN) encoder architecture:
- Deep transformer (8 layers by default)
- CLS token for pooling
- Pre-layer normalization (always)
- Simple mean aggregation across demo pairs (no cross-attention)
- Single vector output per demo pair

Key differences from StandardDemoEncoder:
- Deeper by default (8 layers vs 2)
- CLS token pooling (not mean/attention over sequence)
- No separate set encoder - just mean over demo embeddings
- Outputs single vector (hidden_size), then projects to output_tokens if needed
"""

import math
from typing import Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import einops

from models.encoders.base import BaseDemoEncoder, DemoEncoderConfig, EncoderOutput
from models.common import trunc_normal_init_
from models.layers import (
    CastedLinear,
    CastedEmbedding,
    SwiGLU,
    rms_norm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
)


class LPNTransformerBlock(nn.Module):
    """
    Transformer block for LPN encoder.

    Uses pre-layer normalization (always) and optional layer scale.
    Follows LPN paper architecture choices.
    """

    def __init__(self, config: DemoEncoderConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.norm_eps = config.rms_norm_eps

        # Self-attention projections
        self.qkv_proj = CastedLinear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=False
        )
        self.o_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)

        # MLP
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

        # Layer scale (optional, from CaiT)
        self.use_layer_scale = config.layer_scale_init > 0
        if self.use_layer_scale:
            self.gamma_1 = nn.Parameter(
                config.layer_scale_init * torch.ones(config.hidden_size)
            )
            self.gamma_2 = nn.Parameter(
                config.layer_scale_init * torch.ones(config.hidden_size)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            cos, sin: RoPE embeddings (seq_len, head_dim)

        Returns:
            (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Pre-norm attention
        normed = rms_norm(hidden_states, variance_epsilon=self.norm_eps)

        # QKV projection
        qkv = self.qkv_proj(normed)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, S, H, D)

        # Apply RoPE (skip CLS token at position 0)
        q_rope = q.clone()
        k_rope = k.clone()

        # Apply RoPE to non-CLS positions
        q_pos = q[:, 1:, :, :]  # Skip CLS
        k_pos = k[:, 1:, :, :]
        cos_pos = cos[:seq_len - 1]  # Positions for non-CLS tokens
        sin_pos = sin[:seq_len - 1]

        # RoPE application
        q_pos = q_pos * cos_pos.unsqueeze(-2) + self._rotate_half(q_pos) * sin_pos.unsqueeze(-2)
        k_pos = k_pos * cos_pos.unsqueeze(-2) + self._rotate_half(k_pos) * sin_pos.unsqueeze(-2)

        q_rope[:, 1:] = q_pos
        k_rope[:, 1:] = k_pos

        # Attention
        q_rope = q_rope.transpose(1, 2)  # (B, H, S, D)
        k_rope = k_rope.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q_rope, k_rope, v, is_causal=False)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if self.use_layer_scale:
            attn_output = self.gamma_1 * attn_output
        hidden_states = hidden_states + attn_output

        # Pre-norm MLP
        mlp_output = self.mlp(rms_norm(hidden_states, variance_epsilon=self.norm_eps))
        if self.use_layer_scale:
            mlp_output = self.gamma_2 * mlp_output
        hidden_states = hidden_states + mlp_output

        return hidden_states

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class LPNGridEncoder(nn.Module):
    """
    LPN-style grid encoder for single (input, output) demo pair.

    Key features:
    - CLS token prepended to sequence
    - Deep transformer (8 layers by default)
    - Pre-layer normalization
    - CLS token output as demo embedding
    """

    def __init__(self, config: DemoEncoderConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # Embedding scale
        embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / embed_scale

        # Separate embeddings for input and output grids
        self.input_embed = CastedEmbedding(
            config.vocab_size, config.hidden_size,
            init_std=embed_init_std, cast_to=self.forward_dtype
        )
        self.output_embed = CastedEmbedding(
            config.vocab_size, config.hidden_size,
            init_std=embed_init_std, cast_to=self.forward_dtype
        )

        # CLS token (learnable)
        self.cls_token = nn.Parameter(
            trunc_normal_init_(
                torch.empty(1, 1, config.hidden_size, dtype=self.forward_dtype),
                std=0.02
            )
        )

        # Positional encoding for concatenated sequence (2 * seq_len + 1 for CLS)
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_heads,
            max_position_embeddings=2 * config.seq_len + 1,
            base=10000.0,
        )

        # Transformer layers (deep by default)
        self.layers = nn.ModuleList([
            LPNTransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Final layer norm (applied to CLS output)
        self.final_norm_eps = config.rms_norm_eps

        self.embed_scale = embed_scale

    def forward(
        self,
        input_grid: torch.Tensor,
        output_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a single demo pair.

        Args:
            input_grid: (batch, seq_len) - Input grid tokens
            output_grid: (batch, seq_len) - Output grid tokens

        Returns:
            encoding: (batch, hidden_size) - Demo encoding (CLS token output)
        """
        batch_size = input_grid.shape[0]

        # Embed input and output
        input_emb = self.input_embed(input_grid.to(torch.int32))   # (B, S, D)
        output_emb = self.output_embed(output_grid.to(torch.int32))  # (B, S, D)

        # Concatenate: [input; output]
        hidden = torch.cat([input_emb, output_emb], dim=1)  # (B, 2*S, D)
        hidden = self.embed_scale * hidden

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        hidden = torch.cat([cls_tokens, hidden], dim=1)  # (B, 1 + 2*S, D)

        # Get RoPE embeddings
        cos, sin = self.rotary_emb()

        # Apply transformer layers
        for layer in self.layers:
            hidden = layer(hidden, cos, sin)

        # Extract CLS token output
        cls_output = hidden[:, 0, :]  # (B, D)

        # Final normalization
        cls_output = rms_norm(cls_output.unsqueeze(1), variance_epsilon=self.final_norm_eps).squeeze(1)

        return cls_output


class LPNStandardEncoder(BaseDemoEncoder):
    """
    LPN-style standard (deterministic) demonstration encoder.

    Architecture:
    1. Deep transformer encodes each (input, output) pair with CLS pooling
    2. Simple mean aggregation across demo encodings
    3. Project to output_tokens if needed

    Key differences from StandardDemoEncoder:
    - Deeper (8 layers default)
    - CLS token pooling
    - No cross-attention set encoder
    - Simpler aggregation (mean)
    """

    def __init__(self, config: DemoEncoderConfig):
        super().__init__(config)

        self.grid_encoder = LPNGridEncoder(config)

        # Project from hidden_size to output_tokens * hidden_size
        # This recreates the multi-token output format expected by TRM
        self.output_proj = CastedLinear(
            config.hidden_size,
            config.output_tokens * config.hidden_size,
            bias=False
        )

    def forward(
        self,
        demo_inputs: torch.Tensor,
        demo_labels: torch.Tensor,
        demo_mask: torch.Tensor,
        return_full_output: bool = False,
    ) -> Union[torch.Tensor, EncoderOutput]:
        """
        Encode demonstrations into context embedding.

        Args:
            demo_inputs: (batch, max_demos, seq_len) - Demo input grids
            demo_labels: (batch, max_demos, seq_len) - Demo output grids
            demo_mask: (batch, max_demos) - True for valid demos
            return_full_output: If True, return EncoderOutput with auxiliary info

        Returns:
            If return_full_output=False: context tensor (batch, output_tokens, hidden_size)
            If return_full_output=True: EncoderOutput with context and z_pooled
        """
        batch_size, max_demos, seq_len = demo_inputs.shape

        # Encode each demo pair
        demo_inputs_flat = demo_inputs.view(batch_size * max_demos, seq_len)
        demo_labels_flat = demo_labels.view(batch_size * max_demos, seq_len)

        demo_encodings_flat = self.grid_encoder(demo_inputs_flat, demo_labels_flat)
        demo_encodings = demo_encodings_flat.view(batch_size, max_demos, -1)  # (B, K, D)

        # Mask invalid demos
        demo_mask_expanded = demo_mask.unsqueeze(-1).to(demo_encodings.dtype)  # (B, K, 1)
        demo_encodings = demo_encodings * demo_mask_expanded

        # Mean aggregation across demos (LPN style - simple mean)
        num_valid = demo_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)  # (B, 1, 1)
        z_pooled = demo_encodings.sum(dim=1) / num_valid.squeeze(-1)  # (B, D)

        # Project to output tokens
        context = self.output_proj(z_pooled)  # (B, T*D)
        context = context.view(batch_size, self.config.output_tokens, self.config.hidden_size)

        if return_full_output:
            return EncoderOutput(
                context=context,
                z_pooled=z_pooled,
                kl_loss=None,
                mu=None,
                logvar=None,
            )

        return context


def test_lpn_standard_encoder():
    """Test the LPN standard encoder."""
    print("=== Testing LPNStandardEncoder ===\n")

    # Use LPN-style defaults (8 layers)
    config = DemoEncoderConfig(
        hidden_size=512,
        num_heads=8,
        num_layers=8,  # LPN uses 8 layers
        output_tokens=16,
        vocab_size=12,
        seq_len=900,
        norm_style="pre",  # LPN uses pre-norm
    )

    encoder = LPNStandardEncoder(config)
    encoder.eval()

    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {num_params:,}")

    # Create dummy input
    batch_size = 4
    max_demos = 3
    seq_len = 900

    demo_inputs = torch.randint(0, 12, (batch_size, max_demos, seq_len))
    demo_labels = torch.randint(0, 12, (batch_size, max_demos, seq_len))
    demo_mask = torch.tensor([
        [True, True, True],
        [True, True, False],
        [True, False, False],
        [True, True, True],
    ])

    # Forward pass (simple)
    with torch.no_grad():
        context = encoder(demo_inputs, demo_labels, demo_mask)

    print(f"Input shapes:")
    print(f"  demo_inputs: {demo_inputs.shape}")
    print(f"  demo_labels: {demo_labels.shape}")
    print(f"  demo_mask: {demo_mask.shape}")
    print(f"\nOutput shape: {context.shape}")
    print(f"Expected: ({batch_size}, {config.output_tokens}, {config.hidden_size})")

    assert context.shape == (batch_size, config.output_tokens, config.hidden_size)

    # Forward pass (full output)
    with torch.no_grad():
        output = encoder(demo_inputs, demo_labels, demo_mask, return_full_output=True)

    print(f"\nFull output:")
    print(f"  context: {output.context.shape}")
    print(f"  z_pooled: {output.z_pooled.shape}")
    print(f"  kl_loss: {output.kl_loss}")
    assert output.context.shape == (batch_size, config.output_tokens, config.hidden_size)
    assert output.z_pooled.shape == (batch_size, config.hidden_size)
    assert output.kl_loss is None

    print("\n=== Test passed! ===")


if __name__ == "__main__":
    test_lpn_standard_encoder()
