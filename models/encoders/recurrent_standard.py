"""
Recurrent demonstration encoder (Variant A: RecurrentAggregationEncoder).

Like HybridStandard but with carry state that evolves across ACT steps.
NO internal H/L loops - just one pass per ACT step (deterministic).

Architecture:
1. DemoGridEncoder: Encodes each (input, output) pair (recomputed each step for gradients)
2. Cross-attention: z_e attends to demo encodings
3. MLP: Further processing

The z_e state persists across ACT steps via carry, allowing the encoder
representation to evolve alongside the decoder's reasoning.
"""

import torch
from torch import nn
import torch.nn.functional as F

from models.encoders.base import DemoEncoderConfig
from models.encoders.recurrent_base import BaseRecurrentEncoder, RecurrentEncoderCarry
from models.encoders.standard import DemoGridEncoder
from models.common import trunc_normal_init_
from models.layers import CastedLinear, SwiGLU, rms_norm


class CrossAttentionLayer(nn.Module):
    """Cross-attention from queries (z_e) to key-value pairs (demo_encodings)."""

    def __init__(self, config: DemoEncoderConfig):
        super().__init__()
        self.config = config

        self.q_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            queries: (B, T, D) - z_e
            keys: (B, K, D) - demo_encodings
            values: (B, K, D) - demo_encodings
            key_mask: (B, K) - True for valid demos

        Returns:
            output: (B, T, D)
        """
        batch_size = queries.shape[0]

        q = self.q_proj(queries).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(keys).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(values).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Create attention mask
        attn_mask = key_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, K)
        attn_bias = torch.zeros_like(attn_mask, dtype=q.dtype)
        attn_bias.masked_fill_(~attn_mask, float("-inf"))

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        return self.o_proj(attn_out)


class RecurrentAggregationEncoder(BaseRecurrentEncoder):
    """
    Recurrent demonstration encoder with simple carry-based recurrence.

    Variant A: No internal H/L loops, just carry state that evolves across ACT steps.

    Architecture:
    - DemoGridEncoder: Per-demo encoding (recomputed each step)
    - z_e: Latent state that attends to demo_encodings
    - Cross-attention + MLP: Update z_e based on demos

    Each forward pass does ONE iteration, z_e evolves across ACT steps.
    """

    def __init__(self, config: DemoEncoderConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # Reuse DemoGridEncoder from standard.py
        self.grid_encoder = DemoGridEncoder(config)

        # Learnable initial state for z_e (like TRM's H_init)
        self.z_e_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.output_tokens, config.hidden_size, dtype=self.forward_dtype),
                std=1.0,
            ),
            persistent=True,
        )

        # Cross-attention layer (z_e queries attend to demo_encodings)
        self.cross_attn = CrossAttentionLayer(config)

        # MLP after cross-attention
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

        self.norm_eps = config.rms_norm_eps

    def initial_carry(self, batch_size: int, device: torch.device) -> RecurrentEncoderCarry:
        """Initialize z_e from learnable init."""
        # Clone to avoid shared storage across batches (prevents "backward through graph twice" error)
        z_e = self.z_e_init.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device)
        return RecurrentEncoderCarry(z_e=z_e)

    def reset_carry(
        self,
        needs_reset: torch.Tensor,
        carry: RecurrentEncoderCarry,
    ) -> RecurrentEncoderCarry:
        """Reset z_e for halted samples."""
        # Clone to avoid shared storage (prevents "backward through graph twice" error)
        init_broadcast = self.z_e_init.unsqueeze(0).clone().to(carry.z_e.device)
        z_e = torch.where(
            needs_reset.view(-1, 1, 1),
            init_broadcast,
            carry.z_e,
        )
        return RecurrentEncoderCarry(z_e=z_e)

    def _encode_demos(
        self,
        demo_inputs: torch.Tensor,
        demo_labels: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode each demo pair (recomputed each step for gradient flow).

        Args:
            demo_inputs: (batch, max_demos, seq_len)
            demo_labels: (batch, max_demos, seq_len)
            demo_mask: (batch, max_demos)

        Returns:
            demo_encodings: (batch, max_demos, hidden_size)
        """
        batch_size, max_demos, seq_len = demo_inputs.shape

        # Flatten and encode
        demo_inputs_flat = demo_inputs.view(batch_size * max_demos, seq_len)
        demo_labels_flat = demo_labels.view(batch_size * max_demos, seq_len)

        demo_encodings_flat = self.grid_encoder(demo_inputs_flat, demo_labels_flat)
        demo_encodings = demo_encodings_flat.view(batch_size, max_demos, -1)

        # Mask invalid demos
        demo_encodings = demo_encodings * demo_mask.unsqueeze(-1).to(demo_encodings.dtype)

        return demo_encodings

    def forward(
        self,
        carry: RecurrentEncoderCarry,
        demo_inputs: torch.Tensor,
        demo_labels: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> RecurrentEncoderCarry:
        """
        Single recurrent step: z_e cross-attends to demo_encodings.

        Pattern:
            z_e = z_e + CrossAttn(z_e, demo_encodings)
            z_e = z_e + MLP(z_e)

        Args:
            carry: Current carry state
            demo_inputs: (batch, max_demos, seq_len)
            demo_labels: (batch, max_demos, seq_len)
            demo_mask: (batch, max_demos)

        Returns:
            new_carry: Updated carry state
        """
        # Encode demos (recomputed each step for gradient flow)
        demo_encodings = self._encode_demos(demo_inputs, demo_labels, demo_mask)

        # Get current z_e
        z_e = carry.z_e

        # Cross-attention: z_e queries attend to demo_encodings
        attn_out = self.cross_attn(
            queries=z_e,
            keys=demo_encodings,
            values=demo_encodings,
            key_mask=demo_mask,
        )
        z_e = rms_norm(z_e + attn_out, variance_epsilon=self.norm_eps)

        # MLP
        mlp_out = self.mlp(z_e)
        z_e = rms_norm(z_e + mlp_out, variance_epsilon=self.norm_eps)

        # Detach to break computational graph (allows reuse across batches)
        return RecurrentEncoderCarry(z_e=z_e.detach())
