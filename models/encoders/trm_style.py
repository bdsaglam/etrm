"""
TRM-style recurrent demonstration encoder (Variant B).

Like the TRM decoder, this encoder has:
- Dual latent states: z_e_H (high-level context) and z_e_L (low-level reasoning)
- Internal H/L loops per ACT step (hierarchical refinement)
- z_e_H serves as the context for the decoder

This is more complex than RecurrentAggregationEncoder but allows richer
pattern reasoning from demonstrations.
"""

import torch
from torch import nn
from typing import List

from models.encoders.base import DemoEncoderConfig
from models.encoders.recurrent_base import BaseRecurrentEncoder, TRMStyleEncoderCarry
from models.encoders.standard import DemoGridEncoder
from models.common import trunc_normal_init_
from models.layers import CastedLinear, SwiGLU, rms_norm, RotaryEmbedding, Attention, CosSin


class TRMStyleEncoderBlock(nn.Module):
    """Single layer for TRM-style encoder (L_level equivalent)."""

    def __init__(self, config: DemoEncoderConfig):
        super().__init__()
        self.config = config

        # Self-attention
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )

        # MLP
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

        self.norm_eps = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor, cos_sin: CosSin = None) -> torch.Tensor:
        """
        Single layer forward pass.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            cos_sin: Optional rotary embeddings

        Returns:
            hidden_states: (batch, seq_len, hidden_size)
        """
        # Self-attention
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )

        # MLP
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)

        return hidden_states


class TRMStyleEncoderReasoningModule(nn.Module):
    """Reasoning module with input injection (L_level equivalent)."""

    def __init__(self, layers: List[TRMStyleEncoderBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self, hidden_states: torch.Tensor, input_injection: torch.Tensor, cos_sin: CosSin = None
    ) -> torch.Tensor:
        """
        Forward with input injection.

        Args:
            hidden_states: Current state
            input_injection: Signal to inject (demo_input or other level's state)
            cos_sin: Optional rotary embeddings

        Returns:
            Updated hidden_states
        """
        hidden_states = hidden_states + input_injection

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin=cos_sin)

        return hidden_states


class TRMStyleEncoder(BaseRecurrentEncoder):
    """
    TRM-style recurrent encoder with hierarchical H/L reasoning.

    Architecture:
    - DemoGridEncoder: Per-demo encoding (recomputed each step)
    - z_e_H: High-level context state (output to decoder)
    - z_e_L: Low-level reasoning state (internal refinement)
    - H/L loops: Hierarchical refinement similar to TRM decoder

    Each forward pass:
    1. Encode demos → demo_input
    2. Run H_cycles with L_cycles nested loops
    3. Refine z_e_H and z_e_L hierarchically
    4. Return TRMStyleEncoderCarry (z_e_H is context)
    """

    def __init__(self, config: DemoEncoderConfig, etrmtrm_config):
        super().__init__()
        self.config = config
        self.etrmtrm_config = etrmtrm_config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # Reuse DemoGridEncoder from standard.py
        self.grid_encoder = DemoGridEncoder(config)

        # Learnable initial states for z_e_H and z_e_L
        self.z_e_H_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.output_tokens, config.hidden_size, dtype=self.forward_dtype),
                std=1.0,
            ),
            persistent=True,
        )

        self.z_e_L_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.output_tokens, config.hidden_size, dtype=self.forward_dtype),
                std=1.0,
            ),
            persistent=True,
        )

        # L_level: Reasoning module (stack of layers)
        self.L_level = TRMStyleEncoderReasoningModule([
            TRMStyleEncoderBlock(config)
            for _ in range(etrmtrm_config.encoder_l_layers)
        ])

        # Position encodings (optional, like decoder)
        if config.seq_len > 0:  # Only if we have positional info
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.output_tokens,
                base=10000.0,
            )
        else:
            self.rotary_emb = None

        self.norm_eps = config.rms_norm_eps

    def initial_carry(self, batch_size: int, device: torch.device) -> TRMStyleEncoderCarry:
        """Initialize z_e_H and z_e_L from learnable inits."""
        # Clone to avoid shared storage across batches
        z_e_H = self.z_e_H_init.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device)
        z_e_L = self.z_e_L_init.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device)
        return TRMStyleEncoderCarry(z_e_H=z_e_H, z_e_L=z_e_L)

    def reset_carry(
        self,
        needs_reset: torch.Tensor,
        carry: TRMStyleEncoderCarry,
    ) -> TRMStyleEncoderCarry:
        """Reset z_e_H and z_e_L for halted samples."""
        # Clone to avoid shared storage
        init_H_broadcast = self.z_e_H_init.unsqueeze(0).clone().to(carry.z_e_H.device)
        init_L_broadcast = self.z_e_L_init.unsqueeze(0).clone().to(carry.z_e_L.device)

        z_e_H = torch.where(
            needs_reset.view(-1, 1, 1),
            init_H_broadcast,
            carry.z_e_H,
        )

        z_e_L = torch.where(
            needs_reset.view(-1, 1, 1),
            init_L_broadcast,
            carry.z_e_L,
        )

        return TRMStyleEncoderCarry(z_e_H=z_e_H, z_e_L=z_e_L)

    def _encode_demos(
        self,
        demo_inputs: torch.Tensor,
        demo_labels: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode demos and aggregate into single representation.

        Args:
            demo_inputs: (batch, max_demos, seq_len)
            demo_labels: (batch, max_demos, seq_len)
            demo_mask: (batch, max_demos)

        Returns:
            demo_input: (batch, output_tokens, hidden_size) - aggregated demo representation
        """
        batch_size, max_demos, seq_len = demo_inputs.shape

        # Flatten and encode each demo pair
        demo_inputs_flat = demo_inputs.view(batch_size * max_demos, seq_len)
        demo_labels_flat = demo_labels.view(batch_size * max_demos, seq_len)

        demo_encodings_flat = self.grid_encoder(demo_inputs_flat, demo_labels_flat)
        demo_encodings = demo_encodings_flat.view(batch_size, max_demos, -1)

        # Mask invalid demos
        demo_encodings = demo_encodings * demo_mask.unsqueeze(-1).to(demo_encodings.dtype)

        # Aggregate: mean pool across demos to get (batch, hidden_size)
        # Then broadcast to (batch, output_tokens, hidden_size)
        valid_count = demo_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)  # (batch, 1, 1)
        demo_agg = demo_encodings.sum(dim=1, keepdim=True) / valid_count  # (batch, 1, hidden_size)

        # Broadcast to match output_tokens length
        demo_input = demo_agg.expand(-1, self.config.output_tokens, -1)  # (batch, output_tokens, hidden_size)

        return demo_input

    def forward(
        self,
        carry: TRMStyleEncoderCarry,
        demo_inputs: torch.Tensor,
        demo_labels: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> TRMStyleEncoderCarry:
        """
        TRM-style hierarchical refinement with H/L loops.

        Pattern (same as TRM decoder):
        for H_step in range(H_cycles - 1):  # Without grad for efficiency
            for L_step in range(L_cycles):
                z_e_L = L_level(z_e_L, z_e_H + demo_input)
            z_e_H = L_level(z_e_H, z_e_L)

        # Last H cycle with grad
        for L_step in range(L_cycles):
            z_e_L = L_level(z_e_L, z_e_H + demo_input)
        z_e_H = L_level(z_e_H, z_e_L)

        Args:
            carry: Current carry state (z_e_H, z_e_L)
            demo_inputs: (batch, max_demos, seq_len)
            demo_labels: (batch, max_demos, seq_len)
            demo_mask: (batch, max_demos)

        Returns:
            new_carry: Updated TRMStyleEncoderCarry
        """
        # Encode demos → aggregated representation
        demo_input = self._encode_demos(demo_inputs, demo_labels, demo_mask)

        # Get positional encodings
        cos_sin = self.rotary_emb() if self.rotary_emb is not None else None

        # Current states
        z_e_H, z_e_L = carry.z_e_H, carry.z_e_L

        # H_cycles-1 without grad (for efficiency, like TRM)
        with torch.no_grad():
            for _H_step in range(self.etrmtrm_config.encoder_h_cycles - 1):
                for _L_step in range(self.etrmtrm_config.encoder_l_cycles):
                    z_e_L = self.L_level(z_e_L, z_e_H + demo_input, cos_sin=cos_sin)
                z_e_H = self.L_level(z_e_H, z_e_L, cos_sin=cos_sin)

        # Last H cycle with grad
        for _L_step in range(self.etrmtrm_config.encoder_l_cycles):
            z_e_L = self.L_level(z_e_L, z_e_H + demo_input, cos_sin=cos_sin)
        z_e_H = self.L_level(z_e_H, z_e_L, cos_sin=cos_sin)

        # EMA smoothing: gradual evolution (same as Variant A, but for both H and L)
        # This helps decoder track slowly moving target
        # Smooth with previous state (90% old, 10% new)
        z_e_H_smoothed = 0.9 * carry.z_e_H.detach() + 0.1 * z_e_H
        z_e_L_smoothed = 0.9 * carry.z_e_L.detach() + 0.1 * z_e_L

        # Detach to break computational graph (allows reuse across batches)
        return TRMStyleEncoderCarry(
            z_e_H=z_e_H_smoothed.detach(),
            z_e_L=z_e_L_smoothed.detach()
        )
