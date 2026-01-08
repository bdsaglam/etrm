"""
Hybrid variational demonstration encoder.

Combines best of both architectures:
- LPN's deep grid encoder (8 layers, CLS pooling, pre-norm)
- Standard's cross-attention set aggregation (sees all demos at once)
- Variational bottleneck after aggregation

Why this matters:
- LPN's mean aggregation loses cross-demo relationships
- Standard's shallow encoder (2 layers) limits per-demo representation quality
- Hybrid gets deep per-demo encoding AND cross-demo interaction
"""

from typing import Union

import torch
from torch import nn

from models.encoders.base import BaseDemoEncoder, DemoEncoderConfig, EncoderOutput
from models.encoders.lpn_standard import LPNGridEncoder
from models.encoders.standard import DemoSetEncoder
from models.layers import CastedLinear, rms_norm


class HybridVariationalEncoder(BaseDemoEncoder):
    """
    Hybrid encoder combining LPN's deep grid encoder with cross-attention aggregation.

    Architecture:
    1. LPNGridEncoder: Deep (8L) transformer with CLS pooling per demo
    2. DemoSetEncoder: Cross-attention aggregation (sees all demos)
    3. Variational bottleneck: After aggregation

    Data flow:
        demo1 → [LPN 8L] → z1 ──┐
        demo2 → [LPN 8L] → z2 ──┼─→ [CrossAttn SetEncoder] → context (B, T, D)
        demo3 → [LPN 8L] → z3 ──┘           ↓
                                       pool to (B, D)
                                            ↓
                                       μ, σ projection
                                            ↓
                                       reparameterize z
                                            ↓
                                       decode to (B, T, D)
    """

    def __init__(self, config: DemoEncoderConfig):
        super().__init__(config)

        # Deep grid encoder from LPN (8 layers, CLS pooling, pre-norm)
        self.grid_encoder = LPNGridEncoder(config)

        # Cross-attention set encoder from Standard (sees all demos at once)
        self.set_encoder = DemoSetEncoder(config)

        # Variational bottleneck (after aggregation)
        self.mu_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=True)
        self.logvar_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=True)

        # Initialize logvar_proj to output near-zero initially (prevents NaN from exp overflow)
        nn.init.zeros_(self.logvar_proj.weight)
        nn.init.zeros_(self.logvar_proj.bias)

        # Decode from latent back to output tokens
        self.decode_proj = CastedLinear(
            config.hidden_size,
            config.output_tokens * config.hidden_size,
            bias=False
        )

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + √Σ * ε"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence from N(μ, Σ) to N(0, I)."""
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()

    def forward(
        self,
        demo_inputs: torch.Tensor,
        demo_labels: torch.Tensor,
        demo_mask: torch.Tensor,
        return_full_output: bool = False,
    ) -> Union[torch.Tensor, EncoderOutput]:
        """
        Encode demonstrations with hybrid architecture + variational bottleneck.

        Args:
            demo_inputs: (batch, max_demos, seq_len) - Demo input grids
            demo_labels: (batch, max_demos, seq_len) - Demo output grids
            demo_mask: (batch, max_demos) - True for valid demos
            return_full_output: If True, return EncoderOutput with KL loss

        Returns:
            If return_full_output=False: context tensor (batch, output_tokens, hidden_size)
            If return_full_output=True: EncoderOutput with context, z_pooled, kl_loss, mu, logvar
        """
        batch_size, max_demos, seq_len = demo_inputs.shape

        # 1. Encode each demo with deep LPN encoder (8 layers, CLS pooling)
        demo_inputs_flat = demo_inputs.view(batch_size * max_demos, seq_len)
        demo_labels_flat = demo_labels.view(batch_size * max_demos, seq_len)

        demo_encodings_flat = self.grid_encoder(demo_inputs_flat, demo_labels_flat)
        demo_encodings = demo_encodings_flat.view(batch_size, max_demos, -1)  # (B, K, D)

        # Mask invalid demos
        demo_encodings = demo_encodings * demo_mask.unsqueeze(-1).to(demo_encodings.dtype)

        # 2. Cross-attention aggregation (sees all demos at once)
        context = self.set_encoder(demo_encodings, demo_mask)  # (B, T, D)

        # 3. Variational bottleneck
        # Pool context to single vector
        z_pre = context.mean(dim=1)  # (B, D)

        # Normalize before variational projection (critical for stability)
        # The set encoder has no final norm, so z_pre magnitude can drift
        z_pre = rms_norm(z_pre, variance_epsilon=self.config.rms_norm_eps)

        # Project to μ and log(Σ)
        mu = self.mu_proj(z_pre)  # (B, D)
        logvar = self.logvar_proj(z_pre)  # (B, D)

        # Clamp logvar to prevent numerical instability (exp overflow)
        logvar = torch.clamp(logvar, min=-10, max=10)

        # Reparameterize
        z = self._reparameterize(mu, logvar)  # (B, D)

        # Compute KL loss
        kl_loss = self._compute_kl_loss(mu, logvar)

        # 4. Decode back to output tokens
        context = self.decode_proj(z)  # (B, T*D)
        context = context.view(batch_size, self.config.output_tokens, self.config.hidden_size)

        if return_full_output:
            return EncoderOutput(
                context=context,
                z_pooled=z,
                kl_loss=kl_loss,
                mu=mu,
                logvar=logvar,
            )

        return context


def test_hybrid_variational_encoder():
    """Test the hybrid variational encoder."""
    print("=== Testing HybridVariationalEncoder ===\n")

    config = DemoEncoderConfig(
        hidden_size=512,
        num_heads=8,
        num_layers=8,  # Deep like LPN
        output_tokens=16,
        vocab_size=12,
        seq_len=900,
        norm_style="pre",  # LPN style
        set_encoder_layers=2,  # Cross-attention depth
    )

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

    encoder = HybridVariationalEncoder(config)
    encoder.train()

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {num_params:,}")

    # Forward pass
    output = encoder(demo_inputs, demo_labels, demo_mask, return_full_output=True)
    print(f"context: {output.context.shape}")
    print(f"z_pooled: {output.z_pooled.shape}")
    print(f"mu: {output.mu.shape}")
    print(f"logvar: {output.logvar.shape}")
    print(f"kl_loss: {output.kl_loss.item():.4f}")

    assert output.context.shape == (batch_size, config.output_tokens, config.hidden_size)
    assert output.z_pooled.shape == (batch_size, config.hidden_size)
    assert output.kl_loss is not None

    # Test eval mode (no sampling)
    encoder.eval()
    with torch.no_grad():
        output_eval = encoder(demo_inputs, demo_labels, demo_mask, return_full_output=True)
    print(f"\nEval mode kl_loss: {output_eval.kl_loss.item():.4f}")

    print("\n=== Test passed! ===")


if __name__ == "__main__":
    test_hybrid_variational_encoder()
