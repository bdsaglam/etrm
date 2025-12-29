"""
LPN-style variational demonstration encoder.

Based on "Searching Latent Program Spaces" (LPN) encoder architecture with VAE:
- Deep transformer (8 layers by default)
- CLS token for pooling
- Pre-layer normalization (always)
- Mean aggregation across demo pairs
- Variational bottleneck: projects to μ and log(Σ) with diagonal covariance
- Reparameterization trick: z = μ + √Σ * ε

Key differences from our VariationalDemoEncoder:
- Deeper by default (8 layers vs 2)
- CLS token pooling (not mean/attention over sequence)
- No separate set encoder
- Per-demo variational encoding before aggregation (LPN style)
"""

import math
from typing import Union

import torch
from torch import nn
import torch.nn.functional as F

from models.encoders.base import BaseDemoEncoder, DemoEncoderConfig, EncoderOutput
from models.encoders.lpn_standard import LPNGridEncoder
from models.layers import CastedLinear


class LPNVariationalEncoder(BaseDemoEncoder):
    """
    LPN-style variational demonstration encoder.

    Architecture (following LPN paper):
    1. Deep transformer encodes each (input, output) pair with CLS pooling
    2. Project each demo encoding to μ and log(Σ)
    3. Reparameterize: z_i = μ_i + √Σ_i * ε_i for each demo
    4. Mean aggregation across demo latents
    5. Project to output_tokens

    This follows the LPN paper where variational inference is done
    per-demo, then latents are aggregated.
    """

    def __init__(self, config: DemoEncoderConfig):
        super().__init__(config)

        self.grid_encoder = LPNGridEncoder(config)

        # Variational projection heads (per-demo)
        # LPN projects from hidden_size to latent_dim (256 in paper)
        # We keep hidden_size for compatibility
        self.mu_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=True)
        self.logvar_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=True)

        # Output projection (from latent to context tokens)
        self.output_proj = CastedLinear(
            config.hidden_size,
            config.output_tokens * config.hidden_size,
            bias=False
        )

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + √Σ * ε

        Args:
            mu: (batch, dim) or (batch, demos, dim) - mean
            logvar: same shape - log variance

        Returns:
            z: same shape - sampled latent
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL divergence from N(μ, Σ) to N(0, I).

        KL = -0.5 * sum(1 + log(Σ) - μ² - Σ)

        Args:
            mu: (..., dim)
            logvar: (..., dim)

        Returns:
            Scalar KL loss (mean over batch)
        """
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
        Encode demonstrations with variational bottleneck.

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

        # Encode each demo pair
        demo_inputs_flat = demo_inputs.view(batch_size * max_demos, seq_len)
        demo_labels_flat = demo_labels.view(batch_size * max_demos, seq_len)

        demo_encodings_flat = self.grid_encoder(demo_inputs_flat, demo_labels_flat)
        demo_encodings = demo_encodings_flat.view(batch_size, max_demos, -1)  # (B, K, D)

        # Project to μ and log(Σ) for each demo (LPN style: per-demo variational)
        mu = self.mu_proj(demo_encodings)  # (B, K, D)
        logvar = self.logvar_proj(demo_encodings)  # (B, K, D)

        # Reparameterize each demo
        z_demos = self._reparameterize(mu, logvar)  # (B, K, D)

        # Mask invalid demos
        demo_mask_expanded = demo_mask.unsqueeze(-1).to(z_demos.dtype)  # (B, K, 1)
        z_demos = z_demos * demo_mask_expanded
        mu_masked = mu * demo_mask_expanded
        logvar_masked = logvar * demo_mask_expanded

        # Mean aggregation across demos (LPN style)
        num_valid = demo_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)  # (B, 1, 1)
        z_pooled = z_demos.sum(dim=1) / num_valid.squeeze(-1)  # (B, D)

        # Compute KL loss (average over valid demos, then over batch)
        # Note: KL is computed per demo, then averaged
        kl_per_demo = -0.5 * torch.sum(
            1 + logvar_masked - mu_masked.pow(2) - logvar_masked.exp(),
            dim=-1
        )  # (B, K)
        kl_per_demo = kl_per_demo * demo_mask.to(kl_per_demo.dtype)
        kl_loss = kl_per_demo.sum() / demo_mask.sum().clamp(min=1)

        # Project to output tokens
        context = self.output_proj(z_pooled)  # (B, T*D)
        context = context.view(batch_size, self.config.output_tokens, self.config.hidden_size)

        if return_full_output:
            # Return aggregated mu/logvar for logging
            mu_agg = mu_masked.sum(dim=1) / num_valid.squeeze(-1)
            logvar_agg = logvar_masked.sum(dim=1) / num_valid.squeeze(-1)

            return EncoderOutput(
                context=context,
                z_pooled=z_pooled,
                kl_loss=kl_loss,
                mu=mu_agg,
                logvar=logvar_agg,
            )

        return context


class LPNVariationalEncoderV2(BaseDemoEncoder):
    """
    Alternative LPN-style variational encoder.

    Variation: Aggregate first, then apply variational bottleneck.
    This is simpler and may be more stable.

    Architecture:
    1. Deep transformer encodes each (input, output) pair with CLS pooling
    2. Mean aggregation across demo encodings
    3. Project aggregated encoding to μ and log(Σ)
    4. Reparameterize: z = μ + √Σ * ε
    5. Project to output_tokens
    """

    def __init__(self, config: DemoEncoderConfig):
        super().__init__(config)

        self.grid_encoder = LPNGridEncoder(config)

        # Variational projection heads (after aggregation)
        self.mu_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=True)
        self.logvar_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=True)

        # Output projection
        self.output_proj = CastedLinear(
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
        Encode demonstrations with variational bottleneck (aggregate-first style).
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

        # Mean aggregation across demos FIRST
        num_valid = demo_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)  # (B, 1, 1)
        z_agg = demo_encodings.sum(dim=1) / num_valid.squeeze(-1)  # (B, D)

        # THEN apply variational bottleneck
        mu = self.mu_proj(z_agg)  # (B, D)
        logvar = self.logvar_proj(z_agg)  # (B, D)

        # Reparameterize
        z_pooled = self._reparameterize(mu, logvar)  # (B, D)

        # Compute KL loss
        kl_loss = self._compute_kl_loss(mu, logvar)

        # Project to output tokens
        context = self.output_proj(z_pooled)  # (B, T*D)
        context = context.view(batch_size, self.config.output_tokens, self.config.hidden_size)

        if return_full_output:
            return EncoderOutput(
                context=context,
                z_pooled=z_pooled,
                kl_loss=kl_loss,
                mu=mu,
                logvar=logvar,
            )

        return context


def test_lpn_variational_encoder():
    """Test both LPN variational encoder variants."""
    print("=== Testing LPNVariationalEncoder ===\n")

    config = DemoEncoderConfig(
        hidden_size=512,
        num_heads=8,
        num_layers=8,
        output_tokens=16,
        vocab_size=12,
        seq_len=900,
        norm_style="pre",
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

    # Test LPNVariationalEncoder (per-demo variational)
    print("--- LPNVariationalEncoder (per-demo VAE) ---")
    encoder1 = LPNVariationalEncoder(config)
    encoder1.train()

    num_params = sum(p.numel() for p in encoder1.parameters())
    print(f"Encoder parameters: {num_params:,}")

    output1 = encoder1(demo_inputs, demo_labels, demo_mask, return_full_output=True)
    print(f"context: {output1.context.shape}")
    print(f"z_pooled: {output1.z_pooled.shape}")
    print(f"mu: {output1.mu.shape}")
    print(f"logvar: {output1.logvar.shape}")
    print(f"kl_loss: {output1.kl_loss.item():.4f}")
    assert output1.context.shape == (batch_size, config.output_tokens, config.hidden_size)
    assert output1.kl_loss is not None

    # Test LPNVariationalEncoderV2 (aggregate-first)
    print("\n--- LPNVariationalEncoderV2 (aggregate-first VAE) ---")
    encoder2 = LPNVariationalEncoderV2(config)
    encoder2.train()

    num_params = sum(p.numel() for p in encoder2.parameters())
    print(f"Encoder parameters: {num_params:,}")

    output2 = encoder2(demo_inputs, demo_labels, demo_mask, return_full_output=True)
    print(f"context: {output2.context.shape}")
    print(f"z_pooled: {output2.z_pooled.shape}")
    print(f"mu: {output2.mu.shape}")
    print(f"logvar: {output2.logvar.shape}")
    print(f"kl_loss: {output2.kl_loss.item():.4f}")
    assert output2.context.shape == (batch_size, config.output_tokens, config.hidden_size)
    assert output2.kl_loss is not None

    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    test_lpn_variational_encoder()
