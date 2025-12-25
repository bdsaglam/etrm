"""
Variational demonstration encoder (VAE-style).

Extends the standard encoder with a variational bottleneck:
1. Standard encoder produces context
2. Variational layer projects to mu/logvar
3. Reparameterization trick samples z
4. Decoder projects back to context tokens

KL divergence loss regularizes toward N(0, I).
"""

from typing import Union

import torch
from torch import nn

from models.encoders.base import BaseDemoEncoder, DemoEncoderConfig, EncoderOutput
from models.encoders.standard import DemoGridEncoder, DemoSetEncoder
from models.layers import CastedLinear


class VariationalDemoEncoder(BaseDemoEncoder):
    """
    Variational demonstration encoder with KL regularization.

    Architecture:
    1. DemoGridEncoder: Encodes each (input, output) pair
    2. DemoSetEncoder: Aggregates via cross-attention
    3. Variational bottleneck: context -> mu, logvar -> sample z -> decode to context

    Output shape: (batch, output_tokens, hidden_size)
    """

    def __init__(self, config: DemoEncoderConfig):
        super().__init__(config)

        # Standard encoder components
        self.grid_encoder = DemoGridEncoder(config)
        self.set_encoder = DemoSetEncoder(config)

        # Variational bottleneck
        # Project from mean-pooled context to mu and logvar
        self.mu_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=True)
        self.logvar_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=True)

        # Decode z back to context tokens
        self.decode_proj = CastedLinear(
            config.hidden_size,
            config.output_tokens * config.hidden_size,
            bias=True
        )

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence from N(mu, sigma^2) to N(0, I)."""
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
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
        demo_encodings = demo_encodings_flat.view(batch_size, max_demos, -1)

        # Zero out invalid demos
        demo_encodings = demo_encodings * demo_mask.unsqueeze(-1).to(demo_encodings.dtype)

        # Aggregate via cross-attention
        context = self.set_encoder(demo_encodings, demo_mask)  # (B, T, D)

        # Variational bottleneck
        # Pool context to single vector
        z_pre = context.mean(dim=1)  # (B, D)

        # Project to mu and logvar
        mu = self.mu_proj(z_pre)  # (B, D)
        logvar = self.logvar_proj(z_pre)  # (B, D)

        # Reparameterize
        z = self._reparameterize(mu, logvar)  # (B, D)

        # Decode to context tokens
        context = self.decode_proj(z)  # (B, T*D)
        context = context.view(batch_size, self.config.output_tokens, self.config.hidden_size)

        # Compute KL loss
        kl_loss = self._compute_kl_loss(mu, logvar)

        if return_full_output:
            return EncoderOutput(
                context=context,
                z_pooled=z,
                kl_loss=kl_loss,
                mu=mu,
                logvar=logvar,
            )

        return context


def test_variational_encoder():
    """Test the variational encoder."""
    print("=== Testing VariationalDemoEncoder ===\n")

    config = DemoEncoderConfig(
        hidden_size=512,
        num_heads=8,
        num_layers=2,
        output_tokens=16,
        vocab_size=12,
        seq_len=900,
    )

    encoder = VariationalDemoEncoder(config)
    encoder.train()

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
    context = encoder(demo_inputs, demo_labels, demo_mask)
    print(f"Context shape: {context.shape}")
    assert context.shape == (batch_size, config.output_tokens, config.hidden_size)

    # Forward pass (full output)
    output = encoder(demo_inputs, demo_labels, demo_mask, return_full_output=True)
    print(f"\nFull output:")
    print(f"  context: {output.context.shape}")
    print(f"  z_pooled: {output.z_pooled.shape}")
    print(f"  mu: {output.mu.shape}")
    print(f"  logvar: {output.logvar.shape}")
    print(f"  kl_loss: {output.kl_loss.item():.4f}")

    assert output.kl_loss is not None
    assert output.mu is not None
    assert output.logvar is not None

    print("\n=== Test passed! ===")


if __name__ == "__main__":
    test_variational_encoder()
