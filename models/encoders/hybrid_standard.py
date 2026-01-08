"""
Hybrid standard (non-variational) demonstration encoder.

Combines best of both architectures without VAE bottleneck:
- LPN's deep grid encoder (configurable layers, CLS pooling, pre-norm)
- Standard's cross-attention set aggregation (sees all demos at once)

This is the deterministic version of HybridVariationalEncoder for ablation studies.

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


class HybridStandardEncoder(BaseDemoEncoder):
    """
    Hybrid encoder combining LPN's deep grid encoder with cross-attention aggregation.

    Architecture:
    1. LPNGridEncoder: Deep transformer with CLS pooling per demo
    2. DemoSetEncoder: Cross-attention aggregation (sees all demos)
    3. Direct output (no variational bottleneck)

    Data flow:
        demo1 → [LPN] → z1 ──┐
        demo2 → [LPN] → z2 ──┼─→ [CrossAttn SetEncoder] → context (B, T, D)
        demo3 → [LPN] → z3 ──┘

    This is simpler than HybridVariationalEncoder - no VAE, no decode_proj.
    The set encoder directly produces the output context.
    """

    def __init__(self, config: DemoEncoderConfig):
        super().__init__(config)

        # Deep grid encoder from LPN (CLS pooling, pre-norm)
        self.grid_encoder = LPNGridEncoder(config)

        # Cross-attention set encoder from Standard (sees all demos at once)
        self.set_encoder = DemoSetEncoder(config)

    def forward(
        self,
        demo_inputs: torch.Tensor,
        demo_labels: torch.Tensor,
        demo_mask: torch.Tensor,
        return_full_output: bool = False,
    ) -> Union[torch.Tensor, EncoderOutput]:
        """
        Encode demonstrations with hybrid architecture (no variational bottleneck).

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

        # 1. Encode each demo with deep LPN encoder (CLS pooling)
        demo_inputs_flat = demo_inputs.view(batch_size * max_demos, seq_len)
        demo_labels_flat = demo_labels.view(batch_size * max_demos, seq_len)

        demo_encodings_flat = self.grid_encoder(demo_inputs_flat, demo_labels_flat)
        demo_encodings = demo_encodings_flat.view(batch_size, max_demos, -1)  # (B, K, D)

        # Mask invalid demos
        demo_encodings = demo_encodings * demo_mask.unsqueeze(-1).to(demo_encodings.dtype)

        # 2. Cross-attention aggregation (sees all demos at once)
        context = self.set_encoder(demo_encodings, demo_mask)  # (B, T, D)

        if return_full_output:
            # Pooled representation for contrastive/diversity metrics
            z_pooled = context.mean(dim=1)  # (B, D)
            return EncoderOutput(
                context=context,
                z_pooled=z_pooled,
                kl_loss=None,
                mu=None,
                logvar=None,
            )

        return context


def test_hybrid_standard_encoder():
    """Test the hybrid standard encoder."""
    print("=== Testing HybridStandardEncoder ===\n")

    config = DemoEncoderConfig(
        hidden_size=512,
        num_heads=8,
        num_layers=4,  # Configurable depth
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

    encoder = HybridStandardEncoder(config)
    encoder.eval()

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {num_params:,}")

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
    assert output.kl_loss is None  # Standard encoder has no KL loss

    print("\n=== Test passed! ===")


if __name__ == "__main__":
    test_hybrid_standard_encoder()
