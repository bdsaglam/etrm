"""
Demonstration encoders for few-shot learning.

These encoders process demonstration input-output pairs and produce
a context embedding that replaces the puzzle_id embedding in TRM.

Available encoders:

Original architecture (grid encoder + cross-attention set encoder):
- StandardDemoEncoder: Deterministic encoder (default)
- VariationalDemoEncoder: VAE-style encoder with KL regularization

LPN-style architecture (deep transformer + CLS pooling + mean aggregation):
Based on "Searching Latent Program Spaces" paper.
- LPNStandardEncoder: Deeper transformer (8 layers), CLS pooling, simpler aggregation
- LPNVariationalEncoder: LPN-style with per-demo variational bottleneck
- LPNVariationalEncoderV2: LPN-style with aggregate-first variational bottleneck
"""

from models.encoders.base import (
    DemoEncoderConfig,
    BaseDemoEncoder,
    EncoderOutput,
    compute_contrastive_loss,
    create_encoder,
)
from models.encoders.standard import StandardDemoEncoder
from models.encoders.variational import VariationalDemoEncoder
from models.encoders.lpn_standard import LPNStandardEncoder
from models.encoders.lpn_variational import LPNVariationalEncoder, LPNVariationalEncoderV2
from models.encoders.hybrid_variational import HybridVariationalEncoder
from models.encoders.hybrid_standard import HybridStandardEncoder

__all__ = [
    # Base and factory
    "DemoEncoderConfig",
    "BaseDemoEncoder",
    "EncoderOutput",
    "compute_contrastive_loss",
    "create_encoder",
    # Original architecture
    "StandardDemoEncoder",
    "VariationalDemoEncoder",
    # LPN-style architecture
    "LPNStandardEncoder",
    "LPNVariationalEncoder",
    "LPNVariationalEncoderV2",
    # Hybrid architecture
    "HybridStandardEncoder",
    "HybridVariationalEncoder",
]
