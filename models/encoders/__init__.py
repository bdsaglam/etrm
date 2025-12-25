"""
Demonstration encoders for few-shot learning.

These encoders process demonstration input-output pairs and produce
a context embedding that replaces the puzzle_id embedding in TRM.

Available encoders:
- StandardDemoEncoder: Deterministic encoder (default)
- VariationalDemoEncoder: VAE-style encoder with KL regularization
"""

from models.encoders.base import (
    DemoEncoderConfig,
    BaseDemoEncoder,
    EncoderOutput,
    compute_contrastive_loss,
)
from models.encoders.standard import StandardDemoEncoder
from models.encoders.variational import VariationalDemoEncoder

__all__ = [
    "DemoEncoderConfig",
    "BaseDemoEncoder",
    "EncoderOutput",
    "compute_contrastive_loss",
    "StandardDemoEncoder",
    "VariationalDemoEncoder",
]
