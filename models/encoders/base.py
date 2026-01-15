"""
Base class for demonstration encoders.
"""

from typing import Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from pydantic import BaseModel


class DemoEncoderConfig(BaseModel):
    """Configuration for demonstration encoders.

    Two encoder architecture families are available:

    1. Original (StandardDemoEncoder, VariationalDemoEncoder):
       - Grid encoder: shallow transformer (2 layers default)
       - Set encoder: cross-attention aggregation
       - Configurable pooling: mean, attention, or weighted

    2. LPN-style (LPNStandardEncoder, LPNVariationalEncoder, LPNVariationalEncoderV2):
       - Based on "Searching Latent Program Spaces" paper
       - Deep transformer (8 layers default)
       - CLS token pooling
       - Simple mean aggregation (no cross-attention set encoder)

    Use `encoder_type` to select the architecture.
    """

    # === Encoder type selection ===
    # "standard": Original architecture with cross-attention set encoder
    # "variational": Original with VAE bottleneck
    # "lpn_standard": LPN-style deep encoder with CLS pooling
    # "lpn_variational": LPN-style with per-demo VAE (LPN paper style)
    # "lpn_variational_v2": LPN-style with aggregate-first VAE (simpler)
    encoder_type: str = "standard"

    # === Common settings ===
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 2        # Layers in grid encoder (2 for original, 8 for LPN)
    output_tokens: int = 16    # Number of output context tokens (matches puzzle_emb_len)
    vocab_size: int = 12       # 0=PAD, 1=EOS, 2-11=colors
    seq_len: int = 900         # 30x30 grid flattened
    expansion: float = 4.0     # MLP expansion factor
    rms_norm_eps: float = 1e-5
    forward_dtype: str = "bfloat16"

    # === Architecture improvements (best practices) ===

    # Pooling method for grid encoder (original architecture only)
    # "mean": simple mean pooling (loses positional info) - default for backward compat
    # "attention": learned query cross-attends to sequence (preserves position)
    # "weighted": attention-weighted mean (lightweight alternative)
    # Note: LPN encoders always use CLS pooling, this is ignored
    pooling_method: str = "mean"

    # Set encoder depth (cross-attention layers for aggregating demos)
    # Original: 1, recommended: 2-3 for complex patterns
    # Note: LPN encoders use mean aggregation, this is ignored
    set_encoder_layers: int = 1

    # Layer scale (from CaiT/DeiT) - multiply residual by learnable scalar
    # Starts small (init_value) and grows during training. Improves stability.
    # 0 = disabled, typical value: 1e-4
    layer_scale_init: float = 0.0

    # Pre-norm vs post-norm (pre-norm is more stable)
    # "pre": x = x + attn(norm(x))  -- GPT-2+, LLaMA style
    # "post": x = norm(x + attn(x)) -- original transformer
    # Note: LPN encoders always use pre-norm, this is ignored
    norm_style: str = "post"  # "post" for backward compat

    # QK normalization (prevents attention logit explosion)
    # Used in ViT-22B, Gemma. Normalizes Q and K before dot product.
    qk_norm: bool = False

    # === Variational encoder settings ===
    variational: bool = False  # Use VAE-style encoding (deprecated, use encoder_type)
    kl_weight: float = 0.001   # Weight for KL divergence loss (beta-VAE)

    # === Contrastive learning settings ===
    contrastive: bool = False  # Use contrastive loss
    contrastive_weight: float = 0.1  # Weight for contrastive loss
    contrastive_temperature: float = 0.1  # Temperature for InfoNCE


@dataclass
class EncoderOutput:
    """Output from demo encoder with optional auxiliary information."""
    context: torch.Tensor  # (batch, output_tokens, hidden_size) - main context embedding
    z_pooled: Optional[torch.Tensor] = None  # (batch, hidden_size) - pooled representation for contrastive
    kl_loss: Optional[torch.Tensor] = None  # KL divergence loss (for variational)
    mu: Optional[torch.Tensor] = None  # Mean (for variational)
    logvar: Optional[torch.Tensor] = None  # Log variance (for variational)


def compute_contrastive_loss(
    z_pooled: torch.Tensor,
    temperature: float = 0.1,
    group_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute contrastive loss (InfoNCE or Supervised Contrastive).

    If group_labels is None: standard InfoNCE where each sample is its own class.
    If group_labels is provided: supervised contrastive where samples with same
    group label are positive pairs (important when batch contains augmentations
    of the same base puzzle).

    Args:
        z_pooled: (batch, hidden_size) - pooled encoder representations
        temperature: softmax temperature
        group_labels: (batch,) - group IDs for supervised contrastive (optional)

    Returns:
        Scalar contrastive loss
    """
    batch_size = z_pooled.shape[0]
    device = z_pooled.device

    # Normalize
    z_norm = F.normalize(z_pooled, dim=-1)

    # Compute similarity matrix
    sim = z_norm @ z_norm.T  # (batch, batch)

    # Scale by temperature
    logits = sim / temperature

    if group_labels is None:
        # Standard InfoNCE: each sample is its own class
        labels = torch.arange(batch_size, device=device)
        loss = F.cross_entropy(logits, labels)
    else:
        # Supervised contrastive: samples with same group are positives
        # Create mask of positive pairs (same group, excluding self)
        group_labels = group_labels.view(-1, 1)
        positive_mask = (group_labels == group_labels.T).float()  # (B, B)
        self_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - self_mask  # Exclude self from positives

        # For numerical stability, subtract max logits
        logits_max = logits.max(dim=1, keepdim=True)[0].detach()
        logits = logits - logits_max

        # Compute log_softmax over all negatives (all samples except self)
        exp_logits = torch.exp(logits)
        # Denominator: sum over all samples except self
        neg_mask = 1.0 - self_mask
        log_prob = logits - torch.log((exp_logits * neg_mask).sum(dim=1, keepdim=True) + 1e-8)

        # Average log probability over positive pairs
        # For samples with no positives (unique group), this will be 0
        num_positives = positive_mask.sum(dim=1).clamp(min=1)  # Avoid div by zero
        loss = -(positive_mask * log_prob).sum(dim=1) / num_positives

        # Average over batch, but only count samples that have positives
        has_positives = (positive_mask.sum(dim=1) > 0).float()
        if has_positives.sum() > 0:
            loss = (loss * has_positives).sum() / has_positives.sum().clamp(min=1)
        else:
            # No positives in batch - fall back to standard InfoNCE
            labels = torch.arange(batch_size, device=device)
            loss = F.cross_entropy(logits + logits_max, labels)

    return loss


class BaseDemoEncoder(nn.Module, ABC):
    """
    Base class for demonstration encoders.

    All encoders take demonstration input-output pairs and produce
    a context embedding of shape (batch, output_tokens, hidden_size).
    """

    def __init__(self, config: DemoEncoderConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

    @abstractmethod
    def forward(
        self,
        demo_inputs: torch.Tensor,
        demo_labels: torch.Tensor,
        demo_mask: torch.Tensor,
        return_full_output: bool = False,
    ) -> torch.Tensor | EncoderOutput:
        """
        Encode demonstration pairs into context embedding.

        Args:
            demo_inputs: (batch, max_demos, seq_len) - Demo input grids
            demo_labels: (batch, max_demos, seq_len) - Demo output grids
            demo_mask: (batch, max_demos) - True for valid demos
            return_full_output: If True, return EncoderOutput with auxiliary info

        Returns:
            If return_full_output=False: context tensor (batch, output_tokens, hidden_size)
            If return_full_output=True: EncoderOutput with context and auxiliary info
        """
        pass

    def encode(
        self,
        demo_inputs: torch.Tensor,
        demo_labels: torch.Tensor,
        demo_mask: torch.Tensor,
        return_full_output: bool = False,
    ) -> torch.Tensor | EncoderOutput:
        """Alias for forward() for clarity."""
        return self.forward(demo_inputs, demo_labels, demo_mask, return_full_output)


def create_encoder(config: DemoEncoderConfig) -> BaseDemoEncoder:
    """
    Factory function to create an encoder based on config.encoder_type.

    Args:
        config: DemoEncoderConfig with encoder_type set

    Returns:
        Encoder instance of the appropriate type

    Raises:
        ValueError: If encoder_type is not recognized
    """
    # Import here to avoid circular imports
    from models.encoders.standard import StandardDemoEncoder
    from models.encoders.variational import VariationalDemoEncoder
    from models.encoders.lpn_standard import LPNStandardEncoder
    from models.encoders.lpn_variational import LPNVariationalEncoder, LPNVariationalEncoderV2
    from models.encoders.hybrid_variational import HybridVariationalEncoder
    from models.encoders.hybrid_standard import HybridStandardEncoder
    from models.encoders.lpn import LPNEncoder, LPNVariationalEncoder as LPNPaperVariationalEncoder

    encoder_map = {
        "standard": StandardDemoEncoder,
        "variational": VariationalDemoEncoder,
        "lpn_standard": LPNStandardEncoder,
        "lpn_variational": LPNVariationalEncoder,
        "lpn_variational_v2": LPNVariationalEncoderV2,
        "hybrid_standard": HybridStandardEncoder,
        "hybrid_variational": HybridVariationalEncoder,
        # LPN paper-matching encoders (2 layers, 128 hidden, LayerNorm, SiLU)
        "lpn": LPNEncoder,
        "lpn_var": LPNPaperVariationalEncoder,
    }

    encoder_type = config.encoder_type.lower()

    # Handle legacy 'variational' flag for backward compatibility
    if config.variational and encoder_type == "standard":
        encoder_type = "variational"

    if encoder_type not in encoder_map:
        valid_types = ", ".join(encoder_map.keys())
        raise ValueError(
            f"Unknown encoder_type: '{config.encoder_type}'. "
            f"Valid types: {valid_types}"
        )

    return encoder_map[encoder_type](config)
