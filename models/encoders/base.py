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
    """Configuration for demonstration encoders."""
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 2
    output_tokens: int = 16  # Number of output context tokens (matches puzzle_emb_len)
    vocab_size: int = 12     # 0=PAD, 1=EOS, 2-11=colors
    seq_len: int = 900       # 30x30 grid flattened
    expansion: float = 4.0   # MLP expansion factor
    rms_norm_eps: float = 1e-5
    forward_dtype: str = "bfloat16"

    # Variational encoder settings
    variational: bool = False  # Use VAE-style encoding
    kl_weight: float = 0.001   # Weight for KL divergence loss (beta-VAE)

    # Contrastive learning settings
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
