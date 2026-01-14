"""
Base classes for recurrent encoders.

Recurrent encoders have a carry state that persists across ACT steps,
allowing the encoder representation to evolve during model iteration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import torch
from torch import nn


@dataclass
class RecurrentEncoderCarry:
    """Carry state for simple recurrent encoder (Variant A)."""

    z_e: torch.Tensor  # (batch, output_tokens, hidden_size)

    def get_context(self) -> torch.Tensor:
        """
        Extract context for decoder.

        Returns:
            context: (batch, output_tokens, hidden_size) - z_e IS the context
        """
        return self.z_e


@dataclass
class TRMStyleEncoderCarry:
    """Carry state for TRM-style encoder (Variant B)."""

    z_e_H: torch.Tensor  # High-level encoder state (context)
    z_e_L: torch.Tensor  # Low-level encoder state (reasoning)

    def get_context(self) -> torch.Tensor:
        """
        Extract context for decoder.

        Returns:
            context: (batch, output_tokens, hidden_size) - z_e_H is the context
        """
        return self.z_e_H


class BaseRecurrentEncoder(nn.Module, ABC):
    """
    Abstract base class for recurrent demonstration encoders.

    Recurrent encoders have carry state that persists across ACT steps.
    Each forward pass:
    1. Takes carry state + demo data
    2. Processes demos (may re-encode for gradient flow)
    3. Updates carry state
    4. Returns new carry (context extracted via carry.get_context())
    """

    @abstractmethod
    def initial_carry(self, batch_size: int, device: torch.device) -> Union[RecurrentEncoderCarry, TRMStyleEncoderCarry]:
        """
        Create initial carry state.

        Args:
            batch_size: Batch size
            device: Device to create carry on

        Returns:
            Initial carry state
        """
        pass

    @abstractmethod
    def reset_carry(
        self,
        needs_reset: torch.Tensor,
        carry: Union[RecurrentEncoderCarry, TRMStyleEncoderCarry]
    ) -> Union[RecurrentEncoderCarry, TRMStyleEncoderCarry]:
        """
        Reset carry state for halted samples.

        Args:
            needs_reset: (batch,) bool tensor - True for samples to reset
            carry: Current carry state

        Returns:
            Updated carry with reset samples reinitialized
        """
        pass

    @abstractmethod
    def forward(
        self,
        carry: Union[RecurrentEncoderCarry, TRMStyleEncoderCarry],
        demo_inputs: torch.Tensor,
        demo_labels: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> Union[RecurrentEncoderCarry, TRMStyleEncoderCarry]:
        """
        Single forward pass with carry state.

        Args:
            carry: Current carry state
            demo_inputs: (batch, max_demos, seq_len) - Demo input grids
            demo_labels: (batch, max_demos, seq_len) - Demo output grids
            demo_mask: (batch, max_demos) - True for valid demos

        Returns:
            new_carry: Updated carry state

        Note:
            Use carry.get_context() to extract context for decoder.
        """
        pass
