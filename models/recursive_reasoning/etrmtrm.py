"""
ETRMTRM - Encoder-based TRM with Recurrent Encoder.

This model combines a recurrent encoder with the TRM decoder.
The encoder has carry state that evolves across ACT steps, allowing
the encoder representation to adapt alongside the decoder's reasoning.

Key difference from ETRM:
- ETRM: Static encoder (no carry), context computed once and reused
- ETRMTRM: Recurrent encoder (has carry), context evolves each ACT step

The encoder is called every ACT step and returns updated carry state.
Context is extracted from the carry via carry.get_context().
"""

from typing import Tuple, Dict, Optional, Union
from dataclasses import dataclass
import torch
from torch import nn
from pydantic import BaseModel

from models.recursive_reasoning.etrm import (
    TRMEncoderInner,
    TRMEncoderInnerCarry,
    TRMEncoderConfig,
)
from models.encoders import DemoEncoderConfig
from models.encoders.recurrent_base import (
    BaseRecurrentEncoder,
    RecurrentEncoderCarry,
    TRMStyleEncoderCarry,
)
from models.encoders.recurrent_standard import RecurrentAggregationEncoder


@dataclass
class ETRMTRMCarry:
    """Carry state for ETRMTRM (encoder + decoder)."""
    encoder_carry: Union[RecurrentEncoderCarry, TRMStyleEncoderCarry]
    inner_carry: TRMEncoderInnerCarry  # z_H, z_L (same as ETRM)
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class ETRMTRMConfig(TRMEncoderConfig):
    """Configuration for ETRMTRM (extends TRMEncoderConfig)."""
    # Recurrent encoder settings
    recurrent_encoder_type: str = "recurrent_standard"  # "recurrent_standard" or "trm_style"

    # For TRM-style encoder (Variant B)
    encoder_h_cycles: int = 3
    encoder_l_cycles: int = 1
    encoder_l_layers: int = 2


def create_recurrent_encoder(config: ETRMTRMConfig) -> BaseRecurrentEncoder:
    """Create recurrent encoder based on config."""
    encoder_config = DemoEncoderConfig(
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_layers=config.encoder_num_layers,
        output_tokens=config.puzzle_emb_len,
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        expansion=config.expansion,
        rms_norm_eps=config.rms_norm_eps,
        forward_dtype=config.forward_dtype,
        # Architecture improvements
        pooling_method=config.encoder_pooling_method,
        set_encoder_layers=config.encoder_set_layers,
        layer_scale_init=config.encoder_layer_scale_init,
        norm_style=config.encoder_norm_style,
        qk_norm=config.encoder_qk_norm,
    )

    if config.recurrent_encoder_type == "recurrent_standard":
        return RecurrentAggregationEncoder(encoder_config)
    elif config.recurrent_encoder_type == "trm_style":
        # Import here to avoid circular dependency
        from models.encoders.trm_style import TRMStyleEncoder
        return TRMStyleEncoder(encoder_config, config)
    else:
        raise ValueError(f"Unknown recurrent_encoder_type: {config.recurrent_encoder_type}")


class TRMWithRecurrentEncoder(nn.Module):
    """
    TRM with recurrent demonstration encoder.

    Similar structure to TRMWithEncoder but encoder has carry state
    that evolves across ACT steps.

    Architecture:
    - Recurrent encoder (Variant A or B) with carry state
    - TRM decoder (same as ETRM)
    - Encoder called every ACT step, context evolves
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = ETRMTRMConfig(**config_dict)

        # Create recurrent encoder
        self.encoder = create_recurrent_encoder(self.config)

        # Freeze encoder if requested
        if self.config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Create TRM inner model (same as ETRM)
        self.inner = TRMEncoderInner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> ETRMTRMCarry:
        """Initialize carry state (encoder + decoder)."""
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return ETRMTRMCarry(
            encoder_carry=self.encoder.initial_carry(batch_size, device),
            inner_carry=self.inner.empty_carry(batch_size, device=device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),  # Start halted
            current_data={
                k: torch.empty_like(v)
                for k, v in batch.items()
                if k in ["inputs", "labels", "puzzle_identifiers",
                         "demo_inputs", "demo_labels", "demo_mask"]
            },
        )

    def forward(
        self, carry: Optional[ETRMTRMCarry], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Optional[ETRMTRMCarry], Dict[str, torch.Tensor]]:
        """
        Forward pass with recurrent encoder.

        Training mode:
            - Each forward call = ONE ACT step
            - Encoder and decoder both update their carry states
            - Context evolves each step

        Eval mode:
            - Adaptive halting with carry continuation
        """
        if self.training:
            return self._forward_train(carry, batch)
        else:
            assert carry is not None, "Carry required for eval mode"
            return self._forward_eval_step(carry, batch)

    def _forward_train(
        self, carry: ETRMTRMCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[ETRMTRMCarry, Dict[str, torch.Tensor]]:
        """
        Training forward - ONE ACT step.

        Key difference from ETRM:
        - Encoder is recurrent, has carry state
        - Encoder called every step, context evolves

        Args:
            carry: Carry from previous batch
            batch: Current batch data

        Returns:
            new_carry: Updated carry (encoder + decoder)
            outputs: Single-step outputs
        """
        # Determine which samples need reset (were halted last batch)
        needs_reset = carry.halted

        # Update current_data for reset samples
        new_current_data = {
            k: torch.where(
                needs_reset.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
            if k in batch
        }

        # Track encoder diagnostics
        encoder_diagnostics = {}

        # Reset encoder carry for halted samples
        encoder_carry = self.encoder.reset_carry(needs_reset, carry.encoder_carry)

        # ENCODER FORWARD (recurrent - returns updated carry)
        encoder_carry = self.encoder(
            encoder_carry,
            new_current_data["demo_inputs"],
            new_current_data["demo_labels"],
            new_current_data["demo_mask"],
        )

        # Extract context from encoder carry
        context = encoder_carry.get_context()

        # Compute encoder diagnostics
        with torch.no_grad():
            encoder_diagnostics["encoder_output_mean"] = context.mean().item()
            encoder_diagnostics["encoder_output_std"] = context.std().item()
            encoder_diagnostics["encoder_output_norm"] = context.norm(dim=-1).mean().item()
            batch_mean = context.mean(dim=0, keepdim=True)
            cross_sample_var = ((context - batch_mean) ** 2).mean()
            encoder_diagnostics["encoder_cross_sample_var"] = cross_sample_var.item()
            encoder_diagnostics["encoder_token_std"] = context.std(dim=0).mean().item()

        # Reset inner carry for halted samples
        new_inner_carry = self.inner.reset_carry(needs_reset, carry.inner_carry)
        new_steps = torch.where(needs_reset, 0, carry.steps)

        # DECODER FORWARD (same as ETRM)
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data, context
        )

        # Build outputs
        outputs = {
            "logits": logits,
            "labels": new_current_data["labels"],
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            **encoder_diagnostics,
        }

        # Halting logic (same as ETRM)
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # Dynamic halting during training
            if self.config.halt_max_steps > 1:
                # Halt when Q-head says stop
                halted = halted | (q_halt_logits > 0)

                # Exploration: random minimum steps before allowing halt
                exploration_mask = (
                    torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                )
                min_halt_steps = exploration_mask * torch.randint_like(
                    new_steps, low=2, high=self.config.halt_max_steps + 1
                )
                halted = halted & (new_steps >= min_halt_steps)

        # Track steps for metrics
        outputs["steps"] = new_steps.float()

        # Return new carry (encoder + decoder)
        new_carry = ETRMTRMCarry(
            encoder_carry=encoder_carry,
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data,
        )

        return new_carry, outputs

    def _forward_eval_step(
        self, carry: ETRMTRMCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[ETRMTRMCarry, Dict[str, torch.Tensor]]:
        """
        Eval forward with adaptive halting.

        Same structure as _forward_train but without exploration.
        """
        needs_reset = carry.halted

        # Update current_data for reset samples
        new_current_data = {
            k: torch.where(
                needs_reset.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
            if k in batch
        }

        # Reset encoder carry for halted samples
        encoder_carry = self.encoder.reset_carry(needs_reset, carry.encoder_carry)

        # ENCODER FORWARD
        encoder_carry = self.encoder(
            encoder_carry,
            new_current_data["demo_inputs"],
            new_current_data["demo_labels"],
            new_current_data["demo_mask"],
        )
        context = encoder_carry.get_context()

        # Compute encoder diagnostics
        encoder_diagnostics = {}
        with torch.no_grad():
            encoder_diagnostics["encoder_output_mean"] = context.mean().item()
            encoder_diagnostics["encoder_output_std"] = context.std().item()
            encoder_diagnostics["encoder_output_norm"] = context.norm(dim=-1).mean().item()
            batch_mean = context.mean(dim=0, keepdim=True)
            cross_sample_var = ((context - batch_mean) ** 2).mean()
            encoder_diagnostics["encoder_cross_sample_var"] = cross_sample_var.item()
            encoder_diagnostics["encoder_token_std"] = context.std(dim=0).mean().item()

        # Reset inner carry for halted samples
        new_inner_carry = self.inner.reset_carry(needs_reset, carry.inner_carry)
        new_steps = torch.where(needs_reset, 0, carry.steps)

        # DECODER FORWARD
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data, context
        )

        # Halting logic (no exploration in eval)
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step | (q_halt_logits > 0)

        # Build outputs
        outputs = {
            "logits": logits,
            "labels": new_current_data["labels"],
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "steps": new_steps.float(),
            **encoder_diagnostics,
        }

        # Return new carry
        new_carry = ETRMTRMCarry(
            encoder_carry=encoder_carry,
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data,
        )

        return new_carry, outputs
