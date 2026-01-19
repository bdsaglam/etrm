# ETRMTRM Implementation Plan

## Overview
Implement ETRMTRM - a model with **recurrent encoder + TRM decoder** where both iterate together in an interleaved loop.

## Key Design Decisions
- **Two encoder variants** to implement (user requested both)
- **Interleaved loop**: Each decoder L_step triggers an encoder step
- **Separate networks**: Encoder has its own L_level (no weight sharing with decoder)
- **Parallel path**: New files only, don't modify existing `etrm.py`

---

## Architecture

### Current ETRM (for reference)
```
demos → Encoder (once) → context → [TRM Decoder iterates with context]
```

### New ETRMTRM
```
OUTER LOOP (in pretrain_etrmtrm.py):
    for step in training_loop:
        carry, outputs = model(carry, batch)  # calls _forward_train

_forward_train (ONE ACT step - same structure as etrm.py):
    ┌─────────────────────────────────────────────────────────────────────┐
    │  # ENCODER (now has carry!)                                         │
    │  encoder_carry, context = self.encoder(                             │
    │      carry.encoder_carry,                                           │
    │      demos, demo_mask                                               │
    │  )                                                                  │
    │                                                                     │
    │  # DECODER (unchanged from ETRM)                                    │
    │  inner_carry, logits, q_logits = self.inner(                        │
    │      carry.inner_carry,                                             │
    │      batch,                                                         │
    │      context  # Now evolves each step!                              │
    │  )                                                                  │
    │                                                                     │
    │  # Return combined carry                                            │
    │  return ETRMTRMCarry(encoder_carry, inner_carry, ...), outputs      │
    └─────────────────────────────────────────────────────────────────────┘
```

**Key insight**: The only change is encoder now has carry state.
Everything else (outer loop, halting, loss) stays the same as ETRM.

---

## Files to Create

### 1. `models/encoders/recurrent_base.py`
Base classes for recurrent encoders:
```python
@dataclass
class RecurrentEncoderCarry:
    z_e: torch.Tensor  # (batch, output_tokens, hidden_size)

    def get_context(self) -> torch.Tensor:
        return self.z_e  # z_e IS the context

@dataclass
class TRMStyleEncoderCarry:
    z_e_H: torch.Tensor  # High-level (context)
    z_e_L: torch.Tensor  # Low-level (reasoning)

    def get_context(self) -> torch.Tensor:
        return self.z_e_H  # z_e_H is the context (like z_H in decoder)

class BaseRecurrentEncoder(nn.Module, ABC):
    def initial_carry(batch_size, device) -> carry
    def reset_carry(needs_reset, carry) -> carry
    def forward(carry, demo_inputs, demo_labels, demo_mask) -> new_carry
    # forward() returns ONLY the updated carry. Use carry.get_context() for decoder.
```

### 2. `models/encoders/recurrent_standard.py` (Variant A)
**RecurrentAggregationEncoder**: Like HybridStandard but with carry state
- Keep `DemoGridEncoder` (non-recurrent, encodes each demo pair)
- `z_e` initialized from learnable `z_e_init`
- **NO internal H/L loops** - just ONE pass per ACT step (deterministic)
- z_e evolves across ACT steps via carry
  ```python
  def forward(carry, demo_inputs, demo_labels, demo_mask):
      # Encode demos (recomputed each step for gradient flow)
      demo_encodings = self.grid_encoder(demo_inputs, demo_labels)

      # ONE cross-attention step (no internal loop)
      z_e = carry.z_e
      z_e = z_e + CrossAttn(z_e, demo_encodings)
      z_e = z_e + MLP(z_e)

      return RecurrentEncoderCarry(z_e=z_e)
  ```
- Output: `z_e` (batch, output_tokens, hidden_size)

### 3. `models/encoders/trm_style.py` (Variant B)
**TRMStyleEncoder**: Full TRM-like encoder with z_e_H/z_e_L and H/L loops
- Reuses `DemoGridEncoder` for per-demo encoding
- Has `z_e_H` (high-level context) and `z_e_L` (low-level reasoning) latents
- **HAS internal H/L loops** (TRM architecture) per ACT step:
  ```python
  def forward(carry, demo_inputs, demo_labels, demo_mask):
      # Encode demos
      demo_encodings = self.grid_encoder(demo_inputs, demo_labels)
      demo_input = aggregate(demo_encodings, demo_mask)  # (B, T, D)

      z_e_H, z_e_L = carry.z_e_H, carry.z_e_L

      # TRM-style H/L loops
      for H_step in range(H_cycles):
          for L_step in range(L_cycles):
              z_e_L = L_enc(z_e_L, z_e_H + demo_input)
          z_e_H = L_enc(z_e_H, z_e_L)

      return TRMStyleEncoderCarry(z_e_H=z_e_H, z_e_L=z_e_L)
  ```
- Output: `z_e_H` (batch, output_tokens, hidden_size) via `get_context()`

### 4. `models/recursive_reasoning/etrmtrm.py`
Main model combining recurrent encoder + TRM decoder:
```python
@dataclass
class ETRMTRMCarry:
    encoder_carry: Union[RecurrentEncoderCarry, TRMStyleEncoderCarry]  # NEW!
    inner_carry: TRMEncoderInnerCarry  # z_H, z_L (same as ETRM)
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]

class TRMWithRecurrentEncoder(nn.Module):
    """Full ETRMTRM model - same structure as TRMWithEncoder."""

    def __init__(self, config):
        self.encoder = create_recurrent_encoder(config)  # Variant A or B
        self.inner = TRMEncoderInner(config)  # Same decoder as ETRM

    def _forward_train(self, carry, batch):
        """ONE ACT step - same structure as etrm.py _forward_train."""
        needs_reset = carry.halted
        # ... update current_data for reset samples (same as etrm.py)

        # Reset encoder carry for halted samples
        encoder_carry = self.encoder.reset_carry(needs_reset, carry.encoder_carry)

        # ENCODER (returns only carry, context IS the carry state)
        encoder_carry = self.encoder(
            encoder_carry,
            new_current_data["demo_inputs"],
            new_current_data["demo_labels"],
            new_current_data["demo_mask"],
        )
        context = encoder_carry.get_context()  # z_e or z_e_H depending on variant

        # DECODER (unchanged from ETRM)
        inner_carry = self.inner.reset_carry(needs_reset, carry.inner_carry)
        inner_carry, logits, q_logits = self.inner(inner_carry, new_current_data, context)

        # Halting logic (same as etrm.py)
        # ...

        return ETRMTRMCarry(encoder_carry, inner_carry, steps, halted, current_data), outputs
```

### 5. `config/arch/etrmtrm.yaml`
```yaml
name: recursive_reasoning.etrmtrm@TRMWithRecurrentEncoder
recurrent_encoder_type: recurrent_standard  # or "trm_style"
encoder_l_cycles: 1
encoder_l_layers: 2
# ... rest same as etrm.yaml
```

### 6. `config/cfg_pretrain_etrmtrm_arc_agi_1.yaml`
Training config (parallel to `cfg_pretrain_etrm_arc_agi_1.yaml`)

### 7. `pretrain_etrmtrm.py`
New training script (copy from `pretrain_etrm.py`, adapt for ETRMTRM):
- Import `TRMWithRecurrentEncoder` instead of `TRMWithEncoder`
- Handle combined encoder+decoder carry
- Same training loop structure, different model instantiation

---

## Files to Update (imports only)

- `models/encoders/__init__.py` - Add exports
- `models/encoders/base.py` - Add to `encoder_map` in `create_encoder()`

---

## Implementation Sequence

### Phase 1: Foundation
1. Create `recurrent_base.py` with base classes
2. Create `recurrent_standard.py` (Variant A - simpler, start here)
3. Create `etrmtrm.py` with basic structure

### Phase 2: Integration
4. Implement `TRMWithRecurrentEncoder` (copy from `TRMWithEncoder`, add encoder carry)
5. Update carry dataclass to include `encoder_carry`
6. Add `reset_carry` to encoder (same pattern as `inner.reset_carry`)
7. Create config files (`config/arch/etrmtrm.yaml`, `config/cfg_pretrain_etrmtrm_arc_agi_1.yaml`)
8. Create `pretrain_etrmtrm.py` (copy from `pretrain_etrm.py`, minimal changes)
9. Test training runs

### Phase 3: Variant B
10. Create `trm_style.py` (Variant B)
11. Add config switch between variants
12. Test both variants

### Phase 4: Refinement
13. Add diagnostics/logging
14. Unit tests

---

## Critical Reference Files

| File | Reference For |
|------|---------------|
| `models/recursive_reasoning/etrm.py` | Carry management, ACT loop, forward structure |
| `models/recursive_reasoning/trm.py:196-216` | TRM z_H/z_L loop pattern |
| `models/encoders/standard.py` | DemoGridEncoder to reuse |
| `models/encoders/base.py` | Base class patterns |
| `pretrain_etrm.py` | Training infrastructure compatibility |

---

## Carry State Summary

| Component | Shape | Description |
|-----------|-------|-------------|
| `encoder_carry.z_e` (Variant A) | (B, 16, 512) | Encoder latent state |
| `encoder_carry.z_e_H/L` (Variant B) | (B, 16, 512) each | Encoder high/low level states |
| `inner_carry.z_H/z_L` (decoder) | (B, 916, 512) | Decoder states (unchanged) |
| `steps`, `halted`, `current_data` | - | ACT state (unchanged from ETRM) |

---

## Key Differences from ETRM

| Aspect | ETRM | ETRMTRM |
|--------|------|---------|
| Encoder | Non-recurrent (no carry) | Recurrent (has carry) |
| Context | Static (same each ACT step) | Dynamic (z_e evolves each step) |
| Carry | `inner_carry` only | `encoder_carry` + `inner_carry` |
| `_forward_train` | `context = encoder(demos)` | `carry = encoder(carry, demos)`; `context = carry.get_context()` |

**Variant A vs B:**
| Aspect | Variant A (RecurrentAggregation) | Variant B (TRMStyle) |
|--------|----------------------------------|----------------------|
| Internal loops | NO H/L loops | YES, H/L loops (TRM arch) |
| Latent state | Single `z_e` | Dual `z_e_H` + `z_e_L` |
| Complexity | Simpler, fewer params | More complex, TRM-like |
| Context | `z_e` | `z_e_H` |

**Key design**: `z_e` (or `z_e_H`) IS the context. No separate return value - just extract from carry.
