# LPN Encoder: Configurable Latent Dimension

**Date**: 2026-01-22
**Status**: ✅ IMPLEMENTED
**Motivation**: Test if lpn_var's LOW cross-sample variance is due to 32-dim bottleneck

---

## Problem

The lpn_var encoder (O4) showed **LOW cross-sample variance** (0.05 vs 0.15 for O1/O3), indicating representation collapse where different tasks get similar encoder outputs.

**Hypothesis 1**: The 32-dimensional latent bottleneck is too small to capture diverse task representations.

**Comparison**:
- O1 (standard): 16 tokens × 512 dims = **8,192 total dims** → high diversity
- O3 (ETRMTRM): 16 tokens × 512 dims = **8,192 total dims** → high diversity
- O4 (lpn_var): **32 dims** → LOW diversity

**Capacity ratio**: 32 / 8,192 = **256× fewer degrees of freedom**

---

## Solution

Made `lpn_latent_dim` **configurable** while keeping the LPN paper architecture intact (2 layers, 128 hidden, mean aggregation, per-demo sampling).

### Changes Made

**1. Added config parameter** (`models/encoders/base.py`):
```python
# === LPN-specific settings ===
lpn_latent_dim: int = 32  # Default: 32 (paper value)
# Can increase to 512 to match other encoders (16 × 512 = 8,192 dims)
```

**2. Updated LPNVariationalEncoder** (`models/encoders/lpn.py`):
```python
# Before (hardcoded):
LPN_LATENT_DIM = 32
self.latent_dim = self.LPN_LATENT_DIM

# After (configurable):
self.latent_dim = config.lpn_latent_dim

# Projections now use self.latent_dim:
self.mu_proj = nn.Linear(128, self.latent_dim, bias=False)
self.logvar_proj = nn.Linear(128, self.latent_dim, bias=False)
self.output_proj = nn.Linear(self.latent_dim, 16 × 512, bias=False)
```

**3. Scheduled two experiments** (`jobs-overfit-corrected.txt`):
- **O4a**: `lpn_latent_dim=32` (original paper value)
- **O4b**: `lpn_latent_dim=512` (16× larger, matches O1/O3 capacity)

---

## What This Tests

### O4a (32 dims) - Paper Original
```bash
arch.encoder_type=lpn_var arch.lpn_latent_dim=32
```

**Expected**:
- `encoder_cross_sample_var`: ~0.05 (LOW, same as before)
- `encoder_within_puzzle_var`: ~0 or >0.01 (diagnoses determinism)
- `ARC/pass@1`: 5-10% (limited by small bottleneck?)

**Purpose**: Baseline with NEW variance metrics to understand what's happening

### O4b (512 dims) - Increased Capacity
```bash
arch.encoder_type=lpn_var arch.lpn_latent_dim=512
```

**Expected**:
- `encoder_cross_sample_var`: >0.1 (HIGH, should match O1/O3)
- `encoder_within_puzzle_var`: >0.01 (stochastic with proper capacity)
- `ARC/pass@1`: 10-15% (improved generalization?)

**Purpose**: Test if 32-dim bottleneck was causing representation collapse

---

## Interpretation Guide

### Case 1: O4b >> O4a (Significant Improvement)
**Conclusion**: ✅ **Hypothesis 1 CONFIRMED** - 32-dim bottleneck was the issue

**Evidence**:
- O4b has high cross-sample variance (>0.1)
- O4b shows better test accuracy (10-15% vs 5-10%)
- Increasing capacity fixes representation collapse

**Action**: Use O4b (512 dims) for staged training (S4)

### Case 2: O4b ≈ O4a (No Improvement)
**Conclusion**: ❌ **Hypothesis 1 REJECTED** - Issue is not capacity

**Possible alternative causes**:
1. kl_weight=0.0001 too low (collapsed to deterministic)
2. Mean aggregation regularizes toward global average (Hypothesis 2)
3. 2-layer architecture too shallow (Hypothesis 3)
4. LPN design not suitable for ARC-AGI tasks (Hypothesis 4 - less likely)

**Action**: Investigate other hypotheses or keep hybrid_variational as best encoder

### Case 3: Both O4a and O4b Poor (<5% test accuracy)
**Conclusion**: LPN architecture itself may not be suitable

**Evidence**: Neither capacity matters, suggesting fundamental design issue

**Action**: Focus on O1 (standard) and O3 (ETRMTRM) for staged training

---

## Why Keep LPN Architecture Intact?

You correctly pointed out:
- **Hypothesis 2, 3, 5 are intentional** (from LPN paper design)
- **hybrid_variational exists** for alternative designs (cross-attention, deeper, etc.)
- **LPN's purpose**: Learn transformation rule representation for gradient-based search

**What we changed**: Only the latent dimension (bottleneck size)
**What we kept**: Everything else from LPN paper
- 2 layers, 128 hidden
- Per-demo sampling, mean aggregation
- LayerNorm, SiLU activation
- CLS token pooling

This isolates the capacity hypothesis while respecting the original LPN design.

---

## Expected Parameter Counts

### O4a (lpn_latent_dim=32):
```
Grid encoder: ~500K params (2 layers, 128 hidden)
Variational: ~4K params (128 → 32 projections)
Output proj: ~260K params (32 → 8192)
Total: ~764K params
```

### O4b (lpn_latent_dim=512):
```
Grid encoder: ~500K params (same)
Variational: ~65K params (128 → 512 projections)
Output proj: ~4.2M params (512 → 8192)
Total: ~4.8M params
```

**Note**: O4b is significantly larger (6× params) but still smaller than hybrid_variational (~8M params with 4 layers).

---

## New Metrics to Watch

Both experiments will report:

**1. Cross-sample variance** (renamed from `cross_sample_var`):
```python
encoder_cross_sample_var = var(context across batch)
```
- Measures diversity across different puzzle groups
- HIGH = good task-specific representations
- LOW = representation collapse

**2. Within-puzzle variance** (NEW):
```python
encoder_within_puzzle_var = var(encoding1, encoding2 of same puzzle)
```
- Measures encoder stochasticity
- ~0 = deterministic (expected for standard encoders)
- >0.01 = truly variational (expected for VAE encoders)
- Computed on 1% of steps (minimal overhead)

See `ENCODER_VARIANCE_METRICS.md` for full details.

---

## Related Files

- Implementation: `models/encoders/lpn.py` (lines 342-395)
- Config: `models/encoders/base.py` (lines 86-91)
- Experiments: `jobs-overfit-corrected.txt` (O4a, O4b)
- Metrics: `ENCODER_VARIANCE_METRICS.md`
- Original analysis: `ANALYSIS_LPN_VAR_LOW_VARIANCE.md` (now outdated)

---

## Key Takeaway

We're testing if lpn_var's representation collapse is due to:
- ✅ **Testable**: Small latent bottleneck (32 dims)
- ⏸️ **Untested**: Intentional design choices (mean agg, shallow depth)

By making latent_dim configurable, we can isolate the capacity hypothesis while respecting the LPN paper's architectural decisions.

Results from O4a vs O4b comparison will definitively answer whether the 32-dim bottleneck was limiting performance.
