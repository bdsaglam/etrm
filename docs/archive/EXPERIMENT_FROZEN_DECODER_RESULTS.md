# Frozen Decoder Experiment - Results

**Date**: 2026-01-21
**Run ID**: `wandb/run-20260121_201601-vb7ojcz9`
**Checkpoint**: `checkpoints/etrm-investigation/frozen-decoder-overfit/`
**Training Steps**: 2500 / 20000 (12.5% complete)

---

## Executive Summary

**Result**: **Scenario C - Encoder Trains But Useless** ⚠️

The encoder successfully learns with frozen decoder (gradients flow, loss decreases, diversity increases), BUT produces zero test accuracy. This suggests the encoder architecture may be too weak to learn meaningful task representations.

---

## Detailed Metrics

### 1. Gradient Flow ✅

| Metric | Value | Status |
|--------|-------|--------|
| `grad/encoder_norm` | 1.000 | ✅ Excellent (target: 0.1-1.0) |
| `grad/total_norm` | 1.000 | ✅ Stable |
| `grad/inner_norm` | 0.000 | ✅ Decoder frozen |

**Conclusion**: Gradients ARE flowing to encoder correctly. No gradient blocking bug.

### 2. Training Loss ✅

| Metric | Value | Status |
|--------|-------|--------|
| `train/lm_loss` | 0.701 | ✅ Very low (started ~8.5) |
| `train/accuracy` | 76.67% | ✅ High |
| `train/exact_accuracy` | 2.73% | ✅ Learning |

**Conclusion**: Encoder is learning something. Training loss decreased dramatically.

### 3. Encoder Output Diversity ✅

| Metric | Value | Status |
|--------|-------|--------|
| `encoder_token_std` | 0.373 | ✅ Good variance (target: >0.1) |
| `encoder_cross_sample_var` | 0.150 | ✅ Diverse across samples |
| `encoder_output_mean` | -0.051 | ✅ Centered |
| `encoder_output_std` | 1.000 | ✅ Normalized |
| `encoder_output_norm` | 22.625 | ✅ Healthy magnitude |

**Conclusion**: Encoder outputs are diverse and well-behaved. Not outputting constant embeddings.

### 4. Test Accuracy ❌

| Metric | Value | Status |
|--------|-------|--------|
| `ARC/pass@1` | 0.0% | ❌ Zero |
| `ARC/pass@2` | 0.0% | ❌ Zero |
| `ARC/pass@5` | 0.0% | ❌ Zero |
| `ARC/pass@10` | 0.0% | ❌ Zero |
| `ARC/pass@100` | 0.0% | ❌ Zero |

**Random baseline**: ~0.25% (1/400)
**Expected if learning**: >2-3%
**Actual**: 0.0%

**Conclusion**: Despite good training metrics, encoder produces ZERO test accuracy.

### 5. Decoder Verification ✅

```bash
python scripts/verify_decoder_loading.py \
    --trm-checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071 \
    --etrm-checkpoint ./checkpoints/etrm-investigation/frozen-decoder-overfit/step_2500
```

**Result**: ✅ 14/14 decoder weights matched (100%)
**Conclusion**: Decoder stayed frozen throughout training.

---

## Scenario Analysis

According to `EXPERIMENT_FROZEN_DECODER.md`, we have three possible scenarios:

### ❌ Scenario A: Encoder Learns
```
✅ grad/encoder_norm: 0.3-1.0 (stable)
✅ train/lm_loss: 8.5 → 0.7 (decreasing)
✅ encoder_token_std: 0.05 → 0.37 (increasing)
❌ ARC/pass@1: 0% (NO improvement)
```
**Does not fully match**: Training looks good, but test accuracy is zero.

### ❌ Scenario B: Encoder Stuck
```
✅ grad/encoder_norm: 1.0 (NOT zero/vanishing)
✅ train/lm_loss: 0.7 (NOT flat)
✅ encoder_token_std: 0.37 (NOT flat)
❌ ARC/pass@1: 0%
```
**Does not match**: Encoder is clearly training, not stuck.

### ✅ Scenario C: Encoder Trains But Useless
```
✅ grad/encoder_norm: 1.0 (stable)
✅ train/lm_loss: 0.7 (decreasing)
✅ encoder_token_std: 0.37 (increasing)
✅ ARC/pass@1: 0% (no improvement) ← MATCHES!
```
**MATCHES**: Encoder learns *something*, but not useful task representations.

---

## Interpretation

### What's Working

1. **Gradient Flow**: No blocking bugs. Encoder receives gradients from frozen decoder.
2. **Optimization**: Encoder parameters are updating correctly.
3. **Training Dynamics**: Loss decreases, accuracy increases, outputs diversify.
4. **Decoder Loading**: Pretrained decoder weights correctly loaded and frozen.

### What's NOT Working

1. **Test Generalization**: Encoder produces zero test accuracy despite good training metrics.
2. **Task Representation Learning**: Encoder not learning useful abstractions from demos.

### Key Insight

The encoder can **overfit to the training distribution** (hence low training loss), but cannot **extract generalizable task rules** from demonstrations (hence zero test accuracy).

This is different from a gradient flow bug—the encoder IS learning, just not learning the right thing.

---

## Root Cause Hypothesis

**Hypothesis**: Encoder architecture is too weak to learn task-level abstractions.

### Evidence

1. ✅ Encoder works mechanically (gradients, optimization, diversity)
2. ❌ Encoder fails to generalize to test set
3. ⚠️ Only 2500 steps (12.5% of planned 20k), but already overfitting

### Why This Makes Sense

- **TRM uses puzzle_id embeddings**: Direct lookup, no abstraction needed
- **ETRM uses encoder**: Must compress demos → task representation
- **Current encoder**: 2 layers, mean pooling, 16 output tokens
- **Task**: May be too complex for this shallow encoder

---

## Next Steps (Per Investigation Doc)

According to `EXPERIMENT_FROZEN_DECODER.md`, for Scenario C:

1. ✅ Check encoder output variance → **DONE: Variance is healthy**
2. ⚠️ Compare to original TRM puzzle_emb statistics → **TODO**
3. ⚠️ Try stronger encoder: `encoder_num_layers=4, encoder_set_layers=2` → **TODO**
4. ⚠️ Check if encoder output range matches puzzle_emb range in TRM → **TODO**

### Recommended Actions

#### Option 1: Evaluate on Full Test Set
- Current eval: Only 4 test groups (~4154 samples)
- Full test set: 400 puzzles
- Reason: Confirm zero accuracy is not due to small sample

```bash
python scripts/evaluate_trm_checkpoint.py \
    --checkpoint ./checkpoints/etrm-investigation/frozen-decoder-overfit/step_2500 \
    --model-type etrm \
    --config-name cfg_test_frozen_decoder \
    --max-eval-groups 64
```

#### Option 2: Continue Training to 20k Steps
- Current: 2500 steps (12.5%)
- Planned: 20000 steps
- Reason: Check if longer training helps generalization

#### Option 3: Try Stronger Encoder Architecture
- Create `cfg_test_frozen_decoder_strong.yaml`:
  ```yaml
  arch:
    encoder_num_layers: 4      # Was: 2
    encoder_set_layers: 2       # Was: 1
    puzzle_emb_len: 32          # Was: 16
  ```
- Run new experiment with stronger encoder

#### Option 4: Investigate Hypothesis 7 (Architecture Too Weak)
- Move to detailed architecture investigation in `docs/etrm-low-performance-investigation.md`
- Compare encoder capacity to task complexity
- Consider alternative encoder designs (cross-attention, recurrent, etc.)

---

## Conclusions

### For Original Investigation (Hypothesis 1-2)

- **Hypothesis 1 (Decoder Loading)**: ✅ VALIDATED - Decoder loads and freezes correctly
- **Hypothesis 2 (Gradient Flow)**: ❌ INVALIDATED - Gradients flow correctly

### For Overall ETRM Performance Issue

The frozen decoder experiment successfully isolated the problem:

1. **NOT a gradient flow bug**: Encoder receives gradients and learns
2. **NOT a decoder loading bug**: Decoder correctly frozen at pretrained weights
3. **LIKELY an architecture issue**: Encoder learns but doesn't generalize

This narrows the investigation to **Hypothesis 7: Encoder Architecture Too Weak**.

---

## Files

- **Config**: `config/cfg_test_frozen_decoder.yaml`
- **Checkpoint**: `checkpoints/etrm-investigation/frozen-decoder-overfit/step_2500`
- **WandB Run**: `wandb/run-20260121_201601-vb7ojcz9`
- **Verification Script**: `scripts/verify_decoder_loading.py`
- **Investigation Doc**: `docs/etrm-low-performance-investigation.md`
