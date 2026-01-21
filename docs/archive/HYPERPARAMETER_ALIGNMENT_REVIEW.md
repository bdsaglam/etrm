# Hyperparameter Alignment Review

**Date**: 2026-01-21
**Issue**: ETRM/ETRMTRM configs using different hyperparameters than official TRM

---

## Critical Misalignment Found

### 1. `halt_exploration_prob` (CRITICAL)

| Config | Value | Status |
|--------|-------|--------|
| **Official TRM** | **0.1** | ✅ Reference |
| `cfg_pretrain_arc_agi_1.yaml` | (uses arch default) | ✅ Inherits 0.1 from arch/trm.yaml |
| `arch/trm.yaml` | **0.1** | ✅ Correct |
| **`cfg_pretrain_etrm_arc_agi_1.yaml`** | **0.5** | ❌ WRONG (5x too high!) |
| **`arch/etrm.yaml`** | **0.5** | ❌ WRONG (5x too high!) |
| **`cfg_pretrain_etrmtrm_arc_agi_1.yaml`** | **0.5** | ❌ WRONG (5x too high!) |
| **`arch/etrmtrm.yaml`** | **0.5** | ❌ WRONG (5x too high!) |

**Impact**: This is CRITICAL. Exploration probability controls Q-head training during ACT halting.
- **TRM uses 0.1**: Model explores (ignores Q-head prediction) 10% of the time
- **ETRM/ETRMTRM use 0.5**: Model explores 50% of the time → Q-head learns much slower!

This could explain poor performance - the Q-head never learns to predict correctness properly.

---

## Other Hyperparameter Comparison

### 2. `L_cycles` (Architecture)

| Config | Value | Status |
|--------|-------|--------|
| **Official TRM** | **4** | ✅ Reference (trained with this) |
| `cfg_pretrain_arc_agi_1.yaml` | **6** | ⚠️ Different but intentional? |
| `cfg_pretrain_etrm_arc_agi_1.yaml` | **6** | ⚠️ Different |
| `cfg_pretrain_etrmtrm_arc_agi_1.yaml` | **6** | ⚠️ Different |
| `cfg_test_frozen_decoder_corrected.yaml` | **4** | ✅ Matches pretrained decoder |

**Impact**: More reasoning cycles = more capacity. But if loading pretrained decoder trained with L_cycles=4, this creates architectural mismatch.

**Recommendation**:
- **When using `load_pretrained_decoder`**: MUST use `L_cycles: 4` to match decoder's training
- **When training from scratch**: Can use L_cycles=6 (more capacity)
- **Decision needed**: Should base ETRM configs default to L_cycles=4 for easier transfer learning?

### 3. `global_batch_size`

| Config | Value | Status |
|--------|-------|--------|
| **Official TRM** | **768** | ✅ Reference |
| `cfg_pretrain_arc_agi_1.yaml` | **768** | ✅ Correct |
| `cfg_pretrain_etrm_arc_agi_1.yaml` | **256** | ⚠️ 3x smaller |
| `cfg_pretrain_etrmtrm_arc_agi_1.yaml` | **256** | ⚠️ 3x smaller |

**Impact**: Smaller batch size can hurt training stability and gradient estimates. ETRM/ETRMTRM use 3x smaller batches.

**Possible Reason**: Memory constraints due to encoder? But should document this explicitly.

### 4. All Other Hyperparameters ✅

These match correctly across all configs:

| Parameter | Value | Status |
|-----------|-------|--------|
| `lr` | 1e-4 | ✅ All match |
| `lr_warmup_steps` | 2000 | ✅ All match |
| `lr_min_ratio` | 1.0 | ✅ All match |
| `beta1` | 0.9 | ✅ All match |
| `beta2` | 0.95 | ✅ All match |
| `weight_decay` | 0.1 | ✅ All match |
| `ema` | True | ✅ All match |
| `ema_rate` | 0.999 | ✅ All match |
| `epochs` | 100000 | ✅ All match |
| `eval_interval` | 10000 | ✅ All match |
| `halt_max_steps` | 16 | ✅ All match |
| `H_cycles` | 3 | ✅ All match |
| `L_layers` | 2 | ✅ All match |
| `H_layers` | 0 | ✅ All match |
| `hidden_size` | 512 | ✅ All match |
| `num_heads` | 8 | ✅ All match |
| `expansion` | 4 | ✅ All match |
| `puzzle_emb_len` | 16 | ✅ All match |
| `pos_encodings` | rope | ✅ All match |
| `forward_dtype` | bfloat16 | ✅ All match |
| `mlp_t` | False | ✅ All match |
| `no_ACT_continue` | True | ✅ All match |

---

## Files Requiring Changes

### Priority 1: CRITICAL FIX

#### 1. `config/arch/etrm.yaml`
**Line 28**: Change `halt_exploration_prob: 0.5` → `0.1`

#### 2. `config/arch/etrmtrm.yaml`
**Line 37**: Change `halt_exploration_prob: 0.5` → `0.1`

#### 3. `config/cfg_pretrain_etrm_arc_agi_1.yaml`
**Line 22**: Remove override `halt_exploration_prob: 0.5` (inherit from arch)

#### 4. `config/cfg_pretrain_etrmtrm_arc_agi_1.yaml`
**Line 20**: Remove override `halt_exploration_prob: 0.5` (inherit from arch)

### Priority 2: Transfer Learning Fix

#### 5. `config/cfg_pretrain_etrm_arc_agi_1.yaml`
When using `load_pretrained_decoder`, should override:
```yaml
arch:
  L_cycles: 4  # Match pretrained decoder's training config
```

Current config has L_cycles=6, but pretrained TRM decoder was trained with L_cycles=4. This creates architectural mismatch.

### Priority 3: Document Batch Size Choice

#### 6. `config/cfg_pretrain_etrm_arc_agi_1.yaml` & `config/cfg_pretrain_etrmtrm_arc_agi_1.yaml`
Add comment explaining why batch size is 256 instead of 768:
```yaml
# Hyperparams - Training
global_batch_size: 256  # Reduced from 768 due to encoder memory overhead
```

---

## Impact Analysis

### Frozen Decoder Experiment

The frozen decoder experiment used `cfg_test_frozen_decoder.yaml`, which inherits from `cfg_pretrain_etrm_arc_agi_1.yaml`:

```yaml
# cfg_test_frozen_decoder.yaml
defaults:
  - cfg_pretrain_etrm_arc_agi_1  # Inherits halt_exploration_prob: 0.5 ❌
```

**This means the frozen decoder experiment ran with wrong exploration probability!**

The experiment showed:
- ✅ Good training metrics (loss 0.70, accuracy 76.7%)
- ❌ Zero test accuracy

With `halt_exploration_prob=0.5`, the Q-head is exploring 50% of the time instead of 10%. This means:
1. Q-head learns much slower
2. Model doesn't learn proper halting behavior
3. Could explain overfitting (model doesn't know when to stop reasoning)

### All Previous ETRM/ETRMTRM Experiments

**Every ETRM/ETRMTRM experiment ran with wrong exploration probability (0.5 instead of 0.1).**

This could be a major contributor to the low performance (~5-10% vs TRM's 40%).

---

## Recommended Action Plan

### Step 1: Fix Configs (Immediate)
1. Fix `arch/etrm.yaml`: `halt_exploration_prob: 0.5` → `0.1`
2. Fix `arch/etrmtrm.yaml`: `halt_exploration_prob: 0.5` → `0.1`
3. Remove overrides in `cfg_pretrain_etrm_arc_agi_1.yaml` and `cfg_pretrain_etrmtrm_arc_agi_1.yaml`
4. Add `L_cycles: 4` override when using `load_pretrained_decoder`

### Step 2: Re-run Frozen Decoder Experiment (High Priority)
Run the frozen decoder experiment with corrected `halt_exploration_prob: 0.1`:

```yaml
# config/cfg_test_frozen_decoder_correct.yaml
defaults:
  - cfg_pretrain_etrm_arc_agi_1
  - _self_

# Override: Load official TRM checkpoint and freeze decoder
load_pretrained_decoder: ./checkpoints/official_trm/arc_v1_public/step_518071
freeze_decoder_steps: 999999

# CRITICAL: Use correct hyperparameters
arch:
  halt_exploration_prob: 0.1  # Match TRM (was 0.5!)
  L_cycles: 4  # Match pretrained decoder training

# Quick test config
global_batch_size: 256
epochs: 20000
eval_interval: 10000
max_train_groups: 32
max_eval_groups: 4
lr: 3e-4
lr_warmup_steps: 1000

project_name: "ETRM-FrozenDecoder-Test"
run_name: "frozen_decoder_correct_hyperparams"
```

**Expected Outcome**: If exploration probability was the issue, we should see improved test accuracy.

### Step 3: Re-evaluate Previous Checkpoints (Optional)
Previous ETRM/ETRMTRM checkpoints were trained with wrong hyperparameters. However, we can still evaluate them:
- Evaluation doesn't use `halt_exploration_prob` (greedy inference)
- But the checkpoints' Q-heads may have learned poorly due to wrong training exploration

### Step 4: Full ETRM Training with Correct Hyperparameters
After verifying frozen decoder works, run full ETRM training with corrected config.

---

## Why This Matters

### Q-Head Learning with ACT

The Q-head predicts whether the model should halt (output is correct). During training:
- **With exploration_prob=0.1**: Model trusts Q-head 90% of the time → Q-head learns quickly
- **With exploration_prob=0.5**: Model only trusts Q-head 50% of the time → Q-head learns slowly

If Q-head doesn't learn properly:
1. Model doesn't know when to stop reasoning
2. May overcompute (waste steps) or undercompute (halt too early)
3. Training becomes unstable
4. Test performance suffers

### TRM Paper's Choice

TRM paper chose 0.1 for a reason - it balances:
- Enough exploration to avoid local minima
- Enough exploitation for Q-head to learn from its predictions

Using 0.5 (5x higher) significantly changes this balance.

---

## Additional Notes

### Why Was This Overlooked?

1. **Inheritance**: `cfg_pretrain_etrm_arc_agi_1.yaml` has explicit override that shadows the arch config
2. **Comments**: The config has a comment "Controls Q-head exploration during training" but doesn't mention TRM's value
3. **No Validation**: No config validation script to check against TRM baseline

### Prevention

1. ✅ Create this alignment review document
2. ✅ Fix all configs
3. TODO: Add config validation script
4. TODO: Add comment in configs: "# Must match TRM paper value (0.1)"

---

## Summary

**Critical Issue Found**: `halt_exploration_prob: 0.5` in ETRM/ETRMTRM configs (should be 0.1 like TRM)

**Impact**: All previous ETRM/ETRMTRM experiments ran with 5x higher exploration, which likely hurt Q-head learning and contributed to poor performance.

**Action**: Fix configs and re-run frozen decoder experiment with correct hyperparameters.

**Expected Improvement**: If this was a major factor, we should see significant improvement in test accuracy.
