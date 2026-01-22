# Encoder Variance Metrics: Cross-Puzzle vs Within-Puzzle

**Date**: 2026-01-22
**Status**: ✅ IMPLEMENTED
**Files Modified**: `models/recursive_reasoning/etrm.py`, `models/recursive_reasoning/etrmtrm.py`

---

## Problem Statement

Previously, we tracked a single metric `encoder_cross_sample_var` which measured variance across the batch. This was ambiguous because each batch contains **different puzzle groups** (different tasks).

When analyzing lpn_var's surprisingly LOW cross-sample variance, we initially thought the mean aggregation within each puzzle (averaging K demos) would reduce cross-sample variance. **This was wrong** - the mean aggregation happens within each puzzle, not across puzzles.

---

## Solution: Two Separate Metrics

We now track **two distinct variance metrics** to disambiguate what's happening:

### 1. `encoder_cross_sample_var` (Renamed from `encoder_cross_sample_var`)

**What it measures**: Variance of encoder outputs across **different puzzle groups** in the batch

**Formula**:
```python
batch_mean = context.mean(dim=0, keepdim=True)  # (1, T, D)
cross_sample_var = ((context - batch_mean) ** 2).mean()
```

**Expected behavior**:
- **Should be HIGH** if encoder learns diverse task representations
- Different tasks should get different encoder outputs
- LOW cross-sample variance indicates representation collapse (BAD!)

**Interpretation**:
- O1 (standard): 0.15 → ✅ Good diversity
- O3 (ETRMTRM): 0.15 → ✅ Good diversity
- O4 (lpn_var): 0.05 → ❌ Low diversity (potential issue)

### 2. `encoder_within_puzzle_var` (NEW)

**What it measures**: Variance across **different forward passes of the same puzzle**

**Implementation**:
```python
# Computed ~1% of training steps (every ~100 steps)
if random.random() < 0.01 and self.training:
    # Re-encode first 4 samples
    encoder_output2 = self.encoder(subset_batch)
    within_puzzle_var = ((context[:4] - context2) ** 2).mean()
```

**Expected behavior by encoder type**:

| Encoder Type | Expected Value | Interpretation |
|--------------|----------------|----------------|
| **Deterministic** (standard, lpn_standard, hybrid_standard) | **~0** | Same puzzle → same output (correct!) |
| **Variational** (lpn_var, hybrid_variational) | **>0.01** | Stochastic sampling → different outputs |

**Why only compute 1% of steps?**
- Running encoder twice per step adds 100% overhead
- Sampling is sufficient for diagnostics
- Reduces training time impact

---

## What This Reveals About lpn_var

### Original (Wrong) Analysis
> "lpn_var has LOW variance due to mean aggregation AFTER sampling"

This was **incorrect reasoning**. The mean aggregation happens **within each puzzle** (averaging demos), not **across puzzles** in the batch.

### Corrected Analysis

**lpn_var shows LOW cross-sample variance**, which indicates:

1. **Primary Issue**: Different tasks get **similar encoder representations**
   - This is representation collapse (BAD for generalization)
   - Encoder is not learning diverse task embeddings

2. **Root Causes**:
   - **Very low KL weight** (0.0001): No pressure to maintain variance
   - **Zero initialization**: logvar starts at 0
   - **Small latent bottleneck** (32 dims): Less capacity for diversity
   - **Result**: Model learns similar μ values for different tasks, even with low σ

3. **Within-Puzzle Variance** (to be measured):
   - If also LOW → effectively deterministic (kl_weight too low)
   - If HIGH → stochastic but learning poor representations

### Expected Results

When we run new experiments with these metrics:

**O1 (standard) - Deterministic baseline:**
```
encoder_cross_sample_var: 0.15  ✅ High diversity across tasks
encoder_within_puzzle_var: ~0   ✅ Deterministic (as expected)
```

**O3 (ETRMTRM) - Deterministic recurrent:**
```
encoder_cross_sample_var: 0.15  ✅ High diversity across tasks
encoder_within_puzzle_var: ~0   ✅ Deterministic (as expected)
```

**O4 (lpn_var) - Variational:**
```
encoder_cross_sample_var: 0.05  ❌ LOW diversity (representation collapse!)
encoder_within_puzzle_var: ???
  - If ~0: Collapsed to deterministic (kl_weight too low)
  - If >0.01: Stochastic but learning similar representations
```

---

## Implications for lpn_var

### If within-puzzle variance is LOW (~0):
**Conclusion**: lpn_var has effectively collapsed to deterministic due to kl_weight=0.0001

**Fix options**:
1. Increase kl_weight to 0.001 or 0.01 (enforce variance)
2. Or accept it as deterministic and remove variational overhead

### If within-puzzle variance is HIGH (>0.01):
**Conclusion**: lpn_var is stochastic but learning poor representations (all tasks map to similar μ)

**Fix options**:
1. Increase capacity (larger latent_dim: 32 → 128 or 256)
2. Increase encoder depth (2 layers → 4 layers)
3. Different architecture (may not be suitable for this task)

### Why O4 might still work despite low cross-sample variance:
- Small encoder (673K params) forces compression
- May be learning a "universal task embedding" + decoder does the work
- Frozen decoder experiments test encoder quality directly
- If O4 performs well, the decoder might be extracting task info from the small representation

---

## Implementation Details

### Changes Made

**File**: `models/recursive_reasoning/etrm.py`
- Lines 513-542: Added cross-sample + within-puzzle metrics (training mode)
- Lines 654-659: Added cross-sample metric (eval mode)
- Fixed: Use `new_current_data["demo_inputs"]` instead of `batch["demos"]`

**File**: `models/recursive_reasoning/etrmtrm.py`
- Lines 208-235: Added cross-sample + within-puzzle metrics (training mode)
- Lines 339-344: Added cross-sample metric (eval mode)
- Fixed: Use correct encoder call signature and data structure

### Sampling Strategy

Within-puzzle variance is computed **stochastically**:
- Only on 1% of training steps (`random.random() < 0.01`)
- Uses first 4 samples from batch (reduces cost)
- Not computed in eval mode (adds no value, deterministic anyway)

### Compatibility

Old metric `encoder_cross_sample_var` is **replaced** by `encoder_cross_sample_var`:
- Same calculation, clearer name
- Old logs will show `encoder_cross_sample_var`
- New logs will show `encoder_cross_sample_var`
- Both measure the same thing

---

## Verification Steps

### After running new experiments:

1. **Check WandB for new metrics**:
   ```
   train/encoder_cross_sample_var
   train/encoder_within_puzzle_var  (sparse, only 1% of steps)
   ```

2. **Compare architectures**:
   ```python
   # O1 vs O3 vs O4
   cross_sample_var_O1 = ...  # Should be ~0.15
   cross_sample_var_O4 = ...  # Likely ~0.05 (LOW)

   within_puzzle_var_O1 = ...  # Should be ~0 (deterministic)
   within_puzzle_var_O4 = ...  # TBD (key diagnostic!)
   ```

3. **Interpret O4 results**:
   - If both LOW → collapsed to deterministic, no advantage over standard
   - If cross LOW, within HIGH → stochastic but poor diversity
   - Either way suggests lpn_var architecture needs adjustment

---

## Related Documentation

- Original (incorrect) analysis: `ANALYSIS_LPN_VAR_LOW_VARIANCE.md` (lines 16-50)
- Hybrid variational bug fix: `BUG_FIX_HYBRID_VARIATIONAL.md`
- Overfit experiments: `jobs-overfit-corrected.txt`

---

## Key Takeaway

**The lpn_var encoder's LOW cross-sample variance is a BUG, not a feature.**

Different puzzle groups should get **diverse encoder representations**. Low cross-sample variance indicates the encoder is **not learning task-specific embeddings**, which will hurt generalization.

The new metrics will help us diagnose whether the issue is:
1. Collapsed to deterministic (kl_weight too low)
2. Stochastic but learning poor representations (architecture too small)
3. Something else entirely

We'll know after the next round of experiments with these new metrics.
