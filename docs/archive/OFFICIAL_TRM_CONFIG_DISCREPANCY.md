# Official TRM Config Discrepancy Analysis

**Date**: 2026-01-21
**Issue**: Checkpoint config differs from official repo's default config

---

## Summary

There's a **discrepancy** between:
1. **Official checkpoint config** (what they actually trained with)
2. **Official repo default config** (what's in their GitHub)

This explains why the checkpoint has `L_cycles: 4` but the repo defaults to `L_cycles: 6`.

---

## Comparison Table

| Parameter | Checkpoint Config | Repo Default (arch/trm.yaml) | README | Notes |
|-----------|-------------------|------------------------------|--------|-------|
| **L_cycles** | **4** ✅ | **6** ❌ | **4** ✅ | **MISMATCH!** |
| H_cycles | 3 ✅ | 3 ✅ | 3 ✅ | Match |
| halt_exploration_prob | 0.1 ✅ | 0.1 ✅ | Not specified | Match |
| global_batch_size | 768 ✅ | 768 ✅ | Not specified | Match |
| lr | 1e-4 ✅ | 1e-4 ✅ | Not specified | Match |
| lr_warmup_steps | 2000 ✅ | 2000 ✅ | Not specified | Match |
| ema | True ✅ | False ❌ | True ✅ | **MISMATCH!** |
| L_layers | 2 ✅ | 2 ✅ | 2 ✅ | Match |
| H_layers | 0 ✅ | 0 ✅ | Not specified | Match |

---

## Key Findings

### 1. L_cycles Discrepancy (CRITICAL)

**What they trained with**: `L_cycles: 4`
**What repo defaults to**: `L_cycles: 6`
**What README says**: `L_cycles: 4`

**Conclusion**: The official checkpoint was trained with `L_cycles: 4`, but the repo's default architecture config uses `L_cycles: 6`. The README is correct.

**Impact on our configs**:
- ✅ Our `cfg_pretrain_arc_agi_1.yaml` uses `L_cycles: 6` (matches repo default)
- ❌ But when loading pretrained decoder, we need `L_cycles: 4` (matches checkpoint)

### 2. EMA Discrepancy

**What they trained with**: `ema: True`
**What repo defaults to**: `ema: False`
**What README says**: `ema: True`

**Conclusion**: They trained with EMA enabled, but repo defaults to disabled.

**Impact on our configs**:
- ✅ All our configs have `ema: True` (correct)

---

## Why This Matters

### For Transfer Learning (Loading Pretrained Decoder)

When using `load_pretrained_decoder`, we MUST match the checkpoint's training config:
- ✅ Use `L_cycles: 4` (what checkpoint was trained with)
- ❌ NOT `L_cycles: 6` (repo default)

Otherwise, we get architectural mismatch (decoder expects 4 cycles but model runs 6).

### For Training From Scratch

When training from scratch (no pretrained decoder):
- Can use `L_cycles: 6` (more capacity, repo default)
- Or use `L_cycles: 4` (matches official checkpoint architecture)

---

## Our Current Config Status

### ✅ Correctly Configured

1. **`cfg_test_frozen_decoder_corrected.yaml`**: Uses `L_cycles: 4` ✅
   - Loads pretrained decoder → matches checkpoint

2. **All configs use `halt_exploration_prob: 0.1`** ✅
   - Matches checkpoint and repo default

3. **All configs use `ema: True`** ✅
   - Matches checkpoint (even though repo default is False)

### ⚠️ Needs Attention

1. **`cfg_pretrain_etrm_arc_agi_1.yaml`**: Uses `L_cycles: 6`
   - When using `load_pretrained_decoder`: Should override to `L_cycles: 4`
   - When training from scratch: `L_cycles: 6` is fine (more capacity)

2. **`cfg_pretrain_etrmtrm_arc_agi_1.yaml`**: Uses `L_cycles: 6`
   - Same issue as ETRM

---

## Recommendation

### Option 1: Document Current Behavior (Preferred)

Keep base configs at `L_cycles: 6` (more capacity) but:
- Add prominent comment warning about L_cycles mismatch
- When loading pretrained decoder, explicitly override to `L_cycles: 4`

```yaml
# config/cfg_pretrain_etrm_arc_agi_1.yaml
arch:
  L_cycles: 6  # Default: more capacity. IMPORTANT: Use L_cycles: 4 when load_pretrained_decoder is set!
```

### Option 2: Change Default to Match Checkpoint

Change base ETRM configs to default to `L_cycles: 4`:
- Matches official checkpoint architecture
- Easier transfer learning (no override needed)
- But slightly less capacity than repo default

```yaml
# config/cfg_pretrain_etrm_arc_agi_1.yaml
arch:
  L_cycles: 4  # Match official checkpoint (repo default is 6 but checkpoint was trained with 4)
```

---

## Why Does Official Repo Have L_cycles: 6?

**Hypothesis**: The authors may have:
1. Initially used `L_cycles: 6` during development
2. Later decided to reduce to `L_cycles: 4` for the official checkpoint (faster, less parameters)
3. Updated README but forgot to update `arch/trm.yaml` default

Or alternatively:
1. Trained multiple checkpoints with different L_cycles
2. Published the `L_cycles: 4` checkpoint as "official"
3. Kept `L_cycles: 6` as default for users who want more capacity

---

## Action Items

### Immediate (for current experiment)

✅ **DONE**: `cfg_test_frozen_decoder_corrected.yaml` uses `L_cycles: 4`
- Correctly matches pretrained decoder training

### Short-term (documentation)

1. Add warning comment to `cfg_pretrain_etrm_arc_agi_1.yaml`
2. Add warning comment to `cfg_pretrain_etrmtrm_arc_agi_1.yaml`
3. Document in `HYPERPARAMETER_ALIGNMENT_REVIEW.md`

### Long-term (decide default behavior)

**Question for user**: Should base ETRM configs default to:
- **Option A**: `L_cycles: 6` (repo default, more capacity, requires override for transfer learning)
- **Option B**: `L_cycles: 4` (checkpoint default, easier transfer learning, matches README)

---

## Related Files

- **Official Checkpoint**: `checkpoints/official_trm/arc_v1_public/all_config.yaml`
- **Official Repo**: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- **Repo Default Arch**: `config/arch/trm.yaml` (has L_cycles: 6)
- **Official README**: Specifies L_cycles: 4
- **Our Configs**:
  - `config/cfg_pretrain_etrm_arc_agi_1.yaml` (L_cycles: 6)
  - `config/cfg_test_frozen_decoder_corrected.yaml` (L_cycles: 4)

---

## Conclusion

The discrepancy is **in the official repo itself**:
- Their published checkpoint was trained with `L_cycles: 4`
- But their repo's default arch config uses `L_cycles: 6`
- README correctly states `L_cycles: 4`

This is not a bug in our configs, but something to be aware of when:
1. Loading pretrained weights (must use L_cycles: 4)
2. Comparing to official results (they used L_cycles: 4)
3. Training from scratch (can choose either 4 or 6)

Our corrected frozen decoder config correctly uses `L_cycles: 4` to match the pretrained decoder.
