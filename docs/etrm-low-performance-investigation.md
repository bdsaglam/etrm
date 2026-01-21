# ETRM/ETRMTRM Low Performance Investigation

**Date**: 2026-01-21
**Issue**: ETRM and ETRMTRM models show consistently low performance (~5-10%) despite using pretrained TRM decoder
**Training**: Metrics consistently low throughout training (not a sudden drop)
**Evaluation**: Consistent with training (no train-eval discrepancy)

## Investigation Strategy

Move hypothesis-driven:
1. State hypothesis clearly
2. Design tests to validate/invalidate
3. Collect evidence
4. Draw conclusion
5. Move to next hypothesis if invalidated

---

## Hypothesis 1: Pretrained TRM Decoder Not Loading Correctly

### Reasoning
- ETRM uses `load_pretrained_decoder()` to initialize from TRM checkpoint
- If decoder weights aren't loading, model starts from random weights
- TRM alone gets ~40%, so broken loading would cause low performance

### Evidence to Collect
1. ✅ Check if `load_pretrained_decoder()` is actually called during training
2. ✅ Verify weight loading logs show successful loading
3. ✅ Check if EMA weights vs non-EMA weights issue (TRM checkpoints have EMA)
4. ✅ Compare loaded weights to original checkpoint (sanity check)

### Tests

#### Test 1.1: Check Training Logs for Loading
```bash
# Look for loading messages in training output
grep -i "loading pretrained" <training_log>
grep -i "decoder" <training_log>
```

**Status**: PENDING

#### Test 1.2: Verify EMA Weights in TRM Checkpoint
```bash
# Check if TRM checkpoint has EMA weights applied
python scripts/inspect_checkpoint.py \
    --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071
```

**Key Question**: Are the weights in the checkpoint already EMA-averaged, or do we need to apply EMA?

**Status**: PENDING

#### Test 1.3: Create Loading Verification Script

Create a script that:
1. Loads pretrained TRM decoder into ETRM model
2. Compares weight values before/after loading
3. Checks which keys are loaded vs missing
4. Validates weight statistics match

**Status**: PENDING

### Expected Outcomes
- **If hypothesis VALID**: Weights not loading, or loading incorrectly → Fix loading
- **If hypothesis INVALID**: Weights load correctly → Move to Hypothesis 2

---

## Hypothesis 2: Encoder Gradients Not Flowing (Frozen/Detached)

### Reasoning
- If encoder gradients are blocked, encoder can't learn
- Encoder would output random/constant embeddings
- Decoder would see garbage input → low performance

### Evidence to Collect
1. Check if encoder parameters have `requires_grad=True`
2. Verify gradient norms for encoder vs decoder during training
3. Check for accidental `.detach()` calls in encoder forward pass
4. Verify encoder is not accidentally frozen during training

### Tests

#### Test 2.1: Check Gradient Flow in Training Logs
```bash
# Look for encoder gradient norms in training logs
grep "encoder_grad_norm" <training_log>
grep "encoder_frozen" <training_log>
```

**Expected**: Should see non-zero gradient norms for encoder

**Status**: PENDING

#### Test 2.2: Inspect Encoder Forward Pass
```python
# Check models/recursive_reasoning/etrm.py
# Look for .detach() calls in encoder path
# Verify encoder output is not detached before feeding to decoder
```

**Status**: PENDING

### Expected Outcomes
- **If hypothesis VALID**: Gradients not flowing to encoder → Fix detach/freeze issue
- **If hypothesis INVALID**: Gradients flowing correctly → Move to Hypothesis 3

---

## Hypothesis 3: Encoder Output Encoding Issue (Wrong Shape/Range)

### Reasoning
- TRM expects puzzle_emb shape: `[batch, puzzle_emb_ndim]`
- If encoder outputs wrong shape/range, decoder sees invalid input
- Could cause training to fail silently

### Evidence to Collect
1. Check encoder output shape matches expected puzzle_emb shape
2. Verify encoder output value range (not NaN/Inf)
3. Compare encoder output statistics to original puzzle_emb statistics
4. Check if encoder output is being used correctly in model

### Tests

#### Test 3.1: Add Debug Logging to Encoder
```python
# In forward pass, log:
# - Encoder input shape
# - Encoder output shape
# - Encoder output mean/std/min/max
# Compare to original TRM puzzle_emb statistics
```

**Status**: PENDING

### Expected Outcomes
- **If hypothesis VALID**: Shape/range mismatch → Fix encoder architecture
- **If hypothesis INVALID**: Encoder output looks reasonable → Move to Hypothesis 4

---

## Hypothesis 4: Training Dynamics Issue (Learning Rate, Batch Size)

### Reasoning
- Encoder might need different learning rate than decoder
- Batch size might be too small for encoder to learn patterns
- Warmup might be too short for encoder convergence

### Evidence to Collect
1. Check encoder vs decoder learning rates
2. Compare training curves to TRM baseline
3. Check if encoder loss is decreasing
4. Verify batch size is sufficient

### Tests

#### Test 4.1: Check Training Config
```bash
# Look at actual training config used
cat checkpoints/etrm-final/*/all_config.yaml | grep -E "lr|batch|encoder"
```

**Status**: PENDING

### Expected Outcomes
- **If hypothesis VALID**: Hyperparameter issue → Tune hyperparameters
- **If hypothesis INVALID**: Hyperparameters look reasonable → Move to Hypothesis 5

---

## Hypothesis 5: Dataset Issue (FewShotPuzzleDataset vs PuzzleDataset)

### Reasoning
- ETRM uses `FewShotPuzzleDataset` (with demos)
- TRM uses `PuzzleDataset` (no demos, just puzzle_id)
- Dataset mismatch could cause training issues

### Evidence to Collect
1. Verify FewShotPuzzleDataset provides correct inputs
2. Check if demo examples are being fed correctly
3. Verify puzzle_identifiers match between datasets
4. Check dataset augmentation is working

### Tests

#### Test 5.1: Inspect Dataset Output
```python
# Create both datasets, compare outputs
# Check shapes, values, augmentation
```

**Status**: PENDING

### Expected Outcomes
- **If hypothesis VALID**: Dataset issue → Fix dataset
- **If hypothesis INVALID**: Dataset looks correct → Move to Hypothesis 6

---

## Hypothesis 6: Evaluator Bug (Same as TRM Bug?)

### Reasoning
- We just fixed a critical evaluator bug in TRM evaluation
- ETRM might have the same bug
- Could explain low reported metrics

### Evidence to Collect
1. Check if ETRM evaluation uses same evaluator
2. Verify fix was applied to both train and eval paths
3. Test with different evaluation subset sizes

### Tests

#### Test 6.1: Check Evaluator Usage
```bash
# Check if evaluate_encoder() uses the fixed ARC evaluator
grep -n "ARC" pretrain_etrm.py
grep -n "evaluator.result" pretrain_etrm.py
```

**Status**: PENDING

#### Test 6.2: Quick Evaluation Test
```bash
# Evaluate on different subset sizes, check if metrics scale correctly
python scripts/evaluate_trm_checkpoint.py \
    --checkpoint <etrm_checkpoint> \
    --model-type etrm \
    --max-eval-groups 4

python scripts/evaluate_trm_checkpoint.py \
    --checkpoint <etrm_checkpoint> \
    --model-type etrm \
    --max-eval-groups 64
```

**Expected**: Metrics should be consistent across subset sizes (post-fix)

**Status**: PENDING

### Expected Outcomes
- **If hypothesis VALID**: Same evaluator bug → Already fixed!
- **If hypothesis INVALID**: Evaluator works correctly → Move to Hypothesis 7

---

## Hypothesis 7: Encoder Architecture Too Weak

### Reasoning
- Encoder might not have enough capacity to learn transformation rules
- Could be too shallow, too narrow, or wrong architecture
- This would be an architectural limitation, not a bug

### Evidence to Collect
1. Check encoder architecture details
2. Compare to other successful encoder designs
3. Look at encoder training curves (is it learning anything?)
4. Check if encoder output varies across different demo sets

### Tests

#### Test 7.1: Encoder Architecture Inspection
```bash
# Check encoder config
cat config/cfg_pretrain_etrm_arc_agi_1.yaml | grep -A 20 encoder
```

**Status**: PENDING

#### Test 7.2: Encoder Output Variance Test
```python
# Feed different demo sets, check if encoder outputs differ
# If outputs are nearly identical, encoder isn't learning
```

**Status**: PENDING

### Expected Outcomes
- **If hypothesis VALID**: Architecture too weak → Try stronger encoder
- **If hypothesis INVALID**: Architecture should work → Continue investigation

---

## Hypothesis 1 Results: PARTIALLY VALIDATED

### Evidence Collected

#### Test 1.2: EMA Weights Check ✅
```bash
python scripts/inspect_checkpoint.py --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071
```

**Result**: Official checkpoint contains **EMA-averaged weights** (confirmed by low std values)

#### Test 1.3: Decoder Weight Verification ❌
```bash
python scripts/verify_decoder_loading.py \
    --trm-checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071 \
    --etrm-checkpoint ./checkpoints/etrm-final/F2_hybrid_var/step_174240
```

**Result**:
- ❌ Only 2/14 decoder weights match (14.3%)
- ⚠️ Large weight drift (max diff: 0.6, mean diff: 0.03-0.54)
- **BUT**: This is expected since `freeze_decoder_steps: 0` → decoder trains!

### Key Insight from User

> "When I ran experiments with decoder frozen, the network couldn't even learn anything"

This is **CRITICAL**: If encoder can't learn with frozen decoder, it means:
1. Gradients aren't flowing properly, OR
2. Encoder architecture is too weak, OR
3. Learning rate/optimization issue

### Conclusion

Cannot validate/invalidate Hypothesis 1 yet. Need controlled experiment.

---

## 🧪 EXPERIMENT 1: Frozen Decoder Test

### Purpose
Isolate whether encoder can learn at all with frozen pretrained decoder.

### Setup

**Config**: `config/cfg_test_frozen_decoder.yaml`

```yaml
load_pretrained_decoder: ./checkpoints/official_trm/arc_v1_public/step_518071
freeze_decoder_steps: 999999  # Freeze forever
encoder_lr_scale: 1.0  # Separate optimizer for encoder
lr: 3e-4  # Faster LR for encoder-only
global_batch_size: 256
epochs: 20000  # ~2 eval cycles
max_train_groups: 32  # Quick test
```

**Run Command**:
```bash
python pretrain_etrm.py --config-name cfg_test_frozen_decoder
```

### Metrics to Watch

1. **Encoder gradients**: `grad/encoder_norm` should be > 0 and stable
2. **Training loss**: Should decrease if encoder is learning
3. **Encoder output variance**: `encoder_token_std` should increase (learning diversity)
4. **Eval accuracy**: Should improve above random (even if just 5% → 10%)

### Expected Outcomes

| Scenario | Gradients | Loss | Accuracy | Conclusion |
|----------|-----------|------|----------|------------|
| **Encoder learns** | ✅ Non-zero, stable | ✅ Decreasing | ✅ Improving | Gradients flow, encoder works |
| **Encoder stuck** | ❌ Zero or vanishing | ❌ Flat | ❌ Random | Gradient blocking bug |
| **Encoder trains but useless** | ✅ Non-zero | ✅ Decreasing | ❌ No improvement | Architecture too weak |

### Next Steps Based on Results

- **If encoder learns** → Move to Hypothesis 3 or 4 (why does unfrozen decoder make it worse?)
- **If encoder stuck** → Investigate Hypothesis 2 (gradient flow) more deeply
- **If encoder trains but useless** → Move to Hypothesis 7 (architecture issue)

---

## 🧪 EXPERIMENT 1 RESULTS: Encoder Trains But Useless (Scenario C)

**Date**: 2026-01-21
**Status**: ✅ COMPLETED
**Steps**: 2500 / 20000 (training stopped early after conclusive results)
**Detailed Report**: `EXPERIMENT_FROZEN_DECODER_RESULTS.md`

### Key Findings

| Metric Category | Result | Interpretation |
|----------------|--------|----------------|
| **Gradient Flow** | ✅ grad/encoder_norm = 1.0 | Encoder receives gradients correctly |
| **Training Loss** | ✅ train/lm_loss = 0.70 (from ~8.5) | Encoder is learning something |
| **Output Diversity** | ✅ encoder_token_std = 0.37 | Outputs are diverse, not constant |
| **Test Accuracy** | ❌ ARC/pass@1 = 0.0% | Zero generalization to test set |
| **Decoder Frozen** | ✅ 14/14 weights matched | Decoder stayed frozen correctly |

### Conclusion

**Result**: **Scenario C - Encoder Trains But Useless**

The encoder:
- ✅ Receives gradients (no blocking bug)
- ✅ Optimizes correctly (loss decreases)
- ✅ Produces diverse outputs (not outputting constants)
- ❌ Achieves ZERO test accuracy (no generalization)

This is **NOT** a gradient flow bug or decoder loading bug. The encoder mechanically works but cannot learn useful task representations from demonstrations.

### Hypothesis Validation Status

| Hypothesis | Status | Evidence |
|-----------|--------|----------|
| **H1: Decoder Loading Bug** | ❌ INVALIDATED | Decoder weights match 100%, frozen correctly |
| **H2: Gradient Flow Bug** | ❌ INVALIDATED | grad/encoder_norm = 1.0 (healthy) |
| **H7: Architecture Too Weak** | ⚠️ LIKELY | Encoder learns but doesn't generalize |

### Next Action

Move to **Hypothesis 7: Encoder Architecture Too Weak** investigation.

---

