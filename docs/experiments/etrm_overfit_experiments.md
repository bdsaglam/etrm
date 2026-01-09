# ETRM Overfit Phase Experiments

**Goal**: Systematically explore ETRM hyperparameter space to find configurations that can overfit to small dataset (32 groups).

**Project**: `etrm-overfit`

**Success Criterion**: Train accuracy >90% within reasonable time (~5000 steps)

---

## Background

After discovering and fixing the gradient starvation issue (Jan 9, 2026), we now have working ETRM with re-encoding. The standard encoder achieved 86% at step 240 and is still improving.

**Next step**: Systematically explore the hyperparameter space to understand:
1. Which encoder architectures work with ETRM re-encoding?
2. How do halting dynamics affect learning?
3. Do we need pretrained decoder or can we match TRM from scratch?

---

## Common Settings

All experiments use:
- **32 train groups, 32 eval groups** (small subset for fast iteration)
- **Pretrained decoder** (unless explicitly testing without)
- **20,000 epochs** max
- **Eval every 1,000 steps**
- **grad_clip_norm=1.0** (essential for stability)

### Pretrained Decoder Path
```
/home/baris/repos/trm/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4/step_518071
```

---

## Experiment Series

### E1: Baseline Validation (Standard Encoder)

Continue and validate the working standard encoder configuration.

| ID | Description | halt_max_steps | halt_exploration_prob | Expected Outcome |
|----|-------------|-----------------|----------------------|------------------|
| **E1a** | Continue current baseline | 16 | 0.5 | >90% train acc (validation) |
| **E1b** | Lower exploration | 16 | 0.3 | May converge faster, Q-head has more control |
| **E1c** | No exploration | 16 | 0.0 | Pure Q-head learning, may be harder initially |
| **E1d** | Higher exploration | 16 | 0.7 | More diverse training signal |

**Purpose**: Validate standard encoder works reliably and understand exploration impact.

---

### E2: Encoder Architecture Exploration

Re-test encoder architectures that failed with cached ETRM, now with re-encoding.

#### E2a: Variational Encoder

**Purpose**: Test if VAE regularization helps with pattern learning.

**Key settings**:
- `encoder_type: hybrid_variational`
- `encoder_num_layers: 4`
- `encoder_norm_style: pre`
- `encoder_set_layers: 2`
- `halt_exploration_prob: 0.5`
- `global_batch_size: 128` (larger encoder)

**Previous result**: 50.6% with cached ETRM (gradient starvation)

**Expected**: Should now learn properly with full gradients. If >90%, VAE regularization compatible with ETRM.

#### E2b: LPN Variational Encoder

**Purpose**: Test LPN-style deep encoder.

**Key settings**:
- `encoder_type: lpn_variational`
- `encoder_num_layers: 6` (deep)
- `encoder_norm_style: pre`
- `encoder_set_layers: 3`
- `halt_exploration_prob: 0.5`
- `global_batch_size: 128`

**Previous result**: 51.5% with cached ETRM (gradient starvation)

**Expected**: Should learn with re-encoding. LPN designed for program synthesis, may work well.

#### E2c: Hybrid Standard Encoder

**Purpose**: Test deeper standard encoder without VAE.

**Key settings**:
- `encoder_type: hybrid_standard`
- `encoder_num_layers: 4`
- `encoder_norm_style: pre`
- `encoder_set_layers: 2`
- `halt_exploration_prob: 0.5`
- `global_batch_size: 128`

**Expected**: If VAE isn't needed, this should match or beat E2a with fewer parameters.

#### E2d: Deeper Standard Encoder

**Purpose**: Test if simple scaling helps.

**Key settings**:
- `encoder_type: standard`
- `encoder_num_layers: 4` (vs 2 in baseline)
- `halt_exploration_prob: 0.5`

**Expected**: More capacity may learn richer patterns, but may be harder to train.

---

### E3: Halting Dynamics Exploration

Understand how halt_max_steps affects learning.

| ID | halt_max_steps | halt_exploration_prob | Purpose |
|----|----------------|----------------------|---------|
| **E3a** | 8 | 0.5 | Shorter max steps - force efficiency |
| **E3b** | 32 | 0.5 | Longer max steps - allow complexity |
| **E3c** | 16 | 0.5 | Baseline for comparison |

**Analysis**: Compare avg steps used, convergence speed, final accuracy.

---

### E4: From Scratch (No Pretrained Decoder)

Test if ETRM can learn without pretrained decoder - fair comparison with TRM paper.

| ID | Description | Expected Outcome |
|----|-------------|------------------|
| **E4a** | Standard encoder, from scratch | Should learn but slower than with pretrained |
| **E4b** | Deeper encoder (4 layers), from scratch | More capacity may help when learning from scratch |

**Purpose**:
- Ensure pretrained decoder isn't doing all the work
- Fair comparison with TRM paper (45% on ARC-AGI-1)

**Note**: These may need more steps to converge (try 50,000 epochs if needed).

---

### E5: Hyperparameter Sensitivity

Test sensitivity to other hyperparameters.

#### E5a: Learning Rate Sweep

| Variant | Learning Rate | Notes |
|---------|---------------|-------|
| E5a_low | 5e-5 | Half of default (if default is 1e-4) |
| E5a_baseline | 1e-4 | Default |
| E5a_high | 2e-4 | Double default |

**Purpose**: Find optimal learning rate for ETRM.

#### E5b: Batch Size Impact

| Variant | Global Batch Size | Notes |
|---------|-------------------|-------|
| E5b_small | 128 | Smaller, more frequent updates |
| E5b_baseline | 256 | Default |
| E5b_large | 512 | Larger, more stable gradients |

**Purpose**: Understand batch size impact on encoder learning.

---

## Experiment Priority Queue

### Phase 1: Immediate (Week 1)

**Priority**: Validate baseline and re-test failed encoders

1. ✅ **E1a**: Continue current baseline to >90% (already at 86%)
2. **E2a**: Variational encoder (previously failed at 50.6%)
3. **E2b**: LPN encoder (previously failed at 51.5%)
4. **E2c**: Hybrid standard encoder

**Rationale**: These were the experiments that failed with cached ETRM. Need to see if re-encoding fixes them.

### Phase 2: Near-term (Week 2)

**Priority**: Understand halting and exploration

5. **E1b**: Lower exploration (0.3)
6. **E1c**: No exploration (0.0)
7. **E3a**: Shorter max steps (8)
8. **E3b**: Longer max steps (32)

**Rationale**: Understanding halting dynamics crucial for ETRM design.

### Phase 3: Validation (Week 3)

**Priority**: Ensure robustness

9. **E4a**: From scratch (no pretrained decoder)
10. **E2d**: Deeper standard encoder
11. **E5a_low/high**: Learning rate variants

**Rationale**: Validate that findings generalize and aren't overfitted to specific setup.

---

## Expected Outcomes Summary

### Success Scenarios

**Scenario A: Standard encoder is best**
- E1a achieves >90%, E2a/b/c don't improve → stick with standard
- Implication: Simple architecture sufficient, focus on data and training

**Scenario B: Deeper encoders help**
- E2a/b/c achieve >92% vs E1a ~90% → deeper encoders learn better
- Implication: Scale up encoder for full dataset experiments

**Scenario C: Exploration matters**
- E1b (0.3) or E1c (0.0) significantly better than E1a (0.5)
- Implication: Adjust exploration for optimal Q-head learning

**Scenario D: From scratch works**
- E4a achieves >90% without pretrained decoder
- Implication: ETRM is truly learning, not just fine-tuning decoder

### Failure Scenarios

**Red Flag 1: All encoders fail to overfit**
- None reach >80% train accuracy
- Action: Check for bugs, verify gradient flow, check data pipeline

**Red Flag 2: Only with pretrained decoder**
- E4a fails completely (<50% train acc)
- Action: May need better encoder initialization or more training

**Red Flag 3: Variational encoders fail again**
- E2a/E2b still stuck at 50-60%
- Action: VAE regularization may be incompatible with ETRM, abandon

---

## Monitoring and Analysis

### Key Metrics to Track

**Training Dynamics**:
- `train/accuracy`: Token-level (target: >90%)
- `train/exact_accuracy`: Sequence-level EM (target: >50%)
- `train/loss`: Overall loss (should decrease steadily)
- `train/q_halt_loss`: Q-head loss (should decrease)
- `train/q_halt_accuracy`: Q-head accuracy (target: >85%)

**Halting Behavior**:
- `train/steps`: Avg steps used (should be <halt_max_steps)
- Step distribution: Are samples using diverse steps?

**Encoder Diagnostics** (if available):
- Encoder gradient norms
- Cross-sample variance (check encoder differentiation)

### Analysis Questions

For each experiment:
1. **Convergence**: How many steps to reach 80%? 90%?
2. **Stability**: Are losses smooth or noisy?
3. **Halting**: What avg steps used? Is Q-head learning?
4. **Final performance**: Train acc, train EM at convergence?

### Comparison Matrix

After all experiments, create comparison:

| Experiment | Train Acc | Train EM | Avg Steps | Steps to 90% | Notes |
|------------|-----------|----------|-----------|--------------|-------|
| E1a | ? | ? | ? | ? | Baseline |
| E2a | ? | ? | ? | ? | VAE |
| ... | ... | ... | ... | ... | ... |

---

## Decision Points

### After Phase 1 (E1a, E2a, E2b, E2c)

**If E1a reaches >90%**:
✅ Standard encoder validated → proceed to Phase 2 (halting exploration)

**If E2a/b/c significantly better (>92%)**:
✅ Deeper encoders help → use for full dataset
⚠️ But check: is improvement worth compute cost?

**If E2a/b/c fail again (<80%)**:
❌ Variational/LPN architectures incompatible → abandon for now
✅ Focus on standard encoder variants

### After Phase 2 (E1b, E1c, E3a, E3b)

**If exploration helps (E1b/c better than E1a)**:
✅ Use optimal exploration_prob for full dataset

**If max_steps matters (E3a/b significantly different)**:
✅ Tune max_steps based on puzzle complexity

### After Phase 3 (E4a, E2d, E5a)

**If E4a succeeds (>90%)**:
✅ ETRM validated, can learn without pretrained decoder
✅ Ready for full dataset experiments

**If E4a fails but E1a succeeded**:
⚠️ Pretrained decoder essential
⚠️ May need better encoder initialization strategy

---

## Next Steps After Overfit Phase

Once we have validated configurations that overfit reliably:

1. **Select best 2-3 configurations** based on:
   - Train accuracy (>90%)
   - Training efficiency (steps to converge)
   - Halting behavior (good Q-head learning)

2. **Run full dataset experiments** (Phase 2 from future_experiments.md)
   - Train on ~560 puzzles
   - Evaluate on ~400 held-out puzzles
   - Success: ≥5-10% test EM

3. **Document findings** in comparison doc

---

## References

- **Future experiments plan**: `docs/future_experiments.md`
- **Training modes comparison**: `docs/training_modes_comparison.md`
- **Progress from Jan 9**: `docs/progress_2026_01_09.md`
- **Job commands**: `jobs-etrm-overfit.txt`
