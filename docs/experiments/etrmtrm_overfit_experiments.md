# ETRMTRM Overfit Phase Experiments

**Goal**: Test recurrent encoder architecture that evolves context across ACT steps.

**Project**: `etrmtrm-overfit`

**Success Criterion**: Train accuracy >90% within reasonable time (~5000 steps), comparable to ETRM baseline

---

## Background

ETRMTRM introduces a recurrent encoder with carry state that evolves across ACT steps:
- **ETRM**: Static encoder (no carry), context computed once
- **ETRMTRM**: Recurrent encoder (has carry), context evolves each ACT step

**Research Question**: Does allowing the encoder to refine its representation across reasoning steps improve pattern learning?

**Two Variants**:
- **Variant A (recurrent_standard)**: Simple carry-based recurrence, no H/L loops
- **Variant B (trm_style)**: TRM-like encoder with z_e_H/z_e_L and H/L loops

**This phase**: Focus on Variant A (simpler, faster iteration)

---

## Common Settings

All experiments use:
- **32 train groups, 32 eval groups** (small subset for fast iteration)
- **Pretrained decoder** from ETRM experiments
- **20,000 epochs** max
- **Eval every 1,000 steps**
- **grad_clip_norm=1.0** (essential for stability)
- **encoder_type=standard** (base DemoGridEncoder)

### Pretrained Decoder Path
```
/home/baris/repos/trm/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4/step_518071
```

---

## Experiment Series

### E0: Baseline Comparison (ETRM vs ETRMTRM)

**Purpose**: Establish baseline and verify ETRMTRM implementation correctness.

| ID | Model | recurrent_encoder_type | Description | Expected Outcome |
|----|-------|------------------------|-------------|------------------|
| **E0a** | ETRM | N/A (static encoder) | Baseline - standard encoder | >90% train acc (reference) |
| **E0b** | ETRMTRM | recurrent_standard | Recurrent encoder, otherwise same | Should match or beat E0a |

**Key comparison**:
- Same hyperparameters except encoder type
- If E0b matches E0a → Recurrent encoder works, no regression
- If E0b beats E0a → Recurrent encoder improves learning
- If E0b fails → Bug or fundamental issue with recurrent design

**Config for E0b**:
```yaml
arch:
  recurrent_encoder_type: recurrent_standard
  encoder_num_layers: 2
  halt_max_steps: 16
  halt_exploration_prob: 0.5
global_batch_size: 256
```

---

### E1: Encoder Architecture Exploration

**Purpose**: Test if recurrent encoder benefits from different base architectures.

#### E1a: Deeper Grid Encoder

**Hypothesis**: More capacity in per-demo encoding may help recurrent aggregation.

**Config**:
```yaml
arch:
  recurrent_encoder_type: recurrent_standard
  encoder_num_layers: 4  # vs 2 in baseline
  halt_max_steps: 16
  halt_exploration_prob: 0.5
```

**Expected**: May learn richer patterns, but could be harder to train.

#### E1b: Hybrid Grid Encoder

**Hypothesis**: LPN-style deep encoder for per-demo encoding.

**Config**:
```yaml
arch:
  recurrent_encoder_type: recurrent_standard
  encoder_num_layers: 4
  encoder_norm_style: pre  # LPN uses pre-norm
  halt_max_steps: 16
  halt_exploration_prob: 0.5
global_batch_size: 128  # Deeper encoder, reduce batch size
```

**Expected**: LPN designed for program synthesis, may work well with recurrence.

---

### E2: Halting Dynamics with Recurrent Encoder

**Purpose**: Does recurrent encoder benefit from different halting strategies?

**Hypothesis**: Recurrent encoder may need more steps to refine context.

| ID | halt_max_steps | halt_exploration_prob | Purpose |
|----|----------------|----------------------|---------|
| **E2a** | 8 | 0.5 | Shorter max steps - force efficiency |
| **E2b** | 16 | 0.5 | Baseline (same as ETRM) |
| **E2c** | 32 | 0.5 | Longer max steps - allow encoder refinement |
| **E2d** | 16 | 0.7 | Higher exploration - more encoder updates |
| **E2e** | 16 | 0.0 | No exploration - deterministic halting |

**Analysis**:
- Compare avg steps used vs ETRM baseline
- Does recurrent encoder use more steps to refine?
- Is encoder refinement helping or just adding noise?

---

### E3: Learning Rate Sensitivity

**Purpose**: Recurrent encoder may need different learning rate than static encoder.

| ID | Learning Rate | encoder_lr_scale | Notes |
|----|---------------|------------------|-------|
| **E3a** | 1e-4 | 0.0 | Baseline (default ETRM lr) |
| **E3b** | 5e-5 | 0.0 | Lower lr - more stable |
| **E3c** | 2e-4 | 0.0 | Higher lr - faster convergence? |
| **E3d** | 1e-4 | 0.5 | Encoder 2x slower than decoder |

**Hypothesis**: Recurrent encoder's carry state may benefit from slower learning.

---

### E4: Batch Size Impact

**Purpose**: Recurrent encoder processes demos every step - does batch size matter more?

| ID | Global Batch Size | Notes |
|----|-------------------|-------|
| **E4a** | 128 | Smaller - more frequent updates |
| **E4b** | 256 | Baseline |
| **E4c** | 512 | Larger - more stable gradients |

**Analysis**: Compare convergence speed and final accuracy.

---

## Experiment Priority Queue

### Week 1: Validation

**Priority**: Verify ETRMTRM works and doesn't regress from ETRM

1. **E0a**: ETRM baseline (if not already done)
2. **E0b**: ETRMTRM baseline - CRITICAL
3. **E2b**: Confirm baseline halting params work

**Success criteria**:
- E0b reaches >90% train acc
- E0b converges in similar time as E0a
- No gradient flow issues

**If E0b fails**:
- Check encoder carry reset logic
- Verify gradient flow to encoder
- Compare encoder diagnostics with ETRM

### Week 2: Optimization

**Priority**: Find optimal configuration for recurrent encoder

4. **E1a**: Deeper grid encoder (4 layers)
5. **E2c**: Longer max_steps (32)
6. **E2d**: Higher exploration (0.7)
7. **E3b**: Lower learning rate (5e-5)

**Analysis**: Which changes improve over E0b baseline?

### Week 3: Refinement

**Priority**: Test extremes and sensitivities

8. **E2a**: Shorter max_steps (8)
9. **E2e**: No exploration (0.0)
10. **E3c**: Higher learning rate (2e-4)
11. **E4c**: Larger batch (512)

**Analysis**: Understand boundaries and sensitivities.

---

## Key Metrics to Monitor

### Recurrent Encoder Specific

**New Metrics** (should add to logging):
- `encoder_carry_norm`: L2 norm of encoder carry state
- `encoder_carry_change`: Change in carry between steps
- `encoder_convergence_steps`: Steps until carry stabilizes

**Hypothesis**:
- Carry should evolve early, stabilize later
- Large carry changes may indicate instability

### Standard Metrics

**Training Dynamics**:
- `train/accuracy`: Token-level (target: >90%)
- `train/exact_accuracy`: Sequence-level EM (target: >50%)
- `train/loss`: Overall loss
- `train/steps`: Avg ACT steps used

**Encoder Health**:
- `train/encoder_cross_sample_var`: Should be >0.3
- `grad/encoder_norm`: Should be >0.2

**Comparison with ETRM**:
- Does recurrent encoder converge faster/slower?
- Does it use more/fewer ACT steps?
- Is encoder variance higher/lower?

---

## Analysis Questions

For each experiment, document:

1. **Convergence**: Steps to reach 80%? 90%?
2. **Stability**: Are losses smooth or noisy?
3. **Halting**: Avg steps vs ETRM baseline?
4. **Encoder behavior**:
   - Does carry state evolve meaningfully?
   - Cross-sample variance compared to ETRM?
   - Gradient flow healthy?
5. **Final performance**: Train acc, train EM at convergence?

### Comparison Matrix (vs ETRM)

After all experiments, create comparison:

| Experiment | Model | Train Acc | Train EM | Avg Steps | Steps to 90% | Encoder Var | Notes |
|------------|-------|-----------|----------|-----------|--------------|-------------|-------|
| E0a | ETRM | ? | ? | ? | ? | ? | Baseline |
| E0b | ETRMTRM | ? | ? | ? | ? | ? | Recurrent |
| ... | ... | ... | ... | ... | ... | ... | ... |

**Key comparisons**:
- **E0b vs E0a**: Does recurrence help or hurt?
- **Best ETRMTRM vs E0a**: Maximum improvement from recurrence
- **Convergence speed**: Faster or slower learning?

---

## Decision Points

### After E0b (Critical Validation)

**If E0b succeeds (>90% train acc)**:
✅ ETRMTRM validated, proceed to optimization (E1, E2, E3)
✅ Document: convergence speed, avg steps, encoder variance vs ETRM

**If E0b fails (<80% train acc)**:
❌ Debug immediately:
- Check encoder carry reset logic (needs_reset handling)
- Verify gradient flow (encoder should get ~40% of gradients)
- Check encoder output variance (should be >0.3)
- Compare with ETRM run to isolate issue

**If E0b works but slower than ETRM**:
⚠️ Recurrence adds overhead without clear benefit
⚠️ Proceed cautiously, focus on finding configurations where recurrence helps

### After Week 1 (E0a, E0b, E2b)

**If recurrence shows improvement**:
✅ Proceed with full optimization (E1, E2, E3, E4)
✅ Consider testing Variant B (TRM-style encoder)

**If recurrence shows no improvement**:
⚠️ May not be worth the added complexity
⚠️ Focus on finding specific scenarios where it helps
⚠️ Consider abandoning if no clear wins

### After Week 2 (Optimization Experiments)

**If found configuration significantly better than ETRM**:
✅ Document best config
✅ Proceed to full dataset experiments
✅ Start planning Variant B exploration

**If all ETRMTRM configs similar to ETRM**:
⚠️ Recurrence may not matter for this task
⚠️ Document null result (still valuable)
⚠️ Consider: is recurrence hypothesis wrong, or wrong implementation?

---

## Expected Outcomes

### Success Scenarios

**Scenario A: Recurrence Helps Significantly**
- E0b achieves >92% vs E0a ~90%
- Recurrent encoder uses more steps productively
- Encoder variance higher (more diverse representations)
- **Implication**: Recurrence is beneficial, invest in Variant B and full dataset

**Scenario B: Recurrence Helps Slightly**
- E0b achieves ~91% vs E0a ~90%
- Small but consistent improvement
- **Implication**: Worth exploring further, but not a game-changer

**Scenario C: No Difference**
- E0b achieves ~90% same as E0a
- Similar convergence speed and behavior
- **Implication**: Recurrence not harmful but not helpful for overfit task

**Scenario D: Recurrence Hurts**
- E0b achieves <85% when E0a achieves >90%
- Slower convergence or instability
- **Implication**: Implementation bug or fundamental issue

### Failure Scenarios

**Red Flag 1: Implementation Bug**
- E0b fails completely (<50% train acc)
- Encoder variance collapses (<0.1)
- Gradient flow issues
- **Action**: Debug carry management, gradient flow, reset logic

**Red Flag 2: Instability**
- E0b training diverges or highly noisy
- Carry state exploding/vanishing
- **Action**: Check initialization, learning rate, gradient clipping

**Red Flag 3: No Improvement Despite Optimization**
- All ETRMTRM experiments ≤ ETRM baseline
- Recurrence adds complexity without benefit
- **Action**: Document null result, reconsider hypothesis

---

## Next Steps After Overfit Phase

### If ETRMTRM Shows Promise

1. **Implement Variant B** (TRM-style encoder with H/L loops)
   - Create `models/encoders/trm_style.py`
   - Test if full TRM architecture in encoder helps

2. **Full Dataset Experiments**
   - Train on ~560 puzzles
   - Evaluate on ~400 held-out puzzles
   - Success: ≥5-10% test EM (true generalization)

3. **Analysis**
   - Visualize encoder carry evolution across ACT steps
   - Analyze which puzzles benefit from recurrence
   - Compare encoder representations (ETRM vs ETRMTRM)

### If ETRMTRM Shows No Benefit

1. **Document Findings**
   - Why recurrence didn't help
   - What we learned about encoder design

2. **Alternative Directions**
   - Focus on ETRM encoder architecture improvements
   - Explore other encoder innovations (cross-demo attention, hierarchical encoding)

---

## Implementation Status

### ✅ Completed
- Base classes (`recurrent_base.py`)
- Variant A encoder (`recurrent_standard.py`)
- ETRMTRM model (`etrmtrm.py`)
- Config files (`arch/etrmtrm.yaml`, `cfg_pretrain_etrmtrm_arc_agi_1.yaml`)
- Training script (`pretrain_etrmtrm.py`)

### 📋 TODO
- [ ] Test E0a (ETRM baseline) if not done
- [ ] Run E0b (ETRMTRM baseline validation)
- [ ] Add encoder carry diagnostics to logging
- [ ] Implement Variant B if Variant A shows promise

---

## References

- **ETRM overfit experiments**: `docs/experiments/etrm_overfit_experiments.md`
- **ETRM baseline results**: `docs/experiments/etrm_overfit_results.md`
- **Job commands**: `jobs-etrmtrm-overfit.txt`
- **Implementation plan**: `/home/baris/.claude/plans/starry-inventing-tide.md`
