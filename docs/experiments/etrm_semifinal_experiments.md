# ETRM Semi-Final Experiments Design

**Date**: January 12, 2026
**Project**: ETRM (Encoder-based Tiny Recursive Models)
**Phase**: Semi-Final (Full Dataset)
**Goal**: Identify best architecture + hyperparameters for final 50k epoch training

---

## Executive Summary

The semi-final experiments test **6 configurations** on the **full training set** (~560 puzzle groups, ~100k training samples) to identify the top 1-2 performers for final long-run training (50k epochs).

**Primary Question**: Which encoder architecture and exploration setting achieves the best test set generalization?

**Duration**: 1000 epochs per experiment (~6-8 hours each on 4×A100 GPUs)
**Total Time**: ~12-24 hours for all 6 experiments (running in parallel)

---

## Background and Motivation

### Why Semi-Final Phase?

The **overfit phase** (32 groups, results in `notebooks/results/etrm-overfit_comparison.csv`) identified promising configurations:

| Exp | Encoder | Layers | Explore | Train EM @ 5K | Variance | Grad Ratio | Health |
|-----|---------|--------|---------|---------------|----------|------------|---------|
| E1a | standard | 2 | 0.5 | 23.8% | 0.13 | 0.20 | ⚠️ Warning |
| E1b | standard | 2 | 0.3 | 37.1% | 0.34 | 0.41 | ✅ Healthy |
| **E1d** | **standard** | **2** | **0.7** | **48.4%** | **0.42** | **0.41** | **✅ Healthy** |
| E2a | hybrid_var | 4 | 0.5 | 19.5% | 2.45 | 0.24 | ✅ Healthy |
| **E2c** | **hybrid_std** | **4** | **0.5** | **29.7%** | **1.70** | **0.26** | **✅ Healthy** |
| E2b | lpn_var | 6 | 0.5 | 23.4% | 1.75 | 0.07 | ⚠️ Warning |

**Key Findings**:
1. **High exploration (0.7)** significantly improves standard encoder (48.4% vs 23.8%)
2. **Hybrid architectures** show healthy gradient flow (grad/encoder ratio ~0.25)
3. **High variance correlates** with better learning (E1d: 0.42, E2c: 1.70)
4. **LPN variational** suffers from gradient starvation (grad/encoder ratio: 0.07)

### Next Step: Full Dataset Validation

Overfit experiments can overfit to 32 groups. Need to validate on **full dataset** to:
- Confirm which architectures generalize to 560 groups
- Identify train/test gap (overfitting indicators)
- Select top performers for final 50k epoch training

---

## Research Questions

### Primary Question
**Which encoder architecture and exploration probability achieves the best test set accuracy (ARC/pass@1) on the full dataset?**

### Secondary Questions

1. **Architecture Comparison**:
   - Does `hybrid_standard` (4-layer + cross-attention) outperform `standard` (2-layer + mean)?
   - Does `hybrid_variational` (variational bottleneck) improve over `hybrid_standard` (deterministic)?

2. **Hyperparameter Sensitivity**:
   - Is `explore=0.7` consistently better than `explore=0.5` across architectures?
   - Does higher exploration help simple encoders as much as complex ones?

3. **Generalization**:
   - What is the train/test accuracy gap for each configuration?
   - Which architectures maintain healthy gradients on full dataset?

4. **Training Dynamics**:
   - Are learning curves still improving at 1k epochs (suggesting room for 50k training)?
   - How do ACT steps vary during training (2-16 dynamic range)?

---

## Experimental Design

### Dataset

- **Training set**: ~560 puzzle groups (ARC-AGI training + concept subsets)
- **Test set**: ~400 puzzle groups (ARC-AGI evaluation subset)
- **Augmentation**: ~1000× per puzzle (color permutation + dihedral transforms)
- **Samples**: ~100k training samples (after augmentation)

**Important**: Test set demos are **never seen during training** (true few-shot generalization).

### Architecture Matrix

| Exp | Encoder Type | Layers | Architecture | Bottleneck |
|-----|--------------|--------|--------------|------------|
| SF1 | hybrid_standard | 4 | Per-demo → cross-attention | None (deterministic) |
| SF2 | hybrid_standard | 4 | Per-demo → cross-attention | None (deterministic) |
| SF3 | hybrid_variational | 4 | Per-demo → cross-attention | Variational (μ/σ + reparam) |
| SF4 | hybrid_variational | 4 | Per-demo → cross-attention | Variational (μ/σ + reparam) |
| SF5 | standard | 2 | Per-demo → mean aggregation | None (deterministic) |
| SF6 | standard | 2 | Per-demo → mean aggregation | None (deterministic) |

### Hyperparameter Grid

| Exp | Max Steps | Explore Prob | Batch Size | KL Weight |
|-----|-----------|--------------|------------|-----------|
| SF1 | 16 | 0.5 | 128 | N/A |
| SF2 | 16 | 0.7 | 128 | N/A |
| SF3 | 16 | 0.5 | 128 | 0.0 |
| SF4 | 16 | 0.7 | 128 | 0.0 |
| SF5 | 16 | 0.5 | 256 | N/A |
| SF6 | 16 | 0.7 | 256 | N/A |

**Rationale**:
- **Max steps=16**: From TRM paper, sufficient for most ARC tasks
- **Explore=0.5**: TRM paper baseline
- **Explore=0.7**: Best empirical result from overfit (E1d: 48.4% EM)
- **KL=0**: Deterministic variational (avoids stochastic sampling issues discovered in variational experiments)
- **Batch size**: 128 for 4-layer (OOM at 256), 256 for 2-layer

### Training Configuration

```yaml
# Common to all experiments
epochs: 1000                    # ~6-8 hours on 4×A100
eval_interval: 200              # 5 evaluation points
lr: 1e-4                       # From overfit experiments
grad_clip_norm: 1.0            # Prevents gradient explosion
halt_max_steps: 16             # Dynamic halting limit
encoder_reencode: True         # Full gradient coverage
pretrained_decoder: Yes        # Transfer learning from TRM checkpoint
```

### Evaluation Protocol

1. **Timing**: Every 200 epochs (0, 200, 400, 600, 800, 1000)
2. **Metrics**:
   - **Primary**: Test set EM (ARC/pass@1) on evaluation subset
   - **Secondary**: Train EM, train/test gap
   - **Diagnostic**: Gradient ratios, encoder variance, Q-halt accuracy

3. **Voting**: Majority voting across ~912 augmented versions per test puzzle

4. **Health Indicators**:
   - ✅ Healthy: `grad/encoder_norm > 0.2`
   - ⚠️ Warning: `0.1 < grad/encoder_norm < 0.2`
   - ❌ Unhealthy: `grad/encoder_norm < 0.1` (gradient starvation)

---

## Hypotheses

### Hypothesis 1: Hybrid Standard Wins

**Prediction**: `hybrid_standard` with `explore=0.7` (SF2) achieves highest test EM.

**Rationale**:
- Highest variance in overfit (1.70) indicates diverse representations
- Healthy gradient flow (grad/encoder ratio: 0.26)
- Set transformer layer should improve over simple mean aggregation
- Explore=0.7 showed 2× improvement for standard encoder (48.4% vs 23.8%)

**Expected Test EM**: 35-45%

---

### Hypothesis 2: Exploration Matters More Than Architecture

**Prediction**: `explore=0.7` outperforms `explore=0.5` for all encoder types.

**Rationale**:
- Overfit showed 48.4% vs 23.8% for standard encoder (2× improvement)
- Higher exploration encourages more diverse reasoning paths
- Prevents premature convergence to suboptimal halting strategies

**Expected Pattern**: SF2 > SF1, SF4 > SF3, SF6 > SF5

---

### Hypothesis 3: Variational Bottleneck Doesn't Help

**Prediction**: `hybrid_variational` performs similarly to or worse than `hybrid_standard`.

**Rationale**:
- Variational encoders failed with KL > 0 (stochastic sampling prevents convergence)
- With KL=0, both are deterministic - only difference is extra bottleneck parameters
- Overfit showed lower EM for variational (19.5% vs 29.7%) despite healthy gradients
- Variational bottleneck adds complexity without clear benefit for deterministic ARC tasks
- Both use identical cross-attention architecture up to the bottleneck

**Expected**: SF3/SF4 ≤ SF1/SF2 (variational doesn't improve performance)

---

### Hypothesis 4: Simple Encoder Competitive

**Prediction**: `standard` encoder (SF6) is competitive with `hybrid` architectures.

**Rationale**:
- Overfit showed 48.4% EM (highest of all experiments)
- Fewer parameters (2 layers vs 4) → less overfitting risk
- Simpler representation may generalize better

**Expected**: SF6 within 5-10% of SF2

---

## Expected Outcomes

### Performance Ranking (Predicted)

| Rank | Exp | Encoder | Explore | Expected Test EM | Expected Train/Test Gap |
|------|-----|---------|---------|------------------|------------------------|
| 1 | SF2 | hybrid_std | 0.7 | **35-45%** | Small (5-10%) |
| 2 | SF6 | standard | 0.7 | **30-40%** | Small (5-10%) |
| 3 | SF4 | hybrid_var | 0.7 | **30-38%** | Medium (10-15%) |
| 4 | SF1 | hybrid_std | 0.5 | **25-35%** | Medium (10-15%) |
| 5 | SF3 | hybrid_var | 0.5 | **20-30%** | Medium (10-15%) |
| 6 | SF5 | standard | 0.5 | **15-25%** | Large (15-20%) |

### Learning Curve Patterns

**Expected at 1k epochs**:
- All configurations still improving (not converged)
- Test EM increasing monotonically (no catastrophic overfitting)
- Train/test gap stabilizing by 800-1000 epochs

**Implication**: Supports running to 50k epochs for final training

---

## Success Criteria

### Primary Success Metric

**Test set exact match accuracy (ARC/pass@1)** on evaluation subset:
- ✅ **Excellent**: > 40% test EM (competitive with TRM paper's 45%)
- ✅ **Good**: 30-40% test EM (viable for final training)
- ⚠️ **Marginal**: 20-30% test EM (needs hyperparameter tuning)
- ❌ **Poor**: < 20% test EM (architecture or config issue)

### Secondary Metrics

1. **Gradient Health**:
   - ✅ `grad/encoder_norm > 0.2` (healthy gradient flow)
   - ⚠️ `0.1 < grad/encoder_norm < 0.2` (acceptable but monitor)
   - ❌ `grad/encoder_norm < 0.1` (gradient starvation)

2. **Train/Test Gap**:
   - ✅ < 10% gap (good generalization)
   - ⚠️ 10-20% gap (some overfitting)
   - ❌ > 20% gap (severe overfitting)

3. **Learning Dynamics**:
   - ✅ Still improving at 1k epochs (promising for 50k training)
   - ⚠️ Plateaued by 1k epochs (may converge before 50k)
   - ❌ Decreasing after 1k epochs (overfitting)

4. **ACT Behavior**:
   - ✅ Steps vary 2-16 during training (dynamic halting working)
   - ✅ Q-halt accuracy > 90% (learning when to stop)
   - ❌ Steps stuck at constant value (not dynamic)

---

## Decision Rules for Final Training

### Selection Criteria

Top 1-2 configurations selected for **50k epoch final training** based on:

1. **Test EM** (primary): Highest test set accuracy
2. **Gradient health**: Must have `grad/encoder_norm > 0.15`
3. **Learning curve**: Still improving at 1k epochs
4. **Train/test gap**: Prefer < 15% gap

### Tie-Breaking

If two configs have similar test EM (< 3% difference):
1. Prefer **simpler architecture** (fewer parameters)
2. Prefer **lower exploration** (more stable training)
3. Prefer **higher gradient health** (better optimization)

### Expected Final Selection

Based on hypotheses and overfit results:
- **Primary choice**: SF2 (hybrid_std, explore=0.7)
- **Backup choice**: SF6 (standard, explore=0.7) if SF2 overfits

---

## Analysis Plan

### Automated Analysis

Run `notebooks/etrm_experiment_analysis.ipynb` after all experiments complete:

```python
PROJECT = "etrm-semi-final"  # W&B project name
```

**Notebook outputs**:
1. Performance comparison table (test EM, train EM, gap)
2. Learning curves (train/test EM over epochs)
3. Gradient health metrics (encoder norm, inner norm, ratios)
4. Pareto frontier (accuracy vs model complexity)
5. Confusion matrices and failure analysis

### Key Visualizations

1. **Test EM comparison** (bar chart, 6 configs)
2. **Learning curves** (line chart, train/test over time)
3. **Gradient health heatmap** (6 configs × health metrics)
4. **Train/test scatter** (overfitting analysis)
5. **ACT steps distribution** (histogram, dynamic range check)

### Report Generation

Create `docs/experiments/etrm-semifinal-results.md` with:
- Executive summary
- Performance ranking table
- Winner selection rationale
- Recommendations for final training
- Hyperparameter tuning suggestions (if needed)

---

## Risks and Mitigations

### Risk 1: OOM Errors

**Risk**: 4-layer encoders with batch size 128 may OOM on some GPUs.

**Mitigation**: Already reduced from 256 to 128 for hybrid architectures. Standard encoders use 256.

**Status**: ✅ SF1 completed successfully, no OOM issues.

---

### Risk 2: Poor Generalization

**Risk**: High overfit performance doesn't translate to test set.

**Mitigation**:
- Train/test gap analysis identifies overfitting early
- If gap > 20%, consider regularization or different architecture

**Monitor**: Track train/test gap at each evaluation point.

---

### Risk 3: Training Instability

**Risk**: Some configurations may diverge or show loss spikes.

**Mitigation**:
- Gradient clipping enabled (1.0)
- Learning rate annealing (cosine schedule)
- Monitor loss curves for instability

**Recovery**: If loss spikes > 10×, reduce LR by 2× and continue.

---

### Risk 4: Insufficient Training Time

**Risk**: 1k epochs may not be enough to see true performance differences.

**Mitigation**:
- Learning curve analysis shows if still improving
- If all still improving at 1k, extend to 2k before selecting winners

**Decision point**: Review at 600 and 1000 epochs.

---

## Timeline

| Milestone | Target | Status |
|-----------|--------|--------|
| SF1 start | Jan 12, 2026 | ✅ Complete |
| SF2 start | Jan 12, 2026 | ✅ Running |
| SF3-SF6 start | Jan 12, 2026 | ⏳ Pending |
| All experiments complete | Jan 13, 2026 | ⏳ Pending |
| Notebook analysis | Jan 13, 2026 | ⏳ Pending |
| Winner selection | Jan 13, 2026 | ⏳ Pending |
| Final training (50k) | Jan 14+, 2026 | ⏳ Pending |

**Estimated completion**: All semi-final experiments complete by Jan 13 evening.

---

## Next Steps After Semi-Final

### 1. Analysis (Day 1)

- Run experiment analysis notebook
- Generate comparison tables and visualizations
- Document results in report

### 2. Winner Selection (Day 1)

- Select top 1-2 configurations based on criteria
- Validate selection with domain expert review
- Document rationale for choice

### 3. Final Training (Week 1-2)

- Train selected configs for 50k epochs (~5-7 days each)
- Monitor for convergence and overfitting
- Checkpoint every 5k epochs

### 4. Evaluation (Day 14)

- Final evaluation on full test set
- Compare to TRM paper baseline (45% EM)
- Document final performance

### 5. Paper Writing (Week 2-3)

- Results section with tables and figures
- Architecture comparison
- Ablation studies (explore prob, encoder type)
- Failure analysis and discussion

---

## References

### Related Documents

- **Overfit results**: `docs/experiments/etrm_overfit_results.md`
- **Variational analysis**: `docs/variational-encoder-analysis.md`
- **Experiment guide**: `docs/training-path-review-guide.md`
- **Training verification**: `docs/training-path-verification.md`

### Data Files

- **Job commands**: `jobs-etrm-semi-final.txt`
- **Results CSV**: `notebooks/results/etrm-semifinal_comparison.csv` (generated after completion)
- **W&B project**: `bdsaglam/etrm-semi-final`

### Key Papers

- TRM Paper: "Less is More: Recursive Reasoning with Tiny Networks"
- LPN Paper: "Learning Perceptual Priors for Program Synthesis" (variational encoder reference)
- ARC-AGI Paper: "The Abstraction and Reasoning Corpus"

---

## Appendix: Configuration Details

### Full Command Examples

**SF1** (hybrid_std, explore=0.5):
```bash
torchrun --nproc-per-node 4 \
    pretrain_etrm.py \
    --config-name cfg_pretrain_etrm_arc_agi_1 \
    +project_name="etrm-semi-final" \
    +run_name="SF1_hybrid_std_baseline" \
    arch.encoder_type=hybrid_standard \
    arch.encoder_num_layers=4 \
    arch.halt_max_steps=16 \
    arch.halt_exploration_prob=0.5 \
    global_batch_size=128 \
    epochs=1000 \
    eval_interval=200
```

**SF2** (hybrid_std, explore=0.7):
```bash
torchrun --nproc-per-node 4 \
    pretrain_etrm.py \
    --config-name cfg_pretrain_etrm_arc_agi_1 \
    +project_name="etrm-semi-final" \
    +run_name="SF2_hybrid_std_explore0.7" \
    arch.encoder_type=hybrid_standard \
    arch.encoder_num_layers=4 \
    arch.halt_max_steps=16 \
    arch.halt_exploration_prob=0.7 \
    global_batch_size=128 \
    epochs=1000 \
    eval_interval=200
```

### Encoder Architecture Details

**Standard (2-layer, mean aggregation)**:
```
Demo 1 → [Transformer 2L] → z₁ ──┐
Demo 2 → [Transformer 2L] → z₂ ──┼─→ Mean(z₁, z₂, z₃, ...) → context
Demo 3 → [Transformer 2L] → z₃ ──┘
```

**Hybrid Standard (4-layer, cross-attention, deterministic)**:
```
Demo 1 → [LPN Grid Encoder] → z₁ ──┐
Demo 2 → [LPN Grid Encoder] → z₂ ──┼─→ [Cross-Attention Set Encoder] → context
Demo 3 → [LPN Grid Encoder] → z₃ ──┘      (query tokens attend to all demos)
```

**Hybrid Variational (4-layer, cross-attention, variational)**:
```
Demo 1 → [LPN Grid Encoder] → z₁ ──┐
Demo 2 → [LPN Grid Encoder] → z₂ ──┼─→ [Cross-Attention Set Encoder] → context
Demo 3 → [LPN Grid Encoder] → z₃ ──┘      (query tokens attend to all demos)
                                            ↓
                                       Pool to mean → μ, σ
                                            ↓
                                 z ~ N(μ, σ) (reparameterize)
                                            ↓
                               [Decode Projection] → final context
```

**Key Difference**: Both hybrids use cross-attention. Only `hybrid_variational` adds the variational bottleneck (μ/σ → z → decode).

---

**Document Version**: 1.0
**Last Updated**: January 12, 2026
**Authors**: ETRM Research Team
**Status**: Active (experiments in progress)
