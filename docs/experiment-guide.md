# Experiment Guide

## Overview

**Main Research Path**: ETRM (Encoder-based TRM with dynamic halting)

**Why ETRM**: It differs from TRM only in the encoder/embeddings component, which is precisely our research question: *Can we replace learned puzzle embeddings with a neural encoder and achieve true few-shot generalization?*

**Baseline**: TRM (original paper) - 45% on ARC-AGI-1 with learned embeddings

---

## Experimental Dimensions

### 1. Encoder Architecture

The encoder is the core of our research contribution. Different architectures may learn different types of pattern representations from demonstration pairs.

#### Current Architectures

| Architecture | Description | Status | Train Acc (32 groups) |
|--------------|-------------|--------|---------------------|
| **Standard** | 2-layer Transformer, mean pooling | ✅ Baseline | 86% (ETRM, step 240, improving) |
| **Variational** | Variational encoder with KL regularization | ⚠️ Tested (cached ETRM) | 50.6% (failed due to gradient starvation) |
| **LPN** | LPN-style encoder with learned tokens | ⚠️ Tested (cached ETRM) | 51.5% (failed due to gradient starvation) |
| **Hybrid** | Combination of encoder types | ⚠️ Tested (cached ETRM) | Need to retest with re-encoding |

#### New Architecture Ideas

**ETRM² (Encoder with Internal Recurrent State)**

Give encoder its own recurrent latent state (z_E), similar to how decoder has (z_H, z_L):

```python
# Current ETRM: Encoder produces context in one shot
context = encoder(demos)  # No internal state

# ETRM²: Encoder has internal recurrent state
class RecurrentEncoder:
    def __init__(self):
        # Encoder has its own latent state, like decoder's z_H, z_L
        pass

    def forward(self, demos):
        z_E = self.init_state()

        # Refine encoder state (similar to decoder's H/L levels)
        for refinement_step in range(encoder_refinement_steps):
            z_E = self.refine(z_E, demos)

        context = self.project(z_E)
        return context

# Full carry becomes: (z_E, z_H, z_L) per sample
```

**Rationale**:
- Decoder has recurrent state (z_H, z_L) that refines during ACT steps
- Why not encoder too? Could refine pattern extraction from demos
- Encoder could have hierarchical reasoning about patterns (like decoder's H/L levels)

**Trade-off**: More parameters and compute, but could learn richer pattern representations

**Other Ideas**:
- **Cross-attention encoder**: Attend between demo pairs to find commonalities
- **Hierarchical encoder**: Encode at multiple levels (token → grid → pattern)
- **Set transformer**: Explicitly model demos as a set (permutation invariant)

### 2. Pre-trained Decoder

Whether to use a decoder pre-trained on TRM's embedding mode.

| Configuration | Rationale | Status |
|---------------|-----------|--------|
| **With pre-trained decoder** | Proven to improve training | ✅ Current default |
| **Without pre-trained decoder** | True comparison with TRM paper | 📝 Needed for ablation |

**Note**: All experiments so far use pre-trained decoder. We should run at least one experiment from scratch to ensure fair comparison with TRM paper's 45% result.

### 3. Halting Dynamics

Parameters controlling when samples halt and how halting is learned.

| Parameter | Description | Current Default | Exploration Range |
|-----------|-------------|-----------------|-------------------|
| `halt_max_steps` | Maximum ACT steps before forced halt | 16 | 8, 16, 32 |
| `halt_exploration_prob` | Probability of forcing continuation during training | 0.5 | 0.0, 0.3, 0.5, 0.7 |
| Halt loss weight | Weight of Q-halt loss in total loss | (implicit) | Could make explicit |

**Research questions**:
- Does exploration help encoder learning? (forces samples through more steps)
- What's the optimal max_steps for puzzle complexity?
- Should halt loss be weighted differently than prediction loss?

### 4. Training Hyperparameters

Standard ML hyperparameters that could affect learning.

| Parameter | Current | Notes |
|-----------|---------|-------|
| Learning rate | (check config) | Critical for encoder convergence |
| Gradient clipping | 1.0 | Essential for stability (from docs) |
| Batch size | 256 | Per GPU |
| Encoder depth | 2 layers | Could try 3-4 |
| Encoder hidden size | 512 | Could try 768, 1024 |
| Pooling strategy | Mean | Could try max, attention-weighted |

---

## Success Criteria

All experiments must pass **both** criteria:

### Criterion 1: Overfitting Test (Small Subset)

**Setup**: Train on 32 puzzle groups (small subset)

**Success**: Model must achieve **>90% train accuracy** within reasonable training time

**Rationale**: If model can't overfit to small data, something is fundamentally broken (bug or incompatible architecture/hyperparameters)

**Failure modes**:
- Accuracy stuck at 35-50% → Check for gradient starvation (encoder getting gradients?)
- Accuracy stuck at 70-80% → Architecture may be too weak or hyperparameters wrong
- Training diverges → Gradient clipping issue or learning rate too high
- Training extremely slow → Performance bug (check encoder is efficient)

**Action if failed**:
1. Check for bugs (gradient flow, data processing, loss computation)
2. Try different hyperparameters (learning rate, gradient clipping)
3. If still failing, abandon this architecture/configuration

### Criterion 2: Generalization Test (Full Dataset)

**Setup**: Train on full training set (~560 puzzles), evaluate on held-out test set (~400 puzzles)

**Success**: **≥5-10% Exact Match (EM)** on test set

**Rationale**:
- TRM achieves 45% with learned embeddings (not true generalization)
- Our goal is true few-shot generalization to unseen puzzles
- 5-10% EM on truly unseen puzzles would demonstrate encoder is learning useful patterns
- This is a HIGH bar - encoder must generalize from demos alone

**Metrics to track**:
- Exact Match (EM) on test set - primary metric
- Token accuracy on test set - secondary metric
- Pass@2 (best of 2 predictions) - important for final evaluation
- Train/test EM gap - measures overfitting

**Action if failed**:
1. Analyze test predictions - is encoder learning anything?
2. Check train set performance - did it overfit well?
3. If train is good but test is 0% → Encoder not learning transferable patterns
4. May need to try different encoder architecture or training approach

---

## Experiment Workflow

### Phase 1: Overfitting Test (Fast Iteration)

**Goal**: Validate architecture/hyperparameters work

```bash
# Small subset (32 groups) to test overfitting
torchrun --nproc-per-node 4 pretrain_encoder_original.py \
    --config-name cfg_pretrain_encoder_original_arc_agi_1 \
    max_train_groups=32 max_eval_groups=32 \
    +project_name="etrm-dev" \
    +run_name="<experiment_name>_overfit"
```

**Success checkpoint**: >90% train accuracy

**Expected time**: ~2-4 hours

### Phase 2: Generalization Test (Full Dataset)

**Goal**: Measure true few-shot generalization

```bash
# Full dataset
torchrun --nproc-per-node 4 pretrain_encoder_original.py \
    --config-name cfg_pretrain_encoder_original_arc_agi_1 \
    +project_name="etrm-full" \
    +run_name="<experiment_name>_full"
```

**Success checkpoint**: ≥5-10% test EM

**Expected time**: ~1-2 days

### Phase 3: Analysis

For successful experiments:
1. Compare train vs test EM (overfitting degree)
2. Analyze prediction patterns on test set
3. Compare encoder representations (e.g., via t-SNE)
4. Check halting behavior (avg steps, exploration effectiveness)

---

## Current Status (Jan 9, 2026)

### Validated

✅ **ETRM with standard encoder**:
- Overfitting: 86% train accuracy at step 240 (still improving)
- Generalization: Not yet tested (need to continue training to convergence first)
- Status: Main baseline, proving re-encoding works

### Need to Re-test (Previous Cached ETRM Failed)

⚠️ **Variational encoder**: Failed with gradient starvation (50.6% train accuracy)
⚠️ **LPN encoder**: Failed with gradient starvation (51.5% train accuracy)
⚠️ **Hybrid encoder**: Need to retest

**Action**: Re-run these with ETRM re-encoding (not cached)

### Not Yet Tested

📝 **ETRM² (recurrent encoder)**: New idea, needs implementation
📝 **Training from scratch**: Ablation without pre-trained decoder
📝 **Different halting dynamics**: Exploration with different halt parameters

---

## Proposed Experiment Queue

### Immediate (Next 1-2 Weeks)

1. **Continue ETRM standard encoder** to convergence
   - Phase 1: Already at 86%, monitor to >90%
   - Phase 2: Once validated, run full dataset

2. **Re-test variational encoder** (with re-encoding)
   - Phase 1: Overfit test on 32 groups
   - Phase 2: If passes, full dataset

3. **Re-test LPN encoder** (with re-encoding)
   - Phase 1: Overfit test on 32 groups
   - Phase 2: If passes, full dataset

### Short-term (2-4 Weeks)

4. **ETRM from scratch** (no pre-trained decoder)
   - Phase 1: Overfit test
   - Compare with TRM paper's 45% fairly

5. **ETRM² implementation** (recurrent encoder)
   - Phase 1: Implement + overfit test
   - Phase 2: If promising, full dataset

### Medium-term (1-2 Months)

6. **Halting dynamics ablation**
   - Try halt_exploration_prob ∈ {0.0, 0.3, 0.7}
   - Try halt_max_steps ∈ {8, 32}

7. **Encoder architecture variants**
   - Deeper encoders (3-4 layers)
   - Different pooling (attention-weighted)
   - Cross-attention between demos

---

## Experiment Tracking Template

For each experiment, document:

```markdown
### Experiment: <name>

**Date**: YYYY-MM-DD
**Goal**: Brief description
**Changes from baseline**: What's different

**Configuration**:
- Encoder: <architecture>
- Pre-trained decoder: Yes/No
- halt_max_steps: <value>
- halt_exploration_prob: <value>
- Other hyperparams: <list>

**Phase 1 Results (Overfit - 32 groups)**:
- Train accuracy: <value>%
- Steps to converge: <value>
- Issues encountered: <any bugs/problems>
- Status: ✅ Pass / ❌ Fail

**Phase 2 Results (Full dataset)** (if Phase 1 passed):
- Train EM: <value>%
- Test EM: <value>%
- Pass@2: <value>%
- Avg steps: <value>
- Status: ✅ Pass (≥5% test EM) / ⚠️ Partial (some generalization) / ❌ Fail (0% test EM)

**Analysis**:
- Key observations
- Why it worked/failed
- Next steps

**WandB Run**: <link>
```

---

## References

- **Baseline results**: `docs/progress_2026_01_09.md`
- **Training modes**: `docs/training_modes_comparison.md`
- **Architecture details**: `docs/project.md`
- **Job file**: `jobs.txt` (for experiment commands)
