# ETRM Overfit Results Tracker

**Goal**: Track results from overfit phase experiments (32 groups)

**Success Criterion**: Train accuracy >90%

**Update this file** as experiments complete.

---

## Quick Status Dashboard

| Phase | Total | Completed | Success | Failed | Running |
|-------|-------|-----------|---------|--------|---------|
| Phase 1 (Encoders) | 3 | 0 | 0 | 0 | 0 |
| Phase 2 (Halting) | 5 | 0 | 0 | 0 | 0 |
| Phase 3 (Validation) | 3 | 0 | 0 | 0 | 0 |
| **Total** | **11** | **0** | **0** | **0** | **0** |

---

## Experiment Results

### E1a: Baseline Standard Encoder (MONITORING)

**Status**: ✅ In progress (existing run A4_reencode_FIXED)

**Config**:
- Encoder: standard (2 layers)
- halt_max_steps: 16
- halt_exploration_prob: 0.5
- Pretrained decoder: Yes

**Results at Step 240**:
- Train accuracy: 86.15%
- Train EM: 10.9%
- Q-halt accuracy: 91.8%
- Avg steps: 11.34

**Status**: Still improving, monitoring to >90%

**Notes**: Proves re-encoding works. Need to continue to convergence.

---

### E2a: Variational Encoder

**Status**: ⏳ Not started

**Config**:
- Encoder: hybrid_variational (4 layers)
- halt_max_steps: 16
- halt_exploration_prob: 0.5
- Pretrained decoder: Yes
- Batch size: 128

**Previous Result** (cached ETRM): 50.6% train accuracy (gradient starvation)

**Results**:
- Train accuracy: ?
- Train EM: ?
- Q-halt accuracy: ?
- Avg steps: ?
- Steps to 90%: ?

**Status**: ?

**Notes**:

---

### E2b: LPN Variational Encoder

**Status**: ⏳ Not started

**Config**:
- Encoder: lpn_variational (6 layers)
- halt_max_steps: 16
- halt_exploration_prob: 0.5
- Pretrained decoder: Yes
- Batch size: 128

**Previous Result** (cached ETRM): 51.5% train accuracy (gradient starvation)

**Results**:
- Train accuracy: ?
- Train EM: ?
- Q-halt accuracy: ?
- Avg steps: ?
- Steps to 90%: ?

**Status**: ?

**Notes**:

---

### E2c: Hybrid Standard Encoder

**Status**: ⏳ Not started

**Config**:
- Encoder: hybrid_standard (4 layers, no VAE)
- halt_max_steps: 16
- halt_exploration_prob: 0.5
- Pretrained decoder: Yes
- Batch size: 128

**Results**:
- Train accuracy: ?
- Train EM: ?
- Q-halt accuracy: ?
- Avg steps: ?
- Steps to 90%: ?

**Status**: ?

**Notes**:

---

### E1b: Lower Exploration (0.3)

**Status**: ⏳ Not started

**Config**:
- Encoder: standard (2 layers)
- halt_max_steps: 16
- halt_exploration_prob: 0.3
- Pretrained decoder: Yes

**Results**:
- Train accuracy: ?
- Train EM: ?
- Q-halt accuracy: ?
- Avg steps: ?
- Steps to 90%: ?

**Status**: ?

**Notes**:

---

### E1c: No Exploration (0.0)

**Status**: ⏳ Not started

**Config**:
- Encoder: standard (2 layers)
- halt_max_steps: 16
- halt_exploration_prob: 0.0
- Pretrained decoder: Yes

**Results**:
- Train accuracy: ?
- Train EM: ?
- Q-halt accuracy: ?
- Avg steps: ?
- Steps to 90%: ?

**Status**: ?

**Notes**:

---

### E1d: Higher Exploration (0.7)

**Status**: ⏳ Not started

**Config**:
- Encoder: standard (2 layers)
- halt_max_steps: 16
- halt_exploration_prob: 0.7
- Pretrained decoder: Yes

**Results**:
- Train accuracy: ?
- Train EM: ?
- Q-halt accuracy: ?
- Avg steps: ?
- Steps to 90%: ?

**Status**: ?

**Notes**:

---

### E3a: Shorter Max Steps (8)

**Status**: ⏳ Not started

**Config**:
- Encoder: standard (2 layers)
- halt_max_steps: 8
- halt_exploration_prob: 0.5
- Pretrained decoder: Yes

**Results**:
- Train accuracy: ?
- Train EM: ?
- Q-halt accuracy: ?
- Avg steps: ?
- Steps to 90%: ?

**Status**: ?

**Notes**:

---

### E3b: Longer Max Steps (32)

**Status**: ⏳ Not started

**Config**:
- Encoder: standard (2 layers)
- halt_max_steps: 32
- halt_exploration_prob: 0.5
- Pretrained decoder: Yes

**Results**:
- Train accuracy: ?
- Train EM: ?
- Q-halt accuracy: ?
- Avg steps: ?
- Steps to 90%: ?

**Status**: ?

**Notes**:

---

### E4a: From Scratch (No Pretrained)

**Status**: ⏳ Not started

**Config**:
- Encoder: standard (2 layers)
- halt_max_steps: 16
- halt_exploration_prob: 0.5
- Pretrained decoder: **No**
- Max epochs: 50,000

**Results**:
- Train accuracy: ?
- Train EM: ?
- Q-halt accuracy: ?
- Avg steps: ?
- Steps to 90%: ?

**Status**: ?

**Notes**: Fair comparison with TRM paper (45%)

---

### E4b: From Scratch, Deep Encoder

**Status**: ⏳ Not started

**Config**:
- Encoder: standard (4 layers)
- halt_max_steps: 16
- halt_exploration_prob: 0.5
- Pretrained decoder: **No**
- Max epochs: 50,000

**Results**:
- Train accuracy: ?
- Train EM: ?
- Q-halt accuracy: ?
- Avg steps: ?
- Steps to 90%: ?

**Status**: ?

**Notes**: Test if more capacity helps from scratch

---

### E2d: Deeper Standard Encoder

**Status**: ⏳ Not started

**Config**:
- Encoder: standard (4 layers vs 2)
- halt_max_steps: 16
- halt_exploration_prob: 0.5
- Pretrained decoder: Yes

**Results**:
- Train accuracy: ?
- Train EM: ?
- Q-halt accuracy: ?
- Avg steps: ?
- Steps to 90%: ?

**Status**: ?

**Notes**: Simple scaling test

---

## Summary Tables

### Encoder Architecture Comparison

Fill in after Phase 1 completes:

| Experiment | Encoder | Layers | Train Acc | Train EM | Steps to 90% | Status |
|------------|---------|--------|-----------|----------|--------------|--------|
| E1a | standard | 2 | 86% (step 240) | 10.9% | ? | 🔄 Running |
| E2a | hybrid_variational | 4 | ? | ? | ? | ⏳ |
| E2b | lpn_variational | 6 | ? | ? | ? | ⏳ |
| E2c | hybrid_standard | 4 | ? | ? | ? | ⏳ |
| E2d | standard | 4 | ? | ? | ? | ⏳ |

**Winner**: TBD

---

### Halting Dynamics Comparison

Fill in after Phase 2 completes:

| Experiment | halt_max_steps | exploration_prob | Train Acc | Avg Steps | Q-halt Acc | Status |
|------------|----------------|------------------|-----------|-----------|------------|--------|
| E1a | 16 | 0.5 | 86% | 11.34 | 91.8% | 🔄 |
| E1b | 16 | 0.3 | ? | ? | ? | ⏳ |
| E1c | 16 | 0.0 | ? | ? | ? | ⏳ |
| E1d | 16 | 0.7 | ? | ? | ? | ⏳ |
| E3a | 8 | 0.5 | ? | ? | ? | ⏳ |
| E3b | 32 | 0.5 | ? | ? | ? | ⏳ |

**Optimal exploration_prob**: TBD

**Optimal halt_max_steps**: TBD

---

### From Scratch vs Pretrained

Fill in after Phase 3 completes:

| Experiment | Pretrained | Encoder Layers | Train Acc | Steps to 90% | Status |
|------------|------------|----------------|-----------|--------------|--------|
| E1a | Yes | 2 | 86% | ? | 🔄 |
| E4a | No | 2 | ? | ? | ⏳ |
| E4b | No | 4 | ? | ? | ⏳ |

**Conclusion**: TBD

---

## Key Findings

### What Works

(Update as experiments complete)

### What Doesn't Work

(Update as experiments complete)

### Recommendations for Full Dataset

(Update after all experiments complete)

1. Best encoder architecture: ?
2. Best halting settings: ?
3. Need pretrained decoder: ?
4. Other findings: ?

---

## Next Steps

After overfit phase validation:

- [ ] Select top 2-3 configurations
- [ ] Create jobs file for full dataset experiments
- [ ] Update `docs/future_experiments.md` with findings
- [ ] Document lessons learned
