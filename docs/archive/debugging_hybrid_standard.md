# Debugging Hybrid Standard Architecture - Poor Performance Investigation

## Problem Statement

SF1 and SF2 (hybrid_standard with 4 layers) are showing:
- Low train EM (exact match accuracy)
- Low pass@k on test set
- **Most predictions are copies of test input**
- Encoder cross_sample_var looks good (so encoder is learning diverse representations)

This suggests the **TRM decoder is not using the encoder output effectively**.

## Architecture Review

I've reviewed the code and found **no obvious bugs** in:
- ✓ HybridStandardEncoder architecture (models/encoders/hybrid_standard.py)
- ✓ Data loading (dataset/fewshot_puzzle_dataset.py)
- ✓ Loss computation (models/losses.py)
- ✓ Context usage in TRM (models/recursive_reasoning/etrm.py)

However, the "copying input" behavior is a **critical symptom** that needs investigation.

## Potential Issues

### 1. **Encoder Output Not Informative** (Most Likely)
Even though cross_sample_var is good, the encoder might be learning representations that:
- Are diverse but not task-relevant
- Have the wrong scale/magnitude
- Don't capture the pattern from demos

**Evidence needed:**
- Check `train/encoder_output_norm` - should be in range [10, 100]
- Check `grad/encoder_norm` - should be > 0.2
- If encoder output is too small (<1) or too large (>1000), it could be ignored by TRM

### 2. **TRM Not Sensitive to Context**
The TRM reasoning module might have learned to ignore the encoder context:
- This can happen if encoder gradients are too weak early in training
- TRM might find a "shortcut" that doesn't use context

**Evidence needed:**
- Run the debug script to test encoder sensitivity (see below)

### 3. **Wrong Batch Size**
SF1 and SF2 use `global_batch_size=128`, which is **half the default (256)**.
- Smaller batch size → more frequent updates → different training dynamics
- Could affect encoder learning

### 4. **Architecture Mismatch**
Hybrid_standard uses:
- LPNGridEncoder (deep, CLS pooling)
- DemoSetEncoder (cross-attention)

This combination might have issues:
- CLS pooling loses position information
- Cross-attention might not work well with CLS-pooled representations

## Diagnostic Steps

### Step 1: Check W&B Metrics for SF1

Run this to see the metrics:
```bash
python notebooks/etrm_experiment_analysis.ipynb  # Set PROJECT="etrm-semi-final"
```

**Key metrics to check:**
- `train/encoder_cross_sample_var` - you said this looks good (>0.3)
- `grad/encoder_norm` - **CRITICAL**: should be > 0.2 (if < 0.05, encoder has gradient starvation)
- `train/encoder_output_norm` - should be reasonable (10-100 range)
- `train/encoder_output_mean`, `train/encoder_output_std` - check for collapse (std ~ 0)
- `train/exact_accuracy` - how low is it exactly?
- `train/steps` - is the model using all 16 steps or halting early?

### Step 2: Run Data Loading Debug

```bash
python debug_data_loading.py
```

This will verify that demos are being loaded correctly from the same puzzle as the query.

### Step 3: Run Encoder Sensitivity Test (Most Important)

Find the latest checkpoint for SF1:
```bash
# Find checkpoint
ls -lh checkpoints/SF1_hybrid_std_baseline/

# Run test
python debug_encoder_output.py checkpoints/SF1_hybrid_std_baseline/checkpoint_latest.pt
```

This will tell you:
1. Does encoder produce diverse outputs for different demos? (Should: yes, since cross_sample_var is good)
2. **Does TRM output change when encoder context changes?** (Critical test!)
3. Is the model copying the input?

### Step 4: Compare with Standard Encoder

If hybrid_standard is failing, try **standard encoder** (2-layer) as a control:
```bash
# In jobs-etrm-semi-final.txt, add:
[ ] torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain_etrm.py --config-name cfg_pretrain_etrm_arc_agi_1 +project_name="etrm-semi-final" +run_name="SF_DEBUG_standard_2layer" arch.encoder_type=standard arch.encoder_num_layers=2 arch.halt_max_steps=16 arch.halt_exploration_prob=0.5 global_batch_size=256 epochs=1000 eval_interval=200
```

Standard encoder is simpler and has been tested more. If it works but hybrid_standard doesn't, that narrows down the issue.

## Hypotheses Ranked by Likelihood

### Hypothesis 1: Gradient Flow Issue (HIGH)
**Symptom:** `grad/encoder_norm` < 0.1

**Cause:** Encoder receives weak gradients, doesn't learn meaningful representations

**Fix:**
- Increase batch size back to 256 (more stable gradients)
- Use encoder_lr_scale=10 to train encoder faster
- Check if frozen decoder helps: `freeze_decoder_steps=1000`

### Hypothesis 2: Encoder Output Scale Wrong (MEDIUM)
**Symptom:** `train/encoder_output_norm` << 1 or >> 100

**Cause:** Encoder output too small (TRM ignores it) or too large (dominates everything)

**Fix:**
- Add normalization layer after encoder
- Check initialization of LPNGridEncoder (might need different init scale)

### Hypothesis 3: Architecture Incompatibility (MEDIUM)
**Symptom:** Standard encoder works but hybrid_standard doesn't

**Cause:** CLS pooling + cross-attention doesn't work well together

**Fix:**
- Try lpn_standard (mean pooling instead of cross-attention)
- Try hybrid_variational (similar architecture, maybe works better)

### Hypothesis 4: Training Dynamics Bug (LOW)
**Symptom:** Model halts too early or too late

**Cause:** Q-head or exploration settings wrong

**Fix:**
- Check `train/steps` - should average around 8-10
- Try `halt_exploration_prob=0.7` (SF2 has this, wait for results)

## Quick Fixes to Try

### Fix 1: Use Standard Encoder First
Replace hybrid_standard with standard to establish baseline:
```
arch.encoder_type=standard
arch.encoder_num_layers=2
```

### Fix 2: Increase Batch Size
```
global_batch_size=256  # Back to default
```

### Fix 3: Train Encoder Separately First
```
freeze_decoder_steps=2000  # Train encoder only for first 2000 steps
encoder_lr_scale=1.0       # Then unfreeze decoder
```

### Fix 4: Check Checkpoint from SF2
SF2 is still running with `explore=0.7`. If it succeeds, that tells us exploration is the issue.

## Next Steps

1. **Immediately**: Run the diagnostic scripts and report back the metrics
2. **Review**: Check W&B for SF1 gradient metrics
3. **Compare**: Wait for SF2 to finish, see if explore=0.7 helps
4. **Fallback**: Run standard encoder as control experiment

Let me know what the diagnostic scripts show and I can help narrow down the exact issue!
