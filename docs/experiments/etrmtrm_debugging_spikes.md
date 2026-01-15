# Debugging Metric Spikes in ETRMTRM

## Observation
E0b training shows spikes in metrics. Hypothesis: **Decoder struggles with evolving context**.

## Root Cause: By Design, Context Changes Every Step

**The core difference:**
- **ETRM**: `context = encoder(demos)` → **static**, decoder sees same context every ACT step
- **ETRMTRM**: `context = encoder(carry, demos)` → **evolves**, decoder sees different context each ACT step

**Why this causes instability:**
```
Step 1: Decoder sees context_1, produces output_1
Step 2: Encoder refines z_e based on gradients
        Decoder NOW sees context_2 (different!)
        Decoder must adapt to moving target
Step 3: Context changes again → context_3
        ...
```

The decoder is **chasing a moving target**. If encoder updates are too large or inconsistent, decoder can't keep up.

---

## Potential Causes

### 1. **Encoder Updates Too Large** (Most Likely)
Encoder changes z_e dramatically between steps.

**Why this causes spikes:**
- Encoder has high learning rate or large gradients
- z_e changes by large amount each step (e.g., norm change > 0.5)
- Decoder receives very different context signal
- Decoder's previous reasoning becomes invalid
- Metrics spike as decoder re-adjusts

**Check in W&B:**
- Is `encoder_output_norm` fluctuating significantly?
- Are there sudden jumps in `encoder_cross_sample_var`?

---

### 2. **Encoder-Decoder Coupling Instability**
Encoder and decoder updates reinforce each other's instability.

**Why this causes spikes:**
- Decoder gradient → encoder gets large update → z_e changes significantly
- Changed z_e → decoder sees different context → decoder output changes
- New decoder output → even larger encoder gradient
- Positive feedback loop → instability

**Check in W&B:**
- Do `grad/encoder_norm` and `grad/inner_norm` spike together?
- Does `train/loss` have sudden jumps that recover?

---

### 3. **Gradient Clipping Not Effective**
Global gradient clipping (norm=1.0) may clip total but allow encoder to dominate.

**Why this causes spikes:**
- Total gradient clipped to 1.0
- But encoder may get 0.8 of that, decoder only 0.2
- Encoder updates too aggressively relative to decoder
- Imbalanced learning rates

**Check in W&B:**
- Is `grad/total_norm` frequently at 1.0 (clipped)?
- What's the ratio `grad/encoder_norm / grad/total_norm`?
- Is encoder getting >60% of gradients consistently?

---

## Diagnostic Steps

### Step 1: Measure z_e Change Per Step (Critical!)
Add to `models/recursive_reasoning/etrmtrm.py` in `_forward_train()` after line 200:

```python
# Before encoder forward, save previous z_e
prev_z_e = carry.encoder_carry.z_e.clone()

# ... existing encoder forward ...
encoder_carry = self.encoder(...)
context = encoder_carry.get_context()

# After encoder forward, measure change
with torch.no_grad():
    # How much did z_e change?
    z_e_delta = (encoder_carry.z_e - prev_z_e).norm(dim=-1).mean()
    encoder_diagnostics["z_e_step_change"] = z_e_delta.item()

    # Relative change (percentage)
    z_e_norm = prev_z_e.norm(dim=-1).mean()
    if z_e_norm > 1e-6:
        encoder_diagnostics["z_e_step_change_pct"] = (z_e_delta / z_e_norm).item()
```

**What to look for in W&B:**
- `z_e_step_change`: Absolute norm of change per step
  - **Good**: 0.1 - 0.3 (moderate, stable updates)
  - **Warning**: 0.3 - 0.5 (significant but maybe ok)
  - **Bad**: >0.5 (too large, decoder can't adapt)
- `z_e_step_change_pct`: Relative change
  - **Good**: <10% per step
  - **Bad**: >20% per step
- **Correlation**: Do spikes in loss/accuracy align with spikes in `z_e_step_change`?

---

### Step 2: Check Gradient Balance
Already logged, but analyze in W&B:

```python
# Check ratio of encoder vs total gradients
encoder_ratio = grad/encoder_norm / grad/total_norm
```

**What to look for:**
- Encoder getting >70% of gradients → encoder dominating, updates too aggressive
- Encoder getting <20% → encoder starving, not learning enough
- **Ideal**: 30-50% for balanced learning

---

### Step 3: Compare with ETRM (Sanity Check)
Run ETRM to confirm spikes are ETRMTRM-specific:

```bash
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
    pretrain_etrm.py --config-name cfg_pretrain_etrm_arc_agi_1 \
    max_train_groups=32 max_eval_groups=32 epochs=20000 eval_interval=1000 \
    +project_name="etrmtrm-overfit" +run_name="E0a_etrm_stable"
```

**Expected:** ETRM should be much more stable (static context = no moving target)

---

## Potential Fixes

### Fix 1: Slow Down Encoder Updates (Recommended First Try)
**Problem:** Encoder updating too fast → z_e changes too much → decoder can't adapt.

**Solution:** Use slower learning rate for encoder:

```yaml
# In cfg_pretrain_etrmtrm_arc_agi_1.yaml
encoder_lr_scale: 0.3  # Encoder learns at 30% of decoder's rate
```

**Test variants:**
- `encoder_lr_scale: 0.5` - Moderate slowdown
- `encoder_lr_scale: 0.3` - Aggressive slowdown
- `encoder_lr_scale: 0.1` - Very slow (might hurt performance)

**Rationale:** Smaller encoder updates → smaller z_e changes → decoder sees more stable context.

---

### Fix 2: EMA Smoothing of z_e (Recommended) ✅ IMPLEMENTED

**Problem:** z_e changes abruptly between steps.

**Solution:** Smooth z_e updates with exponential moving average:

```python
# In recurrent_standard.py forward():
# After computing new z_e, before detach:

# Smooth with previous state (90% old, 10% new)
z_e_smoothed = 0.9 * carry.z_e.detach() + 0.1 * z_e

return RecurrentEncoderCarry(z_e=z_e_smoothed.detach())
```

**Status:** Implemented in `models/encoders/recurrent_standard.py:212-218`

**Rationale:** Gradual context evolution instead of abrupt changes. Decoder tracks slowly moving target.

---

### Fix 3: Residual Scaling for Encoder Updates
**Problem:** Encoder updates have same magnitude as initial state.

**Solution:** Scale down residual connections:

```python
# In recurrent_standard.py forward():
# Current:
z_e = rms_norm(z_e + attn_out, ...)

# Fixed:
alpha = 0.3  # Scaling factor
z_e = rms_norm(z_e + alpha * attn_out, ...)
z_e = rms_norm(z_e + alpha * mlp_out, ...)
```

**Rationale:** Smaller residual = smaller change per step, more stable.

---

### Fix 4: Stop Gradient on Previous z_e
**Problem:** Encoder-decoder coupling creates feedback loop.

**Solution:** Break gradient flow through previous z_e:

```python
# In recurrent_standard.py forward():
# Use detached previous state as query (no gradient through carry)
z_e_query = carry.z_e.detach()

# Cross-attention with detached query
attn_out = self.cross_attn(
    queries=z_e_query,  # Detached
    keys=demo_encodings,
    values=demo_encodings,
    key_mask=demo_mask,
)
z_e = rms_norm(carry.z_e + attn_out, ...)  # Update original
```

**Rationale:** Prevents positive feedback loop between encoder and decoder gradients.

---

### Fix 5: Freeze Encoder Initially (Conservative)
**Problem:** Encoder destabilizes decoder in early training.

**Solution:** Train decoder first with frozen encoder:

```yaml
# In cfg_pretrain_etrmtrm_arc_agi_1.yaml
freeze_decoder_steps: 0
encoder_lr_scale: 0.0  # Freeze encoder for first 5000 steps

# Then resume with:
encoder_lr_scale: 0.3  # After step 5000
```

**Rationale:** Let decoder stabilize with static context first, then gradually introduce evolution.

---

### Fix 6: Layer Normalization on Context
**Problem:** z_e magnitude unbounded, explodes/vanishes.

**Solution:** Normalize context before feeding to decoder:

```python
# In etrmtrm.py _forward_train(), after getting context:
context = encoder_carry.get_context()

# Normalize context
context = F.layer_norm(context, context.shape[-1:])  # Normalize last dim

# Now pass normalized context to decoder
inner_carry, logits, q_logits = self.inner(inner_carry, new_current_data, context)
```

**Rationale:** Bounded context magnitude → decoder sees consistent scale.

---

## Experiment Plan

### Phase 1: Diagnose (First Priority)
**Goal:** Understand if/why z_e changes cause instability.

1. **Add z_e change diagnostics** (Diagnostic Step 1)
   - Modify `etrmtrm.py` to log `z_e_step_change` and `z_e_step_change_pct`
   - Re-run E0b or continue current run

2. **Analyze current E0b in W&B:**
   - Check `encoder_output_norm` - is it fluctuating wildly?
   - Check `grad/encoder_norm / grad/total_norm` ratio - is encoder dominating?
   - Identify patterns: when do spikes occur?

3. **Run ETRM comparison** (optional but helpful):
   - Confirms if instability is ETRMTRM-specific
   - Provides stable baseline for comparison

**Decision point after Phase 1:**
- If `z_e_step_change > 0.5` and correlates with spikes → Proceed to Phase 2
- If encoder getting >70% of gradients → Try Fix 1 (slow encoder)
- If spikes are small and model converges well → **No fix needed**, this is acceptable

---

### Phase 2: Test Fixes (Priority Order)

**If diagnosis shows large z_e changes:**

#### Quick Wins (Config-only, no code changes):
1. **E0b_slow_encoder_05**: `encoder_lr_scale=0.5`
2. **E0b_slow_encoder_03**: `encoder_lr_scale=0.3`

**If slowdown not enough:**

#### Code Changes (More invasive):
3. **E0b_ema_smooth**: Implement Fix 2 (EMA smoothing)
4. **E0b_residual_scale**: Implement Fix 3 (residual scaling with α=0.3)
5. **E0b_stop_grad**: Implement Fix 4 (stop gradient on previous z_e)

**If all else fails:**

#### Conservative Approach:
6. **E0b_freeze_encoder**: Freeze encoder for first 5000 steps, then unfreeze
7. **E0b_layer_norm**: Add layer normalization to context

---

## Success Criteria

**Diagnosis successful if:**
- Measured `z_e_step_change` and correlated with metric spikes
- Identified if encoder updates are too large (>0.5 norm change)
- Checked if encoder dominating gradients (>70%)
- Confirmed ETRMTRM is less stable than ETRM (evolving context issue)

**Fix successful if:**
- `z_e_step_change` reduced to <0.3 per step
- Metrics smoother (visual inspection of W&B plots)
- Training still converges to >90% train acc
- Convergence not significantly slower

**Fix NOT needed if:**
- Spikes are small (<5% fluctuation) and don't prevent convergence
- Model reaches >90% train acc despite spikes
- Final metrics (last 1000 steps) are stable

---

## Immediate Action Items

### Option A: Quick Check (5 min)
**If E0b is still running or you have W&B access:**
1. Open E0b W&B run
2. Check these plots:
   - `train/loss` - how big are the spikes?
   - `encoder_output_norm` - is it stable or fluctuating?
   - `grad/encoder_norm` vs `grad/inner_norm` - what's the ratio?
3. **Decision:**
   - If spikes small and model learning → wait for E0b to finish
   - If spikes large and model unstable → proceed to Option B

### Option B: Add Diagnostics (10 min)
1. Add `z_e_step_change` logging to `etrmtrm.py` (code in Diagnostic Step 1)
2. Launch new experiment with diagnostics:
   ```bash
   # In jobs-etrmtrm-overfit.txt, add:
   [ ] torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
       pretrain_etrmtrm.py --config-name cfg_pretrain_etrmtrm_arc_agi_1 \
       max_train_groups=32 max_eval_groups=32 epochs=20000 eval_interval=1000 \
       +project_name="etrmtrm-overfit" +run_name="E0b_diagnostics"
   ```
3. Watch first 100 steps for `z_e_step_change` values
4. **Decision:** If >0.5, proceed to Option C

### Option C: Try Quick Fix (No code change)
1. Launch with slower encoder LR:
   ```bash
   [ ] torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
       pretrain_etrmtrm.py --config-name cfg_pretrain_etrmtrm_arc_agi_1 \
       max_train_groups=32 max_eval_groups=32 epochs=20000 eval_interval=1000 \
       encoder_lr_scale=0.3 \
       +project_name="etrmtrm-overfit" +run_name="E0b_slow_encoder"
   ```
2. Compare stability with original E0b
3. If better → use this for all future experiments
4. If not enough → try EMA smoothing (Fix 2)

---

## Quick Reference: What to Check in W&B

**For current E0b run, focus on these metrics:**

1. **Primary stability indicators:**
   - `train/loss` - should decrease smoothly, not spike
   - `train/exact_accuracy` - should increase smoothly
   - `train/lm_loss` - main prediction loss

2. **Encoder health:**
   - `encoder_output_norm` - should be stable (~5-10 range)
   - `encoder_cross_sample_var` - should be stable (>0.15)

3. **Gradient balance:**
   - `grad/encoder_norm` - should be <0.7 × `grad/total_norm`
   - `grad/total_norm` - if always 1.0, clipping is active

4. **Learning progress:**
   - Does model reach >90% train acc eventually?
   - Are final 1000 steps stable or still spiking?

**If you see:** Large spikes (>10% fluctuation) in loss that persist throughout training → likely needs fixing.

**If you see:** Small spikes (<5% fluctuation) early on, then stabilizes → probably fine, no action needed.
