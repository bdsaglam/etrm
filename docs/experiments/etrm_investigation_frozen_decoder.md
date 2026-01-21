# Experiment: Frozen Decoder Test

## Objective
Test if ETRM encoder can learn with frozen pretrained TRM decoder.

## Hypothesis
If encoder can't learn with frozen decoder, there's likely a:
1. Gradient flow bug
2. Architecture issue (encoder too weak)
3. Optimization/LR issue

## Setup

### Run Command
```bash
python pretrain_etrm.py --config-name cfg_test_frozen_decoder
```

### Key Config
- **Decoder**: Frozen at official TRM checkpoint (step_518071)
- **Encoder**: Trainable, lr=3e-4
- **Data**: 32 training groups (quick test)
- **Duration**: 20k epochs (~2 eval cycles)

## Metrics to Monitor

### Critical Metrics (Real-time)

1. **`grad/encoder_norm`**: Should be > 0 (e.g., 0.1-1.0)
   - Zero → Gradients not flowing ❌
   - Vanishing (< 0.001) → Gradient blocking ❌
   - Stable (0.1-1.0) → Learning is possible ✅

2. **`train/lm_loss`**: Should decrease over time
   - Flat → Encoder not learning ❌
   - Decreasing → Encoder is learning something ✅

3. **`train/encoder_token_std`**: Variance across samples
   - Low (< 0.1) → Encoder outputting similar embeddings (not learning) ❌
   - Increasing → Encoder learning diversity ✅

4. **`train/encoder_output_mean/std`**: Overall statistics
   - Should be stable (not exploding/vanishing)

### Eval Metrics (Every 10k epochs)

5. **`ARC/pass@1`**: Accuracy on test set
   - Random: ~0.25% (1/400)
   - If > 2-3%: Encoder is doing something useful ✅
   - If stuck at ~0%: Encoder not learning meaningful representations ❌

## Expected Outcomes

### Scenario A: Encoder Learns ✅
```
grad/encoder_norm: 0.3-1.0 (stable)
train/lm_loss: 8.5 → 7.0 (decreasing)
train/encoder_token_std: 0.05 → 0.15 (increasing)
ARC/pass@1: 0% → 5-10% (improving)
```
**Conclusion**: Gradients flow, encoder works. Problem is elsewhere.

### Scenario B: Encoder Stuck ❌
```
grad/encoder_norm: < 0.001 or zero
train/lm_loss: ~8.5 (flat)
train/encoder_token_std: ~0.05 (flat)
ARC/pass@1: ~0% (random)
```
**Conclusion**: Gradient blocking bug. Need to investigate forward pass.

### Scenario C: Encoder Trains But Useless ⚠️
```
grad/encoder_norm: 0.3-1.0 (stable)
train/lm_loss: 8.5 → 7.0 (decreasing)
train/encoder_token_std: 0.05 → 0.15 (increasing)
ARC/pass@1: ~0% (no improvement)
```
**Conclusion**: Encoder learns *something*, but not useful task representations. Architecture too weak?

## Debugging Commands

### Check Training Logs (WandB)
```bash
# Look for the run
wandb login  # If not logged in
# Go to: https://wandb.ai/<your-team>/ETRM-FrozenDecoder-Test
```

### Inspect Checkpoint
```bash
# After 10k epochs, check weights
ls checkpoints/ETRM-FrozenDecoder-Test/frozen_decoder_test/

# Verify decoder is frozen (weights should match official checkpoint)
python scripts/verify_decoder_loading.py \
    --trm-checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071 \
    --etrm-checkpoint ./checkpoints/ETRM-FrozenDecoder-Test/frozen_decoder_test/step_10000
```

### Check Gradient Flow Manually
```python
# Add to train_batch_encoder() in pretrain_etrm.py temporarily:
for name, param in train_state.model.named_parameters():
    if param.grad is not None and ".encoder." in name:
        print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
```

## Next Steps Based on Results

### If Scenario A (Encoder Learns)
1. Try training with decoder unfrozen but at different LR
2. Check if decoder+encoder joint training causes interference
3. Investigate Hypothesis 4 (learning rate scheduling)

### If Scenario B (Encoder Stuck)
1. Add explicit gradient logging at each layer
2. Check for `.detach()` calls in forward pass
3. Verify encoder output reaches decoder loss computation
4. Investigate Hypothesis 2 (gradient flow) in detail

### If Scenario C (Trains But Useless)
1. Check encoder output variance (is it too small?)
2. Try stronger encoder architecture (more layers, wider)
3. Check if encoder output range matches puzzle_emb range in TRM
4. Investigate Hypothesis 7 (architecture too weak)

## Files
- Config: `config/cfg_test_frozen_decoder.yaml`
- Training script: `pretrain_etrm.py`
- Investigation doc: `docs/etrm-low-performance-investigation.md`
