# TRM Checkpoint Evaluation Guide

## Quick Start

Evaluate TRM checkpoints with correct architecture parameters automatically loaded:

```bash
# Quick test (32 groups)
./scripts/eval_trm_checkpoint.sh \
    --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071 \
    --max-eval-groups 32

# Full evaluation (all 400 groups)
./scripts/eval_trm_checkpoint.sh \
    --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071
```

## Key Feature: Automatic Config Loading

The evaluation scripts **always use the checkpoint's `all_config.yaml`** when available. This is critical because:

1. **Prevents architecture mismatch** - Uses correct L_cycles, H_cycles, L_layers from training
2. **Ensures correct results** - Wrong architecture parameters cause garbage predictions (~3% instead of ~40%)
3. **No manual overrides needed** - Just point to the checkpoint file

## What Was Fixed

### The Bug
The evaluation script had a path resolution bug that caused config loading to fail:
```python
# OLD (buggy):
checkpoint_config_path = REPO_ROOT / "checkpoints" / args.checkpoint / "all_config.yaml"
# Result: /checkpoints/checkpoints/... (double "checkpoints")

# NEW (fixed):
checkpoint_config_path = Path(args.checkpoint).parent / "all_config.yaml"
# Result: ./checkpoints/official_trm/arc_v1_public/all_config.yaml
```

### The Impact
When config loading failed, the script fell back to default config with **wrong architecture parameters**:

| Parameter | Checkpoint | Default (Wrong) | Impact |
|-----------|------------|-----------------|--------|
| L_cycles  | 4          | 6               | 90% accuracy loss |
| H_cycles  | 3          | 3               | ✓ |
| L_layers  | 2          | 2               | ✓ |

Running with L_cycles=6 when trained with L_cycles=4 caused 3.75% instead of ~40% accuracy.

## Usage

### Shell Script (Recommended)

Automatically detects GPUs and runs distributed evaluation:

```bash
./scripts/eval_trm_checkpoint.sh --checkpoint PATH [OPTIONS]
```

**Arguments:**
- `--checkpoint PATH` - Path to checkpoint file (required)
- `--max-eval-groups N` - Limit to first N puzzle groups (default: all)
- `--global-batch-size N` - Override batch size
- `--output-dir DIR` - Save results to directory
- `--config-name NAME` - Fallback Hydra config (only if checkpoint config missing)

### Python Script

For more control or integration:

```bash
# Single GPU
python scripts/evaluate_trm_checkpoint.py \
    --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071 \
    --max-eval-groups 32

# Multi-GPU
torchrun --nproc-per-node 4 scripts/evaluate_trm_checkpoint.py \
    --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071 \
    --max-eval-groups 32
```

## Checkpoint Verification

Inspect checkpoint integrity before evaluation:

```bash
python scripts/inspect_checkpoint.py \
    --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071
```

Shows:
- ✅ Weight statistics and shapes
- ✅ EMA smoothing indicators
- ✅ NaN/Inf detection
- ✅ Config parameters

## Expected Results

| Evaluation | Pass@1 | Notes |
|------------|--------|-------|
| 32 groups subset | ~12-15% | Quick validation |
| Full test (400 groups) | ~40% | Official result |
| With wrong config | ~3-4% | Bug reproduction |

## Files Modified

| File | Purpose |
|------|---------|
| `scripts/evaluate_trm_checkpoint.py` | Fixed config path resolution, always use checkpoint config |
| `scripts/eval_trm_checkpoint.sh` | Updated interface to match Python script |
| `scripts/inspect_checkpoint.py` | New diagnostic tool |
| `scripts/README.md` | Comprehensive usage guide |

## Migration from Old Interface

**Old (deprecated):**
```bash
./eval_trm_checkpoint.sh --run-name arc_v1_public --checkpoint-step 518071
```

**New (current):**
```bash
./scripts/eval_trm_checkpoint.sh \
    --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071
```

The new interface:
- ✅ More explicit (full checkpoint path)
- ✅ Matches Python script interface
- ✅ Works with any checkpoint location
- ✅ Always uses correct config

## Troubleshooting

### Low accuracy (~3-4%)

**Cause:** Wrong architecture parameters (likely L_cycles mismatch)

**Solution:**
1. Verify checkpoint config exists: `ls $(dirname CHECKPOINT)/all_config.yaml`
2. Check evaluation logs for "Using checkpoint config" message
3. Use inspection tool: `python scripts/inspect_checkpoint.py --checkpoint PATH`

### Config not found warning

**Cause:** Checkpoint directory missing `all_config.yaml`

**Solution:**
- Official checkpoints include this file - ensure download is complete
- For old checkpoints, manually create or specify correct config via `--config-name`

### Shape mismatch errors

**Cause:** Puzzle embedding dimensions don't match test set

**Solution:** The script handles this automatically by averaging embeddings. No action needed.

## References

- Original issue: `docs/trm-checkpoint-issue.md`
- Fix details: `docs/trm-checkpoint-fix.md`
- Official checkpoint: https://huggingface.co/arcprize/trm_arc_prize_verification
- TRM paper: https://arxiv.org/abs/2510.04871
