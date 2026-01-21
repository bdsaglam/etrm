# Evaluation Scripts

## TRM Checkpoint Evaluation

Evaluate original TRM checkpoints (with puzzle embeddings).

### Quick Start

```bash
# Evaluate on 32 groups (quick test)
./scripts/eval_trm_checkpoint.sh \
    --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071 \
    --max-eval-groups 32

# Evaluate on full test set (all 400 groups)
./scripts/eval_trm_checkpoint.sh \
    --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071
```

### Interface Comparison

Both Python script and shell wrapper now have the same interface:

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint PATH` | Path to checkpoint file | **required** |
| `--config-name NAME` | Hydra config for fallback | `cfg_pretrain_arc_agi_1` |
| `--max-eval-groups N` | Limit to first N groups | None (full test) |
| `--global-batch-size N` | Global batch size | From config |
| `--output-dir DIR` | Output directory | Checkpoint dir |

**Note:** The script **always** uses the checkpoint's `all_config.yaml` if available. This ensures correct architecture parameters (L_cycles, H_cycles, etc.).

### Examples

```bash
# Full test set with all defaults
./scripts/eval_trm_checkpoint.sh \
    --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071

# Quick validation on 4 groups
./scripts/eval_trm_checkpoint.sh \
    --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071 \
    --max-eval-groups 4

# Override batch size
./scripts/eval_trm_checkpoint.sh \
    --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071 \
    --global-batch-size 1024
```

### Direct Python Usage

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

## Checkpoint Inspection

Verify checkpoint integrity and view statistics:

```bash
python scripts/inspect_checkpoint.py \
    --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071
```

Shows:
- Weight shapes and statistics
- NaN/Inf detection
- EMA smoothing indicators
- Config parameters

## Notes

### Automatic Config Loading

The evaluation scripts **always** load `all_config.yaml` from the checkpoint directory when available. This ensures the correct architecture parameters (L_cycles, H_cycles, etc.) are used.

The config path is resolved as:
```python
checkpoint_config_path = Path(checkpoint).parent / "all_config.yaml"
```

If the checkpoint config is not found, the script falls back to the Hydra config specified by `--config-name` (default: `cfg_pretrain_arc_agi_1`) with a warning message.

### Previous Interface (Deprecated)

The old shell script interface required separate arguments:
```bash
# OLD (deprecated):
./eval_trm_checkpoint.sh --run-name arc_v1_public --checkpoint-step 518071

# NEW (current):
./scripts/eval_trm_checkpoint.sh --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071
```

The new interface matches the Python script and is more explicit.
