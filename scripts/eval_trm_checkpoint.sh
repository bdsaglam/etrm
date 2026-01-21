#!/bin/bash
# Unified evaluation script for TRM, ETRM, and ETRMTRM checkpoints
#
# Usage:
#   ./eval_trm_checkpoint.sh --checkpoint PATH [OPTIONS]
#
# Required:
#   --checkpoint PATH           Path to checkpoint file
#
# Optional:
#   --config-name NAME          Hydra config name for fallback (default: auto-detect)
#   --model-type TYPE           Model type: trm, etrm, etrmtrm, or auto (default: auto)
#   --max-eval-groups N         Limit to first N puzzle groups (default: none = full test set)
#   --output-dir DIR            Directory to save results (default: checkpoint directory)
#   --config-overrides          Additional Hydra config overrides (e.g., global_batch_size=1024)
#
# Note: Always uses checkpoint's all_config.yaml if available
#
# Examples:
#   # Evaluate TRM checkpoint (auto-detected)
#   ./eval_trm_checkpoint.sh --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071
#
#   # Evaluate ETRM checkpoint (auto-detected)
#   ./eval_trm_checkpoint.sh --checkpoint ./checkpoints/etrm-final/F1_standard/step_50000
#
#   # Evaluate on 32 groups subset
#   ./eval_trm_checkpoint.sh --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071 --max-eval-groups 32
#
#   # Override batch size via config overrides
#   ./eval_trm_checkpoint.sh --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071 --config-overrides global_batch_size=1024

set -e

# Defaults
CONFIG_NAME=""
MODEL_TYPE="auto"
MAX_EVAL_GROUPS=""
CHECKPOINT=""
CONFIG_OVERRIDES=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --config-name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --max-eval-groups)
            MAX_EVAL_GROUPS="$2"
            shift 2
            ;;
        --config-overrides)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                CONFIG_OVERRIDES+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            echo ""
            echo "Usage: $0 --checkpoint PATH [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --checkpoint PATH           Path to checkpoint file"
            echo ""
            echo "Optional:"
            echo "  --config-name NAME          Hydra config for fallback (default: auto-detect)"
            echo "  --model-type TYPE           Model type: trm, etrm, etrmtrm, or auto (default: auto)"
            echo "  --max-eval-groups N         Limit to first N puzzle groups"
            echo "  --config-overrides ...      Config overrides (e.g., global_batch_size=1024)"
            echo ""
            echo "Note: Always uses checkpoint's all_config.yaml if available"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    echo ""
    echo "Usage: $0 --checkpoint PATH [OPTIONS]"
    echo ""
    echo "Examples:"
    echo "  $0 --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071"
    echo "  $0 --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071 --max-eval-groups 32"
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Extract step number and directory for display
STEP=$(basename "$CHECKPOINT" | sed 's/step_//')
CHECKPOINT_DIR=$(dirname "$CHECKPOINT")

# Count GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Warning: No GPUs found. Running on CPU (slow!)"
    NUM_GPUS=1
fi

echo "=============================================="
echo "TRM/ETRM/ETRMTRM Checkpoint Evaluation"
echo "=============================================="
echo "Checkpoint:  $CHECKPOINT"
echo "Step:        $STEP"
echo "Model type:  $MODEL_TYPE"
[ -n "$CONFIG_NAME" ] && echo "Config:      $CONFIG_NAME" || echo "Config:      auto-detect"
echo "GPUs:        $NUM_GPUS"
[ -n "$MAX_EVAL_GROUPS" ] && echo "Max groups:  $MAX_EVAL_GROUPS" || echo "Max groups:  all (full test set)"
[ ${#CONFIG_OVERRIDES[@]} -gt 0 ] && echo "Overrides:   ${CONFIG_OVERRIDES[*]}"
echo "=============================================="
echo ""

# Build command
CMD="torchrun --nproc-per-node $NUM_GPUS \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 \
    scripts/evaluate_trm_checkpoint.py \
    --checkpoint $CHECKPOINT \
    --model-type $MODEL_TYPE"

if [ -n "$CONFIG_NAME" ]; then
    CMD="$CMD --config-name $CONFIG_NAME"
fi

if [ -n "$MAX_EVAL_GROUPS" ]; then
    CMD="$CMD --max-eval-groups $MAX_EVAL_GROUPS"
fi

if [ ${#CONFIG_OVERRIDES[@]} -gt 0 ]; then
    CMD="$CMD --config-overrides ${CONFIG_OVERRIDES[*]}"
fi

echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "=============================================="
echo "Evaluation complete!"
if [ -n "$MAX_EVAL_GROUPS" ]; then
    RESULTS_FILE="${CHECKPOINT_DIR}/eval_results_groups_${MAX_EVAL_GROUPS}_step_${STEP}.json"
else
    RESULTS_FILE="${CHECKPOINT_DIR}/eval_results_full_step_${STEP}.json"
fi
echo "Results: $RESULTS_FILE"
echo "=============================================="
