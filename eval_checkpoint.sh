#!/bin/bash
# Evaluate ETRM/ETRMTRM checkpoint on full test set using all GPUs
#
# Usage:
#   ./eval_checkpoint.sh --run-name NAME [--checkpoint-step N] [--global-batch-size N]
#
# Examples:
#   ./eval_checkpoint.sh --run-name F1_standard              # Latest checkpoint
#   ./eval_checkpoint.sh --run-name F1_standard --checkpoint-step 174622
#   ./eval_checkpoint.sh --run-name F2_hybrid_var
#   ./eval_checkpoint.sh --run-name F3_etrmtrm
#   ./eval_checkpoint.sh --run-name F4_lpn_var
#   ./eval_checkpoint.sh --run-name F1_standard --global-batch-size 1024
#
# The script auto-detects:
#   - Model type (etrm vs etrmtrm) based on run name
#   - Config file based on model type
#   - Checkpoint directory

set -e

# Constants
MAX_EVAL_GROUPS=32

# Parse arguments
RUN_NAME=""
CHECKPOINT_STEP=""
BATCH_SIZE="512"

while [[ $# -gt 0 ]]; do
    case $1 in
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --checkpoint-step)
            CHECKPOINT_STEP="$2"
            shift 2
            ;;
        --batch-size|--global-batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$RUN_NAME" ]; then
    echo "Usage: $0 --run-name NAME [--checkpoint-step N] [--global-batch-size N]"
    echo ""
    echo "Examples:"
    echo "  $0 --run-name F1_standard"
    echo "  $0 --run-name F2_hybrid_var --checkpoint-step 25000"
    echo "  $0 --run-name F3_etrmtrm"
    echo "  $0 --run-name F1_standard --global-batch-size 1024"
    echo ""
    echo "Available runs:"
    ls -1 ./checkpoints/etrm-final/ 2>/dev/null || echo "  None found"
    exit 1
fi

# Base checkpoint directory
CHECKPOINT_BASE="./checkpoints/etrm-final"
CHECKPOINT_DIR="${CHECKPOINT_BASE}/${RUN_NAME}"

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    echo ""
    echo "Available runs:"
    ls -1 ${CHECKPOINT_BASE}/ 2>/dev/null || echo "  None found"
    exit 1
fi

# Auto-detect model type and config based on run name
if [[ "$RUN_NAME" == *"etrmtrm"* ]] || [[ "$RUN_NAME" == *"ETRMTRM"* ]]; then
    MODEL_TYPE="etrmtrm"
    CONFIG_NAME="cfg_pretrain_etrmtrm_arc_agi_1"
else
    MODEL_TYPE="etrm"
    CONFIG_NAME="cfg_pretrain_etrm_arc_agi_1"
fi

# Find checkpoint
if [ -n "$CHECKPOINT_STEP" ]; then
    CHECKPOINT="${CHECKPOINT_DIR}/step_${CHECKPOINT_STEP}"
else
    # Find latest checkpoint (highest step number)
    CHECKPOINT=$(ls -v ${CHECKPOINT_DIR}/step_* 2>/dev/null | tail -1)
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Available checkpoints in ${CHECKPOINT_DIR}:"
    ls -la ${CHECKPOINT_DIR}/step_* 2>/dev/null || echo "  None found"
    exit 1
fi

# Extract step number for display
STEP=$(basename "$CHECKPOINT" | sed 's/step_//')

# Count GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs found"
    exit 1
fi

echo "=============================================="
echo "ETRM/ETRMTRM Full Test Set Evaluation"
echo "=============================================="
echo "Run name:    $RUN_NAME"
echo "Step:        $STEP"
echo "Model type:  $MODEL_TYPE"
echo "Config:      $CONFIG_NAME"
echo "Checkpoint:  $CHECKPOINT"
echo "GPUs:        $NUM_GPUS"
[ -n "$BATCH_SIZE" ] && echo "Batch size:  $BATCH_SIZE (global)"
echo "Max groups:  $MAX_EVAL_GROUPS"
echo "=============================================="
echo ""

# Build command
CMD="torchrun --nproc-per-node $NUM_GPUS \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 \
    evaluate_checkpoint.py \
    --checkpoint $CHECKPOINT \
    --config-name $CONFIG_NAME \
    --model-type $MODEL_TYPE \
    --max-eval-groups $MAX_EVAL_GROUPS"

# Add batch size if specified (global batch size)
if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --global-batch-size $BATCH_SIZE"
fi

# Run evaluation with all GPUs
eval $CMD

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results: ${CHECKPOINT_DIR}/eval_results_groups_${MAX_EVAL_GROUPS}_step_${STEP}.json"
echo "=============================================="
