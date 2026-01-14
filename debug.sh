#!/bin/bash
# Generic debug script - minimal run to catch errors quickly
# Edit the parameters below for your specific experiment

# ============================================================================
# CONFIGURATION - Modify these for your experiment
# ============================================================================
SCRIPT="pretrain_etrmtrm.py"           # Training script to run
CONFIG="cfg_pretrain_etrmtrm_arc_agi_1"  # Hydra config name
PROJECT="etrmtrm-debug"                # W&B project name
RUN_NAME="debug_run"                   # W&B run name

# Architecture-specific overrides (modify as needed)
ARCH_OVERRIDES="arch.recurrent_encoder_type=recurrent_standard arch.encoder_num_layers=2"

# ============================================================================
# MINIMAL CONFIG (usually don't need to change)
# ============================================================================
MAX_TRAIN_GROUPS=4
MAX_EVAL_GROUPS=4
EPOCHS=10
EVAL_INTERVAL=5
BATCH_SIZE=8
HALT_MAX_STEPS=4

# ============================================================================
# RUN
# ============================================================================
echo "Running debug experiment..."
echo "  Script: $SCRIPT"
echo "  Config: $CONFIG"
echo "  Groups: $MAX_TRAIN_GROUPS train, $MAX_EVAL_GROUPS eval"
echo "  Epochs: $EPOCHS"
echo ""

torchrun --nproc-per-node 1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
    $SCRIPT \
    --config-name $CONFIG \
    max_train_groups=$MAX_TRAIN_GROUPS \
    max_eval_groups=$MAX_EVAL_GROUPS \
    epochs=$EPOCHS \
    eval_interval=$EVAL_INTERVAL \
    global_batch_size=$BATCH_SIZE \
    arch.halt_max_steps=$HALT_MAX_STEPS \
    $ARCH_OVERRIDES \
    +project_name="$PROJECT" \
    +run_name="$RUN_NAME" \
    2>&1 | tee debug.log

echo ""
echo "Debug run completed. Check debug.log for details."
echo ""
echo "Quick diagnostics:"
grep -i "error\|failed\|traceback" debug.log | head -20 || echo "  No errors found ✓"
