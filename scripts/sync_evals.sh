#!/bin/bash
# Sync eval results from amax1 to amax7
# This script should be run on amax7 to PULL eval results from amax1

# Set subdirectory under 'checkpoints' to sync eval results for.
SUB_DIRECTORY="Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_4"

# Source is amax1 (remote), target is amax7 (local).
REMOTE_HOST="baris@144.122.52.7"  # amax1
REMOTE_DIR="/home/baris/repos/TinyRecursiveModels/checkpoints/$SUB_DIRECTORY"
LOCAL_DIR="/home/baris/repos/trm-original/checkpoints/$SUB_DIRECTORY"

# Ensure the local directory exists
mkdir -p "$LOCAL_DIR"

# Use rsync to pull results from amax1 to amax7
rsync -avzP --stats "$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"

# Alternative: use scp if rsync is not available
# scp -r "$REMOTE_HOST:$REMOTE_DIR/"* "$LOCAL_DIR/"

echo "Done! Eval results synced from $REMOTE_HOST:$REMOTE_DIR to $LOCAL_DIR"
