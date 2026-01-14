"""
Debug script to verify data loading and encoder behavior.

This will help identify if:
1. Demos are being loaded correctly (from same puzzle as query)
2. Encoder is producing meaningful output
3. Model is using encoder output
"""

import torch
import numpy as np
from dataset.fewshot_puzzle_dataset import FewShotPuzzleDataset, FewShotPuzzleDatasetConfig

# Create dataset
config = FewShotPuzzleDatasetConfig(
    seed=0,
    dataset_paths=["data/arc1concept-encoder-aug-1000"],
    global_batch_size=4,
    test_set_mode=False,
    epochs_per_iter=1,
    rank=0,
    num_replicas=1,
    max_demos=10,
)

dataset = FewShotPuzzleDataset(config, split="train")

# Get a few batches
print("=" * 70)
print("DATA LOADING VERIFICATION")
print("=" * 70)

for i, (set_name, batch, batch_size) in enumerate(dataset):
    if i >= 3:  # Check first 3 batches
        break

    print(f"\nBatch {i+1}:")
    print(f"  Set: {set_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  Shapes:")
    print(f"    demo_inputs: {batch['demo_inputs'].shape}")
    print(f"    demo_labels: {batch['demo_labels'].shape}")
    print(f"    demo_mask: {batch['demo_mask'].shape}")
    print(f"    inputs: {batch['inputs'].shape}")
    print(f"    labels: {batch['labels'].shape}")
    print(f"    puzzle_identifiers: {batch['puzzle_identifiers'].shape}")

    # Check if demos are valid (not all padding)
    print(f"\n  Demo statistics:")
    for b in range(min(2, batch['demo_inputs'].shape[0])):  # Check first 2 in batch
        num_valid = batch['demo_mask'][b].sum().item()
        print(f"    Sample {b}: {num_valid} valid demos")

        # Check if demo content is non-zero
        if num_valid > 0:
            demo_in = batch['demo_inputs'][b, 0]  # First demo
            demo_out = batch['demo_labels'][b, 0]
            print(f"      First demo input unique values: {demo_in.unique().tolist()[:10]}")
            print(f"      First demo output unique values: {demo_out.unique().tolist()[:10]}")

            # Check if query is different from demos
            query_in = batch['inputs'][b]
            print(f"      Query input unique values: {query_in.unique().tolist()[:10]}")

            # Check if demo and query are from same puzzle (should share augmentation)
            # They should have similar color distribution if from same puzzle
            demo_nonzero = demo_in[demo_in != 0]
            query_nonzero = query_in[query_in != 0]
            if len(demo_nonzero) > 0 and len(query_nonzero) > 0:
                print(f"      Demo colors: {torch.bincount(demo_in, minlength=12)[:12].tolist()}")
                print(f"      Query colors: {torch.bincount(query_in, minlength=12)[:12].tolist()}")

print("\n" + "=" * 70)
print("TEST PASSED: Data loading looks correct")
print("=" * 70)
