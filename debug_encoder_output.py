"""
Debug script to verify encoder is producing meaningful output.

Run this on a trained checkpoint to check:
1. Encoder output statistics
2. Whether encoder output varies with different demos
3. Whether TRM is sensitive to encoder output
"""

import torch
import sys
import hydra
from omegaconf import DictConfig
from models.recursive_reasoning.etrm import TRMWithEncoder
import numpy as np

def test_encoder_sensitivity(checkpoint_path: str):
    """Test if the trained model is sensitive to encoder output."""

    print("=" * 70)
    print("ENCODER SENSITIVITY TEST")
    print("=" * 70)

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
    else:
        print("ERROR: No config in checkpoint")
        return

    # Create model
    model = TRMWithEncoder(config_dict['arch'])
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print("Model loaded successfully")

    # Create dummy input
    batch_size = 4
    seq_len = 900
    max_demos = 3

    batch = {
        'demo_inputs': torch.randint(0, 12, (batch_size, max_demos, seq_len)),
        'demo_labels': torch.randint(0, 12, (batch_size, max_demos, seq_len)),
        'demo_mask': torch.ones(batch_size, max_demos, dtype=torch.bool),
        'inputs': torch.randint(0, 12, (batch_size, seq_len)),
        'labels': torch.randint(0, 12, (batch_size, seq_len)),
        'puzzle_identifiers': torch.zeros(batch_size, dtype=torch.long),
    }

    print("\n" + "="*70)
    print("TEST 1: Encoder produces diverse outputs for different demos")
    print("="*70)

    with torch.no_grad():
        # Encode with original demos
        encoder_output_1 = model.encoder(
            batch['demo_inputs'],
            batch['demo_labels'],
            batch['demo_mask'],
            return_full_output=True
        )
        context_1 = encoder_output_1.context if hasattr(encoder_output_1, 'context') else encoder_output_1

        # Encode with different demos (shuffle)
        demo_inputs_shuffled = batch['demo_inputs'][torch.randperm(batch_size)]
        demo_labels_shuffled = batch['demo_labels'][torch.randperm(batch_size)]

        encoder_output_2 = model.encoder(
            demo_inputs_shuffled,
            demo_labels_shuffled,
            batch['demo_mask'],
            return_full_output=True
        )
        context_2 = encoder_output_2.context if hasattr(encoder_output_2, 'context') else encoder_output_2

        # Compute difference
        context_diff = (context_1 - context_2).abs().mean().item()
        context_1_norm = context_1.norm(dim=-1).mean().item()

        print(f"Context 1 norm: {context_1_norm:.4f}")
        print(f"Context 2 norm: {context_2.norm(dim=-1).mean().item():.4f}")
        print(f"Absolute difference: {context_diff:.4f}")
        print(f"Relative difference: {context_diff / (context_1_norm + 1e-8):.4f}")

        if context_diff / (context_1_norm + 1e-8) < 0.01:
            print("⚠️  WARNING: Encoder output barely changes with different demos!")
        else:
            print("✓ Encoder produces different outputs for different demos")

    print("\n" + "="*70)
    print("TEST 2: Model output changes with encoder context")
    print("="*70)

    with torch.no_grad():
        # Initialize carry
        carry = model.initial_carry(batch)

        # Forward with original encoder context
        carry_1, outputs_1 = model._forward_eval_step(carry, batch)
        preds_1 = outputs_1['logits'].argmax(dim=-1)

        # Forward with zeros as encoder context (to simulate dead encoder)
        # Monkey-patch the encoder temporarily
        original_encoder_forward = model.encoder.forward
        def zero_encoder(*args, **kwargs):
            return torch.zeros_like(context_1)
        model.encoder.forward = zero_encoder

        carry_2 = model.initial_carry(batch)
        carry_2, outputs_2 = model._forward_eval_step(carry_2, batch)
        preds_2 = outputs_2['logits'].argmax(dim=-1)

        # Restore encoder
        model.encoder.forward = original_encoder_forward

        # Compute difference
        preds_diff = (preds_1 != preds_2).float().mean().item()

        print(f"Predictions with real encoder: {preds_1[0, :20].tolist()}")
        print(f"Predictions with zero encoder: {preds_2[0, :20].tolist()}")
        print(f"Inputs:                        {batch['inputs'][0, :20].tolist()}")
        print(f"\nFraction of different predictions: {preds_diff:.4f}")

        if preds_diff < 0.01:
            print("❌ CRITICAL: Model output doesn't change with encoder context!")
            print("   → The TRM is not using the encoder output effectively")
        else:
            print(f"✓ Model is sensitive to encoder context")

    print("\n" + "="*70)
    print("TEST 3: Check for copy behavior")
    print("="*70)

    with torch.no_grad():
        carry = model.initial_carry(batch)

        # Run multiple steps to get final predictions
        for step in range(model.config.halt_max_steps):
            carry, outputs = model._forward_eval_step(carry, batch)
            if carry.halted.all():
                break

        preds = outputs['logits'].argmax(dim=-1)

        # Check if predictions match inputs
        for b in range(min(2, batch_size)):
            input_seq = batch['inputs'][b]
            pred_seq = preds[b]
            label_seq = batch['labels'][b]

            # Count matches
            input_match = (pred_seq == input_seq).float().mean().item()
            label_match = (pred_seq == label_seq).float().mean().item()

            print(f"\nSample {b}:")
            print(f"  Predictions match input: {input_match:.2%}")
            print(f"  Predictions match label: {label_match:.2%}")
            print(f"  First 20 predictions: {pred_seq[:20].tolist()}")
            print(f"  First 20 inputs:      {input_seq[:20].tolist()}")
            print(f"  First 20 labels:      {label_seq[:20].tolist()}")

            if input_match > 0.9:
                print(f"  ⚠️  WARNING: Model is copying the input!")
            elif input_match > 0.5:
                print(f"  ⚠️  WARNING: Model predictions heavily match input")

    print("\n" + "="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_encoder_output.py <checkpoint_path>")
        print("Example: python debug_encoder_output.py checkpoints/SF1_hybrid_std_baseline/checkpoint_latest.pt")
        sys.exit(1)

    test_encoder_sensitivity(sys.argv[1])
