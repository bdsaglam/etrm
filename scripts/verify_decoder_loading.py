"""
Verify that ETRM model correctly loaded pretrained TRM decoder weights.

Compares weight statistics between:
1. Original TRM checkpoint (pretrained decoder)
2. ETRM checkpoint (should have same decoder weights)

If loading worked, inner model weights should match (encoder weights will differ).
"""

import argparse
import sys
from pathlib import Path

import torch

def get_weight_stats(state_dict, prefix=""):
    """Get statistics for weights matching prefix."""
    stats = {}

    for key, value in state_dict.items():
        if prefix and not any(p in key for p in prefix.split(",")):
            continue

        if isinstance(value, torch.Tensor) and value.numel() > 1:
            stats[key] = {
                "shape": tuple(value.shape),
                "mean": float(value.float().mean()),
                "std": float(value.float().std()),
                "min": float(value.min()),
                "max": float(value.max()),
            }

    return stats


def normalize_key(key):
    """Normalize key by removing common prefixes."""
    # Remove _orig_mod, module, model prefixes
    key = key.replace("_orig_mod.", "")
    key = key.replace("module.", "")
    return key


def compare_checkpoints(trm_path, etrm_path):
    """Compare TRM and ETRM checkpoints."""
    print("=" * 80)
    print("Pretrained Decoder Loading Verification")
    print("=" * 80)
    print()

    # Load checkpoints
    print(f"Loading TRM checkpoint: {trm_path}")
    trm_state = torch.load(trm_path, map_location="cpu", weights_only=True)

    print(f"Loading ETRM checkpoint: {etrm_path}")
    etrm_state = torch.load(etrm_path, map_location="cpu", weights_only=True)

    print()
    print("=" * 80)
    print("Checkpoint Overview")
    print("=" * 80)
    print(f"TRM keys: {len(trm_state)}")
    print(f"ETRM keys: {len(etrm_state)}")
    print()

    # Normalize keys
    trm_normalized = {normalize_key(k): v for k, v in trm_state.items()}
    etrm_normalized = {normalize_key(k): v for k, v in etrm_state.items()}

    # Find inner model keys (decoder) in TRM
    trm_inner_keys = {k for k in trm_normalized.keys() if "model.inner." in k and "puzzle_emb" not in k}

    print(f"TRM inner model (decoder) keys: {len(trm_inner_keys)}")
    print()

    # Check which decoder keys exist in ETRM
    matched = 0
    mismatched = 0
    missing = 0

    print("=" * 80)
    print("Decoder Weight Comparison")
    print("=" * 80)
    print()

    for trm_key in sorted(trm_inner_keys):
        if trm_key not in etrm_normalized:
            print(f"❌ MISSING in ETRM: {trm_key}")
            missing += 1
            continue

        trm_tensor = trm_normalized[trm_key]
        etrm_tensor = etrm_normalized[trm_key]

        # Check shape match
        if trm_tensor.shape != etrm_tensor.shape:
            print(f"❌ SHAPE MISMATCH: {trm_key}")
            print(f"   TRM: {trm_tensor.shape}, ETRM: {etrm_tensor.shape}")
            mismatched += 1
            continue

        # Check if values are close (should be nearly identical if loaded)
        diff = (trm_tensor.float() - etrm_tensor.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        if max_diff < 1e-5:  # Essentially identical
            matched += 1
        else:
            print(f"⚠️  VALUE DIFFERENCE: {trm_key}")
            print(f"   Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
            print(f"   TRM  - mean: {trm_tensor.float().mean():.6f}, std: {trm_tensor.float().std():.6f}")
            print(f"   ETRM - mean: {etrm_tensor.float().mean():.6f}, std: {etrm_tensor.float().std():.6f}")
            mismatched += 1

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✅ Matched weights: {matched}/{len(trm_inner_keys)} ({matched/len(trm_inner_keys)*100:.1f}%)")
    print(f"⚠️  Mismatched weights: {mismatched}")
    print(f"❌ Missing weights: {missing}")
    print()

    if matched == len(trm_inner_keys):
        print("✅ VERIFICATION PASSED: Decoder weights loaded correctly!")
        print("   All decoder weights match pretrained checkpoint.")
        return True
    else:
        print("❌ VERIFICATION FAILED: Decoder weights NOT loaded correctly!")
        print(f"   {len(trm_inner_keys) - matched} weights don't match pretrained checkpoint.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify decoder loading in ETRM")
    parser.add_argument(
        "--trm-checkpoint",
        type=str,
        required=True,
        help="Path to original TRM checkpoint",
    )
    parser.add_argument(
        "--etrm-checkpoint",
        type=str,
        required=True,
        help="Path to ETRM checkpoint to verify",
    )

    args = parser.parse_args()

    success = compare_checkpoints(args.trm_checkpoint, args.etrm_checkpoint)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
