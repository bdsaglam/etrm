"""
Verify that checkpoint loading will work correctly.

Usage:
    python scripts/verify_checkpoint_loading.py --checkpoint PATH
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_checkpoint_loading(checkpoint_path: str):
    """Verify checkpoint and config are compatible."""
    print("=" * 80)
    print("CHECKPOINT LOADING VERIFICATION")
    print("=" * 80)
    print()

    # Step 1: Check config
    print("Step 1: Checking config...")
    config_path = Path(checkpoint_path).parent / "all_config.yaml"

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        print("   The evaluation will fall back to default config (WRONG!)")
        return False

    config = OmegaConf.load(config_path)
    print(f"✅ Config found: {config_path}")
    print(f"   L_cycles: {config.arch.L_cycles}")
    print(f"   H_cycles: {config.arch.H_cycles}")
    print(f"   L_layers: {config.arch.L_layers}")
    print()

    # Step 2: Check checkpoint
    print("Step 2: Loading checkpoint...")
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    print(f"✅ Checkpoint loaded: {checkpoint_path}")
    print(f"   Total keys: {len(state_dict)}")
    print(f"   Total params: {sum(v.numel() for v in state_dict.values() if hasattr(v, 'numel')):,}")
    print()

    # Step 3: Check prefix stripping
    print("Step 3: Checking key prefixes...")
    has_orig_mod = any(k.startswith("_orig_mod.") for k in state_dict.keys())
    has_module = any(k.startswith("module.") for k in state_dict.keys())

    print(f"   Has _orig_mod. prefix: {has_orig_mod}")
    print(f"   Has module. prefix: {has_module}")

    if has_orig_mod:
        print("   ✅ Will strip _orig_mod. prefix")
    if has_module:
        print("   ✅ Will strip module. prefix")
    print()

    # Step 4: Check after stripping
    print("Step 4: Verifying stripped keys...")
    if has_orig_mod:
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    if has_module:
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Check critical keys
    expected_keys = [
        "model.inner.H_init",
        "model.inner.L_init",
        "model.inner.puzzle_emb.weights",
        "model.inner.lm_head.weight",
    ]

    all_present = True
    for key in expected_keys:
        if key in state_dict:
            shape = state_dict[key].shape if hasattr(state_dict[key], "shape") else "N/A"
            print(f"   ✅ {key}: {shape}")
        else:
            print(f"   ❌ {key}: MISSING")
            all_present = False
    print()

    # Step 5: Architecture match
    print("Step 5: Checking architecture match...")

    # Check L_cycles (the critical bug)
    if config.arch.L_cycles == 4:
        print(f"   ✅ L_cycles = 4 (correct for official checkpoint)")
    else:
        print(f"   ❌ L_cycles = {config.arch.L_cycles} (WRONG! Should be 4)")
        print(f"      This will cause ~3% accuracy instead of ~40%")
        all_present = False

    # Check puzzle embedding
    puzzle_emb_shape = state_dict["model.inner.puzzle_emb.weights"].shape
    print(f"   Puzzle embedding: {puzzle_emb_shape}")
    print(f"   Config expects: puzzle_emb_len={config.arch.puzzle_emb_len}, puzzle_emb_ndim={config.arch.puzzle_emb_ndim}")

    if puzzle_emb_shape[1] == config.arch.puzzle_emb_ndim:
        print(f"   ✅ Embedding dimension matches ({config.arch.puzzle_emb_ndim})")
    else:
        print(f"   ⚠️  Dimension mismatch: checkpoint={puzzle_emb_shape[1]}, config={config.arch.puzzle_emb_ndim}")
        print(f"      Script will handle this by averaging")
    print()

    # Final verdict
    print("=" * 80)
    if all_present:
        print("✅ VERIFICATION PASSED")
        print("   The checkpoint will load correctly with proper architecture.")
        print("   Expected performance: ~40% pass@1 (on full test set)")
    else:
        print("❌ VERIFICATION FAILED")
        print("   Issues detected that will cause poor performance.")
    print("=" * 80)

    return all_present


def main():
    parser = argparse.ArgumentParser(description="Verify checkpoint loading")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    args = parser.parse_args()

    success = verify_checkpoint_loading(args.checkpoint)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
