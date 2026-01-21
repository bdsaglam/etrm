"""
Diagnostic script to inspect TRM checkpoint structure and verify integrity.

Usage:
    python scripts/inspect_checkpoint.py --checkpoint ./checkpoints/official_trm/arc_v1_public/step_518071
"""

import argparse
import torch
from pathlib import Path


def inspect_checkpoint(checkpoint_path: str):
    """Inspect checkpoint structure and weights."""
    print("=" * 80)
    print(f"Inspecting checkpoint: {checkpoint_path}")
    print("=" * 80)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    print(f"\n📊 Checkpoint Statistics:")
    print(f"  Total keys: {len(state_dict)}")
    print(f"  Total parameters: {sum(v.numel() for v in state_dict.values() if hasattr(v, 'numel')):,}")

    print(f"\n🔍 Weight Details:")
    print("-" * 80)

    for key in sorted(state_dict.keys()):
        value = state_dict[key]

        if not hasattr(value, "shape"):
            print(f"{key}: type={type(value)}")
            continue

        # Basic info
        print(f"\n{key}")
        print(f"  Shape: {tuple(value.shape)}, Dtype: {value.dtype}")

        # Statistics for floating point tensors
        if value.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            v_float = value.float()

            min_val = v_float.min().item()
            max_val = v_float.max().item()
            mean_val = v_float.mean().item()
            std_val = v_float.std().item()

            print(f"  Stats: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, std={std_val:.6f}")

            # Check for anomalies
            nan_count = torch.isnan(v_float).sum().item()
            inf_count = torch.isinf(v_float).sum().item()

            if nan_count > 0 or inf_count > 0:
                print(f"  ⚠️  WARNING: {nan_count} NaN, {inf_count} Inf values detected!")

            # Check if weights look like EMA (typically have lower std than raw training weights)
            if "weight" in key.lower() and std_val < 0.05:
                print(f"  ℹ️  Low std ({std_val:.6f}) - consistent with EMA smoothing")

    # Check for config file
    config_path = Path(checkpoint_path).parent / "all_config.yaml"
    print("\n" + "=" * 80)
    print(f"📄 Configuration File:")
    print(f"  Path: {config_path}")
    print(f"  Exists: {config_path.exists()}")

    if config_path.exists():
        from omegaconf import OmegaConf
        config = OmegaConf.load(config_path)

        print(f"\n  Key Parameters:")
        print(f"    EMA enabled: {config.ema}")
        print(f"    EMA rate: {config.ema_rate}")
        print(f"    L_cycles: {config.arch.L_cycles}")
        print(f"    H_cycles: {config.arch.H_cycles}")
        print(f"    L_layers: {config.arch.L_layers}")
        print(f"    Puzzle emb dim: {config.arch.puzzle_emb_ndim}")
        print(f"    Puzzle emb len: {config.arch.puzzle_emb_len}")

    print("\n" + "=" * 80)
    print("✅ Checkpoint inspection complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Inspect TRM checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    args = parser.parse_args()

    inspect_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
