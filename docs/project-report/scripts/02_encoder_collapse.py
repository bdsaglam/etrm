"""
Generate encoder collapse evidence figures from W&B data.

Uses logged encoder metrics:
- train/encoder_cross_sample_var: Variance of encoder outputs across samples in batch
- train/encoder_output_std: Std of encoder outputs
- train/encoder_output_norm: Norm of encoder outputs
- train/encoder_token_std: Per-token std

Output: outputs/encoder_collapse.png, outputs/encoder_collapse.pdf
"""

import pandas as pd
import matplotlib.pyplot as plt
import wandb
from pathlib import Path

# Config
ENTITY = 'bdsaglam'
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

FINAL_RUNS = {
    'F1_standard': {'project': 'etrm-final', 'run_id': 'z31hae14', 'label': 'ETRM-Deterministic', 'color': '#1f77b4'},
    'F2_hybrid_var': {'project': 'etrm-final', 'run_id': '7km7llbl', 'label': 'ETRM-Variational', 'color': '#ff7f0e'},
    'F3_etrmtrm': {'project': 'etrm-final', 'run_id': 'wj3xu8md', 'label': 'ETRM-TRM', 'color': '#2ca02c'},
}

ENCODER_METRICS = [
    'train/encoder_cross_sample_var',
    'train/encoder_output_std',
    'train/encoder_output_norm',
    'train/encoder_token_std',
]


def fetch_encoder_history(entity: str, project: str, run_id: str) -> pd.DataFrame:
    """Fetch encoder metrics history from W&B."""
    api = wandb.Api()
    run = api.run(f'{entity}/{project}/{run_id}')

    keys = ['_step'] + ENCODER_METRICS
    history = run.history(keys=keys, samples=1000)

    return history


def main():
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

    # Fetch histories
    histories = {}

    for name, info in FINAL_RUNS.items():
        print(f"Fetching {name}...")
        try:
            histories[name] = fetch_encoder_history(ENTITY, info['project'], info['run_id'])
            print(f"  Got {len(histories[name])} data points")

            # Show available columns
            cols = [c for c in histories[name].columns if c.startswith('train/encoder')]
            print(f"  Encoder metrics: {cols}")
        except Exception as e:
            print(f"  Error: {e}")

    # Plot 1: Cross-sample variance over training
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot cross-sample variance
    ax = axes[0, 0]
    for name, info in FINAL_RUNS.items():
        if name in histories:
            df = histories[name]
            if 'train/encoder_cross_sample_var' in df.columns:
                df_valid = df[df['train/encoder_cross_sample_var'].notna()]
                if not df_valid.empty:
                    ax.plot(df_valid['_step'] / 1000, df_valid['train/encoder_cross_sample_var'],
                            label=info['label'], color=info['color'], linewidth=2, alpha=0.8)
    ax.set_xlabel('Training Steps (k)')
    ax.set_ylabel('Cross-Sample Variance')
    ax.set_title('Encoder Output Variance Across Samples')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot output std
    ax = axes[0, 1]
    for name, info in FINAL_RUNS.items():
        if name in histories:
            df = histories[name]
            if 'train/encoder_output_std' in df.columns:
                df_valid = df[df['train/encoder_output_std'].notna()]
                if not df_valid.empty:
                    ax.plot(df_valid['_step'] / 1000, df_valid['train/encoder_output_std'],
                            label=info['label'], color=info['color'], linewidth=2, alpha=0.8)
    ax.set_xlabel('Training Steps (k)')
    ax.set_ylabel('Output Std')
    ax.set_title('Encoder Output Standard Deviation')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot output norm
    ax = axes[1, 0]
    for name, info in FINAL_RUNS.items():
        if name in histories:
            df = histories[name]
            if 'train/encoder_output_norm' in df.columns:
                df_valid = df[df['train/encoder_output_norm'].notna()]
                if not df_valid.empty:
                    ax.plot(df_valid['_step'] / 1000, df_valid['train/encoder_output_norm'],
                            label=info['label'], color=info['color'], linewidth=2, alpha=0.8)
    ax.set_xlabel('Training Steps (k)')
    ax.set_ylabel('Output Norm')
    ax.set_title('Encoder Output Norm')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot token std
    ax = axes[1, 1]
    for name, info in FINAL_RUNS.items():
        if name in histories:
            df = histories[name]
            if 'train/encoder_token_std' in df.columns:
                df_valid = df[df['train/encoder_token_std'].notna()]
                if not df_valid.empty:
                    ax.plot(df_valid['_step'] / 1000, df_valid['train/encoder_token_std'],
                            label=info['label'], color=info['color'], linewidth=2, alpha=0.8)
    ax.set_xlabel('Training Steps (k)')
    ax.set_ylabel('Token Std')
    ax.set_title('Per-Token Standard Deviation')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Encoder Output Statistics Over Training', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'encoder_collapse.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'encoder_collapse.pdf', bbox_inches='tight')
    print(f"\nSaved to {OUTPUT_DIR / 'encoder_collapse.png'}")

    # Print final values for table
    print("\n" + "="*60)
    print("Final Encoder Statistics (for report table):")
    print("="*60)
    for name, info in FINAL_RUNS.items():
        if name in histories:
            df = histories[name]
            print(f"\n{info['label']}:")
            for metric in ENCODER_METRICS:
                if metric in df.columns:
                    final_val = df[metric].dropna().iloc[-1] if not df[metric].dropna().empty else None
                    metric_name = metric.replace('train/', '')
                    if final_val is not None:
                        print(f"  {metric_name}: {final_val:.6f}")


if __name__ == '__main__':
    main()
