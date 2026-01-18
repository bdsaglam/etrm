"""
Generate training curves plot from W&B data.

Output: outputs/training_curves.png, outputs/training_curves.pdf
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

TRM_RUN = {'project': 'Arc1concept-aug-1000-ACT-torch', 'run_id': '2jpjeuav', 'label': 'TRM (baseline)', 'color': '#d62728'}


def fetch_training_history(entity: str, project: str, run_id: str, metric: str = 'train/exact_accuracy') -> pd.DataFrame:
    """Fetch training history from W&B."""
    api = wandb.Api()
    run = api.run(f'{entity}/{project}/{run_id}')
    history = run.history(keys=['_step', metric], samples=1000)
    history = history[history[metric].notna()]
    return history


def main():
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

    # Fetch histories
    histories = {}

    for name, info in FINAL_RUNS.items():
        print(f"Fetching {name}...")
        try:
            histories[name] = fetch_training_history(ENTITY, info['project'], info['run_id'])
            print(f"  Got {len(histories[name])} data points")
        except Exception as e:
            print(f"  Error: {e}")

    print("Fetching TRM baseline...")
    try:
        histories['TRM'] = fetch_training_history(ENTITY, TRM_RUN['project'], TRM_RUN['run_id'])
        print(f"  Got {len(histories['TRM'])} data points")
    except Exception as e:
        print(f"  Error: {e}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, info in FINAL_RUNS.items():
        if name in histories and not histories[name].empty:
            df = histories[name]
            ax.plot(df['_step'] / 1000, df['train/exact_accuracy'] * 100,
                    label=info['label'], color=info['color'], linewidth=2)

    if 'TRM' in histories and not histories['TRM'].empty:
        df = histories['TRM']
        ax.plot(df['_step'] / 1000, df['train/exact_accuracy'] * 100,
                label=TRM_RUN['label'], color=TRM_RUN['color'], linewidth=2, linestyle='--')

    ax.set_xlabel('Training Steps (k)', fontsize=14)
    ax.set_ylabel('Training Accuracy (%)', fontsize=14)
    ax.set_title('Training Accuracy Over Time', fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'training_curves.pdf', bbox_inches='tight')
    print(f"\nSaved to {OUTPUT_DIR / 'training_curves.png'}")


if __name__ == '__main__':
    main()
