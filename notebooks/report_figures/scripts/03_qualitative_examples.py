"""
Generate qualitative example figures showing predictions vs ground truth.

Uses:
- test_puzzles.json for ground truth and demo examples
- submission.json from evaluation for model predictions

Output: outputs/qualitative_*.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

# Config
DATA_DIR = Path('../../data/arc1concept-aug-1000')
CHECKPOINT_DIR = Path('../../checkpoints')
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

# ARC color palette
ARC_COLORS = {
    0: '#000000',  # Black
    1: '#0074D9',  # Blue
    2: '#FF4136',  # Red
    3: '#2ECC40',  # Green
    4: '#FFDC00',  # Yellow
    5: '#AAAAAA',  # Gray
    6: '#F012BE',  # Magenta
    7: '#FF851B',  # Orange
    8: '#7FDBFF',  # Cyan
    9: '#870C25',  # Brown
}

SUBMISSIONS = {
    'F1_standard': CHECKPOINT_DIR / 'etrm-final' / 'F1_standard' / 'evaluator_ARC_step_174622' / 'submission.json',
    'TRM': CHECKPOINT_DIR / 'Arc1concept-aug-1000-ACT-torch' / 'pretrain_att_arc1concept_4' / 'evaluator_ARC_step_518071' / 'submission.json',
}


def plot_grid(grid, ax, title=''):
    """Plot a single ARC grid."""
    grid = np.array(grid)
    cmap = ListedColormap([ARC_COLORS[i] for i in range(10)])

    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add grid lines
    for i in range(grid.shape[0] + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5)
    for j in range(grid.shape[1] + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.5)


def load_test_puzzles():
    """Load test puzzles with ground truth."""
    path = DATA_DIR / 'test_puzzles.json'
    with open(path) as f:
        return json.load(f)


def load_submission(path):
    """Load submission predictions."""
    with open(path) as f:
        return json.load(f)


def plot_example(puzzle_id, puzzle_data, predictions, output_path):
    """Plot a complete example with demos, test, and predictions."""
    demos = puzzle_data['train']
    test_cases = puzzle_data['test']

    num_demos = len(demos)
    num_tests = len(test_cases)

    # Determine layout
    num_cols = max(num_demos + 1, 4)  # demos + test/predictions

    fig, axes = plt.subplots(3, num_cols, figsize=(2.5 * num_cols, 7))

    # Row 0: Demo inputs
    for i in range(num_cols):
        if i < num_demos:
            plot_grid(demos[i]['input'], axes[0, i], f'Demo {i+1} Input')
        else:
            axes[0, i].axis('off')

    # Row 1: Demo outputs
    for i in range(num_cols):
        if i < num_demos:
            plot_grid(demos[i]['output'], axes[1, i], f'Demo {i+1} Output')
        else:
            axes[1, i].axis('off')

    # Row 2: Test input, Ground truth, Predictions
    test_input = test_cases[0]['input']
    test_output = test_cases[0]['output']

    plot_grid(test_input, axes[2, 0], 'Test Input')
    plot_grid(test_output, axes[2, 1], 'Ground Truth')

    col = 2
    for model_name, preds in predictions.items():
        if puzzle_id in preds and preds[puzzle_id] and col < num_cols:
            pred = preds[puzzle_id][0]['attempt_1']  # First test case, first attempt
            plot_grid(pred, axes[2, col], f'{model_name}')
            col += 1

    for i in range(col, num_cols):
        axes[2, i].axis('off')

    fig.suptitle(f'Puzzle: {puzzle_id}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # Load data
    print("Loading test puzzles...")
    test_puzzles = load_test_puzzles()
    print(f"  Loaded {len(test_puzzles)} puzzles")

    # Load submissions
    predictions = {}
    for name, path in SUBMISSIONS.items():
        if path.exists():
            print(f"Loading {name} predictions from {path}...")
            predictions[name] = load_submission(path)
            print(f"  Loaded {len(predictions[name])} predictions")
        else:
            print(f"  {name} submission not found at {path}")

    if not predictions:
        print("No predictions found!")
        return

    # Find common puzzles with non-empty predictions
    common_puzzles = set(test_puzzles.keys())
    for preds in predictions.values():
        # Only include puzzles with non-empty predictions
        valid_puzzles = {k for k, v in preds.items() if v}
        common_puzzles &= valid_puzzles

    print(f"\nFound {len(common_puzzles)} puzzles with non-empty predictions from all models")

    # Plot examples for first N puzzles
    n_examples = min(5, len(common_puzzles))
    puzzle_ids = sorted(common_puzzles)[:n_examples]

    for i, puzzle_id in enumerate(puzzle_ids):
        print(f"Plotting {puzzle_id}...")
        output_path = OUTPUT_DIR / f'qualitative_{i+1}_{puzzle_id}.png'
        plot_example(puzzle_id, test_puzzles[puzzle_id], predictions, output_path)
        print(f"  Saved to {output_path}")

    # Create a combined figure with multiple examples
    print("\nCreating combined figure...")
    fig, axes = plt.subplots(n_examples, 4, figsize=(10, 2.5 * n_examples))

    for row, puzzle_id in enumerate(puzzle_ids):
        puzzle = test_puzzles[puzzle_id]
        test_input = puzzle['test'][0]['input']
        test_output = puzzle['test'][0]['output']

        plot_grid(test_input, axes[row, 0], 'Input' if row == 0 else '')
        plot_grid(test_output, axes[row, 1], 'Ground Truth' if row == 0 else '')

        col = 2
        for model_name, preds in predictions.items():
            if puzzle_id in preds and preds[puzzle_id] and col < 4:
                pred = preds[puzzle_id][0]['attempt_1']
                plot_grid(pred, axes[row, col], model_name if row == 0 else '')
                col += 1

        # Add puzzle ID on the left
        axes[row, 0].set_ylabel(puzzle_id[:8], fontsize=8, rotation=0, ha='right', va='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'qualitative_combined.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'qualitative_combined.pdf', bbox_inches='tight')
    print(f"Saved combined figure to {OUTPUT_DIR / 'qualitative_combined.png'}")


if __name__ == '__main__':
    main()
