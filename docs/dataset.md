# Dataset Description: ARC-AGI-1

This document provides a comprehensive description of the ARC-AGI-1 dataset used in this project, including dataset statistics, augmentation strategies, and preprocessing details.

---

## 1. Introduction

The Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) is a benchmark designed to measure an AI system's ability to acquire and efficiently use new skills through few-shot learning [1]. Unlike traditional benchmarks that test memorization or pattern recognition on large training sets, ARC-AGI evaluates **abstract reasoning** and **knowledge efficiency**—the ability to extract underlying transformation rules from just a handful of demonstration examples and generalize to novel instances.

Each task in ARC-AGI presents a visual reasoning puzzle where:
1. **Demonstrations** (2-7 input-output pairs) implicitly define an abstract transformation rule
2. **Test inputs** (1-2 grids) must be solved by applying the learned rule
3. **Success** requires generalizing from demonstrations to solve previously unseen test cases

This design directly tests few-shot learning capability—the core challenge of our research.

---

## 2. Dataset Overview

### ARC-AGI-1 Benchmark

**Source**: Chollet, F. (2019). On the Measure of Intelligence. arXiv:1911.01547 [1]

**Dataset Composition**:
- **Training subset**: 400 tasks (publicly available)
- **Evaluation subset**: 400 tasks (publicly available)
- **Test subset**: 100 tasks (private, held out for competition)
- **ConceptARC**: 160 additional concept-focused tasks [2]

**Task Structure**:
Each task consists of:
- **Demonstration examples** (referred to as "train" in ARC-AGI format): 2-7 input-output pairs that show the transformation pattern
- **Test examples** (referred to as "test" in ARC-AGI format): 1-2 input grids whose outputs must be predicted

**Grid Properties**:
- Maximum size: 30×30 cells
- Cell values: 0-9 (representing 10 distinct colors)
- Typical patterns: Object manipulation, spatial reasoning, color transformations, pattern completion, symmetry operations

### Dataset Statistics (Original, Pre-Augmentation)

Using the **evaluation subset** (400 tasks) as our test set:

| Statistic | Value |
|-----------|-------|
| **Number of puzzles** | 400 |
| **Demonstration examples per puzzle** | min: 2, max: 7, mean: 3.41 |
| **Test examples per puzzle** | min: 1, max: 2, mean: 1.05 |
| **Total examples per puzzle** | min: 3, max: 9, mean: 4.46 |

For training, we combine:
- **Training subset**: 400 tasks
- **ConceptARC subset**: 160 tasks
- **Total training puzzles**: 560 tasks

---

## 3. Data Augmentation Strategy

Following the methodology of Jolicoeur-Martineau (2025) [3], we apply **1000× data augmentation** to each puzzle using three transformation types. This augmentation is critical for:
1. **Data efficiency**: Generating sufficient training examples from limited base puzzles
2. **Invariance learning**: Teaching the model that color assignment and orientation do not change the underlying transformation rule
3. **Robustness**: Enabling test-time ensemble predictions via majority voting

### Augmentation Types

#### 1. Color Permutation
- **Operation**: Randomly shuffle color assignments for values 1-9
- **Constraint**: Black (value 0) remains fixed as it often represents background
- **Rationale**: Abstract transformation rules should be invariant to specific color assignments
- **Example**: A rule "copy red objects" generalizes to "copy objects of type X" for any color X

#### 2. Dihedral Transformations
- **Operations**: 8 geometric transformations from the dihedral group D₄
  - Identity (no change)
  - 90°, 180°, 270° rotations
  - Horizontal flip, vertical flip
  - Diagonal flips (2 diagonals)
- **Rationale**: Most transformation rules should be invariant to grid orientation
- **Example**: A "reflect vertically" rule applies regardless of initial grid rotation

#### 3. Translational Augmentation (Training Split Only)
- **Operation**: Randomly position grids within the 30×30 canvas
- **Application**: Applied to **training split only** to improve spatial invariance
- **Constraint**: One randomly selected example per augmented puzzle remains unaugmented to preserve original spatial relationships
- **Rationale**: Model should learn position-invariant features

### Augmentation Process

For each base puzzle:
1. **Generate 1000 augmented versions**:
   - Sample a random color permutation
   - Sample a random dihedral transformation
   - Apply the **same** transformation to all examples (demonstrations + tests) within that augmented version
2. **Deduplication**: Track augmented puzzle hashes to avoid exact duplicates
3. **Retry mechanism**: If 1000 unique augmentations cannot be generated, use as many unique versions as possible

**Critical Design**: All examples within an augmented puzzle share the same transformation. This preserves the consistency between demonstrations and test cases—the model must learn the rule from transformed demonstrations and apply it to consistently transformed test inputs.

**Augmentation Details** (from `build_arc_dataset_encoder.py`):
```python
# Color permutation: shuffle colors 1-9, keep 0 fixed
color_map = [0] + np.random.permutation(range(1, 10)).tolist()

# Dihedral transform: one of 8 transformations
transform_id = np.random.randint(0, 8)  # t0-t7

# Translational augment (train only): random position in 30×30 grid
# Applied to all but one randomly selected example per puzzle
```

---

## 4. Preprocessed Dataset Statistics

After applying augmentation and preprocessing with `build_arc_dataset_encoder.py`, we obtain:

### Dataset Hierarchy

```
960 base puzzles
  ├─ 560 training groups (training + concept subsets)
  │   └─ ~908 augmented puzzles per group
  │       └─ ~4.67 examples per puzzle (3.07 demos + 1.60 queries)
  │
  └─ 400 test groups (evaluation subset)
      └─ ~920 augmented puzzles per group
          └─ ~4.47 examples per puzzle (3.42 demos + 1.05 queries)
```

### Overall Statistics

| Metric | Train Split | Test Split | Total |
|--------|-------------|------------|-------|
| **Base puzzles (groups)** | 560 | 400 | 960 |
| **Augmented puzzles** | 508,544 | 368,149 | 876,693 |
| **Total examples** | 2,373,921 | 1,646,169 | 4,020,090 |
| **Augmentations per group** | 908 (avg) | 920 (avg) | 914 (avg) |
| **Examples per puzzle** | 4.67 (avg) | 4.47 (avg) | 4.59 (avg) |
| **Demos per puzzle** | 3.07 (avg) | 3.42 (avg) | 3.22 (avg) |

### Puzzle Complexity Distribution

**Train Split**:
- Examples per puzzle: min=3, max=12, mean=4.67
- Demos per puzzle: min=1, max=10, mean=3.07
- Augmentations per group: min=18, max=1001, mean=908

**Test Split**:
- Examples per puzzle: min=3, max=9, mean=4.47
- Demos per puzzle: min=2, max=7, mean=3.42
- Augmentations per group: min=9, max=1001, mean=920

**Interpretation**: The variation in augmentation count (18-1001) reflects the deduplication process—some base puzzles have less diversity in their geometric/color space, resulting in fewer unique augmented versions.

---

## 5. Encoder-Mode Train/Test Split

A **critical distinction** of our approach is the train/test splitting strategy, which differs from the original TRM paper [3].

### Original TRM (Embedding Mode)

```
Training puzzles:  demos → train/, queries → train/
Evaluation puzzles: demos → train/, queries → test/
                    ↑ Seen during training!
```

**Problem**: Evaluation puzzle demonstrations appear in the training set. The model's learned puzzle embeddings encode transformation rules for evaluation puzzles, preventing true few-shot generalization assessment.

### Our Approach (Encoder Mode)

```
Training puzzles:  demos → train/, queries → train/
Evaluation puzzles: demos → test/,  queries → test/
                    ↑ Never seen during training!
```

**Solution**: Evaluation puzzle demonstrations are **strictly excluded** from the training set. The encoder must generalize to:
1. Puzzles it has never seen before
2. Transformation rules not observed during training

This enables genuine few-shot learning evaluation—the model receives demonstrations at test time and must extract the transformation rule on-the-fly, just as the ARC-AGI benchmark intends.

### Split Implementation Details

From `build_arc_dataset_encoder.py`:

```python
def load_puzzles_arcagi_encoder(config):
    for subset_name in config.subsets:
        is_test_subset = subset_name == "evaluation"

        if is_test_subset:
            # ENCODER MODE: Both demos and queries go to test/
            dest_mapping = {
                "train": ("test", "all"),  # demos → test/
                "test":  ("test", "all"),  # queries → test/
            }
        else:
            # Training puzzles: both go to train/
            dest_mapping = {
                "train": ("train", "all"),
                "test":  ("train", "all"),
            }
```

**Result**: Zero overlap between training and evaluation puzzles—no data leakage, true generalization.

---

## 6. Data Format and Representation

### Token Encoding

Grids are flattened into 900-token sequences (30×30):

| Token Value | Meaning |
|-------------|---------|
| 0 | Padding (PAD) |
| 1 | End-of-sequence marker (EOS) |
| 2-11 | Colors 0-9 (ARC cell values + 2) |

**Vocabulary size**: 12 tokens

### File Structure

```
data/arc1concept-encoder-aug-1000/
├── identifiers.json              # Maps puzzle ID → puzzle name
│                                 # Format: "base_id|||tN|||color_perm"
│
├── train/
│   ├── all__inputs.npy           # (2,373,921, 900) - Input grids
│   ├── all__labels.npy           # (2,373,921, 900) - Output grids
│   ├── all__puzzle_identifiers.npy  # (508,544,) - Puzzle ID per example
│   ├── all__puzzle_indices.npy      # (508,545,) - Example boundaries
│   ├── all__group_indices.npy       # (561,) - Group boundaries
│   ├── all__num_demos.npy           # (508,544,) - Demo count per puzzle
│   └── dataset.json                 # Metadata
│
├── test/
│   └── (same structure, test split)
│
└── test_puzzles.json             # Original 400 eval puzzles (no augmentation)
```

### Puzzle Identifier Format

Each augmented puzzle has a unique identifier:
```
<base_puzzle_id>|||t<transform_id>|||<color_permutation>
```

**Example**: `007bbfb7|||t3|||0517283946`
- `007bbfb7`: Original ARC-AGI puzzle ID
- `t3`: Dihedral transformation ID (0-7)
- `0517283946`: Color permutation (10-digit string encoding the color mapping)

---

## 7. Training Data Structure (Encoder Mode)

Each training batch contains:

```python
{
    'demo_inputs':  (B, K, 900),  # K demonstration input grids
    'demo_labels':  (B, K, 900),  # K demonstration output grids
    'demo_mask':    (B, K),       # Valid demo indicators
    'inputs':       (B, 900),     # Test input grid (query)
    'labels':       (B, 900),     # Test output grid (target)
    'puzzle_id':    (B,),         # Puzzle identifier
}
```

Where:
- `B` = batch size
- `K` = maximum number of demos (typically 5)
- All examples in a batch item come from the **same augmented puzzle**

### Key Properties

1. **Intra-puzzle consistency**: All demos and the test input share the same augmentation (color permutation + geometric transform)
2. **Inter-puzzle diversity**: Different batch items come from different puzzles (possibly different augmentations of the same base puzzle)
3. **Few-shot structure**: The model receives 1-5 demonstrations and must predict the output for a held-out test input

---

## 8. Evaluation Protocol

### Test-Time Augmentation and Voting

Following the TRM paper [3], evaluation uses **ensemble prediction via majority voting**:

1. **For each original test puzzle**:
   - Evaluate on all ~920 augmented versions
   - Generate predictions for each augmented version
   - Apply inverse transformation to predictions (un-rotate, un-permute colors)
   - Collect all predictions in original coordinate/color space

2. **Majority vote**:
   - Count frequency of each predicted output grid
   - Report most common prediction as final answer

**Rationale**: Averaging over augmentations improves robustness and reduces prediction variance.

### Metrics

Following ARC-AGI convention:
- **Exact accuracy**: Percentage of test inputs with perfectly correct output grids
- **Pass@k accuracy**: Success rate when model is allowed k attempts per test input (k=1, 2)

---

## 9. References

[1] Chollet, F. (2019). On the Measure of Intelligence. arXiv:1911.01547.

[2] Moskvichev, A., Odouard, V. V., & Mitchell, M. (2023). The ConceptARC Benchmark: Evaluating Understanding and Generalization in the ARC Domain. arXiv:2305.07141.

[3] Jolicoeur-Martineau, A. (2025). Less is More: Recursive Reasoning with Tiny Networks. arXiv:2510.04871.

---

## 10. Summary

The preprocessed ARC-AGI-1 dataset for encoder-based training contains:
- **876,693 augmented puzzles** from 960 base puzzles
- **4.0M total examples** (2.4M train, 1.6M test)
- **~900 augmentations per base puzzle** (color permutation × geometric transform)
- **Strict train/test separation**: Evaluation puzzle demonstrations never seen during training
- **Few-shot structure**: 3-4 demonstration pairs per puzzle, 1-2 test inputs

This augmented dataset enables:
1. Training with sufficient data despite limited base puzzles
2. Learning transformation invariances (color, rotation, position)
3. Robust evaluation via test-time voting
4. **True few-shot generalization assessment** (encoder mode only)

The key innovation of our preprocessing is maintaining **zero overlap** between training and evaluation puzzle sets, ensuring the encoder must generalize to truly novel puzzles at test time—the intended paradigm of ARC-AGI few-shot learning.
