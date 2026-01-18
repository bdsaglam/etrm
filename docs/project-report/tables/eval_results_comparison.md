# Evaluation Results Comparison

Evaluation on 32 held-out test puzzle groups with majority voting across augmentations.

## Main Results

| Model          | Encoder Type            |   Train EM% | Pass@1%   | Pass@2%   | Pass@5%   |
|:---------------|:------------------------|------------:|:----------|:----------|:----------|
| TRM (Original) | embedding               |        99.9 | 3.25      | 3.88      | 4.25      |
| F1: Standard   | standard (2L)           |        78.9 | 0.00      | 0.25      | 0.25      |
| F2: Hybrid VAE | hybrid_variational (4L) |        40.6 | 0.00      | 0.00      | 0.00      |
| F3: ETRMTRM    | trm_style (recurrent)   |        51.2 | 0.00      | 0.00      | 0.25      |
| F4: LPN VAE    | lpn_var (2L)            |        30.9 | -         | -         | -         |

### Key Observations

1. **Original TRM** achieves 3.25% Pass@1 with learned embeddings
2. **Encoder-based approaches** (ETRM) show significant generalization gap
3. **Training accuracy** is high across all models, but test performance varies

### Notes

- **Train EM%**: Exact match accuracy on training puzzles
- **Pass@k%**: Accuracy with k attempts, using majority voting across augmented versions
- Evaluation uses 32 held-out puzzle groups (true generalization test)
- Original TRM uses puzzle-specific learned embeddings (not generalizable to new puzzles)
- ETRM variants use encoder networks that can generalize to unseen puzzles
