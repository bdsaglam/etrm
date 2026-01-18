# ETRM: Encoder-based Tiny Recursive Model

This project extends the [Tiny Recursive Model (TRM)](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) to enable true few-shot learning on the ARC-AGI benchmark by replacing task-specific embeddings with a demonstration encoder.

## Motivation

TRM achieved 45% accuracy on ARC-AGI-1 with only 7M parameters through recursive reasoning. However, [analysis by the ARC Prize Foundation](https://arcprize.org/blog/hrm-analysis) revealed a critical limitation:

> "TRM assigns each task a unique identifier (puzzle_id) that is fed into a learned embedding layer. During inference, the model receives only a single input grid and its puzzle_id—it never sees demonstration pairs. The transformation rule is therefore not inferred from demonstrations at test time; instead, it is encoded into the puzzle_id embedding weights during training."

This means TRM **memorizes** task-specific transformations rather than learning to **infer** them from demonstrations—fundamentally deviating from ARC-AGI's few-shot learning intent.

## Our Approach

We replace the puzzle_id embedding with a **demonstration encoder** that processes input-output pairs directly:

```
Original TRM:
  puzzle_id → Embedding Matrix Lookup → context vector

ETRM:
  demo pairs → Neural Network Encoder → context vector
```

The encoder learns to extract transformation rules from demonstrations, enabling generalization to truly unseen tasks at test time.

### Architecture

```
Demo Pairs: [(input1, output1), (input2, output2), ...]
                              ↓
              ┌─────────────────────────────────────────┐
              │  Demonstration Encoder                  │
              │                                         │
              │  Per-demo encoding:                     │
              │    Transformer (2 layers)               │
              │    Input+Output → concat → encode       │
              │    Mean pool → single vector            │
              │                                         │
              │  Cross-demo aggregation:                │
              │    Cross-attention (1 layer)            │
              │    16 learnable query tokens            │
              │                                         │
              └─────────────────────────────────────────┘
                              ↓
              Context: (batch, 16, 512) - replaces puzzle embedding
                              ↓
              TRM Inner Model (unchanged recursive reasoning)
```

### Key Difference from Original TRM

| Aspect | Original TRM | ETRM |
|--------|--------------|------|
| Task representation | Learned puzzle_id embedding | Encoded from demo pairs |
| Training set | Includes eval task demos | Strictly excludes eval tasks |
| Generalization | To seen puzzle_ids only | To any task with demos |
| Few-shot capability | Not truly few-shot | True few-shot inference |

## Requirements

- Python 3.10+
- CUDA 12.6.0+
- 1-4 GPUs (L40S or H100 recommended)

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
pip install -r requirements.txt
pip install --no-cache-dir --no-build-isolation adam-atan2
wandb login YOUR-LOGIN  # optional, for experiment tracking
```

## Dataset Preparation

```bash
# ARC-AGI-1
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation
```

## Training

### ETRM (Dynamic Halting)

```bash
torchrun --nproc-per-node 4 pretrain_etrm.py \
    --config-name cfg_pretrain_encoder_original_arc_agi_1 \
    +project_name="etrm" \
    +run_name="etrm_baseline"
```

### Encoder Variants

| Type | Description |
|------|-------------|
| `standard` | 2-layer transformer + mean pooling |
| `lpn_standard` | Deeper encoder (4-8 layers) with CLS pooling |
| `lpn_variational` | VAE version with KL regularization |

## Evaluation

The key metric is accuracy on held-out evaluation tasks that were **never seen during training**—measuring true few-shot generalization rather than memorization.

## References

This project builds upon:

**Tiny Recursive Model (TRM)**
```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```

**Hierarchical Reasoning Model (HRM)**
```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

## Acknowledgments

- Original TRM codebase: [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- HRM analysis: [arcprize/hierarchical-reasoning-model-analysis](https://github.com/arcprize/hierarchical-reasoning-model-analysis)
- ARC Prize Foundation for the [HRM analysis](https://arcprize.org/blog/hrm-analysis) that motivated this work
