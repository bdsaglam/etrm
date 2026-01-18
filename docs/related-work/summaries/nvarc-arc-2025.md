# NVARC Solution to ARC-AGI-2 2025

**Paper**: NVARC solution to ARC-AGI-2 2025
**Authors**: Ivan Sorokin & Jean-François Puget (NVIDIA Corporation)
**Year**: 2025 (November)
**Competition Result**: 1st place on public leaderboard (27.64% → 29.72% post-competition)

## TL;DR
NVARC won 1st place in ARC Prize 2025 by creating a massive synthetic dataset (103k puzzles) using LLM-based generation pipeline and applying it to two existing approaches: ARChitects (test-time fine-tuned Qwen3-4B) achieving 29.72%, and TRM (pretrained then test-time fine-tuned) achieving 10%, demonstrating that synthetic data generation is the key bottleneck, not model architecture.

## What They Did
- Built 4-stage synthetic data generation (SDG) pipeline using LLMs (Claude Opus 4, Sonnet 4.5, gpt-oss-120b)
- Generated 103,253 synthetic ARC puzzles with Python programs (input/output logic)
- Scaled ARChitects approach with Qwen3-4B (4B params) using LoRA test-time fine-tuning
- Adapted TRM with reduced puzzle embeddings to fit Kaggle constraints (12 hours, 4 L4 GPUs)
- Implemented batch-invariant Depth First Search for deterministic decoding
- Ensembled ARChitects + TRM (mixed results: TRM solved 2-3 unique puzzles but ensemble selection was problematic)
- **Key insight**: Synthetic data quality matters more than architecture choice

## Key Mechanism

### Synthetic Data Generation Pipeline (4 Stages)

**Stage 1: Puzzle Summaries (3,268 summaries)**
- Seed data: H-ARC (1700 human descriptions for 716 puzzles) + BARC (160 descriptions)
- Manually labeled 29 eval puzzles (20-30 min each)
- Used LLMs to generate summaries for remaining puzzles
- Format: input generation, solution steps, rules summary, key insight, puzzle concepts

**Stage 2: Mix Summaries (266,593 new summaries)**
- Inspired by INSTRUCT-SKILLMIX: combine two puzzle summaries → new complex puzzle
- Only used gpt-oss-120b model (15k tokens/s on 8xH100)

**Stage 3: Input Grid Programs (126,901 programs)**
- Generate Python code for input grid logic + unit tests
- Filter: must produce ≥30 unique input grids, all pass unit tests
- Acceptance rate: 70% (training puzzles) vs 50% (eval puzzles)

**Stage 4: Output Grid Programs (103,253 final puzzles)**
- Generate Python code for transformation rules
- Validation: prompt LLM 20 times, keep only if ≥15 produce same output
- Each synthetic puzzle has ~30 input/output pairs

### ARChitects Improvements

**Training Data (3.2M augmented samples)**:
| Source | Unique Puzzles | Aug/Puzzle | Total Samples | Share |
|--------|---------------|------------|---------------|-------|
| MINI-ARC | 147 | 256 | 37,632 | 1.2% |
| ConceptARC | 160 | 256 | 40,960 | 1.3% |
| RE-ARC | 400 | 256 | 102,392 | 3.2% |
| ARC-AGI-2 | 609 | 256 | 155,904 | 4.8% |
| NVARC training | 47,337 | 24 | 1,132,633 | 34.8% |
| NVARC full | 55,886 | 32 | 1,785,960 | 54.9% |

**Key modifications**:
1. **Dialog template** (Qwen3 format): simpler representation (16 tokens per I/O pair)
   ```
   <|im_start|>user\n123\n456<|im_end|> <|im_start|>assistant\n78\n90<|im_end|>
   ```

2. **LoRA test-time fine-tuning**: r=256, alpha=32, bfloat16, Flash Attention 2
   - Removed gradient checkpointing and 4-bit quantization for speed

3. **Batch DFS**: Scales linearly with batch size (but nondeterministic)
   - Used batch-invariant ops solution to fix nondeterminism (17% slower, not used in final)

4. **Augmentation re-scoring** (improved post-competition):
   ```
   score(s) = count(s) × geometric_mean(log_prob_j(s)) over 8 augmentations
   ```

**Pretraining**: NeMo RL framework, Megatron backend, 4 nodes × 8xH100 for 27 hours
- Tensor model parallelism + context parallelism
- Sequence packing up to 256k tokens from 256 puzzle examples

### TRM Adaptations (Kaggle Constraints)

**The puzzle embedding problem**:
- Original TRM: 51B parameters in embedding table (100k puzzles × 1k aug × 512 params)
- **Cannot fit in Kaggle memory!**

**Solutions**:
1. Reduced puzzles: 3k synthetic + 1.2k original (instead of 100k)
2. Reduced augmentations: 256 per puzzle (instead of 1000)
3. Reduced epochs: 10k (instead of 100k)
4. Faster training: batch_size=3072, lr=3e-4 → 24 hours on 8xH100

**Test-time fine-tuning**:
- Embedding table reset: test puzzles different from pretraining puzzles
- Initialize with average of pretrained embeddings
- 2000 epochs, 200 warmup steps, 4 H cycles, 10 halt_max_steps
- Runs in ~2 hours on Kaggle (4 L4 GPUs)

**Local evaluation strategy**:
- Train TRM twice: (1) with eval data, (2) without eval data
- Score model (2) on eval → 9.44% pass@2
- Select checkpoint from model (1) with matching training steps
- Submitted checkpoint → 7.5% (vs 2.08% baseline)

### Ensembling ARChitects + TRM

**Process**:
- TRM generates 10 attempts per puzzle (instead of 2)
- Add TRM attempts to ARChitects attempts
- Score all with Qwen3 model

**Results** (mixed):
- Most TRM-solved puzzles also solved by Qwen3 (no gain)
- TRM solved 2-3 unique puzzles per run
- Qwen3 scoring picked ~1 TRM solution on average
- **Problem**: Qwen3 didn't always select TRM's unique solutions

**Engineering challenges**:
- Conflicting package versions between ARChitects and TRM code
- Only got working submission day before deadline
- No time for tuning

## Results

### Competition Submissions

**Best submission (post-competition)**:
- Model: Qwen3-4B-Thinking-2507
- Score: **29.72%** on public leaderboard (120 puzzles)
- Time: 12 hours
- State-of-the-art on ARC-AGI-2 (surpasses submissions costing 40× more)

**During competition**:
- Best: 27.64% (1st place)
- ARChitects only (Qwen3-2B): 21.53% → 22.50% with TRM ensemble
- ARChitects only (Qwen3-4B, 10h): 27.22% → 27.22% with TRM (no change)

**TRM standalone**:
- Baseline (no synthetic data): 2.08%
- With 3k synthetic + 1.2k original: 7.5%
- Post-competition (4k epochs): 10.0% (in <4 hours)

**Comparison to other approaches**:
- NVARC (Qwen3-4B): 29.72%
- Previous SOTA: ~24% (costs 40× more compute)

### Synthetic Data Impact

**Loss reduction** (Figure 1 in paper):
- More synthetic data → lower validation loss (measured on 120 eval puzzles)
- Strong correlation with public leaderboard score
- Note: Eval puzzle descriptions "leaked" into synthetic dataset (still generalizes)

**Acceptance rates**:
- Easy puzzles (training-based): 70%
- Hard puzzles (eval-based): 50%

**Key finding**: "Scaling the pretraining stage with good synthetic puzzles was the right direction to succeed in this Kaggle competition."

## Key Quotes

> "The core idea of our solution is the novel synthetic dataset of ARC-AGI tasks. During the competition we successfully applied this dataset to existing approaches: the ARChitects and Tiny Recursive Model."
- Context: Main contribution—synthetic data is the bottleneck, not architecture

> "ARC Prize challenge is unusual because [standard pretraining] does not work at all. The hidden test tasks are different from all the public tasks, and using models pretrained on public tasks isn't effective. To perform well one has to use some form of training on the hidden test set (test-time fine-tuning or TTFT)."
- Context: Why test-time fine-tuning is essential for ARC-AGI-2

> "TRM is claimed to be tiny because it contains 7M parameters. This is misleading because the model contains an additional puzzle embedding table, with 512 parameters per puzzle. When using 100k puzzles with 1000 augmentations we get an additional 51B parameters, which is way too large for a Kaggle submission."
- Context: **Critical insight about TRM's puzzle embedding limitation**—exactly what ETRM addresses

> "We found that BARC was not useful for the 2D transformer we developed. We therefore removed the BARC dataset and used real puzzles with our synthetic puzzles only."
- Context: Not all existing datasets help; synthetic data quality matters

> "We used the following closed and open-weights models: claude-opus-4-20250514, claude-sonnet-4-5-20250929, gpt-oss-120b. The majority of generated data is made with gpt-oss-120b model using the NeMo-Skills framework."
- Context: Practical SDG pipeline—uses powerful closed models for seed data, fast open model for scaling

> "Instead of generating a full puzzle program with input/output logic, we decided to split this into two stages. First - generate a Python program for the input grid logic, second - generate a program of the output grid logic, i.e. transformation rules."
- Context: Two-stage generation with different validation strategies (unit tests vs consistency)

> "Most of the puzzles solved by TRM were solved by Qwen3, hence TRM added nothing there. However, about 2 or 3 puzzles solved by TRM were not solved by Qwen3. Unfortunately, these were not always picked by Qwen3 scoring."
- Context: Ensemble challenge—TRM has unique capabilities but selection is hard

> "Scaling the pretraining stage with good synthetic puzzles was the right direction to succeed in this Kaggle competition. But we also believe that this synthetic data with python programs and reasoning traces could be used for other research directions."
- Context: Future potential of synthetic data beyond competition

## Relevance to ETRM

**NVARC provides critical validation for our research direction:**

### What NVARC Confirms
1. **TRM's puzzle embedding is a fundamental bottleneck**: 51B params can't scale
2. **Test-time fine-tuning is essential**: Pretrained models fail on ARC-AGI-2
3. **TRM has unique problem-solving capabilities**: Solves 2-3 puzzles ARChitects misses
4. **Ensembling is valuable but challenging**: Need better selection mechanisms

### How ETRM Addresses NVARC's TRM Limitations

**The 51B parameter problem**:
```python
# NVARC's forced compromise (doesn't scale):
# 3k puzzles × 256 aug × 512 params = 393M params (barely fits Kaggle)

# ETRM solution (scales infinitely):
# Encoder: processes demo pairs → task representation (no per-puzzle params)
demo_encoding = encoder(demo_inputs, demo_labels)  # Fixed params regardless of # puzzles
x = input_embedding(x_input) + demo_encoding
```

**Why ETRM is better for NVARC's use case**:
- **No memory scaling with puzzles**: Encoder has fixed parameters
- **Can use full synthetic dataset**: All 103k puzzles, no reduction needed
- **Can use 1000 augmentations**: No puzzle_id explosion
- **Better test-time adaptation**: Learn from test puzzle demos, not just fine-tune embeddings

### Potential Impact on NVARC Pipeline

**If NVARC used ETRM instead of TRM**:
1. **No puzzle reduction**: Use all 103k synthetic puzzles (vs 3k)
2. **No augmentation reduction**: Keep 1000 aug/puzzle (vs 256)
3. **Better generalization**: True few-shot learning from demos
4. **Faster test-time adaptation**: No need to fine-tune embedding table
5. **Better ensemble with ARChitects**: Complementary reasoning (ARChitects=pattern matching, ETRM=learn from demos)

### ETRM's Position in the Landscape

**NVARC shows**:
- ARChitects (Qwen3-4B): 29.72% (best overall, test-time fine-tuned LLM)
- TRM (original): 7.5% → 10% with synthetic data (limited by embeddings)
- Ensemble: Marginal gain due to selection issues

**ETRM hypothesis**:
- **ETRM standalone**: Should exceed 10% by using encoder instead of embeddings
- **ETRM + synthetic data**: Could scale to full 103k dataset (vs 3k for TRM)
- **ETRM + ARChitects ensemble**: Better complementarity (demos vs patterns)

### Lessons for ETRM Evaluation

1. **Need test-time adaptation**: Pretraining alone won't work on ARC-AGI-2
2. **Synthetic data quality matters**: NVARC's 4-stage pipeline shows what "good" means
3. **Voting is powerful**: 8-1000 augmentations with majority vote
4. **Ensemble selection is hard**: Need better scoring than just log-likelihood
5. **Compute constraints matter**: 12 hours on 4 L4 GPUs is realistic benchmark

## Limitations/Gaps

### Acknowledged in Paper
1. **Limited time for TRM tuning**: Only got ensemble working day before deadline
2. **Ensemble selection not optimized**: Qwen3 scoring misses unique TRM solutions
3. **Nondeterminism trade-off**: Batch-invariant DFS is 17% slower (not used)
4. **Synthetic data leakage**: Eval puzzle descriptions used in SDG (though still generalizes)
5. **One submission per day**: Limited ability to tune based on leaderboard feedback

### Critical Gaps

**TRM analysis**:
- **No investigation of why embedding reduction works**: 3k puzzles sufficient?
- **No ablation on augmentation count**: Does 256 vs 1000 matter for TRM?
- **No analysis of what TRM learns differently**: Why does it solve unique puzzles?
- **No attempt to improve ensemble selection**: Could better scoring help?

**Synthetic data quality**:
- **No evaluation of synthetic puzzle quality**: How "ARC-like" are they?
- **No analysis of mixing strategy**: Is 2-way mixing optimal? What about 3-way?
- **No ablation on LLM choice**: Does claude-opus-4 vs gpt-oss-120b matter?
- **No discussion of diversity metrics**: How diverse are 103k puzzles?

**Architecture choices**:
- **Why Qwen3?**: No comparison to other LLMs (Llama, Gemma, etc.)
- **Why LoRA r=256?**: No ablation on rank
- **Why 8 augmentations for re-scoring?**: No ablation on augmentation count

**Reproducibility**:
- **Closed models used**: claude-opus-4, claude-sonnet-4.5 not reproducible
- **Manual labeling**: 29 eval puzzles hand-labeled (not released?)
- **Kaggle-specific optimizations**: May not transfer to other settings

### The Fundamental Question ETRM Answers

**NVARC's implicit question**:
> "We had to reduce both the number of samples we use, and the number of augmentations we use, to be able to fit within Kaggle GPU memory."

**Why this happened**: Puzzle embedding table scales with (puzzles × augmentations × 512)

**ETRM's answer**: Replace puzzle_id embeddings with encoder → no scaling problem

**Testable hypothesis**: ETRM with NVARC's full synthetic dataset (103k puzzles, 1000 aug) should significantly outperform TRM with reduced dataset (3k puzzles, 256 aug).

---

**Bottom line**: NVARC demonstrates that (1) synthetic data generation is the key bottleneck for ARC-AGI-2, (2) TRM has unique problem-solving capabilities but is fundamentally limited by puzzle embeddings requiring 51B parameters, and (3) test-time adaptation is essential. ETRM directly addresses limitation #2, potentially unlocking TRM's full potential when combined with NVARC's synthetic data pipeline.
