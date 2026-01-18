# ARC Prize 2025 Results and Analysis

**Source**: https://arcprize.org/blog/arc-prize-2025-results-analysis
**Author**: Mike Knoop
**Date**: December 5, 2025

## TL;DR

ARC Prize 2025 wrapped with significant progress but no Grand Prize winner. The defining theme was "refinement loops" - iterative program optimization based on feedback signals. TRM (Tiny Recursive Model) won 1st place paper award with a 7M parameter network achieving 45% on ARC-AGI-1, demonstrating that test-time training with recursive refinement can match larger models.

## Key Theme: Refinement Loops

2025 became the "Year of the Refinement Loop" in AGI progress. A refinement loop is an iterative process that transforms one program into another, incrementally optimizing toward a goal based on feedback signals.

**Two-phase refinement process:**
1. **Explore**: Generate many candidate solutions
2. **Verify**: Analyze programs to yield feedback signal
3. **Repeat**: Loop until the program solves all training input/output pairs

This approach manifests in three key areas:

1. **Program Synthesis**: Evolutionary approaches (Berman, Pang) that evolve ARC solution programs in natural language or Python, dynamically creating abstraction libraries

2. **Zero-Pretraining Deep Learning**: New training paradigm that directly trains neural weights to represent task-solving programs, with extremely small networks and test-time training (TRM, CompressARC)

3. **Commercial AI Systems**: Chain-of-thought can be interpreted as natural language programs. Longer reasoning traces enable more refinement. Application-layer refinement loops (like Poetiq's approach) meaningfully improve reliability beyond model-native reasoning

The blog notes: "From an information theory perspective, refinement is intelligence."

## Competition Results

### High Scores (Kaggle Competition)
- **1st Place**: NVARC - 24.03% on ARC-AGI-2 ($0.20/task)
  - Synthetic-data ensemble with improved Architects-style test-time training + TRM components
- **2nd Place**: the ARChitects - 16.53%
  - 2D-aware masked-diffusion LLM with recursive self-refinement
- **3rd Place**: MindsAI - 12.64%
  - Test-time training pipeline with augmentation ensembles
- 1,455 teams submitted 15,154 entries (similar to 2024)

### Paper Awards (90 submissions, up from 47 in 2024)
- **1st Place ($50k)**: Alexia Jolicoeur-Martineau - "Less is More: Recursive Reasoning with Tiny Networks"
  - TRM: 7M parameters, 45% on ARC-AGI-1, 8% on ARC-AGI-2
- **2nd Place ($20k)**: Pourcel et al. - SOAR (Self-Improving Language Models)
  - Evolutionary program synthesis with LLM fine-tuning
- **3rd Place ($5k)**: Isaac Liao - "ARC-AGI Without Pretraining"
  - CompressARC: 76K parameters, 20% on ARC-AGI-1
  - No pretraining, test-time training only, MDL-based approach

**5 additional runners-up** and **8 honorable mentions** were added due to exceptional paper quality.

### Industry Progress
- Top commercial model: **Claude Opus 4.5 (Thinking, 64k)** - 37.6% for $2.20/task
- Top refinement solution: **Poetiq on Gemini 3 Pro** - 54% for $30/task
- All 4 major AI labs (OpenAI, xAI, Anthropic, Google DeepMind) now report ARC-AGI scores on model cards

## Key Findings

### What Worked in 2025

1. **Zero-Pretraining Deep Learning Methods**
   - TRM: Recursively improves answers with tiny network (7M params), up to 16 improvement steps
   - CompressARC: 76K params, minimizes description length (MDL principle), just gradient descent
   - Properties: Extremely small networks, test-time training, no pretraining needed

2. **Application-Layer Refinement**
   - Poetiq's refinement harness improved Gemini 3 Pro from 31% → 54%
   - Works on Claude Opus 4.5 too (similar accuracy, ~2× cost)
   - Adds refinement loops at application layer instead of relying solely on model reasoning

3. **Test-Time Training + Augmentation**
   - Still basis for top Kaggle scores (NVARC, ARChitects)
   - Augmentation voting mechanism remains powerful

### Paradigm Shifts

**AI Reasoning Systems** (emerged 2024): Tasks are automatable when they have:
1. Sufficient knowledge coverage in foundational model
2. Verifiable feedback signal

**New form of "overfitting"**: Models trained on broad domain data can now adapt to private test sets that are IID with public data, even without direct memorization. Evidence: Gemini 3 uses correct ARC color mappings in reasoning despite LLM harness not mentioning ARC.

**Benchmark Evolution**: Most useful benchmarks require sustained effort to adapt as technology improves. ARC Prize is applying its own refinement loop - iteratively improving benchmarks to keep "easy for humans, hard for AI" gap meaningful.

## Key Quotes

> "From an information theory perspective, [refinement is intelligence](https://arxiv.org/pdf/1310.8599v4). While we still need new ideas to achieve AGI, ARC has catalyzed several now open-source refinement approaches. I anticipate these will push AI reasoning further in 2026."
- Context: Explaining why refinement loops are the central theme of 2025 progress

> "TRM recursively improves its predicted answer y with a tiny network. It starts with the embedded input question x and initial embedded answer y, and latent z. For up to Nsup = 16 improvement steps, it tries to improve its answer y. It does so by i) recursively updating n times its latent z given the question x, current answer y, and current latent z (recursive reasoning), and then ii) updating its answer y given the current answer y and current latent z. This recursive process allows the model to progressively improve its answer (potentially addressing any errors from its previous answer) in an extremely parameter-efficient manner while minimizing overfitting."
- Context: Description of TRM architecture from the 1st place paper award winner

> "For the ARC-AGI-1/2 format, we believe the Grand Prize accuracy gap is now primarily bottlenecked by engineering while the efficiency gap remains bottlenecked by science and ideas."
- Context: Assessment of what's needed to close remaining gaps

> "You'll know AGI is here when the exercise of creating tasks that are easy for regular humans but hard for AI becomes simply impossible." (Francois Chollet, December 2024)
- Context: The core philosophy driving ARC-AGI benchmark evolution

> "The ARC-AGI playbook: run a refinement loop by iteratively improving benchmarks in response to AI progress in order to drive the gap between 'easy for humans, hard for AI' to zero."
- Context: Meta-commentary on benchmark development as itself a refinement loop

## Relevance to ETRM

**TRM's Recognition**: TRM won 1st place paper award ($50k) and was integrated into the winning Kaggle solution (NVARC). This validates the core recursive reasoning approach that ETRM builds upon.

**Refinement Loops = Recursive Reasoning**: The blog's central theme directly connects to our work:
- TRM performs up to 16 recursive improvement steps on its answer
- Each step refines the latent representation and answer
- This is exactly the "refinement loop" paradigm the blog identifies as defining 2025 progress
- ETRM extends this by replacing learned embeddings with encoder-computed representations

**Positioning ETRM**:
1. **Builds on award-winning architecture**: ETRM inherits TRM's recursive refinement mechanism
2. **Addresses generalization**: TRM uses learned embeddings (in embedding matrix), limiting generalization to unseen puzzles. ETRM computes representations from demo examples, enabling true few-shot generalization
3. **Aligns with 2025 trends**: Both test-time adaptation and learning from demonstrations are validated approaches. ETRM combines them
4. **Parameter efficiency**: If ETRM maintains TRM's small model size while improving generalization, it exemplifies the "zero-pretraining deep learning" paradigm the blog highlights

**Key insight for our report**: Frame ETRM as extending the refinement loop concept - TRM refines answers recursively, ETRM refines puzzle representations from examples before applying recursive reasoning. This positions our work within the recognized paradigm shift of 2025.

**ARC-AGI-3 implications**: New format (early 2026) will test interactive reasoning, exploration, planning, memory, goal acquisition. ETRM's demo-based approach may be better positioned than embedding-based approaches for these new challenges.
