# 2. Background and Related Work

## Outline

### 2.1 The ARC-AGI Benchmark
- Abstraction and Reasoning Corpus created to measure abstract reasoning and skill acquisition efficiency
- Task structure: 2-5 demonstration pairs showing input→output transformation + 1-2 test inputs to predict
- Grids up to 30x30, cell values 0-9 (colors)
- What makes ARC hard:
  - Out-of-distribution generalization required (hidden test set differs from training)
  - Compute efficiency is explicit goal (fixed hardware, capped time)
- Core knowledge priors: objectness, elementary physics, geometry, numerics
- **Cite**: Chollet 2019 "On the Measure of Intelligence"

### 2.2 Framing ARC as Program Synthesis
- Fundamental framing: find shortest program P such that P(input) = output for all demo pairs
- This framing unifies diverse approaches under common lens
- **Cite**: Li et al. 2024 "Combining Induction and Transduction" (2411.02272) - explicitly frames ARC as program synthesis
- **Cite**: ARC Prize 2024 Technical Report (2412.04604) - organizes competition approaches under this framing

#### Taxonomy of Approaches

Li et al. (2024) and Hemens (2025) organize ARC approaches along three axes:

**Axis 1: Inference Mode (Induction vs Transduction)**
- **Induction**: infer explicit program/rule, then apply to test input
  - Process: Examples → Search → Program → Execute → Output
  - Provides interpretable explanation
  - Examples: DSL synthesis, LLM code generation
- **Transduction**: directly predict test output from demos + test input
  - Process: (Examples + Test Input) → Network → Output
  - No explicit intermediate program; rule implicit in network activations
  - Examples: LLM+TTT, TRM/HRM, latent program networks
- **Skill Split** (Li et al.): near-orthogonal performance
  - Induction excels at: counting, arithmetic, long-horizon logic
  - Transduction excels at: noisy/fuzzy inputs, gestalt perception, topology
- **Cite**: Li et al. 2024 (primary source for this distinction in ARC context)

**Axis 2: Program Representation (Discrete vs Continuous)**
- *Discrete/Symbolic*: DSL primitives, Python code
  - Pros: perfect precision, natural compositionality, interpretable
  - Cons: non-differentiable, combinatorial search space
- *Continuous/Neural*: model weights, latent vectors, embeddings
  - Pros: differentiable, gradient-based optimization possible
  - Cons: imprecise execution, completeness hard to guarantee
- **Cite**: Chaudhuri et al. 2021 "Neurosymbolic Programming" - foundational theory for this distinction (predates ARC focus)

**Axis 3: Program Search (Heuristic vs Learned)**
- *Hand-crafted/Heuristic*: brute-force enumeration, MDL heuristics
- *Learned/Gradient-based*: SGD on weights (TTT), gradient ascent in latent space
- *Implicit/Emergent*: reasoning in thinking models, recursive refinement
- **Cite**: Chaudhuri et al. 2021 (general theory); Nye 2022 PhD thesis (goal-conditioned search)

**Synthesis**: Hemens (2025) combines these axes into a unified taxonomy for ARC, mapping all major approaches.
- **Cite**: Hemens 2025 "How to Beat ARC-AGI-2" (lewish.io) - ARC-specific synthesis

**Our work (ETRM)**: transductive approach with continuous representation (encoder output) and learned search (recursive decoder)

### 2.3 Overview of Approaches to ARC
*Brief survey (1-2 paragraphs), organized by representation × search*

| Approach | Program Representation | Program Search |
|----------|----------------------|----------------|
| DSL search (Icecuber) | Discrete (custom DSL) | Hand-crafted heuristics |
| LLM program synthesis | Discrete (Python) | LLM sampling + refinement |
| LLM + TTT (ARChitects, NVARC) | Neural (weights + prompt) | Gradient-based (SGD) |
| Latent Program Networks | Neural (latent vector) | Gradient-based (test-time) |
| TRM/HRM | Neural (puzzle embedding) | Learned (recursive decoder) |
| Thinking models (GPT-5.2, Gemini 3) | Neural (weights + thinking tokens) | Learned (implicit search) |

**Key observations from ARC 2024-2025**:
- Test-time adaptation crucial for generalization
- **Refinement loops** emerged as central theme in 2025 (iteratively improving programs/outputs)
- Ensembling transductive + inductive methods effective
- Augmentation + voting widely used

**Current state (ARC-AGI-2 leaderboard, Jan 2026)**:
- Frontier LLMs with thinking: GPT-5.2 Pro ~54%, Gemini 3 Deep Think ~45%, Opus 4.5 ~38%
- Kaggle winner (NVARC): 24% at $0.20/task (combines TTT + TRM)
- TRM: ~6% on ARC-AGI-2, 40% on ARC-AGI-1
- Human panel: 100% on ARC-AGI-2

**Cite**: ARC Prize 2024 Technical Report, ARC Prize 2025 Results Analysis, individual papers briefly

### 2.4 Hierarchical and Recursive Reasoning Models

#### 2.4.1 HRM: Hierarchical Reasoning Model
- Introduced hierarchical architecture for ARC: high-level module + low-level module
- High-level: reasons about abstract structure/strategy
- Low-level: executes pixel-level transformations
- Key innovation: outer-loop refinement process during training
- Results: 32% on ARC-AGI-1, 2% on ARC-AGI-2
- **Cite**: Wang et al. 2025 (HRM paper, 2506.21734)

#### 2.4.2 TRM: Tiny Recursive Model
- Simplifies HRM: single network with dual-state design (z_H, z_L)
- Only 7M parameters - remarkably small for its performance
- **1st Place Paper Award, ARC Prize 2025**
- Results: 45% on ARC-AGI-1, ~8% on ARC-AGI-2 (before test-time augmentation)
- **Core mechanism**:
  - Recursive reasoning: nested loops (H_cycles × L_cycles)
  - Deep supervision: loss at every recursion step
  - Adaptive computation via learned halting
- **Task conditioning**: puzzle_id embedding
  - Each puzzle assigned unique ID → lookup in learned embedding matrix
  - Embedding provides task context to guide transformation
  - Critical: demos not processed at inference time
- **Cite**: Jolicoeur-Martineau et al. 2025 (TRM paper, 2510.04871)

#### 2.4.3 TRM as a Refinement Loop
The ARC Prize 2025 analysis identified **refinement loops** as the central theme driving progress:
> "At its core, a refinement loop iteratively transforms one program into another, where the objective is to incrementally optimize a program towards a goal based on a feedback signal."

TRM exemplifies this: it recursively refines its predicted output over multiple steps, with deep supervision providing feedback at each step. This connects TRM to the broader trend of iterative refinement in ARC solutions.

**Cite**: ARC Prize 2025 Results Analysis

#### 2.4.4 The Memorization Problem
Analysis papers revealed critical limitation:

**Finding 1**: Puzzle_id is essential but limiting
- Replacing puzzle_id with blank/random token → 0% accuracy
- Model cannot function without task-specific embedding
- **Cite**: "Inductive Biases" paper (2512.11847)

**Finding 2**: Performance comes from memorization, not generalization
- "Cross-task transfer learning has limited benefits"
- "Most performance comes from memorizing solutions to specific tasks"
- HRM trained only on eval tasks still achieves ~31%
- **Cite**: ARC Prize "Hidden Drivers of HRM" analysis

**Finding 3**: Test-time augmentation accounts for significant portion
- ~11% of performance from 1000 samples + voting
- **Cite**: "Inductive Biases" paper (2512.11847)

**Implication**: TRM/HRM achieve strong results but deviate from ARC's intended few-shot paradigm. They memorize task→transformation mappings rather than learning to extract rules from demonstrations.

### 2.5 Latent Program Networks
*Most conceptually related to our encoder approach*

- **Core idea**: encode demonstrations → latent "program" vector → decode to output
- **Cite**: Bonnet et al. 2024 "Searching Latent Program Spaces" (2411.08706)

**Architecture**:
- Encoder: small transformer mapping demo pairs to latent space (dim 256)
- Decoder: neural executor mapping (latent program, test input) → output grid
- Both trained end-to-end

**Key innovation - test-time gradient search**:
- Initial guess from encoder
- Gradient ascent in latent space to refine program
- Uses leave-one-out loss on demo pairs as signal
- This search is critical to performance

**Results**: 3% test, 9.5% eval, trained from scratch without LLMs

**Relation to our work**:
- Similar: encode demos → latent representation used by decoder
- Different: LPN does gradient search at test time; we use encoder output directly
- Different: LPN trains from scratch; we build on TRM's proven decoder architecture
- Trade-off: LPN's search adds compute but may find better programs; our approach is faster but relies on encoder quality

### 2.6 Positioning Our Work: ETRM

**Motivation**: Combine TRM's effective decoder with true few-shot capability

**Our approach**:
- Take TRM's decoder (recursive refinement + deep supervision)
- Replace puzzle_id embedding lookup with learned encoder
- Encoder processes demonstration pairs → latent task representation
- No gradient-based search at test time (unlike LPN)

**Position in taxonomy**:
- **Representation**: continuous/neural (encoder output = latent program vector)
- **Search**: implicit/learned (TRM decoder performs recursive reasoning)
- **Mode**: transductive (directly predicts output, no explicit program)

**Key distinctions**:

| Aspect | TRM | LPN | ETRM (Ours) |
|--------|-----|-----|-------------|
| Task representation | Learned puzzle_id embedding | Encoder + gradient search | Encoder only |
| Processes demos at test time | No | Yes | Yes |
| Test-time optimization | No | Yes (gradient ascent) | No |
| Decoder architecture | Recursive reasoning | Feed-forward transformer | Recursive reasoning (from TRM) |
| Can generalize to unseen tasks | No | Yes | Yes (goal) |

**Research question**: Can replacing TRM's memorization mechanism with a learned encoder enable true few-shot generalization while preserving its reasoning capabilities?

---

*Target length: ~2.5-3 pages*

## Figures Needed
- [ ] Figure: Program synthesis framing taxonomy (representation × search)
- [ ] Figure: TRM architecture overview (simplified, focus on puzzle_id flow)
- [ ] Figure: ETRM architecture showing encoder replacing puzzle_id
- [ ] Table: Comparison of approaches (as shown in 2.6)

## Papers to Cite (Short References)

### ARC Benchmark & Framing
- Chollet 2019 - "On the Measure of Intelligence" - ARC benchmark, core knowledge priors
- Li et al. 2024 - "Combining Induction and Transduction" (2411.02272) - **primary**: frames ARC as program synthesis, induction/transduction distinction, skill split
- ARC Prize 2024 Technical Report (2412.04604) - organizes approaches under program synthesis framing
- Hemens 2025 - "How to Beat ARC-AGI-2" (lewish.io) - synthesizes full taxonomy for ARC context

### Foundational Theory (pre-ARC)
- Chaudhuri et al. 2021 - "Neurosymbolic Programming" - general theory for discrete/continuous representations and search strategies
- Nye 2022 - PhD thesis "Search and Representation in Program Synthesis" - goal-conditioned search theory

### Core Method Papers
- Wang et al. 2025 - HRM paper (2506.21734) - hierarchical reasoning
- Jolicoeur-Martineau et al. 2025 - TRM paper (2510.04871) - recursive reasoning, deep supervision
- Bonnet et al. 2024 - LPN/SLPS paper (2411.08706) - latent program networks, test-time optimization

### Analysis Papers (Back Our Narrative)
- ARC Prize 2025 Results Analysis (blog) - refinement loops theme
- "Inductive Biases" paper (2512.11847) - puzzle_id → 0% finding
- ARC Prize "Hidden Drivers of HRM" (blog) - memorization finding
- McGovern 2025 - "Test-time Adaptation of TRM" (2511.02886)

### Brief Mentions
- Icecuber (2020) - discrete heuristic search baseline
- ARChitects / NVARC - LLM+TTT approaches
- DreamCoder (Ellis et al. 2021) - library learning

### Local Files
See `docs/related-work/paper-index.md` for file path mappings.
