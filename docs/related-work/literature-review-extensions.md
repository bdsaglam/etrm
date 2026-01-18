Literature Review: Papers Extending HRM/TRM

1. Hierarchical Reasoning Models: Perspectives and Misconceptions

Authors: Renee Ge, Qianli Liao, Tomaso Poggio (MIT)
arXiv: https://arxiv.org/abs/2510.00355 (Sep 2025)

Key findings:
- Even without the H module, a plain 8-layer Transformer performs
similarly to the original HRM
- Deep supervision is the primary driver of performance gains, not the
hierarchical structure
- The paper clarifies misconceptions about what makes HRM work

Relevance to your project: Confirms that the recursive refinement
mechanism is valuable, but questions whether the specific hierarchical
architecture is necessary.

---
2. Less is More: Recursive Reasoning with Tiny Networks (TRM)

Author: Alexia Jolicoeur-Martineau (Samsung SAIL Montréal)
arXiv: https://arxiv.org/abs/2510.04871 (Oct 2025)

Key improvements over HRM:
- Single 2-layer network instead of dual 4-layer networks
- Full backpropagation through recursions (vs 1-step approximation)
- 7M params achieving 45% (vs HRM's 27M params at 40%)
- Simpler training recipe with EMA

Relevance: This is the baseline you're building on. The paper does NOT
address the puzzle_id limitation.

---
3. Tiny Recursive Models on ARC-AGI-1: Inductive Biases, Identity 
Conditioning, and Test-Time Compute

Authors: Antonio Roye-Azar, Santiago Vargas-Naranjo, Dhruv Ghai, Nithin
Balamurugan, Rayan Amir
arXiv: https://arxiv.org/abs/2512.11847 (Dec 2025)

Key findings:
- Puzzle ID dependency confirmed: Replacing puzzle_id with blank/random
token → 0% accuracy
- Test-time augmentation (1000 samples + voting) accounts for ~11% of
performance
- Most computation happens in first recursion step; subsequent steps are
shallow refinements
- TRM's performance comes from "efficiency + task-specific conditioning +
aggressive test-time compute, not deep internal reasoning"

Relevance to your project: Directly validates your hypothesis that TRM
relies on puzzle_id memorization rather than genuine reasoning.

---
4. Test-time Adaptation of Tiny Recursive Models

Author: Ronan McGovern (Trelis LTD)
arXiv: https://arxiv.org/abs/2511.02886 (Nov 2025)

Key findings:
- Pre-trained TRM on 1,280 public tasks → ~10% on public eval
- Test-time fine-tuning (12,500 steps) → 6.67% on semi-private tasks
- Full fine-tuning works better than LoRA or task-embedding-only
fine-tuning
- Makes TRM competitive within ARC Prize compute constraints

Relevance: Shows test-time adaptation can help, but still relies on
task-specific learning rather than true few-shot from demonstrations.

---
5. CompressARC (ARC-AGI Without Pretraining)

Author: Isaac Liao
arXiv: https://arxiv.org/abs/2512.06104 (Dec 2025)

Key approach:
- 76K parameter network trained from scratch on EACH task at test time
- Minimizes description length (MDL principle)
- Achieves ~20% on ARC-AGI-1 with zero pretraining

Relevance: Different approach to few-shot—learns per-task at test time
rather than using demonstrations. Shows that task-specific adaptation at
test time can work.

---
Summary Table

| Paper                                    | Approach                  |
Key Contribution                  | Addresses puzzle_id? |
|------------------------------------------|---------------------------|--
---------------------------------|----------------------|
| HRM (Wang et al.)                        | Dual-loop recurrent       |
Original architecture             | No                   |
| TRM (Jolicoeur-Martineau)                | Simplified single network |
Better performance, fewer params  | No                   |
| Perspectives (Ge et al.)                 | Analysis                  |
Shows H-module may be unnecessary | No                   |
| Identity Conditioning (Roye-Azar et al.) | Analysis                  |
Confirms puzzle_id dependency     | Analysis only        |
| Test-time Adaptation (McGovern)          | Fine-tuning               |
Adapts within compute limits      | No                   |
| CompressARC (Liao)                       | MDL at test time          |
Zero pretraining approach         | Different approach   |

---
Gap Your Project Fills

None of these papers propose replacing puzzle_id embeddings with a 
demonstration encoder for true few-shot generalization. The closest is
CompressARC, which trains from scratch per-task, but this is
computationally expensive and different from learning to read
demonstrations.

Your contribution is novel: a demonstration encoder that processes
input-output examples to produce task context, enabling the model to
generalize to truly unseen tasks without any task-specific parameters.

---
Sources:
- https://arxiv.org/abs/2510.00355
- https://arxiv.org/abs/2510.04871
- https://arxiv.org/abs/2512.11847
- https://arxiv.org/abs/2511.02886
- https://arxiv.org/abs/2512.06104
- https://arcprize.org/blog/hrm-analysis


How CompressARC works:
1. Takes a single ARC task with its demonstration pairs (input→output
examples)
2. At test time, trains a tiny 76K parameter network from scratch on that
specific task
3. The network learns to "compress" the demonstrations by overfitting to
them
4. Uses MDL (Minimum Description Length) principle: the network that best
compresses the demos should also predict the correct test output
5. No pretraining—each task starts from random initialization

Key insight: It's essentially learning a task-specific transformation rule
  at test time by fitting to the demonstrations, then applying that learned
  rule to the test input.

Relevance to your project: CompressARC actually validates your intuition
that demonstrations can be used to extract transformation rules. The
difference is:
- CompressARC: Trains a new network from scratch per task at test time
(expensive, ~minutes per task)
- Your approach: Trains one encoder that can process demonstrations in a
single forward pass (fast, generalizes across tasks)

Your approach is more aligned with how humans solve ARC—we don't "retrain
our brain" for each puzzle, we read the examples and apply what we learn.


Literature Review: Extensions and Related Work to HRM/TRM

  Papers Directly Extending HRM/TRM

  1. Hierarchical Reasoning Models: Perspectives and Misconceptions

  Authors: Renee Ge, Qianli Liao, Tomaso Poggio (MIT)
  arXiv: https://arxiv.org/abs/2510.00355 (Sep 2025)

  Contribution: Critical analysis of HRM's architecture. Shows that a plain 8-layer Transformer performs similarly to HRM's dual-module
  design. Identifies deep supervision as the primary driver of performance, not the hierarchical structure.

  Relevance: Confirms the recursive refinement mechanism is valuable, but questions whether the specific H/L module split matters.

  ---
  2. Less is More: Recursive Reasoning with Tiny Networks (TRM)

  Author: Alexia Jolicoeur-Martineau (Samsung SAIL Montréal)
  arXiv: https://arxiv.org/abs/2510.04871 (Oct 2025)

  Contribution: Simplifies HRM to a single 2-layer network (7M params) achieving 45% on ARC-AGI-1 vs HRM's 40% with 27M params. Key
  improvements: full backpropagation through recursions, EMA, simpler ACT.

  Relevance: Our baseline architecture. Does NOT address the puzzle_id limitation.

  ---
  3. Tiny Recursive Models on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute

  Authors: Antonio Roye-Azar et al.
  arXiv: https://arxiv.org/abs/2512.11847 (Dec 2025)

  Contribution: Empirical analysis showing:
  - Replacing puzzle_id with blank/random token → 0% accuracy
  - Test-time augmentation (1000 samples) accounts for ~11% of performance
  - Most computation happens in first recursion step

  Relevance: Directly validates our hypothesis that TRM relies on puzzle_id memorization. Confirms the limitation we aim to address.

  ---
  4. Test-time Adaptation of Tiny Recursive Models

  Author: Ronan McGovern
  arXiv: https://arxiv.org/abs/2511.02886 (Nov 2025)

  Contribution: Shows TRM can be adapted at test time via fine-tuning (12,500 steps) to reach 6.67% on semi-private tasks within competition
  compute limits.

  Relevance: Explores test-time adaptation but still requires gradient updates per task, not demonstration-based conditioning.

  ---
  5. CompressARC (ARC-AGI Without Pretraining)

  Author: Isaac Liao
  arXiv: https://arxiv.org/abs/2512.06104 (Dec 2025)

  Contribution: 76K parameter network trained from scratch on each task at test time using MDL (minimum description length) principle.
  Achieves ~20% on ARC-AGI-1 with zero pretraining.

  Relevance: Uses demonstrations to learn task-specific transformation, but requires expensive per-task training (minutes per task) rather
  than a single forward pass.

  ---
  Related Work: Demonstration Encoding Approach

  6. Searching Latent Program Spaces (LPN)

  Authors: Matthew Macfarlane, Clément Bonnet et al.
  arXiv: https://arxiv.org/abs/2411.08706 (Nov 2024)
  Award: 3rd Place Paper Award at ARC Prize 2024, NeurIPS 2025 Spotlight

  Contribution: Proposes Latent Program Network that:
  - Encodes I/O demonstration pairs → latent distribution (mean + covariance)
  - Uses variational formulation (like VAE)
  - Performs gradient-based search in latent space at test time
  - No task-specific embeddings; generalizes to novel tasks

  Relevance: Highly relevant - shares our core insight that demonstrations can be encoded to produce task context. Key prior work.

  ---
  Gap Analysis: Why Our Work is Still Novel

  | Aspect                 | LPN (2024)             | TRM (2025)            | Our Proposal                |
  |------------------------|------------------------|-----------------------|-----------------------------|
  | Encodes demonstrations | ✓                      | ✗ (uses puzzle_id)    | ✓                           |
  | Recursive refinement   | ✗                      | ✓                     | ✓                           |
  | Test-time search       | ✓ (gradient-based)     | ✗                     | ✗ (single forward pass)     |
  | Variational            | ✓                      | ✗                     | ✓                           |
  | Architecture           | Custom encoder-decoder | Recursive Transformer | TRM + demonstration encoder |

  Our contribution bridges these two lines of work:

  1. Combines the best of both: LPN's demonstration encoding + TRM's recursive refinement
  2. Minimal modification: We don't propose a new architecture—we surgically replace puzzle_id with a demonstration encoder in the existing
  TRM
  3. Efficiency: LPN requires expensive test-time gradient search; ours is a single forward pass
  4. Direct ablation: Provides first direct comparison of "TRM with puzzle_id" vs "TRM with demonstration encoder"—isolating the effect of the
   conditioning mechanism
  5. Validates the hypothesis: Tests whether TRM's recursive refinement can work with dynamically computed task context rather than memorized
  embeddings

  Research Question (refined): Can we preserve TRM's recursive refinement performance while enabling true few-shot generalization by replacing
   memorized puzzle_id embeddings with demonstration-derived task context?

  ---
  Timeline Context

  Nov 2024: LPN - demonstrates encoder approach works for ARC
  Jun 2025: HRM - recursive refinement achieves 40% with small model
  Oct 2025: TRM - simplifies HRM to 7M params, 45% accuracy
  Dec 2025: Analysis papers confirm puzzle_id is critical (0% without it)
  Our work: Combines LPN's encoding insight with TRM's recursive refinement