# ARC-AGI Research Review & Taxonomy (Hemens 2025)

**Source**: lewish.io
**Posts**: "ARC-AGI 2025: A research review" and "How to beat ARC-AGI-2"
**Author**: Lewis Hemens
**Date**: April 2025 (review), July 2025 (taxonomy)

## TL;DR

Hemens provides a comprehensive review of ARC-AGI approaches and synthesizes them into a unified conceptual framework based on three key axes: program representation (discrete vs continuous), program search (heuristic vs learned), and inference mode (induction vs transduction). The core insight is that solving ARC requires navigating the fundamental trade-off between completeness (can your program space express all solutions?) and search tractability (can you find solutions efficiently?).

## The Taxonomy

Hemens synthesizes ARC approaches along three axes:

### Axis 1: Program Representation / Program Space

**Discrete vs Continuous**

- **Discrete/Symbolic** (e.g., Python, custom DSL)
  - Programs are composed of discrete primitives in a formal language
  - Executed by an explicit interpreter/runtime
  - Examples: Icecuber's DSL, ARC-DSL, Python code generation
  - **Strengths**: Inherently generalizable, perfect precision, naturally compositional
  - **Weaknesses**: Non-differentiable (can't use gradient descent), completeness depends on DSL design, large search spaces

- **Continuous/Neural** (e.g., latent vectors, model weights, thinking tokens)
  - Programs implicitly described by neural network states/weights
  - Executed through forward passes or token generation
  - Examples: TTT/TTFT, Searching Latent Program Spaces, thinking models
  - **Strengths**: Differentiable (enables gradient-based search), end-to-end learnable
  - **Weaknesses**: Data-dependent completeness (hard to prove coverage), imperfect precision (probabilistic errors), composition is difficult

**Key Quote**:
> "Programs might be described in Python code, and executed with a Python interpreter. Programs might be implicitly described by the state+weights of a neural net (Transformer), and executed through successive token generation."
- Context: Defining the spectrum of program representations from fully discrete to fully continuous.

**Mapping to Induction/Transduction**:
- Discrete representations → typically **inductive** (produce interpretable programs)
- Continuous representations → typically **transductive** (directly predict outputs)

### Axis 2: Program Search

**Heuristic vs Learned**

- **Hand-Crafted/Heuristic** (Explicit search)
  - Manual heuristics guide program exploration
  - Examples: Icecuber's breadth-first DSL search, MDL-based greedy search
  - Fixed search procedure designed by humans
  - Often very efficient when well-engineered

- **Gradient-Based** (Explicit search via optimization)
  - SGD/Adam performs local search in continuous spaces
  - Examples: TTT (search over model weights), SLPS (search over latent program vectors)
  - Leverages differentiability of continuous program spaces
  - Can get stuck in local optima

- **Learned Search** (Implicit or guided search)
  - Neural networks learn to guide or perform the search
  - Examples: Thinking models (O3, DeepSeek R1), DreamCoder's recognition model, LLM-guided program synthesis
  - **Most powerful**: Search procedure itself is learned from data
  - Can incorporate backtracking, self-verification, refinement

**Key Quote**:
> "What if that search process itself can be learned? We see the process of a thinking model and its thinking tokens as performing a type of search, and as these models were specifically designed to 'think' - they are leveraging that to propose, validate, refine and back track on ideas."
- Context: Explaining why thinking models represent a fundamentally different and potentially more powerful search paradigm.

**The Smart vs Fast Trade-off**:
> "You have a spectrum here - search smartly - search fast. No one has come up with a DL model that can effectively outperform a well engineered brute-force search, yet."
- Context: Noting that as of 2024, highly optimized brute-force search (like Icecuber) still beats neurally-guided search due to computational overhead.

### Axis 3: Inference Mode

**Induction vs Transduction**

- **Induction**: "Inferring latent functions"
  - First find/explain the transformation rule (create a program)
  - Then apply that program to test inputs
  - Produces an interpretable explanation
  - Examples: Program synthesis approaches, DreamCoder

- **Transduction**: "Directly predicting the test output"
  - Skip finding explicit rules
  - Directly generate outputs from examples via pattern matching
  - No interpretable intermediate program
  - Examples: TTT without code generation, pure LLM inference

**Important Nuance**:
> "I don't think this is a well defined distinction and gets particularly blurry when looking at some of the TTT (test-time-training) fine tuning approaches, and also with thinking language models."
- Context: Acknowledging that TTT fine-tunes model weights (arguably a form of program search in weight space), blurring the line.

**Complementarity**:
The 1st prize paper (Combining Induction and Transduction) showed that these approaches solve **different puzzles** - 26 solved only by induction, 35 only by transduction, 19 by both. Ensembling both methods is crucial for top performance.

## Key Approaches Covered

| Approach | Program Representation | Program Search | Inference Mode | 2024 Score |
|----------|----------------------|----------------|----------------|------------|
| **LLM + TTT** (ARChitects) | Continuous (weights + prompt) | Gradient-based (Adam/SGD on weights at test time) | Transductive | 53.5% |
| **Reasoning LLMs** (O3) | Continuous (weights + thinking tokens) | Learned (implicit search via reasoning) | Mixed | 87.5% (ARC-1) |
| **Heuristic DSL search** (Icecuber) | Discrete (custom DSL) | Hand-crafted (breadth-first with pruning) | Inductive | ~20% baseline |
| **LLM-guided synthesis** (Greenblatt) | Discrete (Python programs) | Learned (LLM generates/refines programs) | Inductive | 50% |
| **Searching Latent Program Spaces** (SLPS) | Continuous (latent vectors) | Gradient-based (SGD over latent space) | Inductive | 9.5% eval |
| **Neurally-guided discrete search** (DreamCoder) | Discrete (adaptive DSL library) | Learned (recognition model guides search) | Inductive | ~4.5% |
| **Omni-ARC** | Continuous (small LLM weights) | Gradient-based (per-task TTT) | Transductive | 40% |

## The Completeness vs Search Tractability Trade-off

**The Central Problem**:

> "From the above, what starts to emerge is a trade-off between two options: (1) Use a complete and discrete program representation, losing smoothness, and making search difficult (2) Limit the size of your search space, make it smooth, and search becomes tractable"

**The Challenge**:
- **Discrete programs**: Easy to make complete (Turing completeness is trivial), but search is intractable without gradient descent
- **Continuous programs**: Easy to search (gradient descent works), but hard to prove completeness (can your neural decoder actually compute all required transformations?)

**Two Research Directions**:

1. **Making smooth program spaces complete** (continuous → more expressive)
   - Challenge: Can a transformer in a single forward pass trace all paths in a maze? (requires recursive/iterative computation)
   - Potential solutions: Stacking models (refinement), more powerful differentiable computation primitives
   - Example problem requiring this: [b782dc8a](https://arcprize.org/play?task=b782dc8a) - maze tracing

2. **Making discrete program search tractable** (discrete → more efficient)
   - Use expressive DSLs to reduce search space
   - Build libraries of reusable primitives
   - Learn to navigate program space via feedback (not just sample from it)
   - Use MDL or other heuristics to guide search

**Key Insight on Expressivity**:
Hemens defines **expressivity** relative to a target program set:
> "If you have a set of primitives, and these can be composed together to form a set of full programs, then you can define the expressivity with respect to as: **The maximum number of primitives required to construct any program**"

More expressive DSLs = shorter programs = smaller search space.

## Key Concepts and Techniques

### Test-Time Adaptation

**Definition of Test-Time Adaptivity**:
> "The degree to which the search process at any given step is conditioned on all previous steps"

- Not just using test examples (everyone does that)
- **Adaptive search**: Using feedback from failed attempts to guide next attempts
- Examples: TTT (gradient from errors), thinking models (backtracking), program refinement loops

**Why TTT Works**:
- Fine-tune on test examples (or leave-one-out from demos)
- Model adapts to the specific puzzle's pattern
- Can be per-task or shared across all test tasks
- Logarithmic returns on inference compute (more attempts → better, but diminishing returns)

### Data Augmentations

Critical for both training and candidate selection:
- **Geometric**: Rotations, reflections, transpositions, dihedral transforms
- **Color**: Permutations of colors 1-9 (black=0 fixed)
- **Structural**: Reordering pairs, padding, upscaling
- **For LLMs**: Especially important for rotations (easier to see horizontal patterns than vertical)

**Use in Candidate Selection**:
> "A correct solution should demonstrate greater stability in its sampling probability under augmented conditions compared to an incorrect one"
- Run inference on 96+ augmented versions
- Vote or score by consistency across augmentations
- De-augment and select best

### Refinement

Allowing models to make multiple passes:
```
Input → f() → Partial output → f() → Partial output → f() → Final output
```

- Enables iterative/recursive computation
- Important for problems requiring linearly-dependent steps
- Small but significant performance boost
- Keeps everything differentiable if using log-probabilities

### Domain-Specific Languages (DSLs)

**Why DSLs Matter**:
- Python: Most programs are invalid/useless for ARC
- Good DSL: Most programs are semantically relevant
- Trade-off: Completeness (can solve all test problems?) vs Expressivity (short programs?)

**Challenges**:
- Hand-coded DSLs risk incompleteness (may not cover test set)
- Large DSLs (e.g., ARC-DSL with 160 primitives) are still hard to search
- Need to balance coverage with search efficiency

### Core Knowledge Priors

Minimal innate knowledge required to solve ARC:
1. **Objectness and elementary physics**: cohesion, persistence, contact
2. **Agentness and goal-directedness**: intentions, goal pursuit, efficiency
3. **Natural numbers and arithmetic**: small number representations, operations
4. **Elementary geometry and topology**: distance, orientation, containment

**Implication**: Models need these priors either hard-coded (discrete approaches) or learned from data (neural approaches).

## ARC-AGI-2 Changes (2025)

**Massive Difficulty Increase**:
- ARChitects: 53.5% → 3%
- O3-low: 76% → 4%
- Most LLMs: → 0%

**New Challenge Types**:
1. **Symbolic interpretation**: Symbols have meaning beyond visual patterns
2. **Compositional reasoning**: Simultaneous application of multiple interacting rules
3. **Contextual rule application**: Rules change based on context

**Dataset Changes**:
- Training: 400 → 1000 problems
- All eval sets: → 120 problems each
- Removed brute-force-susceptible problems (solved by Icecuber)

**Hardware**: Improved (more GPU memory available)

## Why Thinking Models Work So Well

**O3's Success** (87.5% on ARC-1, though only 4% on ARC-2):

Three key differentiators from gradient-based TTT:

1. **Explicitly learned search** (not fixed optimizer like Adam)
   - Model was trained to search effectively
   - Learned when to backtrack, verify, refine

2. **Search over discrete tokens** (not continuous weights)
   - Enables more complex, arguably Turing-complete computation
   - Thinking tokens act as a discrete program in text space

3. **Conditioned on entire search history** (not just local gradients)
   - Each step considers all previous attempts
   - True test-time adaptivity

**Conceptual Model**:
```
LLM(weights, input_tokens) → output

Split into:
LLM_think(weights, input) → thinking_tokens
LLM_solve(weights, input + thinking_tokens) → solution
```

The thinking tokens effectively reparameterize the program/model without changing weights.

## Insights and Research Directions

### Hemens' Hypotheses (Guiding His Work)

1. **Start with inductive approaches** (aligned with Chollet's view)
   - Focus on discrete program synthesis
   - Use deep learning for guided search (not as the program executor)

2. **Invest in expressive primitives and recomposition**
   - Build rich DSLs
   - Generate large synthetic datasets by recombining primitives
   - Explore program space offline (during pretraining) and online (at test time)

3. **Bring test-time adaptation to inductive approaches**
   - Learn to move through program space (not just sample from it)
   - Use feedback loops: program → execute → error → modify program
   - Train on synthetic search trajectories

**Program Modification via Feedback**:
```
Generate program p → Execute on demos → Get error signal →
Modify program p' → Execute again → Iterate until correct or timeout
```

This gives inductive approaches the same adaptivity that TTT gives transductive ones.

### What Hasn't Been Tried

From the ARC technical report:
> "One approach that has not been tried so far (likely because it is technically challenging) but that we expect to perform well in the future, is the use of specialist deep learning models to guide the branching decisions of a discrete program search process - similar to what can be seen in the AlphaProof system from Google DeepMind."

**The Gap**: No one has successfully built an AlphaGo-style system for ARC where:
- Discrete programs provide completeness and precision
- Specialist neural networks (policy + value) guide search
- Entire system is efficient enough to beat brute-force

## Key Datasets and Resources

**Extended Training Sets**:
- **Re-ARC**: Generators for all 400 ARC-AGI-1 training problems (more samples, same puzzles)
- **Concept-ARC**: 160 new problems (10 per concept group, 16 concepts)
- **ARC-Heavy (BARC)**: 200K new problems generated by LLMs (large but noisy)

**Simplified Variants**:
- Mini-ARC, 1D-ARC, Sort-of-ARC, LARC (language-annotated)

**Ecosystem**:
- [arcprize.org](https://arcprize.org): Official site, Discord, leaderboards
- [Awesome ARC](https://github.com/neoneye/arc-notes): Tools, papers, prompts, editors

## Relevance to ETRM

**How ETRM fits the taxonomy**:

1. **Program Representation**: **Continuous (hybrid)**
   - Encoder output is a continuous representation (512-dim × 16 tokens)
   - But it's computed from discrete demonstration examples (not learned embedding)
   - Decoder operates on this continuous representation

2. **Program Search**: **Learned (implicit)**
   - Recursive decoder performs learned search through reasoning steps
   - Not explicit gradient descent (like TTT)
   - Not heuristic search (like Icecuber)
   - Most similar to: Thinking models or SLPS, but without explicit test-time optimization

3. **Inference Mode**: **Transductive**
   - Directly predicts output grids (doesn't generate interpretable code)
   - Uses demonstrations to ground the prediction
   - No explicit intermediate program representation

**Position on the Completeness-Tractability Spectrum**:
- ETRM attempts to balance both:
  - **Completeness**: Demo encoder can theoretically capture any pattern present in demonstrations (data-dependent completeness, like neural approaches)
  - **Tractability**: Recursive decoder performs efficient learned search (no expensive gradient optimization at test time)

**Key Differentiator**:
- Unlike embedding-based TRM (learns puzzle-specific embeddings), ETRM is **encoder-based**
- This means it can generalize to truly unseen puzzles (not in training set)
- Similar to how thinking models generalize via in-context learning, but using demonstrations instead of thinking tokens

**Potential Improvements Based on Hemens' Insights**:
1. **Test-time adaptation**: Currently ETRM has none. Could add:
   - Refinement loops (multiple decoder passes)
   - Per-puzzle encoder fine-tuning
   - Feedback from partial predictions

2. **Augmentation voting**:
   - Generate predictions on augmented versions
   - Vote across de-augmented predictions
   - Already standard in TRM codebase but could be optimized

3. **Completeness**:
   - Verify encoder can represent complex transformations
   - Test on problems requiring recursive/iterative computation (like maze tracing)
   - Consider stacking encoders/decoders for more computational power

4. **Learned search**:
   - Decoder already does this, but could make it more explicit
   - Add verification heads (predict if solution is correct)
   - Train on search trajectories (partial → refined → final)
