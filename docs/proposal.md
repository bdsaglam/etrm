## Problem Definition

The ARC-AGI benchmark evaluates abstract reasoning through visual puzzle tasks where each task presents 2-5 demonstration input-output pairs that implicitly define a transformation rule, followed by test inputs [1]. The benchmark is designed to assess few-shot learning: can a system extract the underlying abstract rule from demonstrations and generalize to novel instances?

The Tiny Recursive Model (TRM) [3] achieved 45% accuracy on ARC-AGI-1 using only 7M parameters, demonstrating that recursive refinement with deep supervision enables effective reasoning. However, analysis by the ARC Prize Foundation [4] revealed a critical limitation in how these models process tasks.

**The Core Problem**: TRM assigns each task a unique identifier (puzzle_id) that is fed into a learned embedding layer. During inference, the model receives only a single input grid and its puzzle_id—it never sees demonstration pairs. The transformation rule is therefore not inferred from demonstrations at test time; instead, it is encoded into the puzzle_id embedding weights during training. To evaluate on a task, the model must include that task's demonstration pairs in its training set. When the closely-related HRM model was trained exclusively on 400 evaluation tasks (excluding training tasks), performance only dropped from 41% to 31%, confirming that most capability comes from memorizing task-specific transformations rather than learning few-shot inference [4].

While technically not "cheating" (test outputs remain unseen), this fundamentally deviates from ARC-AGI's intended few-shot learning paradigm. The model cannot handle truly novel tasks—only those whose demonstrations appeared during training (see Figure 1 in Appendix).

**Research Question**: Can we modify TRM's architecture to enable true few-shot generalization by replacing task-specific embeddings with an encoder that extracts transformation rules directly from demonstration pairs at test time?

## Proposed Solution

We propose replacing TRM's puzzle_id embedding with a **demonstration encoder** that processes input-output pairs directly to generate task-conditional representations, shifting from memorizing task-specific rules to learning how to extract rules from demonstrations.

1. **Demonstration Encoder**: A neural network that takes 2-5 demonstration pairs as input and outputs a task context vector z_enc. The encoder learns to identify abstract patterns (object movement, color transformations, spatial relationships, compositional rules). We will evaluate both standard and variational formulations:
   - **Standard**: Direct encoding to fixed-size vector
   - **Variational**: Outputs distribution parameters (μ, σ²), sample z_enc ~ N(μ, σ²), with KL regularization for smoother latent space and uncertainty estimation

2. **Integration with TRM**: Replace puzzle_id embedding lookup with z_enc. This serves as task-conditional context guiding TRM's recursive reasoning. TRM's recursive refinement mechanism and deep supervision remain unchanged.

3. **Training**: Train with loss L = L_reconstruction + β × KL(q(z_enc|demos) || N(0, I)) (for variational formulation). During training, randomly sample demonstration pairs from each task to generate context vectors, then predict outputs for held-out examples from the same task. This forces the encoder to learn generalizable feature extraction rather than task-specific mappings. Following TRM, employ extensive data augmentation (rotations, reflections, color permutations).

**Critical Design Choice**: Evaluation tasks are strictly excluded from training—the model never sees their demonstration pairs during training, testing true few-shot generalization at test time.

## Experiment Plan

**Phase 1: Baseline Reproduction**
- Reproduce TRM's architecture and verify performance on ARC-AGI-1 public evaluation set

**Phase 2: Encoder Design and Integration**
- Implement demonstration encoder: neural network that takes demonstration pairs as input and outputs distribution parameters (μ, σ²) for sampling z_enc
- Integrate z_enc as replacement for puzzle_id embeddings in TRM's architecture

**Phase 3: Training and Evaluation**
- **Critical Training Protocol**: Strictly partition tasks into training and evaluation sets with zero overlap. The model will never see demonstration pairs from evaluation tasks during training.
- Train the modified model following the same dataset configuration and augmentation strategies (rotations, reflections, color permutations) as TRM
- Evaluate on 400 public evaluation tasks to measure true few-shot generalization capability
- Compare performance against reported TRM baselines

**Phase 4: Ablation Studies**
- **Compare variational encoding vs. standard (non-variational) encoding** to isolate the contribution of the generative formulation

**Evaluation Metrics**:
- Primary: Pass@k accuracy on held-out evaluation tasks (k=1,2)
- Comparison: Our model's few-shot performance vs. reported TRM baselines to assess whether learning from demonstrations enables better generalization than task-specific embeddings

## Dataset

**ARC-AGI-1 Benchmark**: 400 training and 400 evaluation tasks. Each task contains 2-5 demonstration pairs (median: 3) and 1-2 test inputs. Grids up to 30×30 with cell values 0-9.

The benchmark tests diverse reasoning patterns (object manipulation, spatial reasoning, pattern completion) while requiring only core knowledge priors. The strict train-test separation (400 training vs. 400 evaluation tasks) enables genuine generalization assessment. The dataset is computationally feasible for the project while providing sufficient scale when combined with data augmentation. Most importantly, this dataset directly tests our hypothesis: if the encoder learns to extract transformation rules from demonstrations, the model should generalize to evaluation tasks without seeing their demonstrations during training.

## References

1. Chollet, F. (2019). On the Measure of Intelligence. arXiv:1911.01547.
2. Wang, G., Li, J., Sun, Y., Chen, X., Liu, C., Wu, Y., Lu, M., Song, S., and Yadkori, Y. A. (2025). Hierarchical Reasoning Model. arXiv:2506.21734.
3. Jolicoeur-Martineau, A. (2025). Less is More: Recursive Reasoning with Tiny Networks. arXiv:2510.04871.
4. ARC Prize Foundation. (2025). The Hidden Drivers of HRM's Performance on ARC-AGI. https://arcprize.org/blog/hrm-analysis

## Appendix

![Figure 1: TRM training and test procedure. Demonstration pairs (left) that define the transformation rule are not fed to the model. Instead, only the puzzle_id embedding and individual input grids are processed through the recursive network (4 iterations shown). Both training and test require the same puzzle_id, meaning the model cannot generalize to tasks with unseen puzzle_ids.](./trm.png)
