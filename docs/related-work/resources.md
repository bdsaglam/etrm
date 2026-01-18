## Analysis

ARC Prize 2024: Technical Report
https://arxiv.org/pdf/2412.04604

https://lewish.io/
https://lewish.io/posts/how-to-beat-arc-agi-2

ARC-AGI 2025 Results and Analysis
https://arcprize.org/blog/arc-prize-2025-results-analysis


## Foundational Methods

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
Vaswani et al. (2017). Introduces the Transformer architecture.

[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
Kingma & Welling (2014). Introduces the Variational Autoencoder (VAE) framework.

[Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](https://arxiv.org/abs/1810.00825)
Lee et al. (2019). Introduces cross-attention with learnable query tokens for set aggregation.


## Related Works

[Latent Program Network](https://arxiv.org/abs/2411.08706)
Introduces a latent program network that learns a latent space of implicit programs-neurally mapping inputs to outputs-through which it can search using gradients at test time. 
One of the top 3 papers in ARC-AGI 2024. 
Repo: https://github.com/clement-bonnet/lpn


[Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734)
Introduces a hierarchical reasoning model that uses a high-level and a low-level module to perform reasoning.


[Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)
Simplifies the hierarchical reasoning model by using a single network to perform reasoning and achieves 45% accuracy on ARC-AGI 1. 
ARC-AGI 2025 top paper award winner. 
Repo: https://github.com/SamsungSAILMontreal/TinyRecursiveModels


[Tiny Recursive Models on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute](https://arxiv.org/abs/2512.11847)
Key findings:
- Puzzle ID dependency confirmed: Replacing puzzle_id with blank/random
token → 0% accuracy
- Test-time augmentation (1000 samples + voting) accounts for ~11% of
performance
- Most computation happens in first recursion step; subsequent steps are
shallow refinements
- TRM's performance comes from "efficiency + task-specific conditioning +
aggressive test-time compute, not deep internal reasoning"


[The Hidden Drivers of HRM's Performance on ARC-AGI](https://arcprize.org/blog/hrm-analysis)
Summary of our findings:
- First of all: we were able to approximately reproduce the claimed numbers. HRM shows impressive performance for its size on the ARC-AGI Semi-Private sets:

ARC-AGI-1: 32% - Though not state of the art, this is impressive for such a small model.
ARC-AGI-2: 2% - While scores >0% show some signal, we do not consider this material progress on ARC-AGI-2.
At the same time, by running a series of ablation analyses, we made some surprising findings that call into question the prevailing narrative around HRM:

The "hierarchical" architecture had minimal performance impact when compared to a similarly sized transformer.
However, the relatively under-documented "outer loop" refinement process drove substantial performance, especially at training time.
Cross-task transfer learning has limited benefits; most of the performance comes from memorizing solutions to the specific tasks used at evaluation time.
Pre-training task augmentation is critical, though only 300 augmentations are needed (not 1K augmentations as reported in the paper). Inference-time task augmentation had limited impact.


[[2511.02886] Test-time Adaptation of Tiny Recursive Models](https://arxiv.org/abs/2511.02886)
This paper shows that, by starting from a tiny recursive model that has been pre-trained on public ARC tasks, one can efficiently fine-tune on competition tasks within the allowed compute limits. Specifically, a model was pre-trained on 1,280 public tasks for 700k+ optimizer steps over 48 hours on 4xH100 SXM GPUs to obtain a ~10% score on the public evaluation set. That model was then post-trained in just 12,500 gradient steps during the competition to reach a score of 6.67% on semi-private evaluation tasks. Notably, such post-training performance is achieved by full-fine tuning of the tiny model, not LoRA fine-tuning or fine-tuning of task embeddings alone.



