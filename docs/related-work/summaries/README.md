# Paper Summaries

Quick reference for citing and understanding related work. Each summary includes:
- TL;DR and key findings
- Mechanism/approach details
- Results and quotes with context
- Relevance to ETRM
- Limitations/gaps

## Method Papers

| Paper | arXiv | Summary | Key Contribution |
|-------|-------|---------|------------------|
| **Searching Latent Program Spaces (LPN)** | 2411.08706 | [Summary](./2411.08706-lpn.md) | Encoder → latent space + test-time gradient search |
| **Tiny Recursive Models (TRM)** | 2510.04871 | [Summary](./2510.04871-trm.md) | 7M param recursive reasoning with deep supervision |
| **Hierarchical Reasoning Model (HRM)** | 2506.21734 | [Summary](./2506.21734-hrm.md) | Brain-inspired dual-module hierarchical recurrence |
| **Test-time Adaptation of TRM** | 2511.02886 | [Summary](./2511.02886-trm-ttt.md) | Pre-train + test-time fine-tuning for generalization |

## Competition Reports & Analysis

| Document | Source | Summary | Key Contribution |
|----------|--------|---------|------------------|
| **ARC Prize 2024 Technical Report** | 2412.04604 | [Summary](./2412.04604-arc-prize-2024.md) | Competition results, taxonomy, 11% ceiling finding |
| **ARC Prize 2025 Results Analysis** | arcprize.org/blog | [Summary](./arc-prize-2025-results.md) | Refinement loops theme, TRM wins 1st paper award |

## Taxonomy & Background

| Document | Source | Summary | Key Contribution |
|----------|--------|---------|------------------|
| **ARC-AGI Research Review & Taxonomy** | lewish.io | [Summary](./lewish-io-arc-taxonomy.md) | 3-axis taxonomy: representation × search × inference |

---

## Quick Reference for Report Writing

### TRM Memorization Problem
See [TRM Summary](./2510.04871-trm.md#the-puzzleid-problem) - documents the puzzle_id embedding limitation that ETRM addresses.

### Refinement Loops / Recursive Reasoning
See [ARC Prize 2025](./arc-prize-2025-results.md#key-theme-refinement-loops) - TRM's recursive approach is the central theme of 2025 progress.

### Positioning ETRM in Taxonomy
See [lewish.io Taxonomy](./lewish-io-arc-taxonomy.md#relevance-to-etrm) - ETRM is continuous/learned/transductive, bridging static inference and TTT.

### LPN vs ETRM Comparison
See [LPN Summary](./2411.08706-lpn.md#relevance-to-etrm) - most conceptually related work, both encode demos to latent space.

### Why Full Fine-tuning vs Embeddings-only
See [TRM TTT Summary](./2511.02886-trm-ttt.md#key-findings) - embeddings-only achieves near-zero, suggesting representation alone insufficient (but ETRM encoder is trained end-to-end, not random init).

### The 11% Ceiling
See [ARC Prize 2024](./2412.04604-arc-prize-2024.md#key-findings) - no static inference solution scores above 11% without TTT.

---

## Citation Quick Reference

### Primary Sources for ETRM Report

**For TRM architecture:**
> Jolicoeur-Martineau et al. 2025. "Less is More: Recursive Reasoning with Tiny Networks." arXiv:2510.04871

**For HRM (TRM's predecessor):**
> Wang et al. 2025. "Hierarchical Reasoning Model." arXiv:2506.21734

**For LPN (most similar encoder approach):**
> Bonnet et al. 2024. "Searching Latent Program Spaces." arXiv:2411.08706

**For TTT alternative:**
> McGovern 2025. "Test-time Adaptation of Tiny Recursive Models." arXiv:2511.02886

**For taxonomy/framing:**
> Chollet et al. 2024. "ARC Prize 2024 Technical Report." arXiv:2412.04604
> Hemens 2025. "ARC-AGI 2025: A research review." lewish.io

**For competition landscape:**
> Knoop 2025. "ARC Prize 2025 Results and Analysis." arcprize.org/blog
