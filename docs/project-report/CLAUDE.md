# Project Report Agent Map

Quick reference for agents working on this report.

---

## Project Summary

**Goal**: Write a conference-paper-style report for a term project on Encoder-Based TRM for ARC-AGI.

**Research Question**: Can we replace TRM's puzzle_id memorization with an encoder that extracts transformation rules from demonstration pairs, enabling true few-shot generalization?

**Key Contribution**: ETRM (Encoder-based TRM) - replaces learned puzzle embeddings with a neural encoder that processes demo pairs at inference time.

---

## Writing Method

**Approach**: "Outside-In" - start with factual sections, then synthesis sections

**Writing Order**:
1. Method (most concrete, doesn't need results)
2. Background & Related Work (factual)
3. Experiments Setup (factual)
4. Results & Analysis (once experiments complete)
5. Introduction (now you know the story)
6. Discussion (reflect on findings)
7. Abstract (write last)

**Per-Section Process**:
- Pass 1: Bullet-point outline (DONE for all sections)
- Pass 2: Expand to prose draft

---

## File Locations

### Report Structure
```
docs/project-report/
├── report.md              # Main index, tracks status
├── CLAUDE.md              # This file - agent reference
├── outlines/              # Bullet-point outlines (reference only)
│   ├── 00_abstract.md
│   ├── 01_introduction.md
│   ├── 02_background_and_related_work.md
│   ├── 04_method.md
│   ├── 05_experiments.md
│   ├── 06_discussion.md
│   └── 07_references.md
├── sections/              # Full prose drafts (create when writing)
│   └── (to be created)
└── figures/               # Diagrams and plots
```

### Section Mapping
| # | Section | Outline | Draft |
|---|---------|---------|-------|
| 0 | Abstract | `outlines/00_abstract.md` | `sections/00_abstract.md` |
| 1 | Introduction | `outlines/01_introduction.md` | `sections/01_introduction.md` |
| 2 | Background & Related Work | `outlines/02_background_and_related_work.md` | `sections/02_background_and_related_work.md` |
| 3 | Method | `outlines/04_method.md` | `sections/03_method.md` |
| 4 | Experiments | `outlines/05_experiments.md` | `sections/04_experiments.md` |
| 5 | Discussion | `outlines/06_discussion.md` | `sections/05_discussion.md` |
| - | References | `outlines/07_references.md` | `sections/06_references.md` |
| A | Appendix | `outlines/08_appendix.md` | `sections/07_appendix.md` |

> **Note**: Outline files keep old numbering for reference. Draft files use clean sequential numbering.

### Project Context
| File | Purpose |
|------|---------|
| `docs/proposal.md` | Original research proposal |
| `docs/project.md` | Detailed project briefing |
| `docs/codebase.md` | Codebase documentation |
| `docs/data_flow.md` | Data pipeline explanation |
| `docs/dataset.md` | Dataset description |
| `docs/future-work.md` | Future work |
| `docs/related-work/` | Related work |



### Experiment Files
| File | Purpose |
|------|---------|
| `jobs-etrm-semi-final.txt` | Semi-final experiment configs & results |
| `jobs-final.txt` | Final experiment configs |

### Key Code Files
| File | Purpose |
|------|---------|
| `models/recursive_reasoning/etrm.py` | Main ETRM model |
| `models/encoders/standard.py` | Standard encoder |
| `models/encoders/hybrid_variational.py` | Hybrid VAE encoder |
| `models/encoders/lpn_variational.py` | LPN encoder |
| `pretrain_etrm.py` | Training script |

---

## Current Status

- [x] Folder structure created
- [x] All section outlines written (including Appendix)
- [ ] Method section draft
- [ ] Background & Related Work section draft
- [ ] Experiments section (waiting for results)
- [ ] Introduction section draft
- [ ] Discussion section draft
- [ ] Appendix (training diagnostics, plots)
- [ ] Abstract (write last)
- [ ] Figures and tables
- [ ] Final compilation

---

## Report Parameters

- **Length**: ~10 pages main + appendices
- **Format**: Markdown
- **Audience**: Very familiar with generative models and machine learning in general, somewhat familiar with TRM/ARC-AGI
- **Style**: Conference paper, but include journey/attempts (term project)

---

## W&B Projects for Results

- `etrm-semi-final-subset-eval` - Semi-final architecture search
- `etrm-final` - Final training runs

Entity: `bdsaglam`
