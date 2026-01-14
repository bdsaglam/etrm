# TRM vs ETRM Puzzle Representation Comparison

## TRM (Embedding Mode)

From config/arch/trm.yaml:
hidden_size: 512
puzzle_emb_ndim: ${.hidden_size}  # = 512
puzzle_emb_len: 16

- Puzzle embedding shape: (num_puzzles, 16, 512)
- Per-puzzle representation: (16, 512)
- 16 tokens, each with 512 dimensions
- Method: Learned embedding matrix stored in model.model.puzzle_emb.weights
- Training: Embeddings are learned parameters that get gradient updates

## ETRM (Encoder Mode)

From config/arch/etrm.yaml:
hidden_size: 512
puzzle_emb_len: 16  # = output_tokens
puzzle_emb_ndim: 0  # disabled

From models/encoders/standard.py:518:
```python
context = self.set_encoder(demo_encodings, demo_mask)  # (B, T, D)
# Returns: (batch, output_tokens, hidden_size)
```

- Encoder output shape: (batch, 16, 512)
- Per-puzzle representation: (16, 512)
- 16 tokens, each with 512 dimensions
- Method: Computed dynamically from demonstration examples via:
a. DemoGridEncoder: Encodes each (input, output) demo pair
b. DemoSetEncoder: Aggregates demos via cross-attention with 16 learnable query tokens
- Training: Encoder parameters are learned (not per-puzzle embeddings)

## Key Insight

Both modes produce identical tensor shapes (16, 512) for puzzle representation:
- 16 context tokens (puzzle_emb_len / output_tokens)
- 512 dimensions per token (hidden_size)

The only difference is how they obtain this representation:
- TRM: Lookup from learned embedding matrix (requires puzzle seen during training)
- ETRM: Computed from demo examples (can generalize to unseen puzzles)

This architectural equivalence ensures fair comparison between the two approaches!