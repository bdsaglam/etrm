"""
Standard (deterministic) demonstration encoder.

Architecture:
1. DemoGridEncoder: Encodes each (input, output) pair into a fixed-size representation
2. DemoSetEncoder: Aggregates multiple demo encodings via cross-attention

The output replaces the puzzle_id embedding in TRM.

Configurable options (see DemoEncoderConfig):
- pooling_method: "mean" (default), "attention", "weighted"
- set_encoder_layers: depth of set aggregation (default 1)
- layer_scale_init: CaiT-style layer scaling (default 0 = disabled)
- norm_style: "pre" or "post" normalization
- qk_norm: normalize Q/K in attention
"""

import math
from typing import Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import einops

from models.encoders.base import BaseDemoEncoder, DemoEncoderConfig, EncoderOutput
from models.common import trunc_normal_init_
from models.layers import (
    CastedLinear,
    CastedEmbedding,
    Attention,
    SwiGLU,
    rms_norm,
    RotaryEmbedding,
)


class DemoGridEncoderBlock(nn.Module):
    """Single transformer block for encoding demo grids.

    Supports:
    - Pre-norm vs post-norm
    - Layer scale (CaiT-style)
    """

    def __init__(self, config: DemoEncoderConfig):
        super().__init__()
        self.config = config
        self.norm_style = config.norm_style

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

        # Layer scale parameters (CaiT-style)
        self.use_layer_scale = config.layer_scale_init > 0
        if self.use_layer_scale:
            self.gamma_1 = nn.Parameter(
                config.layer_scale_init * torch.ones(config.hidden_size)
            )
            self.gamma_2 = nn.Parameter(
                config.layer_scale_init * torch.ones(config.hidden_size)
            )

    def forward(self, hidden_states: torch.Tensor, cos_sin) -> torch.Tensor:
        if self.norm_style == "pre":
            # Pre-norm: norm before attention/mlp (more stable)
            attn_out = self.self_attn(
                cos_sin=cos_sin,
                hidden_states=rms_norm(hidden_states, variance_epsilon=self.norm_eps)
            )
            if self.use_layer_scale:
                attn_out = self.gamma_1 * attn_out
            hidden_states = hidden_states + attn_out

            mlp_out = self.mlp(rms_norm(hidden_states, variance_epsilon=self.norm_eps))
            if self.use_layer_scale:
                mlp_out = self.gamma_2 * mlp_out
            hidden_states = hidden_states + mlp_out
        else:
            # Post-norm: norm after attention/mlp (original transformer)
            attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
            if self.use_layer_scale:
                attn_out = self.gamma_1 * attn_out
            hidden_states = rms_norm(
                hidden_states + attn_out,
                variance_epsilon=self.norm_eps,
            )

            mlp_out = self.mlp(hidden_states)
            if self.use_layer_scale:
                mlp_out = self.gamma_2 * mlp_out
            hidden_states = rms_norm(
                hidden_states + mlp_out,
                variance_epsilon=self.norm_eps,
            )

        return hidden_states


class DemoGridEncoder(nn.Module):
    """
    Encodes a single (input, output) demonstration pair.

    Takes: input grid (seq_len,) + output grid (seq_len,)
    Returns: single vector (hidden_size,)

    Pooling methods:
    - "mean": simple mean pooling (loses positional info)
    - "attention": learned query cross-attends to sequence (preserves position)
    - "weighted": attention-weighted mean (lightweight alternative)
    """

    def __init__(self, config: DemoEncoderConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        self.pooling_method = config.pooling_method

        # Separate embeddings for input and output grids
        embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / embed_scale

        self.input_embed = CastedEmbedding(
            config.vocab_size, config.hidden_size,
            init_std=embed_init_std, cast_to=self.forward_dtype
        )
        self.output_embed = CastedEmbedding(
            config.vocab_size, config.hidden_size,
            init_std=embed_init_std, cast_to=self.forward_dtype
        )

        # Positional encoding for concatenated sequence (2 * seq_len)
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_heads,
            max_position_embeddings=2 * config.seq_len,
            base=10000.0,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            DemoGridEncoderBlock(config) for _ in range(config.num_layers)
        ])

        # Pooling components
        if self.pooling_method == "attention":
            # Attention pooling: learned query cross-attends to sequence
            self.pool_query = nn.Parameter(
                trunc_normal_init_(
                    torch.empty(1, config.hidden_size, dtype=self.forward_dtype),
                    std=0.02
                )
            )
            # Cross-attention components
            head_dim = config.hidden_size // config.num_heads
            self.pool_q_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
            self.pool_k_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
            self.pool_v_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
            self.pool_o_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
            self.pool_num_heads = config.num_heads
            self.pool_head_dim = head_dim
        elif self.pooling_method == "weighted":
            # Weighted pooling: learned attention weights
            self.pool_weight_proj = CastedLinear(config.hidden_size, 1, bias=False)

        # Output projection (all pooling methods)
        self.pool_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)

        self.embed_scale = embed_scale
        self.norm_eps = config.rms_norm_eps

    def _attention_pool(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attention pooling with learned query."""
        batch_size = hidden.shape[0]

        # Expand query for batch
        query = self.pool_query.expand(batch_size, -1, -1)  # (B, 1, D)

        # Project Q, K, V
        q = self.pool_q_proj(query)  # (B, 1, D)
        k = self.pool_k_proj(hidden)  # (B, 2*S, D)
        v = self.pool_v_proj(hidden)  # (B, 2*S, D)

        # Reshape for multi-head attention
        q = q.view(batch_size, 1, self.pool_num_heads, self.pool_head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.pool_num_heads, self.pool_head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.pool_num_heads, self.pool_head_dim).transpose(1, 2)

        # Create attention mask
        attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, 2*S)
        attn_mask = attn_mask.expand(-1, self.pool_num_heads, 1, -1)
        attn_bias = torch.zeros_like(attn_mask, dtype=q.dtype)
        attn_bias.masked_fill_(~attn_mask, float("-inf"))

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1)
        return self.pool_o_proj(attn_output.unsqueeze(1)).squeeze(1)  # (B, D)

    def _weighted_pool(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted mean pooling with learned attention weights."""
        # Compute attention weights
        weights = self.pool_weight_proj(hidden).squeeze(-1)  # (B, 2*S)

        # Mask invalid positions
        weights = weights.masked_fill(~mask, float("-inf"))
        weights = F.softmax(weights, dim=-1)  # (B, 2*S)

        # Weighted sum
        pooled = (weights.unsqueeze(-1) * hidden).sum(dim=1)  # (B, D)
        return pooled

    def _mean_pool(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Simple mean pooling over non-PAD tokens."""
        mask_expanded = mask.unsqueeze(-1).to(hidden.dtype)  # (B, 2*S, 1)
        masked_hidden = hidden * mask_expanded
        token_counts = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
        pooled = masked_hidden.sum(dim=1) / token_counts  # (B, D)
        return pooled

    def forward(
        self,
        input_grid: torch.Tensor,
        output_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a single demo pair.

        Args:
            input_grid: (batch, seq_len) - Input grid tokens
            output_grid: (batch, seq_len) - Output grid tokens

        Returns:
            encoding: (batch, hidden_size) - Demo encoding
        """
        # Embed input and output
        input_emb = self.input_embed(input_grid.to(torch.int32))   # (B, S, D)
        output_emb = self.output_embed(output_grid.to(torch.int32))  # (B, S, D)

        # Concatenate: [input; output]
        hidden = torch.cat([input_emb, output_emb], dim=1)  # (B, 2*S, D)
        hidden = self.embed_scale * hidden

        # Apply transformer layers
        cos_sin = self.rotary_emb()
        for layer in self.layers:
            hidden = layer(hidden, cos_sin)

        # Create mask for non-PAD tokens (PAD = 0)
        full_grid = torch.cat([input_grid, output_grid], dim=1)  # (B, 2*S)
        token_mask = (full_grid != 0)  # (B, 2*S) - True for non-PAD

        # Apply pooling method
        if self.pooling_method == "attention":
            pooled = self._attention_pool(hidden, token_mask)
        elif self.pooling_method == "weighted":
            pooled = self._weighted_pool(hidden, token_mask)
        else:  # "mean"
            pooled = self._mean_pool(hidden, token_mask)

        # Project and return
        return self.pool_proj(pooled)


class DemoSetEncoderLayer(nn.Module):
    """Single cross-attention layer for set encoding.

    Queries attend to demo encodings, then apply MLP.
    Supports pre/post norm and layer scale.
    """

    def __init__(self, config: DemoEncoderConfig):
        super().__init__()
        self.config = config
        self.norm_style = config.norm_style

        # Cross-attention components
        head_dim = config.hidden_size // config.num_heads
        self.q_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)

        self.num_heads = config.num_heads
        self.head_dim = head_dim

        # MLP after cross-attention
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

        # Layer scale
        self.use_layer_scale = config.layer_scale_init > 0
        if self.use_layer_scale:
            self.gamma_1 = nn.Parameter(
                config.layer_scale_init * torch.ones(config.hidden_size)
            )
            self.gamma_2 = nn.Parameter(
                config.layer_scale_init * torch.ones(config.hidden_size)
            )

    def _cross_attention(
        self,
        queries: torch.Tensor,
        demo_encodings: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-attention from queries to demo encodings."""
        batch_size = queries.shape[0]

        # Project Q, K, V
        q = self.q_proj(queries)  # (B, T, D)
        k = self.k_proj(demo_encodings)  # (B, K, D)
        v = self.v_proj(demo_encodings)  # (B, K, D)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.config.hidden_size)

        return self.o_proj(attn_output)

    def forward(
        self,
        queries: torch.Tensor,
        demo_encodings: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            queries: (batch, output_tokens, hidden_size)
            demo_encodings: (batch, num_demos, hidden_size)
            attn_bias: (batch, num_heads, output_tokens, num_demos) - attention mask

        Returns:
            Updated queries: (batch, output_tokens, hidden_size)
        """
        if self.norm_style == "pre":
            # Pre-norm
            attn_out = self._cross_attention(
                rms_norm(queries, variance_epsilon=self.norm_eps),
                demo_encodings,
                attn_bias,
            )
            if self.use_layer_scale:
                attn_out = self.gamma_1 * attn_out
            queries = queries + attn_out

            mlp_out = self.mlp(rms_norm(queries, variance_epsilon=self.norm_eps))
            if self.use_layer_scale:
                mlp_out = self.gamma_2 * mlp_out
            queries = queries + mlp_out
        else:
            # Post-norm
            attn_out = self._cross_attention(queries, demo_encodings, attn_bias)
            if self.use_layer_scale:
                attn_out = self.gamma_1 * attn_out
            queries = rms_norm(queries + attn_out, variance_epsilon=self.norm_eps)

            mlp_out = self.mlp(queries)
            if self.use_layer_scale:
                mlp_out = self.gamma_2 * mlp_out
            queries = rms_norm(queries + mlp_out, variance_epsilon=self.norm_eps)

        return queries


class DemoSetEncoder(nn.Module):
    """
    Aggregates multiple demo encodings via cross-attention.

    Takes: demo encodings (batch, num_demos, hidden_size)
    Returns: context (batch, output_tokens, hidden_size)

    Supports multiple cross-attention layers (set_encoder_layers config).
    """

    def __init__(self, config: DemoEncoderConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            trunc_normal_init_(
                torch.empty(config.output_tokens, config.hidden_size, dtype=self.forward_dtype),
                std=0.02,  # Changed from 1.0 to 0.02 (better init)
            )
        )

        # Stack of cross-attention layers
        num_layers = config.set_encoder_layers
        self.layers = nn.ModuleList([
            DemoSetEncoderLayer(config) for _ in range(num_layers)
        ])

        self.num_heads = config.num_heads
        self.norm_eps = config.rms_norm_eps

    def forward(
        self,
        demo_encodings: torch.Tensor,
        demo_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate demo encodings into context.

        Args:
            demo_encodings: (batch, num_demos, hidden_size) - Encoded demos
            demo_mask: (batch, num_demos) - True for valid demos

        Returns:
            context: (batch, output_tokens, hidden_size)
        """
        batch_size = demo_encodings.shape[0]

        # Expand query tokens for batch
        context = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # (B, T, D)

        # Create attention mask (shared across layers)
        attn_mask = demo_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, K)
        attn_mask = attn_mask.expand(-1, self.num_heads, self.config.output_tokens, -1)
        attn_bias = torch.zeros_like(attn_mask, dtype=context.dtype)
        attn_bias.masked_fill_(~attn_mask, float("-inf"))

        # Apply cross-attention layers
        for layer in self.layers:
            context = layer(context, demo_encodings, attn_bias)

        # Final normalization (CRITICAL FIX)
        # Without this, context can have unbounded scale, causing instability
        # especially in variational encoders that pool and project this output
        context = rms_norm(context, variance_epsilon=self.norm_eps)

        return context


class StandardDemoEncoder(BaseDemoEncoder):
    """
    Standard (deterministic) demonstration encoder.

    Encodes demo pairs via:
    1. Per-demo grid encoding (transformer over concatenated input-output)
    2. Set aggregation via cross-attention with learnable queries

    Output shape: (batch, output_tokens, hidden_size)
    This replaces the puzzle_id embedding in TRM.
    """

    def __init__(self, config: DemoEncoderConfig):
        super().__init__(config)

        self.grid_encoder = DemoGridEncoder(config)
        self.set_encoder = DemoSetEncoder(config)

    def forward(
        self,
        demo_inputs: torch.Tensor,
        demo_labels: torch.Tensor,
        demo_mask: torch.Tensor,
        return_full_output: bool = False,
    ) -> Union[torch.Tensor, EncoderOutput]:
        """
        Encode demonstrations into context embedding.

        Args:
            demo_inputs: (batch, max_demos, seq_len) - Demo input grids
            demo_labels: (batch, max_demos, seq_len) - Demo output grids
            demo_mask: (batch, max_demos) - True for valid demos
            return_full_output: If True, return EncoderOutput with auxiliary info

        Returns:
            If return_full_output=False: context tensor (batch, output_tokens, hidden_size)
            If return_full_output=True: EncoderOutput with context and auxiliary info
        """
        batch_size, max_demos, seq_len = demo_inputs.shape

        # Encode each demo pair
        # Reshape to process all demos in parallel
        demo_inputs_flat = demo_inputs.view(batch_size * max_demos, seq_len)
        demo_labels_flat = demo_labels.view(batch_size * max_demos, seq_len)

        demo_encodings_flat = self.grid_encoder(demo_inputs_flat, demo_labels_flat)
        demo_encodings = demo_encodings_flat.view(batch_size, max_demos, -1)  # (B, K, D)

        # Zero out invalid demos (masked)
        demo_encodings = demo_encodings * demo_mask.unsqueeze(-1).to(demo_encodings.dtype)

        # Aggregate via cross-attention
        context = self.set_encoder(demo_encodings, demo_mask)  # (B, T, D)

        if return_full_output:
            # Pooled representation for contrastive/diversity metrics
            z_pooled = context.mean(dim=1)  # (B, D)
            return EncoderOutput(
                context=context,
                z_pooled=z_pooled,
                kl_loss=None,
                mu=None,
                logvar=None,
            )

        return context


def test_standard_encoder():
    """Test the standard encoder."""
    print("=== Testing StandardDemoEncoder ===\n")

    config = DemoEncoderConfig(
        hidden_size=512,
        num_heads=8,
        num_layers=2,
        output_tokens=16,
        vocab_size=12,
        seq_len=900,
    )

    encoder = StandardDemoEncoder(config)
    encoder.eval()

    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {num_params:,}")

    # Create dummy input
    batch_size = 4
    max_demos = 3
    seq_len = 900

    demo_inputs = torch.randint(0, 12, (batch_size, max_demos, seq_len))
    demo_labels = torch.randint(0, 12, (batch_size, max_demos, seq_len))
    demo_mask = torch.tensor([
        [True, True, True],
        [True, True, False],
        [True, False, False],
        [True, True, True],
    ])

    # Forward pass (simple)
    with torch.no_grad():
        context = encoder(demo_inputs, demo_labels, demo_mask)

    print(f"Input shapes:")
    print(f"  demo_inputs: {demo_inputs.shape}")
    print(f"  demo_labels: {demo_labels.shape}")
    print(f"  demo_mask: {demo_mask.shape}")
    print(f"\nOutput shape: {context.shape}")
    print(f"Expected: ({batch_size}, {config.output_tokens}, {config.hidden_size})")

    assert context.shape == (batch_size, config.output_tokens, config.hidden_size)

    # Forward pass (full output)
    with torch.no_grad():
        output = encoder(demo_inputs, demo_labels, demo_mask, return_full_output=True)

    print(f"\nFull output:")
    print(f"  context: {output.context.shape}")
    print(f"  z_pooled: {output.z_pooled.shape}")
    print(f"  kl_loss: {output.kl_loss}")
    assert output.context.shape == (batch_size, config.output_tokens, config.hidden_size)
    assert output.z_pooled.shape == (batch_size, config.hidden_size)
    assert output.kl_loss is None  # Standard encoder has no KL loss

    print("\n=== Test passed! ===")


if __name__ == "__main__":
    test_standard_encoder()
