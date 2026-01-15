"""
LPN encoder matching the paper "Searching Latent Program Spaces" exactly.

Paper architecture:
- 2 transformer layers (shallow)
- 128 hidden size (8 heads × 16 dim)
- LayerNorm (no scale)
- SiLU activation (standard MLP, not SwiGLU)
- Absolute 2D position embeddings (row + col, learned)
- CLS token pooling
- Per-demo variational encoding (for VAE version)

Output is projected to match TRM's expected format (output_tokens × 512).
"""

import math
from typing import Union, Optional

import torch
from torch import nn
import torch.nn.functional as F

from models.encoders.base import BaseDemoEncoder, DemoEncoderConfig, EncoderOutput
from models.common import trunc_normal_init_
from models.layers import CastedLinear, CastedEmbedding


class LPNPaperMLP(nn.Module):
    """Standard MLP with SiLU activation (matches LPN paper)."""

    def __init__(self, hidden_size: int, expansion: float = 4.0):
        super().__init__()
        mlp_dim = int(hidden_size * expansion)
        self.fc1 = nn.Linear(hidden_size, mlp_dim, bias=False)
        self.fc2 = nn.Linear(mlp_dim, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class LPNPaperTransformerBlock(nn.Module):
    """
    Transformer block matching LPN paper exactly.

    - Pre-LayerNorm (no scale, no bias)
    - Standard MLP with SiLU
    - No RoPE (uses absolute embeddings added before transformer)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expansion: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # LayerNorm without scale (affine=False means no learnable params)
        # LPN uses use_scale=False, use_bias=False
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        # Self-attention (no bias, like LPN)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # MLP with SiLU
        self.mlp = LPNPaperMLP(hidden_size, expansion)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
            attn_mask: optional attention mask

        Returns:
            (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Pre-norm attention
        normed = self.norm1(x)

        # QKV projection
        qkv = self.qkv_proj(normed)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 x (B, H, S, D)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        attn_output = self.dropout(attn_output)

        x = x + attn_output

        # Pre-norm MLP
        mlp_output = self.mlp(self.norm2(x))
        mlp_output = self.dropout(mlp_output)
        x = x + mlp_output

        return x


class LPNPaperGridEncoder(nn.Module):
    """
    Encodes a single (input, output) grid pair using LPN paper architecture.

    Architecture:
    - Color embeddings for input and output grids
    - Absolute 2D position embeddings (row + col)
    - Channel embeddings (input vs output)
    - CLS token prepended
    - 2-layer transformer
    - CLS output as the encoding
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        vocab_size: int = 12,
        max_rows: int = 30,
        max_cols: int = 30,
        expansion: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_rows = max_rows
        self.max_cols = max_cols

        # Embeddings
        self.color_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_row_embed = nn.Embedding(max_rows, hidden_size)
        self.pos_col_embed = nn.Embedding(max_cols, hidden_size)
        self.channel_embed = nn.Embedding(2, hidden_size)  # 0=input, 1=output
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Transformer layers
        self.layers = nn.ModuleList([
            LPNPaperTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                expansion=expansion,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final LayerNorm before projection (like LPN)
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.dropout = nn.Dropout(dropout)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.color_embed.weight, std=0.02)
        nn.init.normal_(self.pos_row_embed.weight, std=0.02)
        nn.init.normal_(self.pos_col_embed.weight, std=0.02)
        nn.init.normal_(self.channel_embed.weight, std=0.02)

    def forward(
        self,
        input_grid: torch.Tensor,
        output_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a single (input, output) pair.

        Args:
            input_grid: (batch, seq_len) - flattened input grid (30x30=900)
            output_grid: (batch, seq_len) - flattened output grid

        Returns:
            (batch, hidden_size) - CLS token encoding
        """
        batch_size, seq_len = input_grid.shape
        device = input_grid.device

        # Create position indices for 2D grid
        rows = torch.arange(self.max_rows, device=device)
        cols = torch.arange(self.max_cols, device=device)

        # Position embeddings: row[i] + col[j] for position (i,j)
        pos_row = self.pos_row_embed(rows)  # (R, D)
        pos_col = self.pos_col_embed(cols)  # (C, D)
        pos_2d = pos_row.unsqueeze(1) + pos_col.unsqueeze(0)  # (R, C, D)
        pos_flat = pos_2d.view(-1, self.hidden_size)  # (R*C, D)

        # Channel embeddings
        input_channel = self.channel_embed(torch.tensor(0, device=device))  # (D,)
        output_channel = self.channel_embed(torch.tensor(1, device=device))  # (D,)

        # Color embeddings
        input_colors = self.color_embed(input_grid)  # (B, S, D)
        output_colors = self.color_embed(output_grid)  # (B, S, D)

        # Combine: color + position + channel
        input_emb = input_colors + pos_flat.unsqueeze(0) + input_channel
        output_emb = output_colors + pos_flat.unsqueeze(0) + output_channel

        # Concatenate input and output
        x = torch.cat([input_emb, output_emb], dim=1)  # (B, 2*S, D)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+2*S, D)

        # Apply dropout
        x = self.dropout(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Extract CLS token and normalize
        cls_output = x[:, 0]  # (B, D)
        cls_output = self.final_norm(cls_output)

        return cls_output


class LPNEncoder(BaseDemoEncoder):
    """
    LPN paper encoder (deterministic/standard version).

    - Uses LPNPaperGridEncoder for each demo
    - Mean aggregation across demos
    - Projects to TRM's expected output format

    Architecture (hardcoded to match paper):
    - 2 layers, 128 hidden (8 heads × 16 dim)
    - LayerNorm, SiLU MLP, absolute 2D embeddings
    """

    # Paper defaults (hardcoded)
    LPN_HIDDEN_SIZE = 128
    LPN_NUM_LAYERS = 2
    LPN_NUM_HEADS = 8

    def __init__(self, config: DemoEncoderConfig):
        super().__init__(config)

        self.internal_hidden = self.LPN_HIDDEN_SIZE

        # Grid encoder matching paper exactly
        self.grid_encoder = LPNPaperGridEncoder(
            hidden_size=self.LPN_HIDDEN_SIZE,
            num_heads=self.LPN_NUM_HEADS,
            num_layers=self.LPN_NUM_LAYERS,
            vocab_size=config.vocab_size,
            max_rows=30,
            max_cols=30,
            expansion=4.0,
            dropout=0.0,
        )

        # Project from internal hidden to TRM's expected output
        # Output: (batch, output_tokens, hidden_size) where hidden_size=512
        self.output_proj = nn.Linear(
            self.LPN_HIDDEN_SIZE,
            config.output_tokens * config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        demo_inputs: torch.Tensor,
        demo_labels: torch.Tensor,
        demo_mask: torch.Tensor,
        return_full_output: bool = False,
    ) -> Union[torch.Tensor, EncoderOutput]:
        """
        Encode demonstrations.

        Args:
            demo_inputs: (batch, max_demos, seq_len)
            demo_labels: (batch, max_demos, seq_len)
            demo_mask: (batch, max_demos) - True for valid demos

        Returns:
            context: (batch, output_tokens, hidden_size)
        """
        batch_size, max_demos, seq_len = demo_inputs.shape

        # Encode each demo
        demo_inputs_flat = demo_inputs.view(batch_size * max_demos, seq_len)
        demo_labels_flat = demo_labels.view(batch_size * max_demos, seq_len)

        encodings_flat = self.grid_encoder(demo_inputs_flat, demo_labels_flat)
        encodings = encodings_flat.view(batch_size, max_demos, -1)  # (B, K, D_internal)

        # Mask invalid demos
        mask_expanded = demo_mask.unsqueeze(-1).to(encodings.dtype)
        encodings = encodings * mask_expanded

        # Mean aggregation
        num_valid = demo_mask.sum(dim=1, keepdim=True).clamp(min=1)
        z_pooled = encodings.sum(dim=1) / num_valid  # (B, D_internal)

        # Project to output format
        context = self.output_proj(z_pooled)  # (B, T*D)
        context = context.view(batch_size, self.config.output_tokens, self.config.hidden_size)

        if return_full_output:
            return EncoderOutput(
                context=context,
                z_pooled=z_pooled,
                kl_loss=None,
                mu=None,
                logvar=None,
            )

        return context


class LPNVariationalEncoder(BaseDemoEncoder):
    """
    LPN paper encoder (variational version).

    Matches LPN paper exactly:
    - Per-demo variational encoding (μ, σ for each demo)
    - Sample each demo independently
    - Mean aggregation after sampling
    - Projects to TRM's expected output format

    Architecture (hardcoded to match paper):
    - 2 layers, 128 hidden (8 heads × 16 dim)
    - Latent dim: 32
    - LayerNorm, SiLU MLP, absolute 2D embeddings
    """

    # Paper defaults (hardcoded)
    LPN_HIDDEN_SIZE = 128
    LPN_NUM_LAYERS = 2
    LPN_NUM_HEADS = 8
    LPN_LATENT_DIM = 32

    def __init__(self, config: DemoEncoderConfig):
        super().__init__(config)

        self.internal_hidden = self.LPN_HIDDEN_SIZE
        self.latent_dim = self.LPN_LATENT_DIM

        # Grid encoder matching paper exactly
        self.grid_encoder = LPNPaperGridEncoder(
            hidden_size=self.LPN_HIDDEN_SIZE,
            num_heads=self.LPN_NUM_HEADS,
            num_layers=self.LPN_NUM_LAYERS,
            vocab_size=config.vocab_size,
            max_rows=30,
            max_cols=30,
            expansion=4.0,
            dropout=0.0,
        )

        # Variational projections (per-demo, like LPN paper)
        # LPN projects to latent_dim (32), not hidden_size
        self.mu_proj = nn.Linear(self.LPN_HIDDEN_SIZE, self.LPN_LATENT_DIM, bias=False)
        self.logvar_proj = nn.Linear(self.LPN_HIDDEN_SIZE, self.LPN_LATENT_DIM, bias=False)

        # Initialize logvar to output near-zero
        nn.init.zeros_(self.logvar_proj.weight)

        # Project from latent to TRM's expected output
        self.output_proj = nn.Linear(
            self.LPN_LATENT_DIM,
            config.output_tokens * config.hidden_size,
            bias=False,
        )

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ * ε"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _compute_kl_loss(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence from N(μ, σ²) to N(0, I), averaged over valid demos.

        Args:
            mu: (batch, max_demos, latent_dim)
            logvar: (batch, max_demos, latent_dim)
            mask: (batch, max_demos)
        """
        # KL per demo: -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_per_demo = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=-1
        )  # (B, K)

        # Mask and average
        kl_masked = kl_per_demo * mask.to(kl_per_demo.dtype)
        kl_loss = kl_masked.sum() / mask.sum().clamp(min=1)

        return kl_loss

    def forward(
        self,
        demo_inputs: torch.Tensor,
        demo_labels: torch.Tensor,
        demo_mask: torch.Tensor,
        return_full_output: bool = False,
    ) -> Union[torch.Tensor, EncoderOutput]:
        """
        Encode demonstrations with per-demo VAE (matching LPN paper).

        Args:
            demo_inputs: (batch, max_demos, seq_len)
            demo_labels: (batch, max_demos, seq_len)
            demo_mask: (batch, max_demos)

        Returns:
            context: (batch, output_tokens, hidden_size)
        """
        batch_size, max_demos, seq_len = demo_inputs.shape

        # Encode each demo
        demo_inputs_flat = demo_inputs.view(batch_size * max_demos, seq_len)
        demo_labels_flat = demo_labels.view(batch_size * max_demos, seq_len)

        encodings_flat = self.grid_encoder(demo_inputs_flat, demo_labels_flat)
        encodings = encodings_flat.view(batch_size, max_demos, -1)  # (B, K, D_internal)

        # Per-demo variational projection (LPN style)
        mu = self.mu_proj(encodings)  # (B, K, latent_dim)
        logvar = self.logvar_proj(encodings)  # (B, K, latent_dim)

        # Clamp for stability
        logvar = torch.clamp(logvar, min=-10, max=10)

        # Reparameterize each demo
        z_demos = self._reparameterize(mu, logvar)  # (B, K, latent_dim)

        # Mask invalid demos
        mask_expanded = demo_mask.unsqueeze(-1).to(z_demos.dtype)
        z_demos = z_demos * mask_expanded

        # Mean aggregation (after sampling, like LPN)
        num_valid = demo_mask.sum(dim=1, keepdim=True).clamp(min=1)
        z_pooled = z_demos.sum(dim=1) / num_valid  # (B, latent_dim)

        # Compute KL loss
        kl_loss = self._compute_kl_loss(mu, logvar, demo_mask)

        # Project to output format
        context = self.output_proj(z_pooled)  # (B, T*D)
        context = context.view(batch_size, self.config.output_tokens, self.config.hidden_size)

        if return_full_output:
            # Aggregate mu/logvar for logging
            mu_masked = mu * mask_expanded
            logvar_masked = logvar * mask_expanded
            mu_agg = mu_masked.sum(dim=1) / num_valid
            logvar_agg = logvar_masked.sum(dim=1) / num_valid

            return EncoderOutput(
                context=context,
                z_pooled=z_pooled,
                kl_loss=kl_loss,
                mu=mu_agg,
                logvar=logvar_agg,
            )

        return context


def test_lpn_paper_encoders():
    """Test both LPN paper encoders."""
    print("=== Testing LPN Paper Encoders ===\n")

    # Create config - LPN paper values are hardcoded in encoder classes
    # Only TRM-relevant settings needed here
    config = DemoEncoderConfig(
        hidden_size=512,  # TRM output size
        num_heads=8,
        num_layers=2,
        output_tokens=16,
        vocab_size=12,
        seq_len=900,
        norm_style="pre",
    )

    # Test data
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

    # Test standard encoder
    print("--- LPNEncoder (Standard) ---")
    encoder_std = LPNEncoder(config)
    num_params = sum(p.numel() for p in encoder_std.parameters())
    print(f"Parameters: {num_params:,}")

    output_std = encoder_std(demo_inputs, demo_labels, demo_mask, return_full_output=True)
    print(f"context shape: {output_std.context.shape}")
    print(f"z_pooled shape: {output_std.z_pooled.shape}")
    assert output_std.context.shape == (batch_size, config.output_tokens, config.hidden_size)
    print("Standard encoder: OK\n")

    # Test variational encoder
    print("--- LPNVariationalEncoder ---")
    encoder_var = LPNVariationalEncoder(config)
    encoder_var.train()
    num_params = sum(p.numel() for p in encoder_var.parameters())
    print(f"Parameters: {num_params:,}")

    output_var = encoder_var(demo_inputs, demo_labels, demo_mask, return_full_output=True)
    print(f"context shape: {output_var.context.shape}")
    print(f"z_pooled shape: {output_var.z_pooled.shape}")
    print(f"mu shape: {output_var.mu.shape}")
    print(f"logvar shape: {output_var.logvar.shape}")
    print(f"kl_loss: {output_var.kl_loss.item():.4f}")
    assert output_var.context.shape == (batch_size, config.output_tokens, config.hidden_size)
    assert output_var.kl_loss is not None
    print("Variational encoder: OK\n")

    # Test eval mode (no sampling)
    encoder_var.eval()
    with torch.no_grad():
        output_eval = encoder_var(demo_inputs, demo_labels, demo_mask, return_full_output=True)
    print(f"Eval mode kl_loss: {output_eval.kl_loss.item():.4f}")

    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    test_lpn_paper_encoders()
