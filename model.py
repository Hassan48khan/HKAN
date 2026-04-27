"""
HKAN Model
==========
Top-level model and configuration for the Holomorphic KAN architecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .layers import HKANLayer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HKANConfig:
    """
    Configuration for HKAN.

    Attributes
    ----------
    d_model       : Model / embedding dimension.
    n_layers      : Number of HKAN residual blocks.
    vocab_size    : Vocabulary size (for language modelling). Set 0 for non-LM use.
    max_seq_len   : Maximum sequence length.
    d_state       : SSM latent state dimension (N).
    grid_size     : B-spline grid intervals per spline activation.
    spline_order  : B-spline polynomial order (3 = cubic).
    holo_heads    : Number of complex heads in HolomorphicGate.
    expand        : SSM inner expansion factor.
    dropout       : Dropout probability.
    tie_embeddings: Tie input/output embeddings (language modelling).
    pad_token_id  : Padding token id.
    """

    d_model: int        = 512
    n_layers: int       = 12
    vocab_size: int     = 32000
    max_seq_len: int    = 2048
    d_state: int        = 16
    grid_size: int      = 8
    spline_order: int   = 3
    holo_heads: int     = 4
    expand: int         = 2
    dropout: float      = 0.1
    tie_embeddings: bool = True
    pad_token_id: int   = 0

    # Derived  (populated post-init)
    num_parameters: Optional[int] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        assert self.d_model % 2 == 0, "d_model must be even (HolomorphicGate requirement)"
        assert self.d_model % (2 * self.holo_heads) == 0, (
            "d_model must be divisible by 2*holo_heads"
        )

    @classmethod
    def small(cls) -> "HKANConfig":
        """~25 M parameter config (benchmark vs Mamba-130M scale)."""
        return cls(d_model=256, n_layers=12, d_state=8, grid_size=6, holo_heads=4, expand=2)

    @classmethod
    def base(cls) -> "HKANConfig":
        """~130 M parameter config."""
        return cls(d_model=512, n_layers=16, d_state=16, grid_size=8, holo_heads=4, expand=2)

    @classmethod
    def large(cls) -> "HKANConfig":
        """~370 M parameter config."""
        return cls(d_model=1024, n_layers=24, d_state=32, grid_size=12, holo_heads=8, expand=2)


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class LearnedPositionalEncoding(nn.Embedding):
    """Simple learned positional embeddings."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L, D]
        B, L, _ = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        return x + super().forward(pos)


# ---------------------------------------------------------------------------
# HKAN Main Model
# ---------------------------------------------------------------------------

class HKAN(nn.Module):
    """
    Holomorphic Kolmogorov-Arnold Network (HKAN).

    A sequence model that unifies:
      1. **Selective State Spaces** (à la Mamba) — efficient long-range dependencies.
      2. **Learnable Spline Activations** (à la KAN) — expressive pointwise functions.
      3. **Holomorphic Gating** (novel) — complex-valued feature interaction via Möbius transforms.

    Usage
    -----
    >>> cfg = HKANConfig.small()
    >>> model = HKAN(cfg)
    >>> tokens = torch.randint(0, cfg.vocab_size, (2, 128))
    >>> logits = model(tokens)          # [2, 128, vocab_size]
    >>> embeddings = model.encode(tokens)  # [2, 128, d_model]

    Parameters
    ----------
    config : HKANConfig
    """

    def __init__(self, config: HKANConfig):
        super().__init__()
        self.config = config
        cfg = config

        # Token embeddings
        if cfg.vocab_size > 0:
            self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
            self.pos_enc   = LearnedPositionalEncoding(cfg.max_seq_len, cfg.d_model)
        else:
            self.embedding = None
            self.pos_enc   = None

        # Main stack
        self.layers = nn.ModuleList([
            HKANLayer(
                d_model      = cfg.d_model,
                d_state      = cfg.d_state,
                grid_size    = cfg.grid_size,
                spline_order = cfg.spline_order,
                holo_heads   = cfg.holo_heads,
                expand       = cfg.expand,
                dropout      = cfg.dropout,
            )
            for _ in range(cfg.n_layers)
        ])

        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.drop     = nn.Dropout(cfg.dropout)

        # LM head
        if cfg.vocab_size > 0:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            if cfg.tie_embeddings:
                self.lm_head.weight = self.embedding.weight
        else:
            self.lm_head = None

        self._init_weights()
        cfg.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode token IDs to hidden states.

        Args:
            input_ids : [B, L] long tensor
        Returns:
            hidden    : [B, L, d_model]
        """
        assert self.embedding is not None, "vocab_size=0 — call forward() with embeddings directly"
        x = self.embedding(input_ids)       # [B, L, d_model]
        x = self.pos_enc(x)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm_out(x)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids      : [B, L] token ids  (mutually exclusive with inputs_embeds)
            inputs_embeds  : [B, L, d_model]   (pre-computed embeddings)
            return_hidden  : if True return hidden states instead of logits
        Returns:
            logits  : [B, L, vocab_size]  or  hidden : [B, L, d_model]
        """
        if input_ids is not None:
            hidden = self.encode(input_ids)
        elif inputs_embeds is not None:
            x = self.drop(inputs_embeds)
            for layer in self.layers:
                x = layer(x)
            hidden = self.norm_out(x)
        else:
            raise ValueError("Provide either input_ids or inputs_embeds")

        if return_hidden or self.lm_head is None:
            return hidden

        return self.lm_head(hidden)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with top-k / top-p sampling.

        Args:
            input_ids      : [B, L] prompt token ids
            max_new_tokens : tokens to generate
            temperature    : sampling temperature
            top_k          : top-k filtering (0 = disabled)
            top_p          : nucleus probability threshold
            eos_token_id   : stop on this token id
        Returns:
            generated : [B, L + max_new_tokens] token ids
        """
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            ctx = generated[:, -self.config.max_seq_len:]
            logits = self(ctx)[:, -1, :] / (temperature + 1e-8)

            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_logits[cumprobs > top_p] = float('-inf')
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_tok], dim=1)

            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break

        return generated

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"HKAN("
            f"d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
            f"vocab_size={cfg.vocab_size}, "
            f"params={cfg.num_parameters:,})"
        )
