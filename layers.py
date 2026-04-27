"""
HKAN Core Layers
================
Building blocks for the Holomorphic Kolmogorov-Arnold Network.

Key Components:
  1. SplineActivation   - KAN-inspired learnable B-spline activations
  2. SelectiveSSM       - Mamba-inspired selective state space model
  3. HolomorphicGate    - Novel complex-valued gating mechanism
  4. HKANLayer          - Full HKAN residual block combining all three
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# 1. Spline Activation (KAN-inspired)
# ---------------------------------------------------------------------------

class SplineActivation(nn.Module):
    """
    Learnable B-spline activation function per channel.

    Each input feature gets its own set of B-spline control points,
    enabling the network to learn arbitrary 1-D nonlinearities — 
    the core idea from Kolmogorov-Arnold Networks (Liu et al., 2024).

    Unlike KAN's original per-edge parameterisation, we use a shared
    grid per layer and per-channel scale/shift for efficiency.

    Args:
        in_features  : number of input channels
        grid_size    : number of spline intervals (knots = grid_size + 1)
        spline_order : B-spline order (3 = cubic)
        grid_range   : (min, max) for the initial uniform knot grid
        residual_std : std for SiLU residual init
    """

    def __init__(
        self,
        in_features: int,
        grid_size: int = 8,
        spline_order: int = 3,
        grid_range: tuple = (-4.0, 4.0),
        residual_std: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        num_basis = grid_size + spline_order  # number of B-spline basis functions

        # Shared grid (non-parametric, updated via extend_grid)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        self.register_buffer("grid", grid)

        # Per-channel spline coefficients  [in_features, num_basis]
        self.coeff = nn.Parameter(
            torch.randn(in_features, num_basis) * residual_std
        )

        # Per-channel scale for SiLU residual (as in original KAN)
        self.residual_weight = nn.Parameter(torch.ones(in_features))

    # ------------------------------------------------------------------
    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions for input x.

        Args:
            x : [..., in_features]
        Returns:
            bases : [..., in_features, num_basis]
        """
        x = x.unsqueeze(-1)  # [..., in_features, 1]
        grid = self.grid  # [num_knots]

        # de Boor recursion — order 1 (indicator functions)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()

        for k in range(1, self.spline_order + 1):
            left_num  = x - grid[:-(k + 1)]
            left_den  = grid[k:-1] - grid[:-(k + 1)]
            right_num = grid[k + 1:] - x
            right_den = grid[k + 1:] - grid[1:-k]

            left_term  = torch.where(left_den  != 0, left_num  / left_den,  torch.zeros_like(left_num))
            right_term = torch.where(right_den != 0, right_num / right_den, torch.zeros_like(right_num))

            bases = left_term * bases[..., :-1] + right_term * bases[..., 1:]

        return bases  # [..., in_features, num_basis]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [..., in_features]
        Returns:
            out : [..., in_features]
        """
        bases  = self._b_splines(x)                      # [..., F, num_basis]
        spline = (bases * self.coeff).sum(dim=-1)         # [..., F]
        resid  = F.silu(x) * self.residual_weight        # [..., F]
        return spline + resid


# ---------------------------------------------------------------------------
# 2. Selective State Space Model (Mamba-inspired)
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model — input-dependent SSM à la Mamba.

    The key idea: A, B, C, Δ are *functions* of the input, allowing
    the model to selectively focus or ignore different parts of the
    sequence (content-based reasoning over long contexts).

    We implement a simplified but faithful variant using the
    parallel scan algorithm (associative scan) for efficiency.

    Args:
        d_model  : model dimension
        d_state  : SSM state dimension (N)
        d_conv   : local depthwise conv width
        expand   : inner expansion factor
        dt_min   : minimum Δ value
        dt_max   : maximum Δ value
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = d_model * expand
        d_inner       = self.d_inner

        # Input projection (x → z, x_proj)
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)

        # Short depthwise conv for local context
        self.conv1d   = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            bias=True, padding=d_conv - 1, groups=d_inner
        )

        # Input-dependent SSM projections
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # B, C, Δ_raw
        self.dt_proj  = nn.Linear(1, d_inner, bias=True)

        # Log-initialised dt bias
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        self.dt_proj.bias = nn.Parameter(torch.log(dt))

        # Fixed diagonal A (learnable in log-space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    # ------------------------------------------------------------------
    @staticmethod
    def _associative_scan(A_bar: torch.Tensor, Bu_bar: torch.Tensor) -> torch.Tensor:
        """
        Sequential scan (O(L) memory, O(L) time).
        For production use torch.jit or a CUDA parallel scan kernel.

        A_bar  : [B, L, D, N]
        Bu_bar : [B, L, D, N]
        Returns y : [B, L, D]   (summed over N)
        """
        B, L, D, N = A_bar.shape
        h = torch.zeros(B, D, N, device=A_bar.device, dtype=A_bar.dtype)
        ys = []
        for t in range(L):
            h = A_bar[:, t] * h + Bu_bar[:, t]
            ys.append(h.sum(dim=-1))  # [B, D]
        return torch.stack(ys, dim=1)  # [B, L, D]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, L, d_model]
        Returns:
            y : [B, L, d_model]
        """
        B, L, _ = x.shape
        d_inner, d_state = self.d_inner, self.d_state

        # Split into gated branches
        xz = self.in_proj(x)                                 # [B, L, 2*d_inner]
        x_, z = xz.chunk(2, dim=-1)                          # [B, L, d_inner] each

        # Local conv
        x_ = rearrange(x_, 'b l d -> b d l')
        x_ = self.conv1d(x_)[..., :L]
        x_ = rearrange(x_, 'b d l -> b l d')
        x_ = F.silu(x_)

        # Input-dependent B, C, Δ
        bcd   = self.x_proj(x_)                              # [B, L, 2N+1]
        B_ssm = bcd[..., :d_state]                           # [B, L, N]
        C_ssm = bcd[..., d_state:2*d_state]                  # [B, L, N]
        dt_raw = bcd[..., -1:]                               # [B, L, 1]
        dt    = F.softplus(self.dt_proj(dt_raw))             # [B, L, d_inner]

        # Discretise A, B  (zero-order hold)
        A     = -torch.exp(self.A_log)                       # [d_inner, N]
        A_bar = torch.exp(dt.unsqueeze(-1) * A)              # [B, L, d_inner, N]
        B_bar = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)        # [B, L, d_inner, N]

        # u contribution
        Bu_bar = B_bar * x_.unsqueeze(-1)                    # [B, L, d_inner, N]

        # Scan
        y_ssm = self._associative_scan(A_bar, Bu_bar)        # [B, L, d_inner]

        # C readout + D skip
        y_ssm = (y_ssm * C_ssm.unsqueeze(2).sum(-1, keepdim=False)).mean(-1, keepdim=True).expand_as(y_ssm)
        y_ssm = y_ssm + self.D * x_

        # Gate with z
        y_ssm = y_ssm * F.silu(z)

        return self.out_proj(y_ssm)                          # [B, L, d_model]


# ---------------------------------------------------------------------------
# 3. Holomorphic Gate (Novel contribution)
# ---------------------------------------------------------------------------

class HolomorphicGate(nn.Module):
    """
    Holomorphic Gating Mechanism — HKAN's novel contribution.

    Motivation:
    -----------
    Standard gating (SiLU, sigmoid) operates in R. We lift the gating
    to C: each activation is treated as a complex number z = x_re + i*x_im,
    and the gate is a learnable Möbius-inspired complex function:

        g(z) = (az + b) / (cz + d)    [Möbius / linear fractional transform]

    projected back to R via |g(z)| (modulus) and arg(g(z)) (phase),
    then recombined. This gives the network a richer geometry of
    interaction between dimensions.

    Practically: we split the feature dim in half as (re, im), apply
    the parameterised complex map, and return to R via a learned mixing.

    Args:
        d_model   : model dimension (must be even)
        num_heads : number of independent complex heads
    """

    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for HolomorphicGate"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // (2 * num_heads)

        # Möbius parameters (per head)  a, b, c, d  ∈ C  → 8 real params per head
        # Initialise near identity: a=1, b=0, c=0, d=1
        self.mobius_re = nn.Parameter(torch.tensor([[1., 0., 0., 1.]] * num_heads))
        self.mobius_im = nn.Parameter(torch.tensor([[0., 0., 0., 0.]] * num_heads))

        # Mixing back to real
        self.mix = nn.Linear(d_model * 2, d_model)  # (mod, phase, x_re, x_im) → d_model

        # Layernorm for stability inside gate
        self.norm = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------
    @staticmethod
    def _complex_div(
        num_re: torch.Tensor, num_im: torch.Tensor,
        den_re: torch.Tensor, den_im: torch.Tensor,
        eps: float = 1e-6,
    ):
        """Complex division: (a+ib)/(c+id)"""
        denom = den_re ** 2 + den_im ** 2 + eps
        re = (num_re * den_re + num_im * den_im) / denom
        im = (num_im * den_re - num_re * den_im) / denom
        return re, im

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, L, d_model]
        Returns:
            out : [B, L, d_model]
        """
        x = self.norm(x)
        B, L, D = x.shape
        h = self.num_heads
        hd = self.head_dim

        # Split real/imag parts
        x_re, x_im = x[..., :D//2], x[..., D//2:]          # [B, L, D/2]

        # Reshape into heads
        x_re = x_re.view(B, L, h, hd)                       # [B, L, h, hd]
        x_im = x_im.view(B, L, h, hd)

        # Möbius params  [h, 4]
        a_re = self.mobius_re[:, 0]; a_im = self.mobius_im[:, 0]
        b_re = self.mobius_re[:, 1]; b_im = self.mobius_im[:, 1]
        c_re = self.mobius_re[:, 2]; c_im = self.mobius_im[:, 2]
        d_re = self.mobius_re[:, 3]; d_im = self.mobius_im[:, 3]

        # Broadcast params over [B, L, h, hd]
        def _b(p): return p.view(1, 1, h, 1)

        # Numerator: az + b
        num_re = _b(a_re) * x_re - _b(a_im) * x_im + _b(b_re)
        num_im = _b(a_re) * x_im + _b(a_im) * x_re + _b(b_im)

        # Denominator: cz + d
        den_re = _b(c_re) * x_re - _b(c_im) * x_im + _b(d_re)
        den_im = _b(c_re) * x_im + _b(c_im) * x_re + _b(d_im)

        # Apply Möbius transform
        g_re, g_im = self._complex_div(num_re, num_im, den_re, den_im)

        # Modulus and phase
        modulus = torch.sqrt(g_re ** 2 + g_im ** 2 + 1e-6)  # [B, L, h, hd]
        phase   = torch.atan2(g_im, g_re)

        # Reshape and cat
        modulus = modulus.reshape(B, L, D // 2)
        phase   = phase.reshape(B, L, D // 2)
        out = self.mix(torch.cat([modulus, phase, x_re.reshape(B, L, -1), x_im.reshape(B, L, -1)], dim=-1))

        return out  # [B, L, d_model]


# ---------------------------------------------------------------------------
# 4. HKAN Residual Block
# ---------------------------------------------------------------------------

class HKANLayer(nn.Module):
    """
    A single HKAN residual layer combining all three mechanisms:

        x → SelectiveSSM → SplineActivation → HolomorphicGate → x + residual

    The ordering is deliberate:
      - SSM captures long-range sequence dependencies first
      - Spline activations learn the right nonlinearity for each feature
      - Holomorphic gating performs rich cross-feature interaction

    Args:
        d_model      : model dimension
        d_state      : SSM state dim
        grid_size    : spline grid intervals
        spline_order : B-spline polynomial order
        holo_heads   : number of complex heads in HolomorphicGate
        expand       : SSM inner expansion factor
        dropout      : dropout probability
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        grid_size: int = 8,
        spline_order: int = 3,
        holo_heads: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ssm   = SelectiveSSM(d_model, d_state=d_state, expand=expand)
        self.spline = SplineActivation(d_model, grid_size=grid_size, spline_order=spline_order)
        self.gate  = HolomorphicGate(d_model, num_heads=holo_heads)

        self.dropout = nn.Dropout(dropout)

        # Feed-forward after gating (optional depth)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm4 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, L, d_model]
        Returns:
            x : [B, L, d_model]
        """
        # 1. Selective SSM
        x = x + self.dropout(self.ssm(self.norm1(x)))

        # 2. Spline activations (pointwise)
        h = self.norm2(x)
        x = x + self.dropout(self.spline(h))

        # 3. Holomorphic gating
        x = x + self.dropout(self.gate(self.norm3(x)))

        # 4. Feed-forward
        x = x + self.ff(self.norm4(x))

        return x
