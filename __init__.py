"""
HKAN: Holomorphic Kolmogorov-Arnold Network
============================================
A novel architecture combining:
  - KAN-style learnable spline activations
  - Mamba-style selective state space sequence modeling
  - Holomorphic gating: complex-valued gating for richer representational capacity

Author: HKAN Contributors
License: MIT
"""

from .model import HKAN, HKANConfig
from .layers import HKANLayer, HolomorphicGate, SplineActivation, SelectiveSSM
from .utils import count_parameters, load_pretrained

__version__ = "0.1.0"
__all__ = [
    "HKAN",
    "HKANConfig",
    "HKANLayer",
    "HolomorphicGate",
    "SplineActivation",
    "SelectiveSSM",
    "count_parameters",
    "load_pretrained",
]
