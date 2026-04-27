"""
HKAN Utilities
==============
"""

from __future__ import annotations
import os
import json
import torch
from typing import Optional


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count (trainable) parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_parameters_by_component(model: torch.nn.Module) -> dict:
    """Return parameter counts broken down by top-level submodule."""
    counts = {}
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts[name] = n
    return counts


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    path: str,
):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "config": model.config.__dict__,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)
    print(f"[HKAN] Checkpoint saved → {path}")


def load_pretrained(path: str, device: str = "cpu") -> "HKAN":
    """Load HKAN from a checkpoint."""
    from .model import HKAN, HKANConfig
    ckpt = torch.load(path, map_location=device)
    cfg_dict = ckpt["config"]
    # Remove derived fields
    cfg_dict.pop("num_parameters", None)
    config = HKANConfig(**cfg_dict)
    model = HKAN(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"[HKAN] Loaded checkpoint (step {ckpt['step']}) from {path}")
    return model


def lr_schedule_cosine(
    step: int,
    total_steps: int,
    warmup_steps: int,
    lr_max: float,
    lr_min: float = 0.0,
) -> float:
    """Cosine LR schedule with linear warmup."""
    import math
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
