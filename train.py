"""
HKAN Training Script
====================
A simple but complete training loop for HKAN on causal language modelling.

Usage:
    python scripts/train.py \
        --data_path data/train.bin \
        --config small \
        --batch_size 32 \
        --max_steps 100000 \
        --out_dir checkpoints/

Supports:
  - Mixed precision (bfloat16 / float16)
  - Gradient accumulation
  - Cosine LR schedule with warmup
  - Periodic evaluation and checkpoint saving
  - Basic WandB logging (optional)
"""

import argparse
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hkan import HKAN, HKANConfig
from hkan.utils import count_parameters, save_checkpoint, lr_schedule_cosine


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train HKAN")
    p.add_argument("--data_path",    type=str,   default="data/train.bin")
    p.add_argument("--val_path",     type=str,   default="data/val.bin")
    p.add_argument("--config",       type=str,   default="small",
                   choices=["small", "base", "large"], help="Model size preset")
    p.add_argument("--d_model",      type=int,   default=None)
    p.add_argument("--n_layers",     type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--seq_len",      type=int,   default=512)
    p.add_argument("--grad_accum",   type=int,   default=4)
    p.add_argument("--max_steps",    type=int,   default=100_000)
    p.add_argument("--warmup_steps", type=int,   default=2_000)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--min_lr",       type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--eval_every",   type=int,   default=1_000)
    p.add_argument("--save_every",   type=int,   default=5_000)
    p.add_argument("--out_dir",      type=str,   default="checkpoints")
    p.add_argument("--dtype",        type=str,   default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--wandb",        action="store_true")
    p.add_argument("--wandb_project",type=str,   default="HKAN")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loader helpers
# ---------------------------------------------------------------------------

def get_batch(data: np.ndarray, batch_size: int, seq_len: int, device: str):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i: i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1: i + seq_len + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, val_data, batch_size, seq_len, device, ctx, eval_iters=50):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(val_data, batch_size, seq_len, device)
        with ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[HKAN] Training on: {device}")

    # Mixed precision context
    ptdtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device, dtype=ptdtype) if device == "cuda" else nullcontext()
    scaler = torch.cuda.amp.GradScaler() if args.dtype == "float16" and device == "cuda" else None

    # Build config
    cfg_preset = {
        "small": HKANConfig.small,
        "base":  HKANConfig.base,
        "large": HKANConfig.large,
    }[args.config]()
    if args.d_model:  cfg_preset.d_model  = args.d_model
    if args.n_layers: cfg_preset.n_layers = args.n_layers
    cfg_preset.max_seq_len = args.seq_len

    model = HKAN(cfg_preset).to(device)
    print(f"[HKAN] Model: {model}")
    print(f"[HKAN] Parameters: {count_parameters(model):,}")

    # Optimizer
    decay_params     = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))

    # WandB
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # Data
    if not os.path.exists(args.data_path):
        print("[HKAN] data_path not found — generating tiny dummy dataset for demonstration")
        rng = np.random.default_rng(0)
        dummy = rng.integers(0, cfg_preset.vocab_size, size=2_000_000, dtype=np.uint16)
        os.makedirs("data", exist_ok=True)
        dummy.tofile("data/train.bin")
        dummy[:200_000].tofile("data/val.bin")
        args.data_path = "data/train.bin"
        args.val_path  = "data/val.bin"

    train_data = np.fromfile(args.data_path, dtype=np.uint16)
    val_data   = np.fromfile(args.val_path,  dtype=np.uint16) if os.path.exists(args.val_path) else train_data[:50_000]

    # Training loop
    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    optimizer.zero_grad()
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        # LR update
        lr = lr_schedule_cosine(step, args.max_steps, args.warmup_steps, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Accumulate gradients
        loss_accum = 0.0
        for micro in range(args.grad_accum):
            x, y = get_batch(train_data, args.batch_size, args.seq_len, device)
            with ctx:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / args.grad_accum
            loss_accum += loss.item()
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if args.grad_clip > 0:
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
            if args.wandb:
                wandb.log({"train/loss": loss_accum, "train/lr": lr, "step": step})

        # Eval
        if step % args.eval_every == 0:
            val_loss = estimate_loss(model, val_data, args.batch_size, args.seq_len, device, ctx)
            print(f"[EVAL] step {step} | val_loss {val_loss:.4f} | val_ppl {math.exp(val_loss):.2f}")
            if args.wandb:
                wandb.log({"val/loss": val_loss, "val/ppl": math.exp(val_loss), "step": step})

        # Save checkpoint
        if step % args.save_every == 0:
            save_checkpoint(model, optimizer, step, f"{args.out_dir}/hkan_step{step}.pt")

    save_checkpoint(model, optimizer, args.max_steps, f"{args.out_dir}/hkan_final.pt")
    print("[HKAN] Training complete.")


if __name__ == "__main__":
    main()
