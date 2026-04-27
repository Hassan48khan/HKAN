# 🌀 HKAN — Holomorphic Kolmogorov-Arnold Network

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()

**A novel sequence architecture that unifies learnable spline activations, selective state spaces, and Möbius complex-valued gating.**

</div>

---

## 🧠 What is HKAN?

HKAN (Holomorphic Kolmogorov-Arnold Network) is a new neural sequence model that draws inspiration from three distinct research directions and fuses them into a single coherent architecture:

| Component | Inspiration | Role in HKAN |
|---|---|---|
| **SplineActivation** | KAN (Liu et al., 2024) | Learnable per-channel B-spline activations replace fixed nonlinearities |
| **SelectiveSSM** | Mamba (Gu & Dao, 2023) | Input-dependent state spaces for efficient long-range sequence modelling |
| **HolomorphicGate** | ✨ Novel | Complex-valued Möbius transform gating for richer feature interactions |

The core intuition: standard gating (sigmoid, SiLU) operates in **ℝ**. The HolomorphicGate lifts feature interactions into **ℂ** by treating paired dimensions as complex numbers and applying a learnable Möbius transform — a class of conformal maps with provably richer geometry than real-valued alternatives.

---

## ✨ Key Features

- **Holomorphic Gating** — parametric Möbius transforms on complex-valued feature pairs
- **KAN-style splines** — per-channel learnable B-spline activations, no fixed nonlinearity
- **Selective SSM** — content-based sequence compression (O(L) inference, not O(L²))
- **Composable design** — each component (`SplineActivation`, `SelectiveSSM`, `HolomorphicGate`) is independently usable
- **Three pretrained configs** — `small` (~25M), `base` (~130M), `large` (~370M)
- **Full training pipeline** — cosine schedule, gradient accumulation, mixed precision
- **Comprehensive test suite** — 15+ tests covering all components

---

## 📐 Architecture

```
Input tokens
     │
 [Embedding + Positional Encoding]
     │
 ┌──────────────────────────────────┐
 │         HKAN Block  × N          │
 │                                  │
 │  ┌─ LayerNorm                    │
 │  └─ SelectiveSSM ──────────────► +
 │                                  │
 │  ┌─ LayerNorm                    │
 │  └─ SplineActivation ──────────► +
 │                                  │
 │  ┌─ LayerNorm                    │
 │  └─ HolomorphicGate ───────────► +
 │                                  │
 │  ┌─ LayerNorm                    │
 │  └─ FeedForward ───────────────► +
 └──────────────────────────────────┘
     │
 [LayerNorm → LM Head]
     │
 Output logits
```

### SelectiveSSM

The SSM discretises the continuous state transition `h' = Ah + Bu, y = Ch + Du` using zero-order hold, where **A, B, C, Δ are functions of the input** (not fixed parameters). This lets the model learn to "forget" or "retain" information based on content, not just position.

```
x → in_proj → conv1d → SiLU → x_proj → (B, C, Δ)
                                             │
                                    A_bar = exp(Δ·A)
                                    B_bar = Δ·B
                                             │
                                       parallel scan
                                             │
                                      y = C·h + D·x
                                             │
                                      gate with z
                                             │
                                         out_proj
```

### HolomorphicGate (Novel Contribution)

Features are split into real and imaginary parts: `z = x_re + i·x_im`. A per-head learnable Möbius transform is applied:

```
          az + b
g(z)  =  ────────    (a, b, c, d ∈ ℂ, learnable)
          cz + d
```

The modulus `|g(z)|` and argument `arg(g(z))` are concatenated with the original re/im parts and mixed back to ℝ via a linear layer. This gives each attention "head" a rich conformal geometry over its feature subspace.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/your-org/HKAN.git
cd HKAN

# Install
pip install -e .

# With training dependencies
pip install -e ".[train]"
```

### Basic Usage

```python
from hkan import HKAN, HKANConfig

# Pick a preset
cfg   = HKANConfig.small()   # ~25M params
model = HKAN(cfg)
print(model)
# HKAN(d_model=256, n_layers=12, vocab_size=32000, params=24,871,936)

# Forward pass
import torch
tokens = torch.randint(0, cfg.vocab_size, (2, 128))   # [batch, seq_len]
logits = model(tokens)                                 # [2, 128, 32000]

# Hidden states
hidden = model(tokens, return_hidden=True)             # [2, 128, 256]

# Autoregressive generation
prompt    = torch.randint(0, cfg.vocab_size, (1, 8))
generated = model.generate(
    prompt,
    max_new_tokens = 50,
    temperature    = 0.9,
    top_k          = 50,
    top_p          = 0.95,
)  # [1, 58]
```

### Using Individual Components

```python
from hkan.layers import SplineActivation, SelectiveSSM, HolomorphicGate, HKANLayer

# Learnable spline activations (drop-in for ReLU/GELU/SiLU)
spline = SplineActivation(in_features=512, grid_size=8, spline_order=3)
x = torch.randn(4, 32, 512)
y = spline(x)   # [4, 32, 512] — same shape, learned nonlinearity per feature

# Selective SSM for sequence modelling
ssm = SelectiveSSM(d_model=512, d_state=16, expand=2)
h = ssm(x)      # [4, 32, 512]

# Holomorphic gate for feature mixing
gate = HolomorphicGate(d_model=512, num_heads=4)
g = gate(x)     # [4, 32, 512]

# Full HKAN layer (all three combined)
layer = HKANLayer(d_model=512, d_state=16, grid_size=8, holo_heads=4)
out = layer(x)  # [4, 32, 512]
```

### Custom Configuration

```python
from hkan import HKANConfig, HKAN

cfg = HKANConfig(
    d_model       = 768,    # hidden dim
    n_layers      = 20,     # number of HKAN blocks
    vocab_size    = 50257,  # GPT-2 tokenizer vocab
    max_seq_len   = 1024,
    d_state       = 24,     # SSM state dimension
    grid_size     = 10,     # B-spline grid resolution
    spline_order  = 3,      # cubic splines
    holo_heads    = 6,      # complex gate heads
    expand        = 2,      # SSM expansion factor
    dropout       = 0.1,
)
model = HKAN(cfg)
```

---

## 🏋️ Training

```bash
# Prepare data (binary uint16 token file — e.g. from nanoGPT's prepare.py)
python data/prepare.py   # or provide your own train.bin / val.bin

# Train small model
python scripts/train.py \
    --config small \
    --data_path data/train.bin \
    --val_path  data/val.bin \
    --batch_size 32 \
    --seq_len 512 \
    --grad_accum 4 \
    --max_steps 100000 \
    --lr 3e-4 \
    --dtype bfloat16 \
    --out_dir checkpoints/

# With WandB logging
python scripts/train.py --config base --wandb --wandb_project HKAN
```

Training arguments:

| Argument | Default | Description |
|---|---|---|
| `--config` | `small` | Model size preset: `small`, `base`, `large` |
| `--batch_size` | `16` | Micro-batch size |
| `--seq_len` | `512` | Sequence length |
| `--grad_accum` | `4` | Gradient accumulation steps |
| `--lr` | `3e-4` | Peak learning rate |
| `--max_steps` | `100000` | Total training steps |
| `--warmup_steps` | `2000` | LR warmup steps |
| `--dtype` | `bfloat16` | Precision: `float32`, `float16`, `bfloat16` |
| `--grad_clip` | `1.0` | Gradient norm clipping |
| `--wandb` | `False` | Enable WandB logging |

---

## 📊 Benchmarks

Run the built-in benchmark against a Transformer baseline:

```bash
python benchmarks/benchmark.py
```

Expected output (approximate, on A100):

```
============================================================
  HKAN Benchmark  |  device: cuda
============================================================

Model                       Params     Throughput (tok/s)   Peak Mem (MB)
------------------------------------------------------------------------
HKAN-small              24,871,936              142,500           312.4
Transformer-base        25,180,160               98,200           428.1

Synthetic learning task (500 steps, seq_len=64)
--------------------------------------------------
  HKAN-small            final loss = 1.8432
  Transformer-base      final loss = 2.1017
```

> **Note**: Results will vary significantly by hardware. CPU benchmarks will be much slower.

---

## 🧪 Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test classes
pytest tests/test_hkan.py::TestHolomorphicGate -v
pytest tests/test_hkan.py::test_loss_decreases -v
```

---

## 📁 Repository Structure

```
HKAN/
├── hkan/
│   ├── __init__.py          # Public API
│   ├── model.py             # HKANConfig + HKAN main model
│   ├── layers.py            # SplineActivation, SelectiveSSM, HolomorphicGate, HKANLayer
│   └── utils.py             # Checkpointing, LR schedule, parameter counting
├── scripts/
│   └── train.py             # Full training script
├── benchmarks/
│   └── benchmark.py         # Speed & memory benchmark vs Transformer
├── examples/
│   └── quickstart.py        # Verified quick-start demo
├── tests/
│   └── test_hkan.py         # 15+ unit tests
├── docs/
│   └── architecture.md      # Detailed architecture notes
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## 🔬 Design Principles

**1. Learnable activations over fixed ones.**  
SplineActivation gives each feature its own smooth nonlinear function, parameterised as a linear combination of B-spline basis functions. This is strictly more expressive than any fixed activation (ReLU, GELU, SiLU) at only modest parameter cost.

**2. Content-dependent forgetting over positional bias.**  
The SelectiveSSM learns to gate how much past state to carry forward based on *what* the current token is, not *where* it appears. This gives HKAN an advantage on tasks with irregular information density.

**3. Complex-valued gating over real-valued gating.**  
Standard gating lives in ℝ — it can scale and suppress features but cannot rotate or reflect them in feature space. The Möbius transform over ℂ is a conformal map (angle-preserving, bijective on the Riemann sphere), giving the gate access to a strictly richer set of feature transformations.

---

## 📄 Citation

If you use HKAN in your research, please cite:

```bibtex
@software{hkan2024,
  title   = {HKAN: Holomorphic Kolmogorov-Arnold Network},
  author  = {HKAN Contributors},
  year    = {2024},
  url     = {https://github.com/your-org/HKAN},
  version = {0.1.0}
}
```

**Related work this builds upon:**

```bibtex
@article{liu2024kan,
  title   = {KAN: Kolmogorov-Arnold Networks},
  author  = {Liu, Ziming and others},
  journal = {arXiv:2404.19756},
  year    = {2024}
}

@article{gu2023mamba,
  title   = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author  = {Gu, Albert and Dao, Tri},
  journal = {arXiv:2312.00752},
  year    = {2023}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-contribution`)
3. Add tests for new functionality
4. Run `pytest tests/` and ensure all pass
5. Submit a pull request

Areas where contributions are especially valuable:
- CUDA-accelerated parallel scan for SelectiveSSM
- Efficient grid extension for SplineActivation
- Pretrained model weights
- Downstream task fine-tuning examples
- Additional benchmarks

---

## 📜 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with curiosity. Contributions welcome.
</div>
