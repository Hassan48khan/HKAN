"""
HKAN Test Suite
===============
Run with: pytest tests/
"""

import pytest
import torch
from hkan import HKAN, HKANConfig
from hkan.layers import SplineActivation, SelectiveSSM, HolomorphicGate, HKANLayer
from hkan.utils import count_parameters


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cfg():
    return HKANConfig(d_model=64, n_layers=2, vocab_size=100, max_seq_len=32,
                      d_state=4, grid_size=4, holo_heads=2, expand=2)


# ---------------------------------------------------------------------------
# Layer tests
# ---------------------------------------------------------------------------

class TestSplineActivation:
    def test_forward_shape(self):
        layer = SplineActivation(in_features=32)
        x = torch.randn(4, 16, 32)
        out = layer(x)
        assert out.shape == (4, 16, 32)

    def test_gradients_flow(self):
        layer = SplineActivation(in_features=16)
        x = torch.randn(2, 8, 16, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None

    def test_different_grid_sizes(self):
        for gs in [4, 8, 16]:
            layer = SplineActivation(32, grid_size=gs)
            out = layer(torch.randn(2, 4, 32))
            assert out.shape == (2, 4, 32)


class TestSelectiveSSM:
    def test_forward_shape(self):
        layer = SelectiveSSM(d_model=64, d_state=8)
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert out.shape == (2, 16, 64)

    def test_gradients_flow(self):
        layer = SelectiveSSM(d_model=32, d_state=4)
        x = torch.randn(2, 8, 32, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None

    def test_variable_sequence_length(self):
        layer = SelectiveSSM(d_model=32, d_state=4)
        for L in [1, 8, 64, 128]:
            out = layer(torch.randn(1, L, 32))
            assert out.shape == (1, L, 32)


class TestHolomorphicGate:
    def test_forward_shape(self):
        gate = HolomorphicGate(d_model=64, num_heads=2)
        x = torch.randn(2, 16, 64)
        out = gate(x)
        assert out.shape == (2, 16, 64)

    def test_gradients_through_mobius(self):
        gate = HolomorphicGate(d_model=32, num_heads=2)
        x = torch.randn(2, 8, 32, requires_grad=True)
        out = gate(x)
        out.sum().backward()
        assert x.grad is not None
        # Möbius params should also get gradients
        assert gate.mobius_re.grad is not None

    def test_odd_d_model_raises(self):
        with pytest.raises(AssertionError):
            HolomorphicGate(d_model=33)


class TestHKANLayer:
    def test_forward_shape(self):
        layer = HKANLayer(d_model=64, d_state=4, grid_size=4, holo_heads=2)
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Output should differ from input (layer is doing something)."""
        layer = HKANLayer(d_model=64, d_state=4, grid_size=4, holo_heads=2)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestHKAN:
    def test_forward_returns_logits(self, small_cfg):
        model = HKAN(small_cfg)
        x = torch.randint(0, small_cfg.vocab_size, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, small_cfg.vocab_size)

    def test_return_hidden(self, small_cfg):
        model = HKAN(small_cfg)
        x = torch.randint(0, small_cfg.vocab_size, (2, 16))
        h = model(x, return_hidden=True)
        assert h.shape == (2, 16, small_cfg.d_model)

    def test_parameter_count_positive(self, small_cfg):
        model = HKAN(small_cfg)
        assert count_parameters(model) > 0

    def test_config_presets(self):
        for preset_fn in [HKANConfig.small, HKANConfig.base]:
            cfg = preset_fn()
            model = HKAN(cfg)
            assert count_parameters(model) > 1_000_000

    def test_generate(self, small_cfg):
        model = HKAN(small_cfg)
        prompt = torch.randint(0, small_cfg.vocab_size, (1, 4))
        out = model.generate(prompt, max_new_tokens=8, temperature=1.0)
        assert out.shape[1] == 4 + 8

    def test_training_step(self, small_cfg):
        import torch.nn.functional as F
        model = HKAN(small_cfg)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randint(0, small_cfg.vocab_size, (2, 8))
        y = torch.randint(0, small_cfg.vocab_size, (2, 8))
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, small_cfg.vocab_size), y.view(-1))
        loss.backward()
        opt.step()
        assert loss.item() > 0

    def test_no_input_raises(self, small_cfg):
        model = HKAN(small_cfg)
        with pytest.raises(ValueError):
            model()

    def test_tied_embeddings(self, small_cfg):
        model = HKAN(small_cfg)
        assert model.lm_head.weight is model.embedding.weight


# ---------------------------------------------------------------------------
# Regression: ensure loss decreases over a few steps
# ---------------------------------------------------------------------------

def test_loss_decreases():
    import torch.nn.functional as F
    cfg = HKANConfig(d_model=32, n_layers=2, vocab_size=50, max_seq_len=16,
                     d_state=4, grid_size=4, holo_heads=2, expand=2, dropout=0.0)
    model = HKAN(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.manual_seed(0)
    x = torch.randint(0, 50, (8, 8))
    y = torch.randint(0, 50, (8, 8))

    first_loss = None
    for _ in range(30):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 50), y.view(-1))
        if first_loss is None:
            first_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert loss.item() < first_loss, "Loss did not decrease over 30 training steps"
