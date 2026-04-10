"""
Tests for src/models/.

Verification targets (CLAUDE.md):
  - Correct density split: sigma = sigma_w + gamma * sigma_s
  - gamma appears in model.parameters() and is updated by optimizer
  - sigma_w and gamma * sigma_s can be inspected separately
  - Parameter count matches expected architecture
  - to_vector / from_vector round-trip is exact
  - gamma=0 recovers plain BINN (sigma = sigma_w)
  - Architecture matches MATLAB: 4 hidden layers, width 80, tanh
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.sigma_w_net import build_sigma_w_network
from src.models.gamma import GammaParameter
from src.models.sebinn import SEBINNModel


# ===========================================================================
# sigma_w network
# ===========================================================================

class TestSigmaWNetwork:

    def test_default_architecture(self):
        """4 hidden layers of width 80, tanh, scalar output."""
        net = build_sigma_w_network()
        # Count Linear layers: should be 5 (4 hidden + 1 output)
        linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
        assert len(linears) == 5

    def test_layer_widths(self):
        net = build_sigma_w_network(hidden_width=80, n_hidden=4)
        linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
        # First 4: (2->80), (80->80), (80->80), (80->80); last: (80->1)
        assert linears[0].in_features == 2
        assert linears[0].out_features == 80
        for l in linears[1:4]:
            assert l.in_features == 80
            assert l.out_features == 80
        assert linears[4].in_features == 80
        assert linears[4].out_features == 1

    def test_tanh_activations(self):
        net = build_sigma_w_network()
        tanhs = [m for m in net.modules() if isinstance(m, nn.Tanh)]
        assert len(tanhs) == 4

    def test_output_shape(self):
        net = build_sigma_w_network().double()
        y = torch.randn(17, 2, dtype=torch.float64)
        out = net(y)
        assert out.shape == (17, 1)

    def test_parameter_count(self):
        """
        MATLAB: 4 layers of 80 neurons.
        Parameters:
          layer 1:  2*80 + 80 = 240
          layers 2-4: 80*80 + 80 = 6480, x3 = 19440
          output: 80*1 + 1 = 81
          Total = 240 + 19440 + 81 = 19761
        """
        net = build_sigma_w_network()
        n = sum(p.numel() for p in net.parameters())
        assert n == 19761

    def test_custom_architecture(self):
        net = build_sigma_w_network(hidden_width=32, n_hidden=2)
        linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
        assert len(linears) == 3
        assert linears[-1].out_features == 1

    def test_differentiable(self):
        """Gradients flow through the network."""
        net = build_sigma_w_network().double()
        y = torch.randn(5, 2, dtype=torch.float64, requires_grad=False)
        out = net(y).sum()
        out.backward()
        for p in net.parameters():
            assert p.grad is not None


# ===========================================================================
# GammaParameter
# ===========================================================================

class TestGammaParameter:

    def test_scalar_default(self):
        g = GammaParameter()
        assert g.gamma.shape == torch.Size([])
        assert g.n_gamma == 1

    def test_init_value(self):
        g = GammaParameter(n_gamma=1, init_value=2.5)
        assert abs(g.item() - 2.5) < 1e-15

    def test_per_corner(self):
        g = GammaParameter(n_gamma=6, init_value=1.0)
        assert g.gamma.shape == torch.Size([6])
        assert g.n_gamma == 6

    def test_default_zero(self):
        g = GammaParameter()
        assert abs(g.item()) < 1e-15

    def test_is_parameter(self):
        """gamma must be an nn.Parameter so optimizers update it."""
        g = GammaParameter()
        assert isinstance(g.gamma, nn.Parameter)

    def test_in_parameters(self):
        g = GammaParameter()
        params = list(g.parameters())
        assert len(params) == 1
        assert params[0] is g.gamma

    def test_forward_returns_gamma(self):
        g = GammaParameter(init_value=3.7)
        assert abs(g().item() - 3.7) < 1e-14

    def test_gradient_flows(self):
        g = GammaParameter(init_value=1.0)
        loss = g() ** 2
        loss.backward()
        assert g.gamma.grad is not None
        assert abs(g.gamma.grad.item() - 2.0) < 1e-12

    def test_optimizer_updates_gamma(self):
        """Adam step changes gamma from its initial value."""
        g = GammaParameter(init_value=0.0)
        opt = torch.optim.Adam(g.parameters(), lr=1e-2)
        # Gradient points away from 0
        loss = (g() - 1.0) ** 2
        loss.backward()
        opt.step()
        assert abs(g.item()) > 1e-6

    def test_double_dtype(self):
        g = GammaParameter()
        assert g.gamma.dtype == torch.float64


# ===========================================================================
# SEBINNModel
# ===========================================================================

class TestSEBINNModel:

    def _make_model(self, n_gamma=1, gamma_init=0.0):
        return SEBINNModel(
            hidden_width=80, n_hidden=4,
            n_gamma=n_gamma, gamma_init=gamma_init,
            dtype=torch.float64,
        )

    def _dummy_inputs(self, N=20, n_gamma=1):
        y = torch.randn(N, 2, dtype=torch.float64)
        if n_gamma == 1:
            sigma_s = torch.randn(N, dtype=torch.float64)
        else:
            sigma_s = torch.randn(N, n_gamma, dtype=torch.float64)
        return y, sigma_s

    # --- architecture ---

    def test_forward_shape(self):
        model = self._make_model()
        y, sigma_s = self._dummy_inputs(N=25)
        out = model(y, sigma_s)
        assert out.shape == (25,)

    def test_forward_shape_per_corner(self):
        model = self._make_model(n_gamma=6)
        y, sigma_s = self._dummy_inputs(N=15, n_gamma=6)
        out = model(y, sigma_s)
        assert out.shape == (15,)

    def test_parameter_count(self):
        """
        Total params = sigma_w_net params + gamma params.
        Default: 19761 (net) + 1 (gamma) = 19762.
        """
        model = self._make_model()
        assert model.n_params() == 19762

    def test_parameter_count_per_corner(self):
        model = self._make_model(n_gamma=6)
        assert model.n_params() == 19761 + 6

    def test_gamma_in_parameters(self):
        """gamma must appear in model.parameters()."""
        model = self._make_model()
        all_params = list(model.parameters())
        gamma_param = model.gamma_module.gamma
        assert any(p is gamma_param for p in all_params)

    # --- density split ---

    def test_gamma_zero_gives_sigma_w(self):
        """
        With gamma=0, sigma = sigma_w exactly.
        """
        model = self._make_model(gamma_init=0.0)
        y, sigma_s = self._dummy_inputs(N=12)
        full = model(y, sigma_s)
        sw = model.sigma_w(y)
        assert torch.allclose(full, sw, atol=1e-15)

    def test_split_adds_up(self):
        """
        sigma_w(y) + singular_part(sigma_s) must equal forward(y, sigma_s).
        """
        model = self._make_model(gamma_init=1.5)
        y, sigma_s = self._dummy_inputs(N=20)
        full = model(y, sigma_s)
        sw = model.sigma_w(y)
        sp = model.singular_part(sigma_s)
        assert torch.allclose(full, sw + sp, atol=1e-14)

    def test_singular_part_scales_with_gamma(self):
        """singular_part = gamma * sigma_s when n_gamma=1."""
        model = self._make_model(gamma_init=2.3)
        _, sigma_s = self._dummy_inputs(N=10)
        sp = model.singular_part(sigma_s)
        expected = 2.3 * sigma_s
        assert torch.allclose(sp, expected, atol=1e-13)

    def test_singular_part_per_corner(self):
        """singular_part = (sigma_s_mat * gamma_vec).sum(-1) for per-corner."""
        n_gamma = 3
        model = self._make_model(n_gamma=n_gamma, gamma_init=1.0)
        # Set gamma to known values
        with torch.no_grad():
            model.gamma_module.gamma.copy_(
                torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
            )
        N = 8
        sigma_s_mat = torch.ones(N, n_gamma, dtype=torch.float64)
        sp = model.singular_part(sigma_s_mat)
        # Each row: 1*1 + 2*1 + 3*1 = 6
        assert torch.allclose(sp, torch.full((N,), 6.0, dtype=torch.float64))

    # --- gradients ---

    def test_gradients_flow_to_gamma(self):
        """Loss backward must produce gradient on gamma."""
        model = self._make_model(gamma_init=0.5)
        y, sigma_s = self._dummy_inputs(N=10)
        loss = model(y, sigma_s).sum()
        loss.backward()
        assert model.gamma_module.gamma.grad is not None

    def test_gradients_flow_to_network(self):
        """Loss backward must produce gradients on all network parameters."""
        model = self._make_model()
        y, sigma_s = self._dummy_inputs(N=10)
        loss = model(y, sigma_s).sum()
        loss.backward()
        for name, p in model.sigma_w_net.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_optimizer_updates_gamma(self):
        """Adam step on the full model updates gamma away from init."""
        model = self._make_model(gamma_init=0.0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        y, sigma_s = self._dummy_inputs(N=8)
        # Target: make sigma large (gamma should move away from 0)
        target = torch.ones(8, dtype=torch.float64)
        loss = ((model(y, sigma_s) - target) ** 2).mean()
        loss.backward()
        opt.step()
        assert abs(model.gamma_value()) > 1e-7

    # --- parameter vector utilities ---

    def test_to_vector_length(self):
        model = self._make_model()
        v = model.to_vector()
        assert v.shape == (model.n_params(),)

    def test_to_vector_is_detached(self):
        model = self._make_model()
        v = model.to_vector()
        assert not v.requires_grad

    def test_from_vector_round_trip(self):
        """from_vector(to_vector()) leaves all parameters unchanged."""
        model = self._make_model()
        v_before = model.to_vector().clone()
        model.from_vector(v_before)
        v_after = model.to_vector()
        assert torch.allclose(v_before, v_after, atol=1e-15)

    def test_from_vector_sets_values(self):
        """Loading a zero vector sets all parameters to zero."""
        model = self._make_model()
        zeros = torch.zeros(model.n_params(), dtype=torch.float64)
        model.from_vector(zeros)
        v = model.to_vector()
        assert torch.allclose(v, zeros, atol=1e-15)

    def test_from_vector_changes_output(self):
        """After loading a new vector, model output must change."""
        model = self._make_model()
        y, sigma_s = self._dummy_inputs(N=5)
        out_before = model(y, sigma_s).detach().clone()

        # Load a different parameter vector
        v = model.to_vector()
        model.from_vector(v * 2.0 + 1.0)
        out_after = model(y, sigma_s).detach()

        assert not torch.allclose(out_before, out_after)

    def test_gamma_value_scalar(self):
        model = self._make_model(gamma_init=3.14)
        assert abs(model.gamma_value() - 3.14) < 1e-14

    def test_gamma_value_per_corner(self):
        model = self._make_model(n_gamma=6, gamma_init=1.0)
        vals = model.gamma_value()
        assert isinstance(vals, list)
        assert len(vals) == 6
        assert all(abs(v - 1.0) < 1e-14 for v in vals)

    # --- dtype consistency ---

    def test_output_dtype_float64(self):
        model = self._make_model()
        y, sigma_s = self._dummy_inputs(N=10)
        out = model(y, sigma_s)
        assert out.dtype == torch.float64

    # --- integration: SE-BINN vs BINN ---

    def test_sebinn_differs_from_binn_when_gamma_nonzero(self):
        """
        When gamma != 0 and sigma_s != 0, the SE-BINN output must differ
        from plain BINN (gamma=0).  This verifies the enrichment is active.
        """
        torch.manual_seed(42)
        binn = self._make_model(gamma_init=0.0)
        sebinn = self._make_model(gamma_init=1.0)

        # Give both models the same network weights
        sebinn.sigma_w_net.load_state_dict(binn.sigma_w_net.state_dict())

        y, sigma_s = self._dummy_inputs(N=15)
        # Make sigma_s nonzero
        sigma_s = torch.randn(15, dtype=torch.float64)

        out_binn   = binn(y, sigma_s)
        out_sebinn = sebinn(y, sigma_s)

        diff = (out_sebinn - out_binn).abs().max().item()
        assert diff > 1e-6, f"SE-BINN and BINN outputs should differ; diff={diff:.2e}"
