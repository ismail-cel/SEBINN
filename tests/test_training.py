"""
Tests for src/training/collocation.py, operator.py, and loss.py.

Verification targets (CLAUDE.md):
  - Correct assembly of enriched density sigma = sigma_w + gamma * sigma_s
  - Stability of quadrature near x=y (self-correction in the non-square matrix)
  - BIE residual is zero when sigma is the exact BEM density
  - Loss decreases when gamma is the correct value vs gamma=0
  - Operator state shapes are consistent
  - Collocation points lie on the boundary
"""

import numpy as np
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.boundary.polygon import make_koch_geometry, koch_snowflake
from src.boundary.panels import (
    build_uniform_panels, label_corner_ring_panels,
    panel_loss_weights,
)
from src.quadrature.panel_quad import build_panel_quadrature
from src.quadrature.nystrom import assemble_nystrom_matrix, solve_bem
from src.singular.enrichment import SingularEnrichment
from src.models.sebinn import SEBINNModel
from src.training.collocation import build_collocation_points, CollocData
from src.training.operator import build_bie_matrix, build_operator_state, OperatorState
from src.training.loss import residual_vector, sebinn_loss


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_geometry(n_per_edge=4, p=8):
    """Small Koch(1) system for fast tests."""
    geom = make_koch_geometry(n=1)
    panels = build_uniform_panels(geom.vertices, n_per_edge)
    label_corner_ring_panels(panels, geom.vertices)
    qdata = build_panel_quadrature(panels, p)
    weights = panel_loss_weights(panels)
    return geom, panels, qdata, weights


def _g_exact(pts):
    """u_exact = x^2 - y^2 (harmonic); MATLAB cfg.u_exact."""
    return pts[:, 0] ** 2 - pts[:, 1] ** 2


# ===========================================================================
# Collocation points
# ===========================================================================

class TestCollocData:

    def test_scalar_m_col(self):
        _, panels, _, _ = _make_geometry(n_per_edge=6)
        colloc = build_collocation_points(panels, m_col_panel=1)
        assert colloc.n_colloc == len(panels)

    def test_vector_m_col(self):
        _, panels, _, _ = _make_geometry(n_per_edge=4)
        m = np.ones(len(panels), dtype=int) * 2
        colloc = build_collocation_points(panels, m_col_panel=m)
        assert colloc.n_colloc == 2 * len(panels)

    def test_shapes(self):
        _, panels, _, _ = _make_geometry()
        colloc = build_collocation_points(panels, 1)
        Nb = len(panels)
        assert colloc.Xc.shape == (Nb, 2)
        assert colloc.pan_of_xc.shape == (Nb,)
        assert colloc.s0_of_xc.shape == (Nb,)

    def test_pan_id_range(self):
        _, panels, _, _ = _make_geometry()
        colloc = build_collocation_points(panels, 1)
        assert colloc.pan_of_xc.min() == 0
        assert colloc.pan_of_xc.max() == len(panels) - 1

    def test_points_on_panels(self):
        """Every collocation point must lie on its assigned panel segment."""
        _, panels, _, _ = _make_geometry(n_per_edge=6)
        colloc = build_collocation_points(panels, 2)
        for i in range(colloc.n_colloc):
            pid = int(colloc.pan_of_xc[i])
            pan = panels[pid]
            xc = colloc.Xc[i]
            d = pan.b - pan.a
            v = xc - pan.a
            cross = abs(v[0] * d[1] - v[1] * d[0])
            assert cross < 1e-12 * pan.length, (
                f"Point {i} not on panel {pid}: cross={cross:.2e}"
            )

    def test_s0_in_range(self):
        """s0_of_xc[i] must lie in [0, L_i]."""
        _, panels, _, _ = _make_geometry()
        colloc = build_collocation_points(panels, 2)
        for i in range(colloc.n_colloc):
            pid = int(colloc.pan_of_xc[i])
            L = panels[pid].length
            s0 = colloc.s0_of_xc[i]
            assert 0.0 <= s0 <= L + 1e-14


# ===========================================================================
# BIE kernel matrix (non-square: collocation x quadrature)
# ===========================================================================

class TestBIEMatrix:

    def setup_method(self):
        _, self.panels, self.qdata, _ = _make_geometry(n_per_edge=4, p=4)
        self.colloc = build_collocation_points(self.panels, m_col_panel=1)

    def test_shapes(self):
        A, corr = build_bie_matrix(self.colloc, self.qdata)
        Nb = self.colloc.n_colloc
        Nq = self.qdata.n_quad
        assert A.shape == (Nb, Nq)
        assert corr.shape == (Nb,)

    def test_corr_positive(self):
        """Self-panel correction is positive for small panels."""
        _, corr = build_bie_matrix(self.colloc, self.qdata)
        assert np.all(corr > 0)

    def test_off_diagonal_mostly_positive(self):
        """G(x,y)*w > 0 when |x-y| < 1 (panels are small)."""
        A, _ = build_bie_matrix(self.colloc, self.qdata)
        frac = (A > 0).sum() / A.size
        assert frac > 0.9

    def test_coincident_with_nystrom_on_square_system(self):
        """
        When collocation == quadrature nodes, the off-diagonal entries of
        build_bie_matrix(A_col) must match those of the Nystrom V matrix.
        The diagonal entries differ by design: assemble_nystrom_matrix sets
        A[i,i] = 0 explicitly, while build_bie_matrix evaluates G(x,y)*w
        with the r_min clamp (nonzero at coincident points).
        The corrections corr must also agree.
        """
        geom = make_koch_geometry(n=1)
        panels = build_uniform_panels(geom.vertices, 4)
        label_corner_ring_panels(panels, geom.vertices)
        qdata = build_panel_quadrature(panels, p=4)

        colloc = CollocData(
            Xc=qdata.Yq.T.copy(),
            pan_of_xc=qdata.pan_id.copy(),
            s0_of_xc=qdata.s_on_panel.copy(),
        )

        A_col, corr_col = build_bie_matrix(colloc, qdata)
        nmat = assemble_nystrom_matrix(qdata)

        # The full matrix V[i,j] = A_nystrom[i,j] + corr[i]*delta_{ij}.
        # For the non-square path: V_full[k,j] = A_col[k,j] + corr_col[k]*delta_{x_k=y_j}.
        # Since colloc == quad here, delta_{k,j} = delta_{kj}, so:
        #   V_full = A_col + diag(corr_col)
        # This must equal nmat.V.
        V_reconstructed = A_col + np.diag(corr_col)
        assert np.allclose(V_reconstructed, nmat.V, atol=1e-12), (
            f"Reconstructed V max diff: {np.max(np.abs(V_reconstructed - nmat.V)):.2e}"
        )


# ===========================================================================
# Operator state
# ===========================================================================

class TestOperatorState:

    def setup_method(self):
        self.geom, self.panels, self.qdata, self.weights = _make_geometry(
            n_per_edge=4, p=4
        )
        self.enrich = SingularEnrichment(self.geom)
        self.colloc = build_collocation_points(self.panels, 1)

    def _build(self, **kwargs):
        return build_operator_state(
            self.colloc, self.qdata, self.enrich,
            _g_exact, self.weights,
            **kwargs,
        )

    def test_returns_operator_state(self):
        op, diag = self._build()
        assert isinstance(op, OperatorState)
        assert isinstance(diag, dict)

    def test_tensor_shapes(self):
        op, _ = self._build()
        Nb = self.colloc.n_colloc
        Nq = self.qdata.n_quad
        assert op.A.shape == (Nb, Nq)
        assert op.corr.shape == (Nb,)
        assert op.f.shape == (Nb,)
        assert op.wCol.shape == (Nb,)
        assert op.Yq.shape == (Nq, 2)
        assert op.Xc.shape == (Nb, 2)
        assert op.sigma_s_q.shape == (Nq,)
        assert op.sigma_s_c.shape == (Nb,)

    def test_dtype_float64(self):
        op, _ = self._build()
        for name in ("A", "corr", "f", "wCol", "Yq", "Xc", "sigma_s_q", "sigma_s_c"):
            tensor = getattr(op, name)
            assert tensor.dtype == torch.float64, f"{name} has dtype {tensor.dtype}"

    def test_no_gradients_on_tensors(self):
        """Fixed tensors must not require grad."""
        op, _ = self._build()
        for name in ("A", "corr", "f", "wCol", "sigma_s_q", "sigma_s_c"):
            tensor = getattr(op, name)
            assert not tensor.requires_grad, f"{name} should not require grad"

    def test_eq_scale_none(self):
        op, diag = self._build(eq_scale_mode="none")
        assert op.eq_scale == 1.0

    def test_eq_scale_fixed(self):
        op, _ = self._build(eq_scale_mode="fixed", eq_scale_fixed=5.0)
        assert op.eq_scale == 5.0

    def test_eq_scale_auto(self):
        """Auto scale = 1 / mean|A|; verify it makes mean|A_scaled| ≈ 1."""
        op, _ = self._build(eq_scale_mode="auto")
        mean_abs = float(op.A.abs().mean())
        assert abs(mean_abs - 1.0) < 0.1, (
            f"After auto-scaling, mean|A| should be ~1; got {mean_abs:.4f}"
        )

    def test_eq_scale_affects_f(self):
        """Scaling must be applied consistently to A, corr, and f."""
        op1, _ = self._build(eq_scale_mode="none")
        op5, _ = self._build(eq_scale_mode="fixed", eq_scale_fixed=5.0)
        assert torch.allclose(op5.f, 5.0 * op1.f, atol=1e-14)
        assert torch.allclose(op5.A, 5.0 * op1.A, atol=1e-14)

    def test_sigma_s_negative(self):
        """Precomputed sigma_s must be negative (canonical formula)."""
        op, _ = self._build()
        assert (op.sigma_s_q < 0).all()
        assert (op.sigma_s_c < 0).all()

    def test_wCol_sum_matches_tensor(self):
        op, _ = self._build()
        assert abs(op.wCol_sum - float(op.wCol.sum())) < 1e-12

    def test_n_fields_consistent(self):
        op, _ = self._build()
        assert op.n_colloc == self.colloc.n_colloc
        assert op.n_quad   == self.qdata.n_quad
        assert op.n_panels == self.qdata.n_panels


# ===========================================================================
# Loss function
# ===========================================================================

class TestLoss:

    def setup_method(self):
        self.geom, self.panels, self.qdata, self.weights = _make_geometry(
            n_per_edge=6, p=8
        )
        self.enrich = SingularEnrichment(self.geom)
        self.colloc = build_collocation_points(self.panels, 1)
        self.op, _ = build_operator_state(
            self.colloc, self.qdata, self.enrich,
            _g_exact, self.weights,
        )

    def _make_model(self, gamma_init=0.0):
        return SEBINNModel(
            hidden_width=20, n_hidden=2,
            gamma_init=gamma_init,
            dtype=torch.float64,
        )

    def test_loss_is_scalar(self):
        model = self._make_model()
        loss, _ = sebinn_loss(model, self.op)
        assert loss.shape == torch.Size([])

    def test_loss_is_positive(self):
        model = self._make_model()
        loss, _ = sebinn_loss(model, self.op)
        assert float(loss.item()) > 0

    def test_loss_differentiable(self):
        """Loss must be differentiable w.r.t. all model parameters."""
        model = self._make_model(gamma_init=0.5)
        loss, _ = sebinn_loss(model, self.op)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_gamma_grad_nonzero(self):
        """
        The gamma gradient must be nonzero when sigma_s is nonzero
        (which it always is on the Koch domain).
        """
        model = self._make_model(gamma_init=0.0)
        loss, _ = sebinn_loss(model, self.op)
        loss.backward()
        assert model.gamma_module.gamma.grad is not None
        assert abs(float(model.gamma_module.gamma.grad)) > 1e-10

    def test_dbg_keys(self):
        model = self._make_model()
        _, dbg = sebinn_loss(model, self.op)
        for key in ("mean_abs_Vstd", "mean_abs_corr", "mean_abs_res",
                    "mse_scaled", "mse_unscaled", "loss", "gamma"):
            assert key in dbg, f"Missing debug key: {key}"

    def test_residual_shape(self):
        model = self._make_model()
        res, _ = residual_vector(model, self.op)
        assert res.shape == (self.op.n_colloc,)

    def test_residual_differentiable(self):
        model = self._make_model()
        res, _ = residual_vector(model, self.op)
        res.sum().backward()
        assert model.gamma_module.gamma.grad is not None

    def test_gamma_zero_matches_pure_network(self):
        """
        When gamma=0, sigma = sigma_w, so the residual is
        A @ sigma_w + corr * sigma_w_c - f.
        Verify this equals the residual computed by the combined model.
        """
        model = self._make_model(gamma_init=0.0)
        res_sebinn, _ = residual_vector(model, self.op)

        # Compute manually: sigma = net(Yq), no singular correction
        with torch.no_grad():
            sigma_q = model.sigma_w(self.op.Yq)
            sigma_c = model.sigma_w(self.op.Xc)
            Vsig = self.op.A @ sigma_q + self.op.corr * sigma_c
            res_manual = Vsig - self.op.f

        assert torch.allclose(res_sebinn.detach(), res_manual, atol=1e-14)

    def test_loss_decreases_with_optimizer_step(self):
        """A single Adam step must reduce the loss (sanity check)."""
        torch.manual_seed(0)
        model = self._make_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        loss0, _ = sebinn_loss(model, self.op)
        loss0.backward()
        opt.step()
        opt.zero_grad()
        loss1, _ = sebinn_loss(model, self.op)

        assert float(loss1) < float(loss0), (
            f"Loss did not decrease: {float(loss0):.6e} -> {float(loss1):.6e}"
        )

    def test_exact_bem_density_gives_small_residual(self):
        """
        If the network outputs the exact BEM density sigma_BEM and gamma=0,
        the BIE residual at collocation points should be small.

        This verifies the operator assembly is consistent with the BEM solve.
        Uses a larger system (more panels, higher GL order) for accuracy.
        """
        # Larger system for accuracy
        geom = make_koch_geometry(n=1)
        panels = build_uniform_panels(geom.vertices, 12)
        label_corner_ring_panels(panels, geom.vertices)
        qdata = build_panel_quadrature(panels, p=16)
        weights = panel_loss_weights(panels)
        enrich = SingularEnrichment(geom)
        colloc = build_collocation_points(panels, 1)

        # BEM reference solve
        nmat = assemble_nystrom_matrix(qdata)
        f_bem = _g_exact(qdata.Yq.T)
        bem_sol = solve_bem(nmat, f_bem)
        sigma_bem = bem_sol.sigma   # (Nq,)

        # Build operator
        op, _ = build_operator_state(
            colloc, qdata, enrich, _g_exact, weights,
        )

        # Build a minimal model; override network to output sigma_bem at Yq
        # by checking the manually computed residual
        # Manual residual: A @ sigma_bem + corr * sigma_bem_at_Xc - f
        sigma_bem_t = torch.tensor(sigma_bem, dtype=torch.float64)

        # Interpolate sigma_bem to Xc using nearest quadrature node
        # (rough proxy since we can't evaluate the true sigma_bem at Xc)
        # Instead compute A_col @ sigma_bem + corr_col * interp - f and
        # check it's small (BEM density satisfies the equation approximately)
        with torch.no_grad():
            Vstd = op.A @ sigma_bem_t        # (Nb,)
            # For the correction term, sigma at Xc ~ sigma at nearest Yq node
            xc_np = colloc.Xc               # (Nb, 2)
            yq_np = qdata.Yq.T             # (Nq, 2)
            dist = np.linalg.norm(
                xc_np[:, None, :] - yq_np[None, :, :], axis=-1
            )  # (Nb, Nq)
            nearest = dist.argmin(axis=1)  # (Nb,)
            sigma_c_approx = sigma_bem_t[nearest]   # (Nb,)
            Vsig = Vstd + op.corr * sigma_c_approx
            res = Vsig - op.f
            rel_res = float(res.abs().mean()) / max(float(op.f.abs().mean()), 1e-14)

        # With the BEM density, residual should be small (not exact because
        # collocation != quadrature nodes, but consistent with BEM accuracy)
        assert rel_res < 0.1, (
            f"Relative residual with BEM density too large: {rel_res:.3e}"
        )
