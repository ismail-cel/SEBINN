"""
Tests for src/quadrature/nystrom.py.

Key verification (CLAUDE.md):
  - Stability of quadrature near x=y (diagonal correction)
  - BEM solve on Koch(1) with u_exact = x^2 - y^2 (harmonic)
  - Density reconstruction: u(x_interior) = sum G(x,y) sigma(y) w(y) ~ u_exact

The exact solution u = x^2 - y^2 satisfies -Delta u = 0 and is harmonic,
so it has a valid single-layer representation on any simply connected domain.
MATLAB uses this same test case (cfg.u_exact = @(x,y) x.^2 - y.^2).
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.boundary.polygon import koch_snowflake
from src.boundary.panels import build_uniform_panels, label_corner_ring_panels
from src.quadrature.gauss import gauss_legendre
from src.quadrature.panel_quad import build_panel_quadrature
from src.quadrature.nystrom import (
    assemble_nystrom_matrix,
    solve_bem,
    NystromMatrix,
    BEMSolution,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def make_bem_system(n_per_edge: int = 12, p: int = 16):
    """Build the standard Koch(1) BEM system used in all tests."""
    P = koch_snowflake(n=1)
    panels = build_uniform_panels(P, n_per_edge)
    label_corner_ring_panels(panels, P)
    qdata = build_panel_quadrature(panels, p)
    nmat = assemble_nystrom_matrix(qdata)
    # Right-hand side: u_exact = x^2 - y^2
    Yq = qdata.Yq
    f = Yq[0, :] ** 2 - Yq[1, :] ** 2
    return qdata, nmat, f, panels, P


# ===========================================================================
# Matrix structure
# ===========================================================================

class TestNystromMatrix:

    def setup_method(self):
        self.qdata, self.nmat, self.f, self.panels, self.P = make_bem_system()

    def test_shape(self):
        Nq = self.qdata.n_quad
        assert self.nmat.V.shape == (Nq, Nq)
        assert self.nmat.corr.shape == (Nq,)

    def test_diagonal_positive(self):
        """
        The analytic self-panel correction is positive (log kernel integrates
        to a positive value for small panels), so the diagonal of V should be
        dominated by positive corr values.
        """
        assert np.all(self.nmat.corr > 0)

    def test_v_equals_a_plus_diag_corr(self):
        """
        V = A + diag(corr).  Off-diagonal part of V should match the
        off-diagonal kernel matrix A; the diagonal of V should equal corr
        plus the (small) self-panel Gauss contribution (which is 0 by
        construction since we set A[i,i] = 0 before correction).
        """
        # The diagonal of V is exactly corr (A[i,i] was set to 0)
        assert np.allclose(np.diag(self.nmat.V), self.nmat.corr, atol=1e-15)

    def test_off_diagonal_kernel_sign(self):
        """
        G(x, y) = -(1/2pi) * log|x-y|.  For nearby points on a small domain
        (|x-y| < 1), log|x-y| < 0, so G > 0.  Off-diagonal entries of V
        (which equal G*w with w > 0) should mostly be positive for Koch(1)
        since all panel lengths << 1.
        """
        V = self.nmat.V
        off_diag = V.copy()
        np.fill_diagonal(off_diag, 0.0)
        frac_positive = (off_diag > 0).sum() / (off_diag != 0).sum()
        assert frac_positive > 0.95, (
            f"Expected >95% positive off-diag; got {frac_positive:.2%}"
        )

    def test_matrix_symmetric_structure(self):
        """
        V is NOT symmetric (because w_j appears only in column j), but the
        kernel G(x,y) = G(y,x) means  V[i,j]/w[j] = V[j,i]/w[i].
        Check this ratio symmetry on a small system.
        """
        P = koch_snowflake(n=1)
        panels = build_uniform_panels(P, n_per_edge=4)
        qdata = build_panel_quadrature(panels, p=4)
        nmat = assemble_nystrom_matrix(qdata)
        V = nmat.V
        wq = qdata.wq
        Nq = qdata.n_quad
        # A[i,j]/wq[j] should equal A[j,i]/wq[i] for i != j
        A = V - np.diag(nmat.corr)
        for i in range(min(Nq, 10)):
            for j in range(i + 1, min(Nq, 10)):
                r1 = A[i, j] / max(wq[j], 1e-20)
                r2 = A[j, i] / max(wq[i], 1e-20)
                assert abs(r1 - r2) < 1e-12, (
                    f"Kernel symmetry failed at ({i},{j}): {r1:.6e} vs {r2:.6e}"
                )


# ===========================================================================
# BEM solve
# ===========================================================================

class TestBEMSolve:

    def setup_method(self):
        self.qdata, self.nmat, self.f, self.panels, self.P = make_bem_system(
            n_per_edge=12, p=16
        )

    def test_returns_bem_solution(self):
        sol = solve_bem(self.nmat, self.f)
        assert isinstance(sol, BEMSolution)
        assert sol.sigma.shape == (self.qdata.n_quad,)

    def test_relative_residual_small(self):
        """
        BEM relative residual should be very small after solve.
        MATLAB: gmresTol = 1e-12; we allow a looser bound since scipy
        GMRES may use slightly different stopping criteria.
        """
        sol = solve_bem(self.nmat, self.f)
        assert sol.rel_res < 1e-8, (
            f"BEM relative residual too large: {sol.rel_res:.3e}"
        )

    def test_v_sigma_equals_f(self):
        """Direct check: V @ sigma = f to high precision."""
        sol = solve_bem(self.nmat, self.f)
        residual = np.linalg.norm(self.nmat.V @ sol.sigma - self.f)
        f_norm = np.linalg.norm(self.f)
        rel = residual / max(f_norm, 1e-14)
        assert rel < 1e-8, f"||V*sigma - f|| / ||f|| = {rel:.3e}"

    def test_interior_reconstruction_accuracy(self):
        """
        Reconstruct u at interior test points using the BEM density and
        verify against u_exact = x^2 - y^2.

        u(x) = sum_j G(x, y_j) * sigma_j * w_j

        Threshold: relative L2 error < 1e-4 (generous bound for Koch(1)
        with corners; MATLAB achieves ~1e-6 with pGL=16 and NpEdge=12,
        but we allow some margin here for test robustness).
        """
        sol = solve_bem(self.nmat, self.f)
        Yq = self.qdata.Yq
        wq = self.qdata.wq
        sigma = sol.sigma

        # Interior test points on a coarse grid, inside the Koch snowflake
        # Use the centroid and a few nearby points known to be inside
        test_pts = np.array([
            [0.0,  0.0],
            [0.1,  0.05],
            [-0.1, 0.05],
            [0.0,  0.15],
            [0.0, -0.15],
        ])

        u_bem = np.empty(len(test_pts))
        u_exact = test_pts[:, 0] ** 2 - test_pts[:, 1] ** 2

        for k, x in enumerate(test_pts):
            diff = Yq - x[:, None]
            r = np.linalg.norm(diff, axis=0)
            r = np.maximum(r, 1e-14)
            G = -(1.0 / (2.0 * np.pi)) * np.log(r)
            u_bem[k] = float(G @ (sigma * wq))

        err = u_bem - u_exact
        rel_l2 = np.linalg.norm(err) / max(np.linalg.norm(u_exact), 1e-14)
        assert rel_l2 < 1e-4, (
            f"BEM interior reconstruction relative L2 = {rel_l2:.3e}"
        )

    def test_direct_fallback_flag(self):
        """
        When use_direct_fallback=True and GMRES is set to 1 iteration
        (forced failure), direct solve should be invoked.
        """
        sol = solve_bem(self.nmat, self.f, max_iter=1, use_direct_fallback=True)
        # Whether GMRES converged in 1 step or not, the residual should be small
        # if direct fallback was triggered.
        res = np.linalg.norm(self.nmat.V @ sol.sigma - self.f)
        f_norm = np.linalg.norm(self.f)
        # Either GMRES converged quickly OR direct solve was accurate
        assert res / max(f_norm, 1e-14) < 1e-8


# ===========================================================================
# Diagonal correction sanity
# ===========================================================================

class TestDiagonalCorrection:

    def test_correction_positive_on_small_panels(self):
        """
        For panel length L << 1 and midpoint s0 = L/2:
            corr = -(1/2pi) * (log(L/2) + log(L/2) - L)
                 = -(1/2pi) * (2*log(L/2) - L)
        Since log(L/2) < 0 for L < 2, the correction is positive.
        All Koch(1) panels are much shorter than 2, so all corr > 0.
        """
        P = koch_snowflake(n=1)
        panels = build_uniform_panels(P, n_per_edge=8)
        qdata = build_panel_quadrature(panels, p=8)
        nmat = assemble_nystrom_matrix(qdata)
        assert np.all(nmat.corr > 0)

    def test_correction_dominates_diagonal(self):
        """
        The diagonal entry V[i,i] equals corr[i] (since A[i,i] = 0).
        Verify this for a small system.
        """
        P = koch_snowflake(n=1)
        panels = build_uniform_panels(P, n_per_edge=4)
        qdata = build_panel_quadrature(panels, p=4)
        nmat = assemble_nystrom_matrix(qdata)
        assert np.allclose(np.diag(nmat.V), nmat.corr, atol=1e-14)
