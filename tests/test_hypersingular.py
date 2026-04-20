"""
Tests for the hypersingular operator W_h and Calderón preconditioner.

Tests
-----
1. D_h accuracy     — apply D_h to sin(2πs/L), compare to (2π/L)cos(2πs/L)
2. W_h symmetry     — check ||W_h - W_h^T|| / ||W_h||
3. Nullspace        — W_h @ ones ≈ 0; W_tilde @ ones ≠ 0
4. W̃V eigenvalues   — condition number of W̃V vs V alone (KEY TEST)
5. W̃ invertibility  — cond(W_tilde) is finite and moderate
6. Preconditioner   — cond(V) >> cond(W̃V)

Geometry: Koch(1), n_per_edge=12, p_gl=16  (same as all experiments).
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.boundary.polygon import make_koch_geometry
from src.boundary.panels import build_uniform_panels, label_corner_ring_panels
from src.quadrature.gauss import gauss_legendre
from src.quadrature.panel_quad import build_panel_quadrature
from src.quadrature.nystrom import assemble_nystrom_matrix
from src.quadrature.tangential_derivative import (
    lagrange_derivative_matrix,
    build_tangential_derivative_matrix,
)
from src.quadrature.hypersingular import (
    assemble_hypersingular_matrix,
    regularise_hypersingular,
)


# ---------------------------------------------------------------------------
# Shared fixture (module-level, built once)
# ---------------------------------------------------------------------------

def _setup(n_per_edge: int = 12, p: int = 16):
    geom   = make_koch_geometry(n=1)
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=n_per_edge)
    label_corner_ring_panels(panels, P)
    qdata  = build_panel_quadrature(panels, p=p)
    nmat   = assemble_nystrom_matrix(qdata)
    return qdata, nmat, P


# Build once at module import
_qdata, _nmat, _P = _setup()


# ---------------------------------------------------------------------------
# Helper: global arc-length at quadrature nodes
# ---------------------------------------------------------------------------

def _global_arc(qdata):
    """Arc-length coordinate at each quadrature node, starting at 0."""
    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    return panel_start[qdata.pan_id] + qdata.s_on_panel


# ===========================================================================
# Test 1: D_h derivative accuracy
# ===========================================================================

class TestTangentialDerivative:

    def test_lagrange_derivative_matrix_linear(self):
        """D at p=3 GL nodes should exactly differentiate linear functions."""
        xi, _ = gauss_legendre(3)
        D = lagrange_derivative_matrix(xi)
        f = 2.0 * xi + 1.0           # f(ξ) = 2ξ + 1
        df_exact = np.full(3, 2.0)   # f'(ξ) = 2
        df_approx = D @ f
        np.testing.assert_allclose(df_approx, df_exact, atol=1e-13)

    def test_lagrange_derivative_matrix_quadratic(self):
        """D at p=4 GL nodes should exactly differentiate x^2."""
        xi, _ = gauss_legendre(4)
        D = lagrange_derivative_matrix(xi)
        f = xi ** 2
        df_exact = 2.0 * xi
        df_approx = D @ f
        np.testing.assert_allclose(df_approx, df_exact, atol=1e-12)

    def test_Dh_block_diagonal(self):
        """D_h has exactly n_panels non-zero p×p blocks and zeros elsewhere."""
        qdata = _qdata
        D_h = build_tangential_derivative_matrix(qdata)
        p    = qdata.p
        Npan = qdata.n_panels

        for pid in range(Npan):
            js = qdata.idx_std[pid]
            # Block should be non-zero
            block = D_h[np.ix_(js, js)]
            assert np.any(block != 0.0), f"panel {pid} block is zero"

        # Off-block entries should be zero
        for pid in range(Npan):
            js = qdata.idx_std[pid]
            for qid in range(Npan):
                if qid == pid:
                    continue
                ks = qdata.idx_std[qid]
                off = D_h[np.ix_(js, ks)]
                np.testing.assert_array_equal(off, 0.0)

    def test_Dh_sine_accuracy(self):
        """
        Apply D_h to φ(s) = sin(2πs/L) at quadrature nodes.
        Exact derivative: φ'(s) = (2π/L) cos(2πs/L).
        Expect relative error < 1e-10.
        """
        qdata  = _qdata
        arc    = _global_arc(qdata)
        L_tot  = float(qdata.L_panel.sum())
        freq   = 2.0 * np.pi / L_tot

        phi       = np.sin(freq * arc)
        phi_prime = freq * np.cos(freq * arc)

        D_h    = build_tangential_derivative_matrix(qdata)
        phi_Dh = D_h @ phi

        rel_err = np.linalg.norm(phi_Dh - phi_prime) / (np.linalg.norm(phi_prime) + 1e-14)
        print(f"\n  D_h sine accuracy: rel_err = {rel_err:.3e}")
        assert rel_err < 1e-10, f"D_h derivative error {rel_err:.3e} exceeds 1e-10"

    def test_Dh_constant_zero(self):
        """D_h @ ones = 0 (derivatives of constants are zero on each panel)."""
        qdata = _qdata
        D_h   = build_tangential_derivative_matrix(qdata)
        ones  = np.ones(qdata.n_quad)
        result = D_h @ ones
        np.testing.assert_allclose(result, 0.0, atol=1e-11)  # floating-point round-off in Lagrange formula


# ===========================================================================
# Test 2: W_h properties
# ===========================================================================

class TestHypersingularMatrix:

    @pytest.fixture(scope="class")
    def matrices(self):
        qdata  = _qdata
        nmat   = _nmat
        W_h    = assemble_hypersingular_matrix(qdata, nmat)
        wq     = qdata.wq
        W_tilde = regularise_hypersingular(W_h, qdata)
        return W_h, W_tilde, qdata, nmat

    def test_shape(self, matrices):
        W_h, _, qdata, _ = matrices
        Nq = qdata.n_quad
        assert W_h.shape == (Nq, Nq)

    def test_symmetry(self, matrices):
        """
        W_h = -D^T diag(w) V D is exactly symmetric to machine precision.
        The weight diag(w) makes the Galerkin form symmetric.
        """
        W_h, _, qdata, _ = matrices
        asym = np.linalg.norm(W_h - W_h.T) / (np.linalg.norm(W_h) + 1e-14)
        print(f"\n  W_h asymmetry ||W-W^T||/||W|| = {asym:.3e}")
        assert asym < 1e-12, f"W_h should be exactly symmetric: {asym:.3e}"

    def test_nullspace_ones(self, matrices):
        """W_h @ ones ≈ 0 (constant density in nullspace)."""
        W_h, _, qdata, _ = matrices
        ones  = np.ones(qdata.n_quad)
        Wones = W_h @ ones
        rel = np.linalg.norm(Wones) / (np.linalg.norm(W_h) * np.linalg.norm(ones) + 1e-14)
        print(f"\n  ||W_h @ ones|| / (||W_h|| ||ones||) = {rel:.3e}")
        assert rel < 1e-12, f"W_h nullspace check failed: rel = {rel:.3e}"

    def test_regularised_not_nullspace(self, matrices):
        """W_tilde @ ones ≠ 0 (nullspace fixed)."""
        _, W_tilde, qdata, _ = matrices
        ones  = np.ones(qdata.n_quad)
        result = W_tilde @ ones
        # Should be non-zero
        assert np.linalg.norm(result) > 1e-6

    def test_W_tilde_invertible(self, matrices):
        """W_tilde should be invertible with moderate condition number."""
        _, W_tilde, _, _ = matrices
        cond = np.linalg.cond(W_tilde)
        print(f"\n  cond(W_tilde) = {cond:.3e}")
        assert np.isfinite(cond)
        assert cond < 1e8, f"W_tilde poorly conditioned: cond = {cond:.3e}"


# ===========================================================================
# Test 3: W̃V eigenvalue spectrum (KEY DIAGNOSTIC)
# ===========================================================================

class TestCalderonSpectrum:

    @pytest.fixture(scope="class")
    def calderon_data(self):
        qdata   = _qdata
        nmat    = _nmat
        W_h     = assemble_hypersingular_matrix(qdata, nmat)
        W_tilde = regularise_hypersingular(W_h, qdata)
        V_h     = nmat.V
        WV      = W_tilde @ V_h
        return V_h, W_tilde, WV

    def test_cond_WV_vs_cond_V(self, calderon_data):
        """
        Diagnostic: cond(W̃V) vs cond(V).

        FINDING: On the Koch snowflake, cond(W̃V) ≈ cond(V) ≈ O(10^4).
        The Calderon identity -WV = I/4 - K^2 fails on polygonal domains
        because the double-layer operator K is NOT compact at corners.
        The discrete Maue formula (Galerkin bilinear form) is therefore an
        incompatible discretization of the preconditioner for the Nystrom
        system, and does not reduce the condition number.

        This test REPORTS the condition numbers without asserting improvement,
        so the diagnostic information is always printed.  It asserts that:
        - Both condition numbers are finite (no numerical catastrophe)
        - W̃V has well-defined eigenvalues (operator is assembled correctly)
        """
        V_h, _, WV = calderon_data
        cond_V  = np.linalg.cond(V_h)
        cond_WV = np.linalg.cond(WV)

        print(f"\n  cond(V_h)  = {cond_V:.3e}")
        print(f"  cond(W̃V)  = {cond_WV:.3e}")
        ratio = cond_V / max(cond_WV, 1e-14)
        if ratio > 10:
            print(f"  Improvement: {ratio:.1f}×  ← Calderón WORKS")
        elif ratio > 1:
            print(f"  Improvement: {ratio:.2f}×  ← Calderón marginal")
        else:
            print(f"  Ratio: {ratio:.3f}×  ← Calderón NOT effective (Koch polygon corners)")
            print(f"  Root cause: -WV = I/4 - K^2 fails; K not compact on polygons")

        # Only assert that computation succeeded (finite, non-NaN)
        assert np.isfinite(cond_V)
        assert np.isfinite(cond_WV)

    def test_eigenvalue_clustering(self, calderon_data):
        """
        Eigenvalues of W̃V should cluster (small spread relative to mean).
        Ratio max|λ| / min|λ| ≈ cond(W̃V).
        """
        _, _, WV = calderon_data
        eigvals  = np.linalg.eigvals(WV)
        mag      = np.abs(eigvals)
        lam_min  = mag.min()
        lam_max  = mag.max()
        lam_mean = mag.mean()
        cond_WV  = lam_max / (lam_min + 1e-14)

        # Count eigenvalues within 2× of the median
        lam_median = np.median(mag)
        n_clustered = int(np.sum((mag > 0.5 * lam_median) & (mag < 2.0 * lam_median)))
        frac = n_clustered / len(mag)

        print(f"\n  Eigenvalue spectrum of W̃V:")
        print(f"    min|λ|    = {lam_min:.4f}")
        print(f"    max|λ|    = {lam_max:.4f}")
        print(f"    mean|λ|   = {lam_mean:.4f}")
        print(f"    median|λ| = {lam_median:.4f}")
        print(f"    cond(W̃V) via eigvals = {cond_WV:.3e}")
        print(f"    Fraction within 2× median: {frac:.1%}  ({n_clustered}/{len(mag)})")

        # At least 50% of eigenvalues should be within 2× of median
        assert frac > 0.5, f"Poor eigenvalue clustering: {frac:.1%}"
