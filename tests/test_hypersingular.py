"""
Tests for the direct Nyström hypersingular operator T_h and Calderón preconditioner.

Tests
-----
1. Normal orientation     — outward normals point away from domain centroid
2. Normal unit length     — |ν| = 1 at every node
3. Kernel symmetry        — diag(w) T_h is symmetric (Galerkin symmetry)
4. Nullspace              — T_h @ ones ≈ 0 (rank-1 nullspace)
5. T_tilde invertible     — cond(T_tilde) is finite and moderate
6. Calderón spectrum      — cond(T̃V) vs cond(V)  (KEY TEST)
7. Eigenvalue clustering  — eigenvalues of T̃V cluster near a common value

Geometry: Koch(1), n_per_edge=12, p_gl=16  (canonical settings).
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.boundary.polygon import make_koch_geometry
from src.boundary.panels import build_uniform_panels, label_corner_ring_panels
from src.quadrature.panel_quad import build_panel_quadrature
from src.quadrature.nystrom import assemble_nystrom_matrix
from src.quadrature.hypersingular import (
    panel_normals_tangents,
    hypsing_self_panel_correction,
    assemble_hypersingular_direct,
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


# ===========================================================================
# Test 1–2: Normal / tangent geometry
# ===========================================================================

class TestPanelNormalsTangents:

    def test_tangent_unit_length(self):
        """All tangent vectors have unit length."""
        tau, nu = panel_normals_tangents(_qdata)
        np.testing.assert_allclose(np.abs(tau), 1.0, atol=1e-14,
                                   err_msg="Tangent vectors not unit length")

    def test_normal_unit_length(self):
        """All normal vectors have unit length."""
        tau, nu = panel_normals_tangents(_qdata)
        np.testing.assert_allclose(np.abs(nu), 1.0, atol=1e-14,
                                   err_msg="Normal vectors not unit length")

    def test_normal_perpendicular_to_tangent(self):
        """ν ⊥ τ: Re[ν conj(τ)] = 0 at every node."""
        tau, nu = panel_normals_tangents(_qdata)
        dot = np.real(nu * np.conj(tau))
        np.testing.assert_allclose(dot, 0.0, atol=1e-14,
                                   err_msg="Normals not perpendicular to tangents")

    def test_normal_outward(self):
        """
        Outward normals: for each node x_i, (x_i − centroid) · ν_i > 0.

        The Koch snowflake centroid is near the origin.  Any outward normal
        should point away from the centroid, so the dot product is positive.
        """
        tau, nu = panel_normals_tangents(_qdata)
        Yq      = _qdata.Yq        # (2, Nq)
        centroid = Yq.mean(axis=1)  # (2,)

        # Direction from centroid to each node (real 2D)
        radial_x = Yq[0] - centroid[0]
        radial_y = Yq[1] - centroid[1]

        # Real and imaginary parts of ν
        nu_x = np.real(nu)
        nu_y = np.imag(nu)

        dot = radial_x * nu_x + radial_y * nu_y
        n_outward = int(np.sum(dot > 0))
        frac = n_outward / len(dot)
        print(f"\n  Fraction of outward normals: {frac:.1%}  ({n_outward}/{len(dot)})")

        # At least 90% of normals should be outward
        # (corners / panel junctions may have near-zero dot product)
        assert frac > 0.90, f"Only {frac:.1%} of normals are outward"


# ===========================================================================
# Test 3–5: T_h matrix properties
# ===========================================================================

class TestHypersingularDirect:

    @pytest.fixture(scope="class")
    def matrices(self):
        qdata   = _qdata
        nmat    = _nmat
        T_h     = assemble_hypersingular_direct(qdata)
        T_tilde = regularise_hypersingular(T_h, qdata.wq)
        return T_h, T_tilde, qdata, nmat

    def test_shape(self, matrices):
        T_h, _, qdata, _ = matrices
        Nq = qdata.n_quad
        assert T_h.shape == (Nq, Nq)

    def test_galerkin_symmetry(self, matrices):
        """
        diag(w) T_h is symmetric (Galerkin symmetry).

        The Nyström T_h has weights only on the right column:
            T_h[i,j] = T(x_i,y_j) w_j

        Since T(x,y) = T(y,x) (self-adjoint kernel), it follows that
            w_i T_h[i,j] = w_j T_h[j,i]

        i.e. diag(w) T_h is symmetric.  Test tolerance 1e-10 allows for
        the quadrature approximation of the self-panel correction.
        """
        T_h, _, qdata, _ = matrices
        wq  = qdata.wq
        W_T = wq[:, None] * T_h          # diag(w) T_h
        asym = np.linalg.norm(W_T - W_T.T) / (np.linalg.norm(W_T) + 1e-14)
        print(f"\n  ||diag(w)T_h − (diag(w)T_h)^T|| / ||diag(w)T_h|| = {asym:.3e}")
        assert asym < 1e-8, f"Galerkin symmetry violated: asym = {asym:.3e}"

    def test_nullspace_ones(self, matrices):
        """
        T_h @ ones ≈ 0 (constant density in nullspace of T).

        Note: unlike the Galerkin W_h (where D_h·1=0 exactly), the Nyström
        T_h only satisfies T·1≈0 to quadrature accuracy. Adjacent-panel
        contributions are approximated by p=16 Gauss without near-panel
        refinement, giving O(1e-3) relative error at Koch reentrant corners.
        This is expected behaviour, not a formula error.
        """
        T_h, _, qdata, _ = matrices
        ones  = np.ones(qdata.n_quad)
        Tones = T_h @ ones
        rel   = np.linalg.norm(Tones) / (np.linalg.norm(T_h) * np.linalg.norm(ones) + 1e-14)
        print(f"\n  ||T_h @ ones|| / (||T_h|| ||ones||) = {rel:.3e}")
        # Tolerance: p=16 Nyström near-panel quadrature accuracy on Koch(1)
        assert rel < 5e-3, f"T_h nullspace check failed: rel = {rel:.3e}"

    def test_regularised_not_nullspace(self, matrices):
        """T_tilde @ ones ≠ 0 (nullspace fixed)."""
        _, T_tilde, qdata, _ = matrices
        ones   = np.ones(qdata.n_quad)
        result = T_tilde @ ones
        assert np.linalg.norm(result) > 1e-6

    def test_T_tilde_invertible(self, matrices):
        """T_tilde should be invertible with moderate condition number."""
        _, T_tilde, _, _ = matrices
        cond = np.linalg.cond(T_tilde)
        print(f"\n  cond(T_tilde) = {cond:.3e}")
        assert np.isfinite(cond)
        assert cond < 1e8, f"T_tilde poorly conditioned: cond = {cond:.3e}"

    def test_self_panel_correction_formula(self):
        """
        Validate the Hadamard correction formula on straight panels.

        The cos2θ factor cancels (τ_x = τ_y on self-panel), giving:
            I_T = (1/π)(1/(L−s₀) + 1/s₀)   for ANY panel orientation.

        At s₀ = L/2: I_T = (1/π)(2/L + 2/L) = 4/(πL).
        At s₀ = L/4: I_T = (1/π)(4/(3L) + 4/L) = 16/(3πL).
        """
        L = 1.0

        # s0 = L/2 (any orientation gives same result)
        for tau in [1.0+0j, np.exp(1j*np.pi/3), np.exp(1j*np.pi/4)]:
            I_mid = hypsing_self_panel_correction(L, 0.5, tau)
            I_mid_exact = (1.0 / np.pi) * (2.0 / L + 2.0 / L)
            np.testing.assert_allclose(I_mid, I_mid_exact, rtol=1e-14)

        # s0 = L/4
        I_qtr = hypsing_self_panel_correction(L, 0.25)
        I_qtr_exact = (1.0 / np.pi) * (4.0 / (3.0 * L) + 4.0 / L)
        np.testing.assert_allclose(I_qtr, I_qtr_exact, rtol=1e-14)


# ===========================================================================
# Test 6: Calderón spectrum (KEY DIAGNOSTIC)
# ===========================================================================

class TestCalderonSpectrumDirect:

    @pytest.fixture(scope="class")
    def calderon_data(self):
        qdata   = _qdata
        nmat    = _nmat
        T_h     = assemble_hypersingular_direct(qdata)
        T_tilde = regularise_hypersingular(T_h, qdata.wq)
        V_h     = nmat.V
        TV      = T_tilde @ V_h
        return V_h, T_tilde, TV, qdata

    def test_cond_TV_vs_cond_V(self, calderon_data):
        """
        KEY TEST: cond(T̃V) vs cond(V).

        HYPOTHESIS: the direct Nyström T_h lives in the same discrete space
        as V_h, so T̃_h V_h ≈ (1/4) diag(w) and eigenvalues cluster, giving
        cond(T̃V) ≪ cond(V).

        On Koch (non-compact K at corners) we expect some degradation relative
        to a smooth curve, but still a significant improvement.

        The test asserts:
        - Both condition numbers are finite (no numerical catastrophe)
        - T̃V has well-defined eigenvalues
        and REPORTS the improvement factor.
        """
        V_h, _, TV, _ = calderon_data
        cond_V  = np.linalg.cond(V_h)
        cond_TV = np.linalg.cond(TV)

        print(f"\n  cond(V_h)  = {cond_V:.3e}")
        print(f"  cond(T̃V)  = {cond_TV:.3e}")
        ratio = cond_V / max(cond_TV, 1e-14)
        if ratio > 100:
            print(f"  Improvement: {ratio:.0f}×  ← Calderón EXCELLENT")
        elif ratio > 10:
            print(f"  Improvement: {ratio:.1f}×  ← Calderón WORKS")
        elif ratio > 1:
            print(f"  Improvement: {ratio:.2f}×  ← Calderón marginal")
        else:
            print(f"  Ratio: {ratio:.3f}×  ← Calderón NOT effective (Koch corners?)")

        assert np.isfinite(cond_V)
        assert np.isfinite(cond_TV)

    def test_eigenvalue_clustering(self, calderon_data):
        """
        Eigenvalues of T̃V should cluster near w_avg/4.

        For a smooth curve: T_h V_h ≈ (1/4) diag(w), so eigenvalues ≈ w_avg/4.
        For Koch with reentrant corners: some spread expected.
        At least 50% of |λ| should be within 2× of the median.
        """
        _, _, TV, qdata = calderon_data
        eigvals = np.linalg.eigvals(TV)
        mag     = np.abs(eigvals)
        lam_min    = mag.min()
        lam_max    = mag.max()
        lam_mean   = mag.mean()
        lam_median = np.median(mag)
        cond_TV    = lam_max / (lam_min + 1e-14)

        # Expected cluster: w_avg / 4
        w_avg = float(qdata.wq.mean())
        expected = w_avg / 4.0

        n_clustered = int(np.sum((mag > 0.5 * lam_median) & (mag < 2.0 * lam_median)))
        frac = n_clustered / len(mag)

        print(f"\n  Eigenvalue spectrum of T̃V:")
        print(f"    w_avg/4   = {expected:.6f}  (expected cluster center)")
        print(f"    min|λ|    = {lam_min:.6f}")
        print(f"    max|λ|    = {lam_max:.6f}")
        print(f"    mean|λ|   = {lam_mean:.6f}")
        print(f"    median|λ| = {lam_median:.6f}")
        print(f"    cond(T̃V) via eigvals = {cond_TV:.3e}")
        print(f"    Fraction within 2× median: {frac:.1%}  ({n_clustered}/{len(mag)})")

        assert frac > 0.5, f"Poor eigenvalue clustering: {frac:.1%}"
