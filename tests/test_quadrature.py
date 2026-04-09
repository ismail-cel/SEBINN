"""
Tests for src/quadrature/.

Verification targets (per CLAUDE.md):
  - GL nodes/weights reproduce exact integrals up to machine precision
  - Panel quadrature sums to correct perimeter
  - Self-correction formula matches known analytic value
  - Refined quadrature covers same area as standard
  - Projection matrix rows sum to 1 (partition of unity)
  - Structural invariants: shapes, index coverage, continuity
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.boundary.polygon import koch_snowflake, make_koch_geometry
from src.boundary.panels import build_uniform_panels, label_corner_ring_panels
from src.quadrature.gauss import gauss_legendre
from src.quadrature.panel_quad import build_panel_quadrature, build_refined_quadrature
from src.quadrature.self_correction import (
    self_panel_log_correction,
    self_panel_log_correction_vec,
)
from src.quadrature.projection import (
    barycentric_weights,
    barycentric_lagrange_matrix,
    refined_to_standard_projection,
)


# ===========================================================================
# Gauss-Legendre
# ===========================================================================

class TestGaussLegendre:

    def test_single_node(self):
        x, w = gauss_legendre(1)
        assert x.shape == (1,) and w.shape == (1,)
        assert abs(x[0]) < 1e-15
        assert abs(w[0] - 2.0) < 1e-15

    def test_weights_sum_to_two(self):
        """Sum of GL weights = 2 = integral of 1 over [-1,1]."""
        for n in [1, 2, 4, 8, 16]:
            _, w = gauss_legendre(n)
            assert abs(w.sum() - 2.0) < 1e-13, f"n={n}: sum(w)={w.sum()}"

    def test_nodes_in_interval(self):
        """All nodes must lie strictly inside (-1, 1)."""
        for n in [2, 4, 8, 16]:
            x, _ = gauss_legendre(n)
            assert np.all(x > -1.0) and np.all(x < 1.0)

    def test_nodes_sorted_ascending(self):
        for n in [4, 8, 16]:
            x, _ = gauss_legendre(n)
            assert np.all(np.diff(x) > 0)

    def test_symmetry(self):
        """GL nodes and weights are symmetric about 0."""
        for n in [4, 8, 16]:
            x, w = gauss_legendre(n)
            assert np.allclose(x, -x[::-1], atol=1e-14)
            assert np.allclose(w, w[::-1], atol=1e-14)

    def test_integrate_polynomial_exact(self):
        """
        n-point GL rule integrates polynomials of degree <= 2n-1 exactly.
        Test with p(x) = x^(2n-1) for n=4: integral over [-1,1] = 0.
        """
        for n in [4, 8, 16]:
            x, w = gauss_legendre(n)
            # integral of x^(2n-2) over [-1,1] = 2/(2n-1)
            deg = 2 * n - 2
            exact = 2.0 / (deg + 1)
            numerical = float(w @ (x ** deg))
            assert abs(numerical - exact) < 1e-12, (
                f"n={n}: degree {deg} integral error = {abs(numerical-exact):.2e}"
            )

    def test_integrate_constant(self):
        """Integral of 1 over [-1,1] = 2."""
        x, w = gauss_legendre(8)
        assert abs(w @ np.ones_like(x) - 2.0) < 1e-14

    def test_caching(self):
        """Same object returned on repeated calls."""
        x1, w1 = gauss_legendre(8)
        x2, w2 = gauss_legendre(8)
        assert x1 is x2 and w1 is w2


# ===========================================================================
# Panel quadrature
# ===========================================================================

class TestBuildPanelQuadrature:

    def setup_method(self):
        P = koch_snowflake(n=1)
        self.panels = build_uniform_panels(P, n_per_edge=12)
        self.p = 8
        self.qdata = build_panel_quadrature(self.panels, self.p)

    def test_shapes(self):
        N_pan = len(self.panels)
        Nq = N_pan * self.p
        assert self.qdata.Yq.shape == (2, Nq)
        assert self.qdata.wq.shape == (Nq,)
        assert self.qdata.pan_id.shape == (Nq,)
        assert self.qdata.s_on_panel.shape == (Nq,)
        assert self.qdata.L_panel.shape == (N_pan,)
        assert len(self.qdata.idx_std) == N_pan

    def test_node_count(self):
        assert self.qdata.n_quad == len(self.panels) * self.p
        assert self.qdata.n_panels == len(self.panels)

    def test_weights_positive(self):
        assert np.all(self.qdata.wq > 0)

    def test_perimeter_from_weights(self):
        """
        Sum of quadrature weights = total arc length (perimeter).
        Reason: wq[k] = (L/2) * w_GL[j], and sum over all GL nodes on a
        panel gives L (since sum(w_GL) = 2).
        """
        perimeter_exact = sum(p.length for p in self.panels)
        assert abs(self.qdata.wq.sum() - perimeter_exact) < 1e-12

    def test_idx_std_covers_all_nodes(self):
        """idx_std partitions {0, ..., Nq-1} with each index appearing once."""
        Nq = self.qdata.n_quad
        all_idx = np.concatenate(self.qdata.idx_std)
        assert sorted(all_idx) == list(range(Nq))

    def test_idx_std_sizes(self):
        """Each panel's index array has exactly p entries."""
        for idx in self.qdata.idx_std:
            assert len(idx) == self.p

    def test_pan_id_range(self):
        assert self.qdata.pan_id.min() == 0
        assert self.qdata.pan_id.max() == len(self.panels) - 1

    def test_s_on_panel_in_range(self):
        """Local arclength must lie in [0, L] for each node."""
        for i in range(self.qdata.n_quad):
            pid = self.qdata.pan_id[i]
            L = self.qdata.L_panel[pid]
            s = self.qdata.s_on_panel[i]
            assert 0.0 <= s <= L + 1e-14, (
                f"Node {i}: s={s:.4e} out of [0, L={L:.4e}]"
            )

    def test_nodes_on_panel_segments(self):
        """Each quadrature node must lie on the corresponding panel segment."""
        for i in range(self.qdata.n_quad):
            pid = self.qdata.pan_id[i]
            pan = self.panels[pid]
            y = self.qdata.Yq[:, i]
            # y = a + t*(b-a), so (y-a) must be parallel to (b-a)
            d = pan.b - pan.a
            v = y - pan.a
            # Cross product in 2D should be ~0
            cross = abs(v[0] * d[1] - v[1] * d[0])
            assert cross < 1e-12 * pan.length, (
                f"Node {i} not on panel {pid}: cross={cross:.2e}"
            )

    def test_l_panel_matches_panel_lengths(self):
        for m, pan in enumerate(self.panels):
            assert abs(self.qdata.L_panel[m] - pan.length) < 1e-14

    def test_integrate_constant_one(self):
        """
        Integrate f=1 over boundary using quadrature = perimeter.
        This is the fundamental correctness check for the quadrature rule.
        """
        perimeter = sum(p.length for p in self.panels)
        integral = float(self.qdata.wq.sum())
        assert abs(integral - perimeter) < 1e-12

    def test_pan_id_consistency_with_idx_std(self):
        """pan_id[k] should match the panel whose idx_std contains k."""
        for m, idx in enumerate(self.qdata.idx_std):
            for k in idx:
                assert self.qdata.pan_id[k] == m


# ===========================================================================
# Refined quadrature
# ===========================================================================

class TestRefinedQuadrature:

    def setup_method(self):
        P = koch_snowflake(n=1)
        self.panels = build_uniform_panels(P, n_per_edge=12)
        self.p = 8
        self.n_sub = 4
        self.rdata = build_refined_quadrature(self.panels, self.p, self.n_sub)

    def test_shapes(self):
        Npan = len(self.panels)
        NqR = Npan * self.n_sub * self.p
        assert self.rdata.YqR.shape == (2, NqR)
        assert self.rdata.wqR.shape == (NqR,)
        assert self.rdata.pan_id_R.shape == (NqR,)
        assert len(self.rdata.idx_ref) == Npan
        assert self.rdata.n_quad_refined == NqR

    def test_idx_ref_sizes(self):
        for idx in self.rdata.idx_ref:
            assert len(idx) == self.n_sub * self.p

    def test_idx_ref_partitions_all_nodes(self):
        NqR = self.rdata.n_quad_refined
        all_idx = np.concatenate(self.rdata.idx_ref)
        assert sorted(all_idx) == list(range(NqR))

    def test_weights_positive(self):
        assert np.all(self.rdata.wqR > 0)

    def test_same_perimeter_as_standard(self):
        """
        Refined quadrature integrates a constant over the same boundary,
        so the total weight must equal the perimeter.
        """
        perimeter = sum(p.length for p in self.panels)
        assert abs(self.rdata.wqR.sum() - perimeter) < 1e-10

    def test_per_panel_weight_equals_panel_length(self):
        """
        Sum of weights within each panel's refined nodes = panel length.
        """
        for m, pan in enumerate(self.panels):
            idx = self.rdata.idx_ref[m]
            panel_weight = self.rdata.wqR[idx].sum()
            assert abs(panel_weight - pan.length) < 1e-12, (
                f"Panel {m}: refined weight sum={panel_weight:.6e}, "
                f"length={pan.length:.6e}"
            )


# ===========================================================================
# Self-panel analytic correction
# ===========================================================================

class TestSelfPanelCorrection:

    def test_known_value_midpoint(self):
        """
        For L=1, s0=0.5 (midpoint):
          I = -(1/2pi) * (0.5*log(0.5) + 0.5*log(0.5) - 1)
            = -(1/2pi) * (log(0.5) - 1)
            = -(1/2pi) * (-log(2) - 1)
            =  (1/2pi) * (log(2) + 1)
        """
        L, s0 = 1.0, 0.5
        I = self_panel_log_correction(L, s0)
        expected = (1.0 / (2.0 * np.pi)) * (np.log(2.0) + 1.0)
        assert abs(I - expected) < 1e-14, f"I={I:.10e}, expected={expected:.10e}"

    def test_symmetry(self):
        """I(s0; L) = I(L - s0; L): formula is symmetric about midpoint."""
        L = 0.7
        for s0 in [0.1, 0.2, 0.3]:
            I1 = self_panel_log_correction(L, s0)
            I2 = self_panel_log_correction(L, L - s0)
            assert abs(I1 - I2) < 1e-14, (
                f"s0={s0}: I(s0)={I1:.6e} != I(L-s0)={I2:.6e}"
            )

    def test_positive_value(self):
        """
        For small panels (L << 1), log(s) < 0 for s in (0,1),
        so -(1/2pi) * (negative) > 0.
        """
        for L in [0.01, 0.1, 0.5]:
            s0 = L / 2
            I = self_panel_log_correction(L, s0)
            assert I > 0, f"L={L}: I={I:.6e} should be positive"

    def test_scales_with_panel_size(self):
        """
        Larger panel => larger correction (monotone in L for fixed s0/L).
        """
        s0_frac = 0.3
        L_vals = [0.01, 0.1, 1.0]
        I_vals = [self_panel_log_correction(L, s0_frac * L) for L in L_vals]
        assert I_vals[0] < I_vals[1] < I_vals[2]

    def test_numerical_agreement(self):
        """
        Compare analytic correction against Gauss quadrature with the
        singular neighbourhood removed analytically.

        Split: [0, s0-eps] and [s0+eps, L] via GL quadrature;
        add the analytic integral over [s0-eps, s0+eps]:
            -(1/2pi) * int_{-eps}^{eps} log|u| du
          = -(1/2pi) * 2 * (eps*log(eps) - eps)
        """
        L, s0 = 0.5, 0.2
        I_analytic = self_panel_log_correction(L, s0)

        eps = 1e-4
        n_gl = 64
        xi, wi = gauss_legendre(n_gl)

        def integrate_piece(a, b):
            t = 0.5 * ((b - a) * xi + (a + b))
            integrand = -(1.0 / (2.0 * np.pi)) * np.log(np.abs(t - s0))
            return 0.5 * (b - a) * float(wi @ integrand)

        I_outer = integrate_piece(0.0, s0 - eps) + integrate_piece(s0 + eps, L)
        # Analytic contribution of the missing strip [s0-eps, s0+eps]
        I_strip = -(1.0 / (2.0 * np.pi)) * 2.0 * (eps * np.log(eps) - eps)
        I_numerical = I_outer + I_strip

        # 1e-6 is the expected accuracy: GL near the cut-off endpoint
        # contributes O(eps^2) error; the formula is correct to > 6 digits.
        assert abs(I_analytic - I_numerical) < 1e-6, (
            f"Analytic={I_analytic:.8e}, Numerical={I_numerical:.8e}"
        )

    def test_vectorised_matches_scalar(self):
        P = koch_snowflake(n=1)
        panels = build_uniform_panels(P, n_per_edge=4)
        qdata = build_panel_quadrature(panels, p=4)

        corr_vec = self_panel_log_correction_vec(
            qdata.L_panel, qdata.s_on_panel, qdata.pan_id
        )
        for i in range(qdata.n_quad):
            pid = qdata.pan_id[i]
            expected = self_panel_log_correction(
                float(qdata.L_panel[pid]), float(qdata.s_on_panel[i])
            )
            assert abs(corr_vec[i] - expected) < 1e-15


# ===========================================================================
# Barycentric Lagrange and projection matrix
# ===========================================================================

class TestBarycentricLagrange:

    def test_interpolate_at_nodes(self):
        """L[k, j] = delta_{k,j} when x_eval = x_nodes."""
        x, _ = gauss_legendre(5)
        L = barycentric_lagrange_matrix(x, x)
        assert np.allclose(L, np.eye(5), atol=1e-13)

    def test_partition_of_unity(self):
        """Each row of L sums to 1."""
        x_nodes, _ = gauss_legendre(8)
        x_eval = np.linspace(-0.9, 0.9, 20)
        L = barycentric_lagrange_matrix(x_nodes, x_eval)
        assert np.allclose(L.sum(axis=1), 1.0, atol=1e-12)

    def test_interpolates_polynomial(self):
        """Lagrange matrix should reproduce polynomials of degree <= n-1."""
        x_nodes, _ = gauss_legendre(6)
        f_nodes = x_nodes ** 4  # degree 4 < 6

        x_eval = np.linspace(-0.95, 0.95, 50)
        L = barycentric_lagrange_matrix(x_nodes, x_eval)
        f_eval = L @ f_nodes
        f_exact = x_eval ** 4
        assert np.allclose(f_eval, f_exact, atol=1e-12)


class TestRefinedToStandardProjection:

    def test_shape(self):
        p, n_sub = 8, 4
        T = refined_to_standard_projection(p, n_sub)
        assert T.shape == (n_sub * p, p)

    def test_partition_of_unity(self):
        """Rows of T sum to 1."""
        T = refined_to_standard_projection(8, 4)
        assert np.allclose(T.sum(axis=1), 1.0, atol=1e-12)

    def test_reproduces_constant(self):
        """
        T maps standard GL values to refined node values via Lagrange interpolation.
        MATLAB usage: kRef.' * T, i.e. row-vec (1×n_sub*p) @ T (n_sub*p × p).
        The operation is T @ f_std = f_ref (interpolate from std to ref positions).

        For a constant f_std = c*ones(p):
            T @ (c*ones_p) = c * T.sum(axis=1) = c * ones(n_sub*p)
        because rows of T sum to 1 (partition of unity).
        """
        p, n_sub = 8, 3
        T = refined_to_standard_projection(p, n_sub)
        c = 3.7
        f_std = c * np.ones(p)
        f_ref = T @ f_std              # shape (n_sub*p,): interpolate to refined nodes
        assert np.allclose(f_ref, c * np.ones(n_sub * p), atol=1e-12)

    def test_caching(self):
        T1 = refined_to_standard_projection(8, 4)
        T2 = refined_to_standard_projection(8, 4)
        assert T1 is T2

    def test_interpolates_to_refined_nodes(self):
        """
        T @ xi_std should give the values of the identity function f(xi)=xi
        at the refined GL nodes, since T interpolates standard -> refined.
        Exact for polynomials of degree <= p-1.
        """
        p, n_sub = 6, 3
        T = refined_to_standard_projection(p, n_sub)
        xi_std, _ = gauss_legendre(p)
        xi_sub, _ = gauss_legendre(p)

        # Build the refined node positions on [-1, 1]
        xi_ref = []
        for ss in range(n_sub):
            t_A, t_B = ss / n_sub, (ss + 1) / n_sub
            for j in range(p):
                t = (xi_sub[j] + 1.0) / 2.0
                t_g = (1.0 - t) * t_A + t * t_B
                xi_ref.append(2.0 * t_g - 1.0)
        xi_ref = np.array(xi_ref)

        # f(xi) = xi is degree 1 < p: T @ f_std should recover f at xi_ref exactly
        f_ref_approx = T @ xi_std        # shape (n_sub*p,)
        assert np.allclose(f_ref_approx, xi_ref, atol=1e-11), (
            f"Max error: {np.max(np.abs(f_ref_approx - xi_ref)):.2e}"
        )
