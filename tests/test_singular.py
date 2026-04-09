"""
Tests for src/singular/.

Verification targets (CLAUDE.md):
  - Correct singular exponent pi/omega - 1 at each corner type
  - Consistency of corner-local coordinates and distance r
  - sigma_s = -(pi/omega) * r^alpha formula matches canonical definition
  - sigma_s -> -inf as r -> 0 for reentrant corners (alpha < 0)
  - sigma_s = 0 at r = 0 not reached (r_min clamp)
  - Multi-corner assembly sums contributions correctly
  - Cutoff is 1 near corner, 0 far away
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.boundary.polygon import make_koch_geometry, koch_snowflake
from src.singular.corner_coords import corner_local_coords, CornerCoords
from src.singular.sigma_s import sigma_s_single, sigma_s_at_points
from src.singular.enrichment import SingularEnrichment, smooth_cutoff


# ---------------------------------------------------------------------------
# Shared geometry
# ---------------------------------------------------------------------------

GEOM = make_koch_geometry(n=1)

# Koch(1): 6 reentrant corners at indices {1,3,5,7,9,11}, omega = 4*pi/3
OMEGA_REENTRANT = 4.0 * np.pi / 3.0
ALPHA_REENTRANT = np.pi / OMEGA_REENTRANT - 1.0   # = -1/4


# ===========================================================================
# Corner coordinate extraction
# ===========================================================================

class TestCornerCoords:

    def test_singular_only_count(self):
        """singular_only=True returns one CornerCoords per reentrant corner."""
        points = np.zeros((5, 2))
        coords = corner_local_coords(GEOM, points, singular_only=True)
        assert len(coords) == 6   # Koch(1) has 6 reentrant corners

    def test_all_corners_count(self):
        coords = corner_local_coords(GEOM, np.zeros((3, 2)), singular_only=False)
        assert len(coords) == 12

    def test_corner_indices_correct(self):
        points = np.zeros((1, 2))
        coords = corner_local_coords(GEOM, points, singular_only=True)
        idxs = {cc.corner_idx for cc in coords}
        assert idxs == {1, 3, 5, 7, 9, 11}

    def test_omega_values(self):
        points = np.zeros((1, 2))
        coords = corner_local_coords(GEOM, points, singular_only=True)
        for cc in coords:
            assert abs(cc.omega - OMEGA_REENTRANT) < 1e-12

    def test_alpha_values(self):
        """Singular exponent must be -1/4 for all reentrant Koch(1) corners."""
        points = np.zeros((1, 2))
        coords = corner_local_coords(GEOM, points, singular_only=True)
        for cc in coords:
            assert abs(cc.alpha - ALPHA_REENTRANT) < 1e-12, (
                f"corner {cc.corner_idx}: alpha={cc.alpha:.6f}, expected {ALPHA_REENTRANT:.6f}"
            )

    def test_alpha_negative_for_reentrant(self):
        """alpha < 0 iff omega > pi (reentrant corner)."""
        points = np.zeros((1, 2))
        for cc in corner_local_coords(GEOM, points, singular_only=True):
            assert cc.alpha < 0
            assert cc.is_singular

    def test_r_at_vertex_is_r_min(self):
        """Evaluating at the vertex itself gives r = r_min (clamped)."""
        r_min = 1e-14
        for c_idx in [1, 3]:
            v = GEOM.vertices[c_idx]
            points = v[None, :]   # shape (1, 2)
            coords = corner_local_coords(GEOM, points, singular_only=True, r_min=r_min)
            cc = next(cc for cc in coords if cc.corner_idx == c_idx)
            assert abs(cc.r[0] - r_min) < 1e-30

    def test_r_shape(self):
        N = 20
        points = np.random.rand(N, 2) * 0.4 - 0.2
        coords = corner_local_coords(GEOM, points, singular_only=True)
        for cc in coords:
            assert cc.r.shape == (N,)
            assert cc.theta.shape == (N,)

    def test_r_positive(self):
        points = np.random.rand(50, 2) * 0.4 - 0.2
        coords = corner_local_coords(GEOM, points, singular_only=True)
        for cc in coords:
            assert np.all(cc.r > 0)

    def test_theta_in_range(self):
        """theta must lie in (-pi, pi]."""
        points = np.random.rand(50, 2) * 0.4 - 0.2
        coords = corner_local_coords(GEOM, points, singular_only=True)
        for cc in coords:
            assert np.all(cc.theta >= -np.pi - 1e-15)
            assert np.all(cc.theta <= np.pi + 1e-15)

    def test_r_euclidean_distance(self):
        """r[i] must equal |points[i] - vertex|."""
        np.random.seed(42)
        points = np.random.rand(10, 2) * 0.3
        coords = corner_local_coords(GEOM, points, singular_only=True)
        for cc in coords:
            v = cc.vertex
            expected_r = np.linalg.norm(points - v[None, :], axis=1)
            expected_r = np.maximum(expected_r, 1e-14)
            assert np.allclose(cc.r, expected_r, atol=1e-14)


# ===========================================================================
# sigma_s formula
# ===========================================================================

class TestSigmaS:

    def test_canonical_formula(self):
        """
        sigma_s = -(pi/omega) * r^alpha, alpha = pi/omega - 1.
        Test against direct evaluation for Koch(1) reentrant corner.
        """
        omega = OMEGA_REENTRANT
        alpha = ALPHA_REENTRANT
        r = np.array([0.1, 0.2, 0.5])
        expected = -(np.pi / omega) * r ** alpha

        v = GEOM.vertices[1]   # first reentrant corner
        # Build a point at distance r in the direction theta=0
        pts = v[None, :] + np.column_stack([r, np.zeros(3)])
        result = sigma_s_at_points(omega, v, pts)

        assert np.allclose(result, expected, rtol=1e-13)

    def test_negative_values(self):
        """sigma_s is always negative (prefactor -(pi/omega) < 0, r^alpha > 0)."""
        v = GEOM.vertices[1]
        pts = np.random.rand(20, 2) * 0.3
        vals = sigma_s_at_points(OMEGA_REENTRANT, v, pts)
        assert np.all(vals < 0)

    def test_diverges_near_corner(self):
        """
        For alpha < 0, sigma_s -> -inf as r -> 0.
        |sigma_s| should be monotonically LARGER for smaller r.
        r_vals is ordered decreasing, so abs_vals should be ordered increasing.
        """
        omega = OMEGA_REENTRANT
        v = GEOM.vertices[1]
        # r_vals in decreasing order: farther -> closer
        r_vals = np.array([0.1, 0.01, 0.001, 0.0001])
        pts = v[None, :] + np.column_stack([r_vals, np.zeros(4)])
        vals = sigma_s_at_points(omega, v, pts)
        # |sigma_s| increases as r decreases (alpha < 0 => r^alpha grows)
        abs_vals = np.abs(vals)
        assert np.all(np.diff(abs_vals) > 0), (
            f"|sigma_s| not monotone increasing as r decreases: {abs_vals}"
        )

    def test_singular_exponent_exact(self):
        """
        For Koch(1) reentrant corners:
            omega = 4*pi/3  =>  alpha = 3/(4) - 1 = -1/4
        Verify the power-law scaling: sigma_s(2r) / sigma_s(r) = 2^alpha.
        """
        omega = OMEGA_REENTRANT
        alpha = ALPHA_REENTRANT
        v = GEOM.vertices[1]

        r = 0.1
        pt1 = (v + np.array([r,   0.0]))[None, :]
        pt2 = (v + np.array([2*r, 0.0]))[None, :]

        s1 = sigma_s_at_points(omega, v, pt1)[0]
        s2 = sigma_s_at_points(omega, v, pt2)[0]

        ratio = s2 / s1
        expected_ratio = (2.0 * r) ** alpha / r ** alpha   # = 2^alpha
        assert abs(ratio - expected_ratio) < 1e-12, (
            f"Scaling ratio={ratio:.8f}, expected 2^alpha={expected_ratio:.8f}"
        )

    def test_sigma_s_single_matches_at_points(self):
        """sigma_s_single(cc) and sigma_s_at_points give the same result."""
        np.random.seed(7)
        pts = np.random.rand(15, 2) * 0.3
        v = GEOM.vertices[1]
        omega = OMEGA_REENTRANT

        # Via sigma_s_at_points
        direct = sigma_s_at_points(omega, v, pts)

        # Via sigma_s_single (needs CornerCoords)
        coords = corner_local_coords(GEOM, pts, singular_only=True)
        cc = next(cc for cc in coords if cc.corner_idx == 1)
        via_cc = sigma_s_single(cc)

        assert np.allclose(direct, via_cc, rtol=1e-13)

    def test_convex_corner_exponent_positive(self):
        """
        For convex corners (omega < pi), alpha > 0: sigma_s -> 0 as r -> 0.
        Test with a right angle (omega = pi/2, alpha = 1).
        """
        omega = np.pi / 2.0
        alpha = np.pi / omega - 1.0   # = 1.0
        assert alpha > 0

        v = np.array([0.0, 0.0])
        r = np.array([0.01, 0.1, 0.5])
        pts = v[None, :] + np.column_stack([r, np.zeros(3)])
        vals = sigma_s_at_points(omega, v, pts)
        # Smaller r -> smaller |sigma_s| (sigma_s -> 0)
        assert np.abs(vals[0]) < np.abs(vals[1]) < np.abs(vals[2])


# ===========================================================================
# Multi-corner enrichment
# ===========================================================================

class TestSingularEnrichment:

    def test_evaluate_shape(self):
        enrich = SingularEnrichment(GEOM)
        pts = np.random.rand(30, 2) * 0.4 - 0.2
        sigma_s = enrich.evaluate(pts)
        assert sigma_s.shape == (30,)

    def test_evaluate_per_corner_shape(self):
        enrich = SingularEnrichment(GEOM, per_corner_gamma=True)
        pts = np.random.rand(20, 2) * 0.4 - 0.2
        out = enrich.evaluate_per_corner(pts)
        assert out.shape == (20, 6)   # 6 singular corners

    def test_n_gamma_single(self):
        enrich = SingularEnrichment(GEOM, per_corner_gamma=False)
        assert enrich.n_gamma == 1

    def test_n_gamma_per_corner(self):
        enrich = SingularEnrichment(GEOM, per_corner_gamma=True)
        assert enrich.n_gamma == 6

    def test_evaluate_equals_sum_of_per_corner(self):
        """evaluate() must equal sum over columns of evaluate_per_corner()."""
        enrich = SingularEnrichment(GEOM, per_corner_gamma=False)
        pts = np.random.rand(25, 2) * 0.4 - 0.2

        total = enrich.evaluate(pts)
        per_c = SingularEnrichment(GEOM, per_corner_gamma=True).evaluate_per_corner(pts)
        assert np.allclose(total, per_c.sum(axis=1), rtol=1e-13)

    def test_precompute_no_cutoff(self):
        enrich = SingularEnrichment(GEOM, use_cutoff=False)
        pts = np.random.rand(15, 2) * 0.3
        result = enrich.precompute(pts)
        expected = enrich.evaluate(pts)
        assert np.allclose(result, expected, rtol=1e-15)

    def test_precompute_per_corner(self):
        enrich = SingularEnrichment(GEOM, per_corner_gamma=True)
        pts = np.random.rand(10, 2) * 0.3
        result = enrich.precompute(pts)
        assert result.shape == (10, 6)

    def test_negative_values(self):
        """sigma_s is negative everywhere (all corners contribute negatively)."""
        enrich = SingularEnrichment(GEOM)
        pts = np.random.rand(30, 2) * 0.3
        vals = enrich.evaluate(pts)
        assert np.all(vals < 0)

    def test_larger_near_corners(self):
        """
        |sigma_s| should be larger near a singular corner than far from it.
        """
        enrich = SingularEnrichment(GEOM)
        v = GEOM.vertices[1]
        near = v[None, :] + np.array([[0.001, 0.0]])
        far  = v[None, :] + np.array([[0.2,   0.0]])
        s_near = enrich.evaluate(near)[0]
        s_far  = enrich.evaluate(far)[0]
        assert abs(s_near) > abs(s_far), (
            f"|sigma_s| near={abs(s_near):.4e}, far={abs(s_far):.4e}"
        )

    def test_symmetry_on_symmetric_polygon(self):
        """
        Koch(1) has 3-fold rotational symmetry.  Evaluating sigma_s at points
        related by 120-degree rotation should give equal values (to numerical
        precision), since all 6 reentrant corners have the same omega.
        """
        enrich = SingularEnrichment(GEOM)
        # Base point near the center
        angle = 0.0
        r_test = 0.05
        pts = np.array([
            [r_test * np.cos(angle + k * 2 * np.pi / 3),
             r_test * np.sin(angle + k * 2 * np.pi / 3)]
            for k in range(3)
        ])
        vals = enrich.evaluate(pts)
        # All three values should be equal (symmetric geometry)
        assert np.allclose(vals, vals[0], rtol=1e-4), (
            f"Symmetry broken: {vals}"
        )


# ===========================================================================
# Smooth cutoff
# ===========================================================================

class TestSmoothCutoff:

    def test_equals_one_at_zero(self):
        """chi(0) = 1."""
        r = np.array([0.0])
        chi = smooth_cutoff(r, R=1.0)
        assert abs(chi[0] - 1.0) < 1e-12

    def test_equals_zero_beyond_R(self):
        """chi(r) = 0 for r >= R."""
        r = np.array([1.0, 1.5, 2.0])
        chi = smooth_cutoff(r, R=1.0)
        assert np.all(chi == 0.0)

    def test_monotone_decreasing(self):
        """chi should decrease monotonically from 1 to 0."""
        r = np.linspace(0, 0.99, 50)
        chi = smooth_cutoff(r, R=1.0)
        assert np.all(np.diff(chi) <= 0)

    def test_cutoff_reduces_sigma_s_far(self):
        """With cutoff enabled, sigma_s far from the corner is reduced."""
        enrich_no_cut  = SingularEnrichment(GEOM, use_cutoff=False)
        enrich_with_cut = SingularEnrichment(
            GEOM, use_cutoff=True, cutoff_radius=0.1
        )
        v = GEOM.vertices[1]
        # Point far from corner (r >> R): cutoff should suppress sigma_s
        far_pt = (v + np.array([0.3, 0.0]))[None, :]
        s_no   = enrich_no_cut.evaluate(far_pt)[0]
        s_cut  = enrich_with_cut.evaluate(far_pt)[0]
        assert abs(s_cut) < abs(s_no), (
            f"Cutoff did not reduce |sigma_s|: no_cut={s_no:.4e}, cut={s_cut:.4e}"
        )

    def test_cutoff_equals_one_near_corner(self):
        """
        For r < R/2, the cutoff is exactly 1, so sigma_s_at_points (no cutoff)
        and the cutoff-weighted value should agree exactly for that corner.
        Test single-corner contribution directly via sigma_s_at_points + chi.
        """
        R = 0.2
        v = GEOM.vertices[1]
        omega = OMEGA_REENTRANT
        # r = 0.001 << R/2 = 0.1: chi should be exactly 1
        near_pt = (v + np.array([0.001, 0.0]))[None, :]
        r_val = np.array([0.001])
        chi = smooth_cutoff(r_val, R)
        assert abs(chi[0] - 1.0) < 1e-12, f"chi={chi[0]:.6e} at r=0.001, R={R}"

        # sigma_s with and without cutoff must agree when chi = 1
        s_direct = sigma_s_at_points(omega, v, near_pt)[0]
        s_cutoff = s_direct * chi[0]
        assert abs(s_cutoff - s_direct) < 1e-14
