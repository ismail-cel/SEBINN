"""
Tests for src/reconstruction/interior.py.

Verification targets:
  - points_inside_polygon: basic winding-number / ray-cast correctness
  - reconstruct_interior: shape of outputs, NaN pattern, error metrics
  - Single-layer representation: for known σ the BIE single-layer potential
    reproduces u_exact on a unit-square domain to within quadrature accuracy
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.reconstruction.interior import (
    InteriorResult,
    points_inside_polygon,
    reconstruct_interior,
)


# ---------------------------------------------------------------------------
# Helper polygons
# ---------------------------------------------------------------------------

def _unit_square():
    """Open unit square vertices [0,1]² CCW."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def _unit_square_grid(n=20):
    """Grid covering [0,1]²."""
    return (-0.1, 1.1), (-0.1, 1.1), n


# ===========================================================================
# points_inside_polygon
# ===========================================================================

class TestPointsInsidePolygon:

    def test_centre_inside_square(self):
        P = _unit_square()
        pts = np.array([[0.5, 0.5]])
        assert points_inside_polygon(pts, P)[0]

    def test_outside_square(self):
        P = _unit_square()
        pts = np.array([[2.0, 2.0], [-0.5, 0.5]])
        assert not any(points_inside_polygon(pts, P))

    def test_multiple_inside(self):
        P = _unit_square()
        pts = np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])
        assert all(points_inside_polygon(pts, P))

    def test_mixed_inside_outside(self):
        P = _unit_square()
        pts = np.array([[0.5, 0.5], [2.0, 2.0], [0.1, 0.9]])
        inside = points_inside_polygon(pts, P)
        assert inside[0]
        assert not inside[1]
        assert inside[2]

    def test_empty_query(self):
        P = _unit_square()
        pts = np.zeros((0, 2))
        result = points_inside_polygon(pts, P)
        assert result.shape == (0,)

    def test_triangle(self):
        """Points inside / outside a right triangle."""
        P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        inside_pt  = np.array([[0.2, 0.2]])
        outside_pt = np.array([[0.7, 0.7]])  # above hypotenuse
        assert points_inside_polygon(inside_pt, P)[0]
        assert not points_inside_polygon(outside_pt, P)[0]

    def test_unit_circle_approx(self):
        """
        Approximate a circle with 128 vertices; a point at r=0.4 should be
        inside and a point at r=1.2 should be outside.
        """
        theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        P = np.column_stack([np.cos(theta), np.sin(theta)])
        inner = np.array([[0.4, 0.0]])
        outer = np.array([[1.2, 0.0]])
        assert points_inside_polygon(inner, P)[0]
        assert not points_inside_polygon(outer, P)[0]

    def test_returns_bool_dtype(self):
        P = _unit_square()
        pts = np.array([[0.5, 0.5]])
        assert points_inside_polygon(pts, P).dtype == bool

    def test_output_shape(self):
        P = _unit_square()
        pts = np.random.default_rng(0).uniform(0, 1, (50, 2))
        result = points_inside_polygon(pts, P)
        assert result.shape == (50,)


# ===========================================================================
# reconstruct_interior — output structure
# ===========================================================================

class TestReconstructInteriorOutputs:

    def _run(self, n_grid=10, with_exact=False):
        P  = _unit_square()
        Nq = 20
        rng = np.random.default_rng(0)
        Yq  = np.column_stack([
            np.linspace(0, 1, Nq),
            np.zeros(Nq),
        ])
        wq    = np.ones(Nq) / Nq
        sigma = rng.standard_normal(Nq)

        if with_exact:
            u_exact = lambda xy: np.ones(len(xy))
        else:
            u_exact = None

        return reconstruct_interior(
            P, Yq, wq, sigma,
            n_grid=n_grid,
            u_exact=u_exact,
            x_range=(-0.1, 1.1),
            y_range=(-0.1, 1.1),
        )

    def test_returns_interior_result(self):
        res = self._run()
        assert isinstance(res, InteriorResult)

    def test_grid_shape(self):
        res = self._run(n_grid=15)
        assert res.Ugrid.shape == (15, 15)
        assert len(res.xv) == 15
        assert len(res.yv) == 15

    def test_nan_outside_polygon(self):
        """Ugrid must be NaN everywhere outside the polygon."""
        P  = _unit_square()
        Nq = 20
        Yq = np.column_stack([np.linspace(0, 1, Nq), np.zeros(Nq)])
        wq = np.ones(Nq) / Nq
        sigma = np.ones(Nq)
        res = reconstruct_interior(
            P, Yq, wq, sigma, n_grid=30,
            x_range=(0.0, 1.0), y_range=(0.0, 1.0),
        )
        # All NaN-free entries must correspond to interior points
        valid_mask = ~np.isnan(res.Ugrid)
        assert valid_mask.sum() == res.n_interior

    def test_n_interior_positive(self):
        res = self._run(n_grid=20)
        assert res.n_interior > 0

    def test_uvals_length_matches_n_interior(self):
        res = self._run(n_grid=20)
        assert len(res.Uvals) == res.n_interior

    def test_error_fields_none_without_exact(self):
        res = self._run(with_exact=False)
        assert res.Uex     is None
        assert res.Uexgrid is None
        assert res.Egrid   is None
        assert res.rel_L2  is None
        assert res.linf    is None

    def test_error_fields_populated_with_exact(self):
        res = self._run(with_exact=True)
        assert res.Uex     is not None
        assert res.Uexgrid is not None
        assert res.Egrid   is not None
        assert res.rel_L2  is not None
        assert res.linf    is not None

    def test_egrid_is_difference(self):
        """Egrid must equal Ugrid - Uexgrid wherever both are non-NaN."""
        res = self._run(n_grid=15, with_exact=True)
        mask = ~np.isnan(res.Ugrid)
        diff = res.Ugrid[mask] - res.Uexgrid[mask] - res.Egrid[mask]
        assert np.allclose(diff, 0, atol=1e-14)

    def test_uexgrid_nan_pattern_matches_ugrid(self):
        res = self._run(n_grid=20, with_exact=True)
        assert np.array_equal(
            np.isnan(res.Ugrid), np.isnan(res.Uexgrid)
        )

    def test_all_interior_values_finite(self):
        res = self._run(n_grid=20)
        assert np.all(np.isfinite(res.Uvals))


# ===========================================================================
# Numerical accuracy: u_exact = x² − y² (harmonic)
# ===========================================================================

class TestSingleLayerAccuracy:
    """
    Verify that the single-layer representation reproduces u_exact = x² − y²
    on a coarse Koch(1)-based quadrature.

    We use the BEM σ obtained by solving the Nyström system (which gives the
    exact density up to discretisation error) and check that the interior
    reconstruction is accurate.
    """

    def test_quadratic_on_square_domain(self):
        """
        On the unit square with u_exact(x,y) = x²−y², the BEM single-layer
        density σ_BEM can be obtained by solving A σ = f.  Here we bypass
        that by using the KNOWN analytic solution u(x,y) = x²−y² and
        checking that a dense Gauss quadrature on the square boundary plus
        the correct σ reproduces u_exact at interior points.

        For this test we use the Nyström solver directly on the square.
        """
        from src.boundary.panels import build_uniform_panels
        from src.boundary.polygon import PolygonGeometry, interior_angles
        from src.quadrature.panel_quad import build_panel_quadrature
        from src.quadrature.nystrom import assemble_nystrom_matrix, solve_bem

        # Unit square (open, CCW)
        P_sq = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]])
        panels = build_uniform_panels(P_sq, n_per_edge=8)  # 4 edges × 8 = 32 panels

        # Quadrature
        qdata = build_panel_quadrature(panels, p=8)
        Yq_T  = qdata.Yq.T          # (Nq, 2)
        wq    = qdata.wq

        # BEM solve for u_exact = x²−y²
        def u_exact_fn(xy):
            return xy[:, 0]**2 - xy[:, 1]**2

        nmat = assemble_nystrom_matrix(qdata)
        f_bnd = u_exact_fn(Yq_T)
        bem_sol = solve_bem(nmat, f_bnd)
        sigma = bem_sol.sigma

        # Interior reconstruction
        # Query grid: interior of [0,1]²
        res = reconstruct_interior(
            P=P_sq,
            Yq=Yq_T,
            wq=wq,
            sigma=sigma,
            n_grid=15,
            u_exact=u_exact_fn,
            x_range=(0.05, 0.95),
            y_range=(0.05, 0.95),
        )

        assert res.n_interior > 0, "No interior points found"
        assert res.rel_L2 is not None
        # A 32-panel / p=8 Nyström solve should give < 1% relative error
        assert res.rel_L2 < 1e-2, (
            f"rel_L2 error too large: {res.rel_L2:.3e}"
        )

    def test_constant_u_reconstructed(self):
        """
        If u = 1 everywhere (harmonic, trivially), there exists a σ such that
        ∫ G(x,y) σ(y) ds = 1.  Without solving for σ we just verify the
        reconstruction gives finite, consistent values — not correctness.
        """
        P = _unit_square()
        Nq = 40
        theta = np.linspace(0, 2*np.pi, Nq, endpoint=False)
        # Put quadrature on the boundary of the unit square
        # (a simple uniform arc-length parameterisation)
        side = Nq // 4
        t   = np.linspace(0, 1, side, endpoint=False)
        Yq  = np.vstack([
            np.column_stack([t,     np.zeros(side)]),
            np.column_stack([np.ones(side),  t]),
            np.column_stack([1-t,   np.ones(side)]),
            np.column_stack([np.zeros(side), 1-t]),
        ])
        wq    = np.ones(Nq) / Nq
        sigma = np.ones(Nq)

        res = reconstruct_interior(
            P, Yq, wq, sigma, n_grid=10,
            x_range=(0.05, 0.95), y_range=(0.05, 0.95),
        )
        assert np.all(np.isfinite(res.Uvals))
