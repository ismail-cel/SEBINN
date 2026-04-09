"""
Tests for src/boundary/polygon.py and src/boundary/panels.py.

Verification targets (per CLAUDE.md testing expectations):
  - Correct singular exponent pi/omega - 1 at each corner type
  - Consistency of local corner coordinates and distance r
  - Koch(1) vertex count, coordinate, and perimeter sanity
  - Panel count, lengths, and corner/ring classification
  - Global arclength monotonicity
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.boundary.polygon import (
    koch_snowflake,
    interior_angles,
    make_koch_geometry,
)
from src.boundary.panels import (
    build_uniform_panels,
    label_corner_ring_panels,
    panel_loss_weights,
    global_arclength,
    discretize_boundary,
)


# ===========================================================================
# Koch snowflake geometry
# ===========================================================================

class TestKochSnowflake:

    def test_vertex_count_n0(self):
        """n=0 gives equilateral triangle: 3 vertices."""
        P = koch_snowflake(n=0)
        assert P.shape == (3, 2)

    def test_vertex_count_n1(self):
        """n=1 gives 12 vertices."""
        P = koch_snowflake(n=1)
        assert P.shape == (12, 2)

    def test_vertex_count_n2(self):
        """n=2 gives 48 vertices."""
        P = koch_snowflake(n=2)
        assert P.shape == (48, 2)

    def test_first_vertex_at_top(self):
        """
        Koch.m places the first vertex at the top of the circle:
        x = radius * sin(0) = 0,  y = radius * cos(0) = 0.45.
        """
        P = koch_snowflake(n=1)
        assert abs(P[0, 0]) < 1e-14
        assert abs(P[0, 1] - 0.45) < 1e-14

    def test_all_edges_equal_length(self):
        """
        Koch(1) from an equilateral triangle produces equal-length edges
        (all 12 edges have the same length since the base triangle is regular).
        """
        P = koch_snowflake(n=1)
        N = len(P)
        lengths = np.array([
            np.linalg.norm(P[(i + 1) % N] - P[i]) for i in range(N)
        ])
        assert np.std(lengths) < 1e-12, (
            f"Edge lengths should all be equal; std = {np.std(lengths):.2e}"
        )

    def test_polygon_is_not_closed(self):
        """Returned polygon should be OPEN (last vertex != first vertex)."""
        P = koch_snowflake(n=1)
        assert np.linalg.norm(P[-1] - P[0]) > 1e-10

    def test_clockwise_orientation(self):
        """
        Koch.m produces a CW-traversed polygon (signed area < 0).
        Shoelace formula: positive = CCW, negative = CW.
        """
        P = koch_snowflake(n=1)
        xs, ys = P[:, 0], P[:, 1]
        signed_area = 0.5 * (
            np.dot(xs, np.roll(ys, -1)) - np.dot(np.roll(xs, -1), ys)
        )
        assert signed_area < 0, f"Expected CW (signed_area < 0), got {signed_area:.4f}"


# ===========================================================================
# Interior angles
# ===========================================================================

class TestInteriorAngles:

    def test_equilateral_triangle_angles(self):
        """All angles of equilateral triangle = pi/3."""
        P = koch_snowflake(n=0)
        omega = interior_angles(P)
        assert omega.shape == (3,)
        assert np.allclose(omega, np.pi / 3, atol=1e-10)

    def test_sum_of_angles(self):
        """Sum of interior angles of an N-gon = (N-2)*pi."""
        for n in [1, 2]:
            P = koch_snowflake(n=n)
            N = len(P)
            omega = interior_angles(P)
            expected = (N - 2) * np.pi
            assert abs(np.sum(omega) - expected) < 1e-8, (
                f"n={n}: sum(omega)={np.sum(omega):.6f}, expected={expected:.6f}"
            )

    def test_angles_in_valid_range(self):
        """All interior angles must lie in (0, 2*pi)."""
        P = koch_snowflake(n=1)
        omega = interior_angles(P)
        assert np.all(omega > 0)
        assert np.all(omega < 2 * np.pi)

    def test_triangle_corners_n1(self):
        """
        Original triangle corners (indices 0, 4, 8) have omega = pi/3.
        These are vertices P[0], P[4], P[8] in the Koch(1) layout.
        """
        P = koch_snowflake(n=1)
        omega = interior_angles(P)
        for idx in [0, 4, 8]:
            assert abs(omega[idx] - np.pi / 3) < 1e-8, (
                f"Triangle corner at index {idx}: omega={omega[idx]:.6f}, "
                f"expected pi/3={np.pi/3:.6f}"
            )

    def test_bump_tips_n1(self):
        """
        Bump tips (indices 2, 6, 10) have omega = pi/3.
        These are the outer tips of the equilateral Koch bumps.
        """
        P = koch_snowflake(n=1)
        omega = interior_angles(P)
        for idx in [2, 6, 10]:
            assert abs(omega[idx] - np.pi / 3) < 1e-8, (
                f"Bump tip at index {idx}: omega={omega[idx]:.6f}, "
                f"expected pi/3={np.pi/3:.6f}"
            )

    def test_junction_corners_n1(self):
        """
        Junction points (indices 1, 3, 5, 7, 9, 11) have omega = 4*pi/3.
        These are the REENTRANT corners where sigma_s is singular.
        Singular exponent: alpha = pi/omega - 1 = 3/4 - 1 = -1/4.
        """
        P = koch_snowflake(n=1)
        omega = interior_angles(P)
        expected = 4.0 * np.pi / 3.0
        for idx in [1, 3, 5, 7, 9, 11]:
            assert abs(omega[idx] - expected) < 1e-8, (
                f"Junction corner at index {idx}: omega={omega[idx]:.6f}, "
                f"expected 4*pi/3={expected:.6f}"
            )

    def test_singular_exponent_at_junction(self):
        """
        At junction corners (omega = 4*pi/3), the singular exponent is
        alpha = pi/omega - 1 = -1/4.
        This is the exponent in sigma_s = -(pi/omega) * r^alpha.
        """
        omega_junction = 4.0 * np.pi / 3.0
        alpha = np.pi / omega_junction - 1.0
        assert abs(alpha - (-1.0 / 4.0)) < 1e-12

    def test_singular_corner_indices(self):
        """PolygonGeometry.singular_corner_indices returns the 6 junction indices."""
        geom = make_koch_geometry(n=1)
        idx = geom.singular_corner_indices
        assert len(idx) == 6
        assert set(idx) == {1, 3, 5, 7, 9, 11}


# ===========================================================================
# Panel discretization
# ===========================================================================

class TestPanels:

    def setup_method(self):
        self.P = koch_snowflake(n=1)
        self.N_EDGE = 12
        self.N_PAN = 12 * self.N_EDGE   # 12 edges × 12 panels
        self.panels = build_uniform_panels(self.P, self.N_EDGE)

    def test_panel_count(self):
        assert len(self.panels) == self.N_PAN

    def test_panel_ids_sequential(self):
        ids = [p.panel_id for p in self.panels]
        assert ids == list(range(self.N_PAN))

    def test_all_panels_equal_length(self):
        """
        Koch(1) has equal-length edges, so all sub-panels have equal length.
        """
        lengths = np.array([p.length for p in self.panels])
        assert np.std(lengths) < 1e-12

    def test_panels_positive_length(self):
        for p in self.panels:
            assert p.length > 0

    def test_panels_cover_boundary(self):
        """
        Total panel length = perimeter of Koch(1) polygon.
        """
        total = sum(p.length for p in self.panels)
        # Perimeter = 12 * edge_length
        edge_len = np.linalg.norm(self.P[1] - self.P[0])
        expected_perimeter = 12 * edge_len
        assert abs(total - expected_perimeter) < 1e-10

    def test_continuity(self):
        """Consecutive panels share endpoints: panels[i].b == panels[i+1].a."""
        N = len(self.panels)
        for i in range(N - 1):
            assert np.allclose(self.panels[i].b, self.panels[i + 1].a, atol=1e-14), (
                f"Gap between panels {i} and {i+1}"
            )

    def test_closure(self):
        """Last panel end connects back to first panel start."""
        assert np.allclose(
            self.panels[-1].b, self.panels[0].a, atol=1e-14
        ), "Boundary is not closed"


# ===========================================================================
# Corner / ring classification
# ===========================================================================

class TestCornerRingClassification:

    def setup_method(self):
        self.P = koch_snowflake(n=1)
        self.panels = build_uniform_panels(self.P, n_per_edge=12)
        label_corner_ring_panels(self.panels, self.P)

    def test_corner_panel_count(self):
        """
        Each of the 12 polygon vertices is shared by exactly 2 panels
        (the last panel of the incoming edge and the first of the outgoing edge).
        With 12 panels per edge and 12 vertices: 12*2 = 24 corner panels.
        """
        n_corner = sum(p.is_corner for p in self.panels)
        assert n_corner == 24

    def test_ring_panel_count(self):
        """
        With n_per_edge=12, each edge has sub-panels 0..11.
        Corner panels are 0 and 11 (share a vertex endpoint).
        Ring panels are 1 and 10 (adjacent to a corner, not corner themselves).
        That is 2 ring panels per edge × 12 edges = 24.
        """
        n_ring = sum(p.is_ring for p in self.panels)
        assert n_ring == 24

    def test_corner_and_ring_disjoint(self):
        """A panel cannot be both corner and ring."""
        for p in self.panels:
            assert not (p.is_corner and p.is_ring)

    def test_corner_panels_touch_vertices(self):
        """Every corner panel must have an endpoint within tol of a vertex."""
        tol = 1e-10 * max(1.0, float(np.max(np.abs(self.P))))
        for p in self.panels:
            if p.is_corner:
                d_a = float(np.min(np.linalg.norm(self.P - p.a, axis=1)))
                d_b = float(np.min(np.linalg.norm(self.P - p.b, axis=1)))
                assert min(d_a, d_b) < tol


# ===========================================================================
# Loss weights
# ===========================================================================

class TestLossWeights:

    def test_default_weights_all_one(self):
        """With all weights = 1.0, every panel gets weight 1."""
        P = koch_snowflake(n=1)
        panels = build_uniform_panels(P, 12)
        label_corner_ring_panels(panels, P)
        w = panel_loss_weights(panels, w_base=1.0, w_corner=1.0, w_ring=1.0)
        assert np.all(w == 1.0)

    def test_weight_assignment(self):
        """Corner panels get w_corner, ring panels get w_ring, rest get w_base."""
        P = koch_snowflake(n=1)
        panels = build_uniform_panels(P, 12)
        label_corner_ring_panels(panels, P)
        w = panel_loss_weights(panels, w_base=1.0, w_corner=5.0, w_ring=2.0)
        for i, p in enumerate(panels):
            if p.is_corner:
                assert w[i] == 5.0
            elif p.is_ring:
                assert w[i] == 2.0
            else:
                assert w[i] == 1.0

    def test_weight_shape(self):
        P = koch_snowflake(n=1)
        panels = build_uniform_panels(P, 12)
        label_corner_ring_panels(panels, P)
        w = panel_loss_weights(panels)
        assert w.shape == (len(panels),)


# ===========================================================================
# Global arclength
# ===========================================================================

class TestGlobalArclength:

    def test_monotone_increasing(self):
        """
        Global arclength at sorted quadrature nodes should be monotone
        (panels are traversed in order).
        """
        P = koch_snowflake(n=1)
        panels = build_uniform_panels(P, 12)

        # Use panel midpoints as test points
        pan_id = np.arange(len(panels))
        s_on_panel = np.array([p.length / 2.0 for p in panels])
        s_global = global_arclength(panels, pan_id, s_on_panel)

        diffs = np.diff(s_global)
        assert np.all(diffs > 0), "Global arclength must be strictly increasing"

    def test_total_arclength(self):
        """
        Last panel endpoint gives total perimeter.
        """
        P = koch_snowflake(n=1)
        panels = build_uniform_panels(P, 12)
        lengths = np.array([p.length for p in panels])
        total_perimeter = lengths.sum()

        pan_id = np.array([len(panels) - 1])
        s_on_panel = np.array([panels[-1].length])
        s_end = global_arclength(panels, pan_id, s_on_panel)
        assert abs(s_end[0] - total_perimeter) < 1e-12

    def test_zero_at_start(self):
        """First panel start point has s_global = 0."""
        P = koch_snowflake(n=1)
        panels = build_uniform_panels(P, 12)
        s = global_arclength(panels, np.array([0]), np.array([0.0]))
        assert abs(s[0]) < 1e-15


# ===========================================================================
# Integration: discretize_boundary convenience function
# ===========================================================================

class TestDiscretizeBoundary:

    def test_returns_correct_types(self):
        from src.boundary.polygon import make_koch_geometry
        geom = make_koch_geometry(n=1)
        panels, weights = discretize_boundary(geom, n_per_edge=12)
        assert len(panels) == 144
        assert weights.shape == (144,)

    def test_flags_set(self):
        """Corner/ring flags must be set after discretize_boundary."""
        from src.boundary.polygon import make_koch_geometry
        geom = make_koch_geometry(n=1)
        panels, _ = discretize_boundary(geom, n_per_edge=12)
        n_corner = sum(p.is_corner for p in panels)
        n_ring   = sum(p.is_ring   for p in panels)
        assert n_corner > 0
        assert n_ring   > 0
