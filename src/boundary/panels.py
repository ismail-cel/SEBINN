"""
Uniform panel discretization of a polygon boundary.

MATLAB reference
----------------
build_uniform_panels              lines 320-336
corner_graded_collocation_counts  lines 415-437
build_panel_loss_weights          lines 439-444
global_arclength_coordinate       lines 1566-1581

Design notes
------------
- A Panel stores the two endpoint coordinates, length, and panel index.
- Corner/ring flags are set by label_corner_ring_panels(), which requires the
  polygon vertices to identify which panel endpoints coincide with a vertex.
- panel_loss_weights() returns a plain ndarray, matching MATLAB's wPanel vector.
- global_arclength() converts (panel_id, s_on_panel) to a boundary-wide
  arclength coordinate, used for plotting sigma vs. s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .polygon import PolygonGeometry


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class Panel:
    """
    One straight panel on partial Omega.

    Attributes
    ----------
    a, b : ndarray, shape (2,)
        Start and end coordinates.
    length : float
        Euclidean length |b - a|.
    panel_id : int
        0-indexed position in the global panel list.
    is_corner : bool
        True if either endpoint coincides with a polygon vertex.
        MATLAB: isCornerPanel
    is_ring : bool
        True if adjacent to a corner panel but not itself a corner panel.
        MATLAB: isRingPanel
    """

    a: np.ndarray = field(repr=False)
    b: np.ndarray = field(repr=False)
    length: float
    panel_id: int
    is_corner: bool = False
    is_ring: bool = False


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def build_graded_panels(
    P: np.ndarray,
    n_per_edge: int,
    graded_edge_configs: list,
) -> List[Panel]:
    """
    Build panels with algebraic (power-law) grading near corners.

    Non-graded edges get n_per_edge uniform panels.
    Graded edges get panels whose lengths grow algebraically away from the
    specified corner endpoint.

    Algebraic grading rule on an edge of length L with n panels and exponent β:

        s_j = L * (j/n)^β   for corner at start (j=0 is corner end)
        s_j = L * (1 - ((n-j)/n)^β)   for corner at end

    β = 1 gives uniform, β > 1 grades toward the corner.
    Optimal for exponent α singularity: β = 1/|α| = 3 for the L-shape.
    Practical values: β = 2 (moderate) to β = 3 (strong).

    Parameters
    ----------
    P : ndarray, shape (N, 2)
        Polygon vertices (open, last != first).
    n_per_edge : int
        Number of panels for non-graded edges.
    graded_edge_configs : list of (edge_idx, n_graded, corner_at_end, beta)
        edge_idx       : 0-indexed edge number.  Edge e: P[e] → P[(e+1)%N].
        n_graded       : number of panels on this edge.
        corner_at_end  : True if the corner is at b_edge (end), False if at a_edge.
        beta           : algebraic grading exponent (> 1 for grading).

    Returns
    -------
    panels : list of Panel
    """
    N = len(P)
    panels: List[Panel] = []
    pid = 0

    graded_map = {cfg[0]: cfg[1:] for cfg in graded_edge_configs}

    for e in range(N):
        a_edge = P[e]
        b_edge = P[(e + 1) % N]
        L_edge = float(np.linalg.norm(b_edge - a_edge))

        if e in graded_map:
            n_gr, corner_at_end, beta = graded_map[e]
            idx = np.arange(n_gr + 1, dtype=float)
            if not corner_at_end:
                # corner at start: finest panels at j=0
                t_bounds = L_edge * (idx / n_gr) ** beta
            else:
                # corner at end: finest panels at j=n_gr
                t_bounds = L_edge * (1.0 - ((n_gr - idx) / n_gr) ** beta)
            t_bounds[0]  = 0.0
            t_bounds[-1] = L_edge
            n = n_gr
        else:
            t_bounds = np.linspace(0.0, L_edge, n_per_edge + 1)
            n = n_per_edge

        direction = (b_edge - a_edge) / max(L_edge, 1e-300)

        for j in range(n):
            a = a_edge + direction * t_bounds[j]
            b = a_edge + direction * t_bounds[j + 1]
            length = float(np.linalg.norm(b - a))
            panels.append(Panel(a=a.copy(), b=b.copy(), length=length, panel_id=pid))
            pid += 1

    return panels


def build_uniform_panels(P: np.ndarray, n_per_edge: int) -> List[Panel]:
    """
    Subdivide each edge of polygon P into n_per_edge uniform panels.

    MATLAB: build_uniform_panels (lines 320-336).

    Parameters
    ----------
    P : ndarray, shape (N, 2)
        Polygon vertices, NOT closed.
    n_per_edge : int
        Number of panels per polygon edge.  MATLAB: cfg.NpEdge = 12.

    Returns
    -------
    panels : list of Panel, length N * n_per_edge
        Ordered edge-by-edge, sub-panel-by-sub-panel.
    """
    N = len(P)
    panels: List[Panel] = []
    pid = 0

    for e in range(N):
        a_edge = P[e]
        b_edge = P[(e + 1) % N]

        for j in range(n_per_edge):
            t0 = j / n_per_edge
            t1 = (j + 1) / n_per_edge
            a = (1.0 - t0) * a_edge + t0 * b_edge
            b = (1.0 - t1) * a_edge + t1 * b_edge
            length = float(np.linalg.norm(b - a))
            panels.append(Panel(a=a.copy(), b=b.copy(), length=length, panel_id=pid))
            pid += 1

    return panels


# ---------------------------------------------------------------------------
# Corner / ring classification
# ---------------------------------------------------------------------------

def label_corner_ring_panels(
    panels: List[Panel],
    P: np.ndarray,
    tol: float = 1e-10,
) -> None:
    """
    Mark each panel as corner or ring in-place.

    A panel is a *corner panel* if either endpoint is within tol of any
    polygon vertex.  A panel is a *ring panel* if it is not a corner panel
    but is adjacent (in the panel list) to a corner panel.

    MATLAB: corner_graded_collocation_counts (lines 415-437).

    Parameters
    ----------
    panels : list of Panel
        Modified in-place.
    P : ndarray, shape (N, 2)
        Polygon vertices.
    tol : float
        Base coincidence tolerance. Scaled by max(1, max|P|) as in MATLAB.
    """
    scale = max(1.0, float(np.max(np.abs(P))))
    tol_scaled = tol * scale
    n_pan = len(panels)

    # --- corner panels ---
    for pan in panels:
        d_a = float(np.min(np.linalg.norm(P - pan.a, axis=1)))
        d_b = float(np.min(np.linalg.norm(P - pan.b, axis=1)))
        pan.is_corner = (d_a < tol_scaled) or (d_b < tol_scaled)

    # --- ring panels: adjacent to a corner panel but not a corner panel ---
    is_corner = np.array([p.is_corner for p in panels], dtype=bool)
    # circshift(isCornerPanel, 1) | circshift(isCornerPanel,-1)
    is_ring = (np.roll(is_corner, 1) | np.roll(is_corner, -1)) & ~is_corner

    for i, pan in enumerate(panels):
        pan.is_ring = bool(is_ring[i])


# ---------------------------------------------------------------------------
# Loss weights
# ---------------------------------------------------------------------------

def panel_loss_weights(
    panels: List[Panel],
    w_base: float = 1.0,
    w_corner: float = 1.0,
    w_ring: float = 1.0,
) -> np.ndarray:
    """
    Build per-panel loss weights vector.

    MATLAB: build_panel_loss_weights (lines 439-444).

    Parameters
    ----------
    panels : list of Panel
        label_corner_ring_panels must have been called first.
    w_base, w_corner, w_ring : float
        Weights for base / corner / ring panels.
        MATLAB defaults: cfg.wBase=1.0, cfg.wCorner=1.0, cfg.wRing=1.0.

    Returns
    -------
    weights : ndarray, shape (n_panels,)
        Entry i is the loss weight for panels[i].
    """
    weights = np.full(len(panels), w_base, dtype=float)
    for i, pan in enumerate(panels):
        if pan.is_corner:
            weights[i] = w_corner
        elif pan.is_ring:
            weights[i] = w_ring
    return weights


# ---------------------------------------------------------------------------
# Global arclength coordinate
# ---------------------------------------------------------------------------

def global_arclength(
    panels: List[Panel],
    pan_id: np.ndarray,
    s_on_panel: np.ndarray,
) -> np.ndarray:
    """
    Convert (panel index, local arc coordinate) to global boundary arclength.

    MATLAB: global_arclength_coordinate (lines 1566-1581).

    Parameters
    ----------
    panels : list of Panel
    pan_id : ndarray of int, shape (Nq,)
        0-indexed panel index for each quadrature/collocation point.
    s_on_panel : ndarray, shape (Nq,)
        Local arclength measured from the panel start point (0 <= s <= L).

    Returns
    -------
    s_global : ndarray, shape (Nq,)
        Global arclength measured from the start of panels[0].
    """
    lengths = np.array([p.length for p in panels])
    # Cumulative offsets: offset[k] = total arclength before panel k
    offsets = np.concatenate([[0.0], np.cumsum(lengths[:-1])])
    return offsets[pan_id] + s_on_panel


# ---------------------------------------------------------------------------
# Convenience: build and label in one call
# ---------------------------------------------------------------------------

def discretize_boundary(
    geom: PolygonGeometry,
    n_per_edge: int,
    w_base: float = 1.0,
    w_corner: float = 1.0,
    w_ring: float = 1.0,
) -> tuple[List[Panel], np.ndarray]:
    """
    Full boundary discretization: panels + loss weights.

    Parameters
    ----------
    geom : PolygonGeometry
    n_per_edge : int
    w_base, w_corner, w_ring : float

    Returns
    -------
    panels : list of Panel
    weights : ndarray, shape (n_panels,)
    """
    panels = build_uniform_panels(geom.vertices, n_per_edge)
    label_corner_ring_panels(panels, geom.vertices)
    weights = panel_loss_weights(panels, w_base, w_corner, w_ring)
    return panels, weights
