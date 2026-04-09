"""
Corner-local polar coordinates for boundary points.

MATLAB counterpart
------------------
None.  This module has no MATLAB equivalent; it is required by the
SE-BINN enrichment that MATLAB does not implement.

Mathematical role
-----------------
For each singular corner c with vertex position v_c, define the
corner-local distance and angle for any boundary point y:

    r_c(y) = |y - v_c|
    theta_c(y) = atan2(y[1] - v_c[1], y[0] - v_c[0])

r_c is used in sigma_s = -(pi/omega_c) * r_c^(pi/omega_c - 1).
theta_c is stored for completeness (used in higher-order enrichment
and debugging, not in the scalar sigma_s formula at O(1)).

Design notes
------------
- Only the SINGULAR corners (omega > pi) require enrichment; this module
  always receives the full list of PolygonGeometry corners but exposes
  a `singular_only` flag to filter.
- r is clamped to r_min > 0 to avoid log(0) or division by zero when
  evaluating the power r^alpha with alpha < 0.
- Output is a CornerCoords dataclass indexed by corner index c, so
  downstream code can loop over corners or index by c explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..boundary.polygon import PolygonGeometry


_R_MIN = 1e-14   # floor for r_c to prevent r^alpha blowup at vertex


@dataclass
class CornerCoords:
    """
    Corner-local polar coordinates at a set of boundary (or interior) points.

    Attributes
    ----------
    corner_idx : int
        0-based index of the corner vertex in the polygon.
    vertex : ndarray, shape (2,)
        Physical coordinates of the corner vertex.
    omega : float
        Interior angle at this corner (radians).
    alpha : float
        Singular exponent: alpha = pi/omega - 1.
        Negative for reentrant corners (omega > pi).
    r : ndarray, shape (N,)
        Distance from the corner to each evaluation point.
        Clamped to r_min = 1e-14.
    theta : ndarray, shape (N,)
        Angle (radians) from the corner to each evaluation point,
        measured in the global coordinate system.
    """

    corner_idx: int
    vertex: np.ndarray    # (2,)
    omega: float
    alpha: float          # pi/omega - 1
    r: np.ndarray         # (N,)
    theta: np.ndarray     # (N,)

    @property
    def is_singular(self) -> bool:
        """True if omega > pi (reentrant corner, alpha < 0)."""
        return self.omega > np.pi


def corner_local_coords(
    geom: PolygonGeometry,
    points: np.ndarray,
    singular_only: bool = True,
    r_min: float = _R_MIN,
) -> list[CornerCoords]:
    """
    Compute corner-local polar coordinates for a batch of points.

    Parameters
    ----------
    geom : PolygonGeometry
        Polygon with precomputed corner angles.
    points : ndarray, shape (N, 2)
        Evaluation points (boundary quadrature nodes, collocation nodes,
        or interior grid points).
    singular_only : bool
        If True (default), return coordinates only for corners with
        omega > pi (the reentrant, singular corners).
        If False, return for all corners.
    r_min : float
        Minimum value for r_c; prevents r^alpha = inf when alpha < 0.

    Returns
    -------
    coords : list of CornerCoords
        One entry per (selected) corner.  Length equals
        len(geom.singular_corner_indices) if singular_only=True,
        else geom.n_vertices.
    """
    if singular_only:
        corner_indices = geom.singular_corner_indices
    else:
        corner_indices = np.arange(geom.n_vertices)

    coords = []
    for c in corner_indices:
        v = geom.vertices[c]         # (2,)
        omega = float(geom.corner_angles[c])
        alpha = np.pi / omega - 1.0  # singular exponent

        diff = points - v[None, :]   # (N, 2)
        r = np.linalg.norm(diff, axis=1)          # (N,)
        r = np.maximum(r, r_min)
        theta = np.arctan2(diff[:, 1], diff[:, 0])  # (N,)

        coords.append(CornerCoords(
            corner_idx=int(c),
            vertex=v.copy(),
            omega=omega,
            alpha=alpha,
            r=r,
            theta=theta,
        ))

    return coords
