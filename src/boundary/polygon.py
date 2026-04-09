"""
Koch snowflake polygon and corner geometry.

MATLAB reference
----------------
Koch.m                                   : iterative snowflake construction
bem_pinn_nystrom_comparison.m line 63    : P = Koch(1)

Design notes
------------
- Koch(1) is the canonical geometry for all SEBINN experiments.
- Corner angles omega_i are computed here because every downstream module
  that uses sigma_s = -(pi/omega) * r^(pi/omega - 1) needs omega.
- The polygon is stored OPEN (last vertex != first vertex).
- Orientation: Koch.m produces a CW-traversed polygon. The signed-area check
  in interior_angles() handles orientation automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class PolygonGeometry:
    """
    Polygon vertices together with interior corner angles.

    Attributes
    ----------
    vertices : ndarray, shape (N, 2)
        Polygon vertices in traversal order, NOT repeated at close.
    corner_angles : ndarray, shape (N,)
        Interior angle omega_i at each vertex, in radians, omega_i in (0, 2*pi).
        omega_i < pi  : convex corner (no BIE singularity).
        omega_i > pi  : reentrant corner (singular density near vertex).
    """

    vertices: np.ndarray       # (N, 2)
    corner_angles: np.ndarray  # (N,)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def singular_corner_indices(self) -> np.ndarray:
        """Indices of reentrant corners (omega > pi)."""
        return np.where(self.corner_angles > np.pi)[0]


# ---------------------------------------------------------------------------
# Koch snowflake
# ---------------------------------------------------------------------------

def koch_snowflake(n: int = 1) -> np.ndarray:
    """
    Construct Koch snowflake polygon vertices.

    Direct port of Koch.m. Default n=1 gives 12 vertices.

    For n=1 starting from an equilateral triangle:
      - 3 triangle corners      : omega = pi/3
      - 3 bump tips             : omega = pi/3
      - 6 junction points       : omega = 4*pi/3  (reentrant, singular)

    Parameters
    ----------
    n : int
        Koch iterations. n=0 gives equilateral triangle (3 vertices).

    Returns
    -------
    P : ndarray, shape (N, 2)
        Polygon vertices, NOT closed.

    Notes
    -----
    MATLAB Koch.m uses sin/cos axes as:
        x = radius * sin(angle),  y = radius * cos(angle)
    so the first vertex sits at the top of the circle (x=0, y=radius).
    This convention is preserved here to match MATLAB node coordinates exactly.
    """
    radius = 0.45
    init_angles = np.linspace(0.0, 2.0 * np.pi, 4)[:3]  # [0, 2pi/3, 4pi/3]

    # MATLAB: x_vertices = sin, y_vertices = cos  (axis-swapped)
    P = np.column_stack([
        radius * np.sin(init_angles),
        radius * np.cos(init_angles),
    ])  # shape (3, 2), open triangle

    for _ in range(n):
        N = len(P)
        new_verts = []
        for i in range(N):
            a = P[i]
            b = P[(i + 1) % N]
            link = b - a
            ang = np.arctan2(link[1], link[0])
            link_len = np.linalg.norm(link)

            p1   = (2.0 * a + b) / 3.0
            peak = p1 + (link_len / 3.0) * np.array([
                np.cos(ang + np.pi / 3.0),
                np.sin(ang + np.pi / 3.0),
            ])
            p2   = (a + 2.0 * b) / 3.0

            # Each edge a->b becomes four segments: a, p1, peak, p2
            new_verts.extend([a.copy(), p1, peak, p2])

        P = np.array(new_verts)  # shape (4*N, 2)

    return P


# ---------------------------------------------------------------------------
# Interior angle computation
# ---------------------------------------------------------------------------

def interior_angles(P: np.ndarray) -> np.ndarray:
    """
    Compute interior angles omega_i at every vertex of polygon P.

    For vertex i, the interior angle is the angle measured inside the domain,
    between the two edges meeting at P[i].

    Handles both CW and CCW polygon orientations via signed-area detection.

    Parameters
    ----------
    P : ndarray, shape (N, 2)
        Polygon vertices, NOT closed.

    Returns
    -------
    angles : ndarray, shape (N,)
        Interior angles in radians, each in (0, 2*pi).
        omega > pi indicates a reentrant (singular) corner.

    Mathematical note
    -----------------
    Let  u = P[i-1] - P[i]  (vector toward previous vertex),
         v = P[i+1] - P[i]  (vector toward next vertex).
    The angle between u and v, measured inside the polygon, is:

        omega_i = atan2(orient * cross(u, v), dot(u, v))

    where orient = +1 for CCW, -1 for CW.
    If the result is <= 0 it is shifted by 2*pi to land in (0, 2*pi].
    """
    N = len(P)

    # Signed area via shoelace: positive = CCW, negative = CW
    xs, ys = P[:, 0], P[:, 1]
    signed_area = 0.5 * (
        np.dot(xs, np.roll(ys, -1)) - np.dot(np.roll(xs, -1), ys)
    )
    orient = 1.0 if signed_area > 0.0 else -1.0

    angles = np.empty(N)
    for i in range(N):
        u = P[(i - 1) % N] - P[i]  # toward previous
        v = P[(i + 1) % N] - P[i]  # toward next
        cross = u[0] * v[1] - u[1] * v[0]
        dot   = u[0] * v[0] + u[1] * v[1]
        omega = np.arctan2(-orient * cross, dot)
        if omega <= 0.0:
            omega += 2.0 * np.pi
        angles[i] = omega

    return angles


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def make_koch_geometry(n: int = 1) -> PolygonGeometry:
    """
    Build Koch snowflake polygon with precomputed corner angles.

    Parameters
    ----------
    n : int
        Koch iterations (default 1).

    Returns
    -------
    geom : PolygonGeometry
    """
    P = koch_snowflake(n)
    omega = interior_angles(P)
    return PolygonGeometry(vertices=P, corner_angles=omega)
