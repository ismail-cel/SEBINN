"""
L-shaped domain polygon and corner geometry.

Domain definition
-----------------
The L-shaped domain is [-1,1]² with the upper-right quadrant [0,1]×[0,1]
removed:

    (-1,1) ── (0,1)
      |           |
    (-1,-1) ─ (1,-1) ─ (1,0) ─ (0,0)
                                   |
                                 (missing quadrant)

Vertices (CCW, open polygon):
    0: (-1, -1)
    1: ( 1, -1)
    2: ( 1,  0)
    3: ( 0,  0)   ← reentrant corner, ω = 3π/2
    4: ( 0,  1)
    5: (-1,  1)

Interior angles:
    ω₀ = ω₁ = ω₂ = ω₄ = ω₅ = π/2   (convex right angles)
    ω₃ = 3π/2                         (reentrant corner at origin)

Singular structure:
    Only vertex 3 = (0,0) is singular.
    ω = 3π/2 → α = π/ω − 1 = 2/3 − 1 = −1/3.
    This exponent is STRONGER than Koch(1)'s −1/4:
        σ_s = −(2/3) r^{−1/3}   vs   Koch: σ_s = −(3/4) r^{−1/4}
    The spike at (0,0) is sharper, making the singular enrichment more
    important for accurate approximation.

Perimeter breakdown:
    Edge 0 (-1,-1)→(1,-1) : length 2
    Edge 1 (1,-1)→(1,0)   : length 1
    Edge 2 (1,0)→(0,0)    : length 1
    Edge 3 (0,0)→(0,1)    : length 1
    Edge 4 (0,1)→(-1,1)   : length 1
    Edge 5 (-1,1)→(-1,-1) : length 2
    Total : 8
"""

from __future__ import annotations

import numpy as np

from .polygon import PolygonGeometry, interior_angles


# ---------------------------------------------------------------------------
# Vertices
# ---------------------------------------------------------------------------

_LSHAPE_VERTICES = np.array([
    [-1.0, -1.0],   # 0 — bottom-left (π/2)
    [ 1.0, -1.0],   # 1 — bottom-right (π/2)
    [ 1.0,  0.0],   # 2 — right notch (π/2)
    [ 0.0,  0.0],   # 3 — REENTRANT corner (3π/2)
    [ 0.0,  1.0],   # 4 — top notch (π/2)
    [-1.0,  1.0],   # 5 — top-left (π/2)
], dtype=float)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def make_lshape_geometry() -> PolygonGeometry:
    """
    Build the L-shaped domain polygon with precomputed corner angles.

    Returns
    -------
    geom : PolygonGeometry
        vertices       : (6, 2) — the 6 L-shape vertices
        corner_angles  : (6,)  — interior angles in radians
            indices 0,1,2,4,5 → π/2  (convex)
            index 3           → 3π/2 (reentrant, singular)
        singular_corner_indices : [3]

    Notes
    -----
    The vertices are ordered CCW (positive signed area).  This is required
    by the interior_angles() function, which detects orientation via the
    shoelace formula.

    The expected corner angles are:
        [π/2, π/2, π/2, 3π/2, π/2, π/2]

    Verification:
        >>> geom = make_lshape_geometry()
        >>> np.allclose(geom.corner_angles / np.pi,
        ...             [0.5, 0.5, 0.5, 1.5, 0.5, 0.5])
        True
        >>> geom.singular_corner_indices
        array([3])
    """
    P = _LSHAPE_VERTICES.copy()
    omega = interior_angles(P)
    return PolygonGeometry(vertices=P, corner_angles=omega)
