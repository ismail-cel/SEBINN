"""
Singular boundary density sigma_s for a single corner.

MATLAB counterpart
------------------
None.  This is the mathematical core added by SE-BINN that does not
exist in the MATLAB baseline.

Canonical formula (CLAUDE.md)
------------------------------
For a corner with interior angle omega, the singular density is

    sigma_s(y) = -(pi / omega) * r^(pi/omega - 1)

where r = |y - vertex| is the distance from the corner vertex.

This is the leading-order term in the Mellin expansion of the
boundary density near the corner singularity.

Reference: Hu, Jin, Zhou (2024), SIAM J. Sci. Comput., eq. for sigma_s.

Mathematical notes
------------------
- The exponent alpha = pi/omega - 1 is negative for reentrant corners
  (omega > pi), so sigma_s -> -inf as r -> 0.
- The prefactor -(pi/omega) ensures the correct normalization so that
  the single-layer potential reproduces the correct leading singularity
  in u near the corner.
- For convex corners (omega < pi, alpha > 0), sigma_s -> 0 as r -> 0,
  so no enrichment is needed.  This module is typically called only for
  singular corners.

Design notes
------------
- sigma_s_single takes a CornerCoords object (already has r, omega, alpha)
  so it never recomputes distances.
- sigma_s_at_points is the high-level interface: takes raw coordinates.
- Both return plain ndarrays — no PyTorch tensors here.  Conversion to
  tensors happens in the training operator.
"""

from __future__ import annotations

import numpy as np

from .corner_coords import CornerCoords


def sigma_s_single(cc: CornerCoords) -> np.ndarray:
    """
    Evaluate sigma_s at the points encoded in a CornerCoords object.

    sigma_s(y) = -(pi / omega) * r^alpha,  alpha = pi/omega - 1.

    Parameters
    ----------
    cc : CornerCoords
        Must have cc.r, cc.omega, cc.alpha set (output of corner_local_coords).

    Returns
    -------
    sigma_s : ndarray, shape (N,)
        Singular density values at the N evaluation points.
        Values are negative (the prefactor -(pi/omega) < 0 for all omega > 0).

    Notes
    -----
    For reentrant Koch(1) corners: omega = 4pi/3, alpha = -1/4.
        sigma_s = -(3/4) * r^(-1/4)
    """
    prefactor = -(np.pi / cc.omega)
    return prefactor * cc.r ** cc.alpha


def sigma_s_at_points(
    omega: float,
    vertex: np.ndarray,
    points: np.ndarray,
    r_min: float = 1e-14,
) -> np.ndarray:
    """
    Evaluate sigma_s for a single corner at arbitrary points.

    Convenience wrapper that computes r internally.

    Parameters
    ----------
    omega : float
        Interior angle at the corner (radians).
    vertex : ndarray, shape (2,)
        Corner vertex coordinates.
    points : ndarray, shape (N, 2)
        Evaluation points.
    r_min : float
        Distance floor to prevent r^alpha blowup at the vertex.

    Returns
    -------
    sigma_s : ndarray, shape (N,)
    """
    diff = points - vertex[None, :]
    r = np.linalg.norm(diff, axis=1)
    r = np.maximum(r, r_min)

    alpha = np.pi / omega - 1.0
    prefactor = -(np.pi / omega)
    return prefactor * r ** alpha
