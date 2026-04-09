"""
Multi-corner singular enrichment: assembles sigma_s over all singular corners.

MATLAB counterpart
------------------
None.  Required by SE-BINN; does not exist in the MATLAB baseline.

Mathematical role
-----------------
The full enriched density ansatz is

    sigma(y) = sigma_w(y) + gamma * sigma_s(y)

where sigma_s aggregates contributions from all singular corners:

    sigma_s(y) = sum_c  chi_c(r_c(y)) * [-(pi/omega_c) * r_c(y)^alpha_c]

chi_c is an optional smooth cutoff function.  For geometries where the
singular corners are well separated (e.g. Koch(1)), chi_c = 1 is safe.

When cutoff is enabled, chi_c(r) = bump(r / R_c) where bump is a C^inf
function that equals 1 near 0 and 0 for r > R_c.

One-gamma vs per-corner-gamma
------------------------------
The simplest SE-BINN formulation uses a single scalar gamma shared across
all singular corners.  This is appropriate when the geometry is symmetric
(e.g. Koch snowflake with equal angles at all 6 reentrant corners).

For asymmetric geometries, each corner can have its own gamma_c, giving
the vector ansatz:

    sigma(y) = sigma_w(y) + sum_c  gamma_c * chi_c * sigma_s^(c)(y)

This module supports both via the `n_gamma` property.

Design notes
------------
- SingularEnrichment is the object passed into the training operator.
  It holds precomputed sigma_s values at fixed node sets (Yq, Xc).
- evaluate() returns the full sigma_s field at arbitrary points (used
  for interior reconstruction and diagnostics).
- precompute() caches the sigma_s values at a fixed point array.
  Call this once per node set; the result is a (N,) numpy array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..boundary.polygon import PolygonGeometry
from .corner_coords import corner_local_coords
from .sigma_s import sigma_s_single


# ---------------------------------------------------------------------------
# Cutoff function
# ---------------------------------------------------------------------------

def smooth_cutoff(r: np.ndarray, R: float) -> np.ndarray:
    """
    C^inf bump: equals 1 for r <= R/2, tapers monotonically to 0 at r = R.

    Construction:
        chi(r) = 1                          for r <= R/2
        chi(r) = exp(-1/u) / (exp(-1/u)    for r in (R/2, R),
                              + exp(-1/(1-u)))
             where u = (r - R/2) / (R/2)  in (0, 1)
        chi(r) = 0                          for r >= R

    This is the standard Hermite/mollifier bump based on phi(t) = exp(-1/t)
    for t > 0.  It is strictly between 0 and 1 on (R/2, R), monotone
    decreasing, and C^inf everywhere.

    Parameters
    ----------
    r : ndarray
        Distance values (>= 0).
    R : float
        Cutoff radius; chi = 0 for r >= R.

    Returns
    -------
    chi : ndarray, same shape as r, values in [0, 1].
    """
    r = np.asarray(r, dtype=float)
    chi = np.zeros_like(r)

    # Region 1: r <= R/2 -> chi = 1
    mask_one = r <= R / 2.0
    chi[mask_one] = 1.0

    # Region 2: R/2 < r < R -> smooth transition
    mask_mid = (r > R / 2.0) & (r < R)
    if np.any(mask_mid):
        u = (r[mask_mid] - R / 2.0) / (R / 2.0)   # u in (0, 1)
        phi_u   = np.exp(-1.0 / u)
        phi_1mu = np.exp(-1.0 / (1.0 - u))
        chi[mask_mid] = phi_1mu / (phi_u + phi_1mu)

    # Region 3: r >= R -> chi = 0 (already zero)
    return chi


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@dataclass
class SingularEnrichment:
    """
    Precomputed singular enrichment object for a fixed geometry.

    Attributes
    ----------
    geom : PolygonGeometry
        Polygon with corner angles.
    use_cutoff : bool
        Whether to apply smooth spatial cutoff chi_c(r).
    cutoff_radius : float or None
        Cutoff radius R_c (same for all corners).  Required if use_cutoff.
    per_corner_gamma : bool
        If True, each singular corner has its own trainable gamma.
        If False (default), one shared gamma is used.
    """

    geom: PolygonGeometry
    use_cutoff: bool = False
    cutoff_radius: Optional[float] = None
    per_corner_gamma: bool = False

    @property
    def singular_indices(self) -> np.ndarray:
        return self.geom.singular_corner_indices

    @property
    def n_singular(self) -> int:
        return len(self.singular_indices)

    @property
    def n_gamma(self) -> int:
        """Number of trainable gamma parameters."""
        return self.n_singular if self.per_corner_gamma else 1

    def evaluate(
        self,
        points: np.ndarray,
        r_min: float = 1e-14,
    ) -> np.ndarray:
        """
        Evaluate the full sigma_s field at arbitrary points.

        sigma_s(y) = sum_c chi_c(r_c) * [-(pi/omega_c) * r_c^alpha_c]

        Parameters
        ----------
        points : ndarray, shape (N, 2)
        r_min : float

        Returns
        -------
        sigma_s : ndarray, shape (N,)
            Sum over all singular corners.  This is the field multiplied
            by gamma to get the singular contribution to sigma.
        """
        coords_list = corner_local_coords(
            self.geom, points, singular_only=True, r_min=r_min
        )
        result = np.zeros(len(points))
        for cc in coords_list:
            contrib = sigma_s_single(cc)           # (N,)
            if self.use_cutoff and self.cutoff_radius is not None:
                chi = smooth_cutoff(cc.r, self.cutoff_radius)
                contrib = contrib * chi
            result += contrib
        return result

    def evaluate_per_corner(
        self,
        points: np.ndarray,
        r_min: float = 1e-14,
    ) -> np.ndarray:
        """
        Evaluate sigma_s^(c) separately for each singular corner.

        Used when per_corner_gamma=True: the training operator forms
        sum_c gamma_c * sigma_s^(c)(y) rather than gamma * sigma_s(y).

        Parameters
        ----------
        points : ndarray, shape (N, 2)

        Returns
        -------
        sigma_s_per_corner : ndarray, shape (N, n_singular)
            Column c contains the contribution from singular corner c.
        """
        coords_list = corner_local_coords(
            self.geom, points, singular_only=True, r_min=r_min
        )
        out = np.empty((len(points), self.n_singular))
        for k, cc in enumerate(coords_list):
            contrib = sigma_s_single(cc)
            if self.use_cutoff and self.cutoff_radius is not None:
                chi = smooth_cutoff(cc.r, self.cutoff_radius)
                contrib = contrib * chi
            out[:, k] = contrib
        return out

    def precompute(
        self,
        points: np.ndarray,
        r_min: float = 1e-14,
    ) -> np.ndarray:
        """
        Precompute and return sigma_s at a fixed point array.

        This is the value stored in the training operator as a fixed
        (non-trainable) tensor.  Call once for Yq and once for Xc.

        Returns
        -------
        sigma_s_vals : ndarray, shape (N,)
            or shape (N, n_singular) when per_corner_gamma=True.
        """
        if self.per_corner_gamma:
            return self.evaluate_per_corner(points, r_min=r_min)
        return self.evaluate(points, r_min=r_min)
