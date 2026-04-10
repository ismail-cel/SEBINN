"""
Collocation point construction for the BINN/SE-BINN training loss.

MATLAB reference
----------------
build_collocation_points_per_panel   lines 446-478

The collocation points Xc are the points at which the boundary integral
equation residual is enforced during training.  They are distinct from
the quadrature nodes Yq: Yq integrates sigma, Xc evaluates the left-hand
side of the BIE at a (potentially different) set of boundary points.

In the MATLAB code, cfg.mColBase = 1 means one GL node per panel.  The
number can differ per panel (more near corners).

Design notes
------------
- CollocData mirrors the MATLAB colloc struct exactly.
- m_col_panel can be a scalar (same count for all panels) or a per-panel
  array, matching the scalar-broadcast behaviour in MATLAB lines 449-451.
- s0_of_xc stores the local arclength on the panel, needed for the
  self-panel analytic correction I2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..boundary.panels import Panel
from ..quadrature.gauss import gauss_legendre


@dataclass
class CollocData:
    """
    Collocation point set.

    Attributes
    ----------
    Xc : ndarray, shape (Nb, 2)
        Collocation point coordinates.
    pan_of_xc : ndarray of int, shape (Nb,)
        0-indexed panel index for each collocation point.
    s0_of_xc : ndarray, shape (Nb,)
        Local arclength from panel start for each collocation point.
    """
    Xc: np.ndarray           # (Nb, 2)
    pan_of_xc: np.ndarray    # (Nb,) int
    s0_of_xc: np.ndarray     # (Nb,)

    @property
    def n_colloc(self) -> int:
        return len(self.Xc)


def build_collocation_points(
    panels: List[Panel],
    m_col_panel: int | np.ndarray = 1,
) -> CollocData:
    """
    Build Gauss-based collocation points on each panel.

    MATLAB: build_collocation_points_per_panel (lines 446-478).

    Each panel gets m_col GL nodes as collocation points.  Using GL nodes
    (rather than uniform points) avoids aliasing between quadrature and
    collocation when they happen to coincide.

    Parameters
    ----------
    panels : list of Panel
    m_col_panel : int or ndarray of int, shape (Npan,)
        Number of collocation points per panel.
        Scalar is broadcast to all panels.  MATLAB: cfg.mColBase = 1.

    Returns
    -------
    colloc : CollocData
    """
    Npan = len(panels)
    if np.isscalar(m_col_panel):
        m_col_panel = np.full(Npan, int(m_col_panel), dtype=int)
    else:
        m_col_panel = np.asarray(m_col_panel, dtype=int)

    Nb = int(m_col_panel.sum())
    Xc = np.empty((Nb, 2))
    pan_of_xc = np.empty(Nb, dtype=int)
    s0_of_xc = np.empty(Nb)

    k = 0
    for pid, pan in enumerate(panels):
        a = pan.a
        b = pan.b
        L = pan.length
        m = int(m_col_panel[pid])

        xi, _ = gauss_legendre(m)
        tcol = (xi + 1.0) / 2.0   # map [-1,1] -> [0,1]

        for j in range(m):
            t = tcol[j]
            x = (1.0 - t) * a + t * b
            Xc[k] = x
            pan_of_xc[k] = pid
            s0_of_xc[k] = t * L
            k += 1

    return CollocData(Xc=Xc, pan_of_xc=pan_of_xc, s0_of_xc=s0_of_xc)
