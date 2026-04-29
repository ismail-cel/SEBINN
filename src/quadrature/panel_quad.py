"""
Panel Gauss quadrature on a polygon boundary.

MATLAB reference
----------------
build_panel_gauss_polygon_with_index           lines 338-372
build_panel_gauss_polygon_refined_with_index   lines 374-413

Design notes
------------
- QuadratureData is the central object passed to the Nystrom assembler,
  the training operator, and the reconstruction module.
- Yq shape is (2, Nq) — matching MATLAB convention — so that matrix-vector
  products A @ sigma are written naturally as A(i, :) * sigma(:).
- idx_std[m] contains the 0-based integer indices into Yq for panel m.
  It is a list of 1-D arrays, matching MATLAB's cell array idxStd.
- RefinedQuadratureData is only used when cfg.useNearPanelRefine is True.
  Its idx_ref[m] similarly indexes into YqR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..boundary.panels import Panel
from .gauss import gauss_legendre


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class QuadratureData:
    """
    Gauss quadrature on the panelized polygon boundary.

    Attributes
    ----------
    Yq : ndarray, shape (2, Nq)
        Quadrature point coordinates.  MATLAB: Yq.
    wq : ndarray, shape (Nq,)
        Quadrature weights including the arc-length Jacobian (L/2) * w_GL.
        MATLAB: wq.
    pan_id : ndarray of int, shape (Nq,)
        0-indexed panel index for each node.  MATLAB: pan_id (1-indexed there).
    s_on_panel : ndarray, shape (Nq,)
        Local arclength from the panel start: s = t * L, t in [0, 1].
        MATLAB: s_on_panel.
    L_panel : ndarray, shape (Npan,)
        Length of each panel.  MATLAB: L_panel.
    idx_std : list of ndarray of int, length Npan
        idx_std[m] gives the indices into Yq/wq belonging to panel m.
        MATLAB: idxStd (1-indexed cell array).
    p : int
        Number of GL points per panel.  MATLAB: cfg.pGL.
    """

    Yq: np.ndarray           # (2, Nq)
    wq: np.ndarray           # (Nq,)
    pan_id: np.ndarray       # (Nq,) int
    s_on_panel: np.ndarray   # (Nq,)
    L_panel: np.ndarray      # (Npan,)
    idx_std: List[np.ndarray]
    p: int
    pan_za: Optional[np.ndarray] = None   # complex (Npan,) — panel start vertex
    pan_zb: Optional[np.ndarray] = None   # complex (Npan,) — panel end vertex

    @property
    def n_quad(self) -> int:
        return self.wq.shape[0]

    @property
    def n_panels(self) -> int:
        return len(self.idx_std)


@dataclass
class RefinedQuadratureData:
    """
    Sub-panel refined Gauss quadrature (near-singular correction).

    Used only when useNearPanelRefine is True.  Each panel is split into
    n_sub sub-panels, each with p GL nodes.

    Attributes
    ----------
    YqR : ndarray, shape (2, NqR)
        Refined quadrature coordinates.  MATLAB: YqR.
    wqR : ndarray, shape (NqR,)
        Refined weights.  MATLAB: wqR.
    pan_id_R : ndarray of int, shape (NqR,)
        Panel index for each refined node.  MATLAB: pan_idR.
    idx_ref : list of ndarray of int, length Npan
        idx_ref[m] gives the indices into YqR/wqR for panel m.
        MATLAB: idxRef.
    n_sub : int
        Number of sub-panels per panel.  MATLAB: cfg.nSubNear.
    p : int
        GL points per sub-panel.
    """

    YqR: np.ndarray          # (2, NqR)
    wqR: np.ndarray          # (NqR,)
    pan_id_R: np.ndarray     # (NqR,) int
    idx_ref: List[np.ndarray]
    n_sub: int
    p: int

    @property
    def n_quad_refined(self) -> int:
        return self.wqR.shape[0]


# ---------------------------------------------------------------------------
# Standard panel quadrature
# ---------------------------------------------------------------------------

def build_panel_quadrature(panels: List[Panel], p: int) -> QuadratureData:
    """
    Gauss quadrature on each panel.

    MATLAB: build_panel_gauss_polygon_with_index (lines 338-372).

    Each panel [a, b] of length L is mapped from the reference interval
    [-1, 1] via   y(xi) = a + ((xi+1)/2) * (b - a),   Jacobian = L/2.

    Parameters
    ----------
    panels : list of Panel
        Output of build_uniform_panels (with labels already set or not —
        they are not required here).
    p : int
        Number of Gauss-Legendre points per panel.  MATLAB: cfg.pGL = 16.

    Returns
    -------
    qdata : QuadratureData
    """
    xi, wi = gauss_legendre(p)
    Npan = len(panels)
    Nq = Npan * p

    Yq = np.empty((2, Nq))
    wq = np.empty(Nq)
    pan_id = np.empty(Nq, dtype=int)
    s_on_panel = np.empty(Nq)
    L_panel = np.empty(Npan)
    idx_std: List[np.ndarray] = []
    pan_za = np.empty(Npan, dtype=complex)
    pan_zb = np.empty(Npan, dtype=complex)

    k = 0
    for m, pan in enumerate(panels):
        a = pan.a
        b = pan.b
        L = pan.length
        L_panel[m] = L
        pan_za[m] = a[0] + 1j * a[1]
        pan_zb[m] = b[0] + 1j * b[1]

        panel_indices = np.arange(k, k + p)
        idx_std.append(panel_indices)

        for j in range(p):
            # Map xi in [-1,1] to t in [0,1]
            t = (xi[j] + 1.0) / 2.0
            y = (1.0 - t) * a + t * b

            Yq[:, k] = y
            wq[k] = (L / 2.0) * wi[j]
            pan_id[k] = m
            s_on_panel[k] = t * L
            k += 1

    return QuadratureData(
        Yq=Yq,
        wq=wq,
        pan_id=pan_id,
        s_on_panel=s_on_panel,
        L_panel=L_panel,
        idx_std=idx_std,
        p=p,
        pan_za=pan_za,
        pan_zb=pan_zb,
    )


# ---------------------------------------------------------------------------
# Refined near-panel quadrature
# ---------------------------------------------------------------------------

def build_refined_quadrature(panels: List[Panel], p: int, n_sub: int) -> RefinedQuadratureData:
    """
    Sub-panel refined Gauss quadrature for near-singular row replacement.

    MATLAB: build_panel_gauss_polygon_refined_with_index (lines 374-413).

    Each panel is split into n_sub equal sub-panels; each sub-panel gets
    p GL nodes.  The total refined node count is Npan * n_sub * p.

    Parameters
    ----------
    panels : list of Panel
    p : int
        GL points per sub-panel.
    n_sub : int
        Number of sub-panels.  MATLAB: cfg.nSubNear.

    Returns
    -------
    rdata : RefinedQuadratureData
    """
    xi, wi = gauss_legendre(p)
    Npan = len(panels)
    pts_per_panel = n_sub * p
    NqR = Npan * pts_per_panel

    YqR = np.empty((2, NqR))
    wqR = np.empty(NqR)
    pan_id_R = np.empty(NqR, dtype=int)
    idx_ref: List[np.ndarray] = []

    k = 0
    for m, pan in enumerate(panels):
        a = pan.a
        b = pan.b

        panel_indices = np.arange(k, k + pts_per_panel)
        idx_ref.append(panel_indices)

        for ss in range(n_sub):
            t_A = ss / n_sub
            t_B = (ss + 1) / n_sub

            # Sub-panel endpoints in physical space
            a_s = (1.0 - t_A) * a + t_A * b
            b_s = (1.0 - t_B) * a + t_B * b
            L_s = float(np.linalg.norm(b_s - a_s))

            for j in range(p):
                t = (xi[j] + 1.0) / 2.0
                y = (1.0 - t) * a_s + t * b_s

                YqR[:, k] = y
                wqR[k] = (L_s / 2.0) * wi[j]
                pan_id_R[k] = m
                k += 1

    return RefinedQuadratureData(
        YqR=YqR,
        wqR=wqR,
        pan_id_R=pan_id_R,
        idx_ref=idx_ref,
        n_sub=n_sub,
        p=p,
    )
