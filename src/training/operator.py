"""
SE-BINN operator state: bundles all fixed tensors consumed by the loss.

MATLAB reference
----------------
build_A_and_corr_multiColloc   lines 551-571
build_pinn_operator_state      lines 866-926

The operator state is the object passed into every call of the loss
function.  It holds everything that does not change between iterations:
  - the BIE kernel matrix A (Nb x Nq) and diagonal correction corr (Nb,)
  - the right-hand side f (boundary data at collocation points)
  - per-collocation loss weights wCol
  - precomputed sigma_s at quadrature nodes (Yq) and collocation nodes (Xc)
  - equation scaling factor eq_scale
  - the raw coordinate tensors Yq, Xc as PyTorch tensors (for the network
    forward pass inside the loss)

SE-BINN additions vs MATLAB baseline
-------------------------------------
MATLAB operator state has no sigma_s fields.  SE-BINN adds:
  op.sigma_s_q   : precomputed sigma_s at Yq, shape (Nq,)  [fixed tensor]
  op.sigma_s_c   : precomputed sigma_s at Xc, shape (Nb,)  [fixed tensor]
  (and, if near-panel refinement is active, op.sigma_s_ref at YqR)

These are registered as plain tensors (not parameters), so they contribute
to the forward pass but never receive gradients.

Equation scaling (MATLAB lines 877-889)
----------------------------------------
When eq_scale != 1, the matrix A, correction corr, and rhs f are all
pre-multiplied by eq_scale before storage.  This keeps the loss
landscape well-conditioned when |A| << 1.
  'none'  : eq_scale = 1.0
  'auto'  : eq_scale = 1 / mean|A|
  'fixed' : eq_scale = user-supplied value
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import torch

from typing import List

from ..boundary.panels import Panel
from ..boundary.polygon import PolygonGeometry
from ..quadrature.panel_quad import QuadratureData
from ..quadrature.self_correction import self_panel_log_correction
from ..singular.enrichment import SingularEnrichment
from .collocation import CollocData


_LOG_KERNEL_SCALE = -1.0 / (2.0 * np.pi)
_R_MIN = 1e-14


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class OperatorState:
    """
    All fixed tensors needed by the SE-BINN loss function.

    Attributes (tensors are float64 on the target device)
    ------
    A        : Tensor (Nb, Nq)  — kernel matrix (equation-scaled)
    corr     : Tensor (Nb,)     — self-panel correction (equation-scaled)
    f        : Tensor (Nb,)     — boundary data at Xc (equation-scaled)
    wCol     : Tensor (Nb,)     — per-collocation loss weights
    wCol_sum : float            — sum of wCol (normalisation denominator)
    Yq       : Tensor (Nq, 2)   — quadrature node coordinates
    Xc       : Tensor (Nb, 2)   — collocation node coordinates
    sigma_s_q: Tensor (Nq,)     — sigma_s precomputed at Yq  [SE-BINN]
    sigma_s_c: Tensor (Nb,)     — sigma_s precomputed at Xc  [SE-BINN]
    pan_of_xc: ndarray int (Nb,)— panel index for each collocation point
    s0_of_xc : ndarray (Nb,)    — local arc coord for each collocation point
    idx_std  : list of ndarray  — quadrature index arrays per panel
    n_colloc : int              — Nb
    n_quad   : int              — Nq
    n_panels : int              — Npan
    eq_scale : float            — equation scaling factor applied
    """

    A: torch.Tensor           # (Nb, Nq)
    corr: torch.Tensor        # (Nb,)
    f: torch.Tensor           # (Nb,)
    wCol: torch.Tensor        # (Nb,)
    wCol_sum: float

    Yq: torch.Tensor          # (Nq, 2)
    Xc: torch.Tensor          # (Nb, 2)

    sigma_s_q: torch.Tensor   # (Nq,)   SE-BINN: precomputed at Yq
    sigma_s_c: torch.Tensor   # (Nb,)   SE-BINN: precomputed at Xc

    pan_of_xc: np.ndarray     # (Nb,) int  — kept as numpy for indexing
    s0_of_xc: np.ndarray      # (Nb,)
    idx_std: List[np.ndarray]

    n_colloc: int
    n_quad: int
    n_panels: int
    eq_scale: float


# ---------------------------------------------------------------------------
# Non-square BIE kernel matrix (collocation rows, quadrature columns)
# ---------------------------------------------------------------------------

def build_bie_matrix(
    colloc: CollocData,
    qdata: QuadratureData,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the Nb x Nq kernel matrix A and self-correction vector corr.

    MATLAB: build_A_and_corr_multiColloc (lines 551-571).

    For each collocation point x_k, row k of A is:
        A[k, j] = G(x_k, y_j) * w_j   (j = 0..Nq-1)
    with G(x, y) = -(1/2pi) * log|x - y|.

    When x_k lies on panel p, the Gauss quadrature over that panel
    underestimates the self-contribution.  The analytic correction:
        corr[k] = I2(L_p, s0_k) - sum_j A[k, js_p]
    is stored separately and added via  corr[k] * sigma(x_k)  in the loss.

    Parameters
    ----------
    colloc : CollocData
    qdata  : QuadratureData

    Returns
    -------
    A    : ndarray (Nb, Nq)
    corr : ndarray (Nb,)
    """
    Xc = colloc.Xc          # (Nb, 2)
    Yq = qdata.Yq           # (2, Nq)
    wq = qdata.wq           # (Nq,)
    Nb = colloc.n_colloc
    Nq = qdata.n_quad

    A = np.empty((Nb, Nq))
    corr = np.empty(Nb)

    for k in range(Nb):
        xk = Xc[k]                                    # (2,)
        diff = Yq - xk[:, None]                        # (2, Nq)
        r = np.linalg.norm(diff, axis=0)               # (Nq,)
        r = np.maximum(r, _R_MIN)
        G = _LOG_KERNEL_SCALE * np.log(r)              # (Nq,)
        A[k, :] = G * wq

        # Self-panel analytic correction
        pid = int(colloc.pan_of_xc[k])
        js = qdata.idx_std[pid]
        I2 = self_panel_log_correction(
            float(qdata.L_panel[pid]),
            float(colloc.s0_of_xc[k]),
        )
        corr[k] = I2 - A[k, js].sum()

    return A, corr


# ---------------------------------------------------------------------------
# Operator state builder
# ---------------------------------------------------------------------------

def build_operator_state(
    colloc: CollocData,
    qdata: QuadratureData,
    enrichment: SingularEnrichment,
    g: callable,
    panel_weights: np.ndarray,
    eq_scale_mode: Literal["none", "auto", "fixed"] = "none",
    eq_scale_fixed: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[OperatorState, dict]:
    """
    Assemble the full SE-BINN operator state.

    MATLAB: build_pinn_operator_state (lines 866-926).

    Parameters
    ----------
    colloc : CollocData
        Collocation points (Xc, pan_of_xc, s0_of_xc).
    qdata : QuadratureData
        Quadrature data (Yq, wq, idx_std, L_panel, pan_id, s_on_panel).
    enrichment : SingularEnrichment
        Used to precompute sigma_s at Yq and Xc.
    g : callable (ndarray (N,2) -> ndarray (N,))
        Dirichlet boundary data function.  MATLAB: staticData.u_exact.
    panel_weights : ndarray, shape (Npan,)
        Per-panel loss weights.  MATLAB: wPanel.
    eq_scale_mode : 'none' | 'auto' | 'fixed'
        Equation scaling.  MATLAB: cfg.useEquationScaling / cfg.eqScaleMode.
    eq_scale_fixed : float
        Used when eq_scale_mode='fixed'.  MATLAB: cfg.eqScale = 10.0.
    dtype, device : PyTorch tensor options.

    Returns
    -------
    op : OperatorState
    diag : dict
        Diagnostic scalars (mean|A|, eq_scale, etc.).
    """
    # --- 1. BIE kernel matrix and correction ---
    A_np, corr_np = build_bie_matrix(colloc, qdata)

    # --- 2. Right-hand side ---
    f_np = g(colloc.Xc).astype(float)               # (Nb,)

    # --- 3. Per-collocation loss weights ---
    wCol_np = panel_weights[colloc.pan_of_xc].astype(float)   # (Nb,)

    # --- 4. Equation scaling (MATLAB lines 877-889) ---
    mean_abs_A_before = float(np.mean(np.abs(A_np)))
    max_abs_A_before  = float(np.max(np.abs(A_np)))
    mean_abs_f_before = float(np.mean(np.abs(f_np)))

    if eq_scale_mode == "auto":
        eq_scale = 1.0 / (mean_abs_A_before + 1e-12)
    elif eq_scale_mode == "fixed":
        eq_scale = float(eq_scale_fixed)
    else:
        eq_scale = 1.0

    A_np    = eq_scale * A_np
    corr_np = eq_scale * corr_np
    f_np    = eq_scale * f_np

    # --- 5. Precompute sigma_s at Yq and Xc ---
    # Yq is stored as (2, Nq) in QuadratureData; enrichment expects (N, 2)
    Yq_pts = qdata.Yq.T                                       # (Nq, 2)
    sigma_s_q_np = enrichment.precompute(Yq_pts).astype(float)  # (Nq,)
    sigma_s_c_np = enrichment.precompute(colloc.Xc).astype(float)  # (Nb,)

    # --- 6. Convert to tensors ---
    def t(x):
        return torch.tensor(x, dtype=dtype, device=device)

    op = OperatorState(
        A=t(A_np),
        corr=t(corr_np),
        f=t(f_np),
        wCol=t(wCol_np),
        wCol_sum=float(wCol_np.sum()),
        Yq=t(Yq_pts),                  # (Nq, 2)
        Xc=t(colloc.Xc),              # (Nb, 2)
        sigma_s_q=t(sigma_s_q_np),    # (Nq,)
        sigma_s_c=t(sigma_s_c_np),    # (Nb,)
        pan_of_xc=colloc.pan_of_xc.copy(),
        s0_of_xc=colloc.s0_of_xc.copy(),
        idx_std=qdata.idx_std,
        n_colloc=colloc.n_colloc,
        n_quad=qdata.n_quad,
        n_panels=qdata.n_panels,
        eq_scale=eq_scale,
    )

    diag = {
        "mean_abs_A_before": mean_abs_A_before,
        "max_abs_A_before": max_abs_A_before,
        "mean_abs_f_before": mean_abs_f_before,
        "mean_abs_A_after": float(np.mean(np.abs(A_np))),
        "eq_scale": eq_scale,
    }

    return op, diag


def select_corner_points(
    qdata: QuadratureData,
    geom: PolygonGeometry,
    radius_factor: float = 0.3,
) -> np.ndarray:
    """
    Select quadrature node indices within distance R of any reentrant corner.

    R = radius_factor * mean_edge_length

    Only singular (reentrant) corners are considered.  Quadrature nodes on
    both sides of a reentrant corner will be included, giving the penalty
    access to the actual spike region.

    Parameters
    ----------
    qdata         : QuadratureData  (has Yq of shape (2, Nq))
    geom          : PolygonGeometry (has vertices, singular_corner_indices)
    radius_factor : float — fraction of mean edge length to use as radius

    Returns
    -------
    indices : ndarray of int, shape (Nc,)
        Indices into the Nq-dim quadrature node arrays (Yq columns, wq, etc.)
        of nodes within R of any reentrant corner.
    """
    Yq_T   = qdata.Yq.T                       # (Nq, 2)
    P      = geom.vertices                     # (Nv, 2)
    n_edges = len(P)

    # Mean edge length
    edge_lengths = np.array([
        float(np.linalg.norm(P[(i + 1) % n_edges] - P[i]))
        for i in range(n_edges)
    ])
    R = radius_factor * float(edge_lengths.mean())

    sing_idx = geom.singular_corner_indices    # array of corner vertex indices
    corner_verts = P[sing_idx]                 # (n_sing, 2)

    # Distance from every quadrature node to every singular corner
    # shape: (Nq, n_sing)
    dists = np.linalg.norm(
        Yq_T[:, None, :] - corner_verts[None, :, :],
        axis=2,
    )
    near = np.any(dists <= R, axis=1)          # (Nq,) bool
    return np.where(near)[0]
