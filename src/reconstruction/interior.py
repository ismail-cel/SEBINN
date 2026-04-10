"""
Interior solution reconstruction from the learned boundary density.

MATLAB reference
----------------
reconstruct_interior_solution   lines 1439-1477

Given the boundary density σ (evaluated at quadrature nodes Yq with weights
wq), the single-layer potential is

    u(x) = ∫_{∂Ω} G(x, y) σ(y) ds(y)
          ≈ Σ_j  G(x, Yq_j) σ_j wq_j,

where G(x, y) = -(1/2π) log|x - y| is the free-space Laplace kernel.

The reconstruction proceeds on a regular Cartesian grid; only points strictly
inside the polygon are evaluated (all others are set to NaN, matching MATLAB).

SE-BINN vs MATLAB baseline
---------------------------
MATLAB uses net_to_vector / forward to extract σ_q.  Here σ_q is any NumPy
array (Nq,) already evaluated at the quadrature nodes — the caller is
responsible for assembling σ = σ_w(Yq) + γ σ_s(Yq) before passing it in.

The vectorised matrix-vector implementation replaces MATLAB's Python-like
per-point loop (lines 1453-1459) with a single kernel matrix evaluation,
which is faster and easier to read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


_LOG_KERNEL_SCALE = -1.0 / (2.0 * np.pi)
_R_MIN            = 1e-14


# ---------------------------------------------------------------------------
# Point-in-polygon (MATLAB: inpolygon)
# ---------------------------------------------------------------------------

def points_inside_polygon(
    xy:     np.ndarray,   # (N, 2)  query points
    P:      np.ndarray,   # (Nv, 2) polygon vertices (open, last != first)
) -> np.ndarray:
    """
    Ray-casting point-in-polygon test.

    Returns a boolean array of shape (N,): True for points strictly inside P.
    Points exactly on the boundary may give either result — this matches the
    behaviour of MATLAB's inpolygon for non-degenerate query points.

    Parameters
    ----------
    xy : ndarray (N, 2)
    P  : ndarray (Nv, 2)  — open polygon (do not repeat first vertex at end)

    Returns
    -------
    inside : ndarray of bool, shape (N,)
    """
    x = xy[:, 0]  # (N,)
    y = xy[:, 1]  # (N,)
    Nv = len(P)
    inside = np.zeros(len(x), dtype=bool)

    # Close the polygon for edge iteration
    xs = P[:, 0]
    ys = P[:, 1]
    xn = np.roll(xs, -1)  # next vertex x
    yn = np.roll(ys, -1)  # next vertex y

    # Vectorised ray cast: for each edge check crossing with horizontal ray
    # from (x, y) to (+inf, y).
    # We broadcast over both query points and edges simultaneously.
    # x: (N,), xs: (Nv,) → broadcast to (N, Nv)
    x_  = x[:, None]   # (N, 1)
    y_  = y[:, None]   # (N, 1)
    xs_ = xs[None, :]  # (1, Nv)
    ys_ = ys[None, :]
    xn_ = xn[None, :]
    yn_ = yn[None, :]

    # Edge crosses horizontal ray if one endpoint is strictly above y and the
    # other at or below y (or vice versa).
    cond_y = ((ys_ > y_) != (yn_ > y_))

    # x-coordinate of the ray intersection with the edge
    # x_cross = xs + (y - ys) * (xn - xs) / (yn - ys)
    # Guard division: denominator is never zero when cond_y is True
    denom = yn_ - ys_
    # Avoid division by zero for horizontal edges (cond_y False → not used)
    denom_safe = np.where(np.abs(denom) < 1e-300, 1.0, denom)
    x_cross = xs_ + (y_ - ys_) * (xn_ - xs_) / denom_safe  # (N, Nv)

    # Crossing is to the right of the query point
    cond_x = x_ < x_cross

    crossings = np.sum(cond_y & cond_x, axis=1)   # (N,)
    inside = (crossings % 2) == 1
    return inside


# ---------------------------------------------------------------------------
# Kernel matrix evaluation
# ---------------------------------------------------------------------------

def _log_kernel_matrix(
    Xpts: np.ndarray,  # (M, 2)  evaluation points
    Yq:   np.ndarray,  # (Nq, 2) quadrature nodes  [NOTE: (Nq, 2) not (2, Nq)]
) -> np.ndarray:
    """
    Build the M × Nq kernel matrix  K[k, j] = G(Xpts_k, Yq_j).

    G(x, y) = -(1/2π) log|x - y|.
    """
    # diff[k, j, :] = Xpts[k] - Yq[j]
    diff = Xpts[:, None, :] - Yq[None, :, :]   # (M, Nq, 2)
    r    = np.linalg.norm(diff, axis=2)          # (M, Nq)
    r    = np.maximum(r, _R_MIN)
    return _LOG_KERNEL_SCALE * np.log(r)         # (M, Nq)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class InteriorResult:
    """
    Output of reconstruct_interior.

    Attributes
    ----------
    xv      : ndarray (nGrid,)  — x grid coordinates
    yv      : ndarray (nGrid,)  — y grid coordinates
    Ugrid   : ndarray (nGrid, nGrid)  — u_num at grid (NaN outside)
    Uexgrid : ndarray (nGrid, nGrid)  — u_exact at grid (NaN outside, None if no exact)
    Egrid   : ndarray (nGrid, nGrid)  — u_num - u_exact (NaN outside, None if no exact)
    Uvals   : ndarray (M,)  — u_num at interior query points
    Uex     : ndarray (M,) or None    — u_exact at interior query points
    rel_L2  : float or None  — relative L2 error vs u_exact
    linf    : float or None  — L∞ error vs u_exact
    n_interior : int  — number of interior evaluation points
    """
    xv:         np.ndarray
    yv:         np.ndarray
    Ugrid:      np.ndarray
    Uexgrid:    Optional[np.ndarray]
    Egrid:      Optional[np.ndarray]
    Uvals:      np.ndarray
    Uex:        Optional[np.ndarray]
    rel_L2:     Optional[float]
    linf:       Optional[float]
    n_interior: int


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def reconstruct_interior(
    P:       np.ndarray,                   # (Nv, 2) polygon vertices (open)
    Yq:      np.ndarray,                   # (Nq, 2) quadrature nodes
    wq:      np.ndarray,                   # (Nq,)   quadrature weights
    sigma:   np.ndarray,                   # (Nq,)   density at Yq
    n_grid:  int              = 100,
    u_exact: Optional[Callable] = None,
    x_range: tuple            = (-1.0, 1.0),
    y_range: tuple            = (-1.0, 1.0),
) -> InteriorResult:
    """
    Evaluate u(x) = Σ_j G(x, Yq_j) σ_j wq_j on a regular interior grid.

    MATLAB: reconstruct_interior_solution (lines 1439-1477).

    The MATLAB per-point loop over grid points is replaced by a single
    vectorised kernel-matrix product:

        u[k] = K[k, :] @ (σ * wq)

    where K[k, j] = G(x_k, Yq_j).  This is mathematically identical.

    Parameters
    ----------
    P       : ndarray (Nv, 2)     — polygon vertices, open, last ≠ first
    Yq      : ndarray (Nq, 2)     — quadrature node coordinates
    wq      : ndarray (Nq,)       — quadrature weights (panel arc-length included)
    sigma   : ndarray (Nq,)       — density values at Yq (σ = σ_w + γ σ_s)
    n_grid  : int                 — number of grid points per axis
    u_exact : callable (N,2) → (N,) or None
              Exact solution for error computation.  If None, error fields
              in the result are None.
    x_range : (float, float)      — x extent of grid.  MATLAB: [-1, 1]
    y_range : (float, float)      — y extent of grid.  MATLAB: [-1, 1]

    Returns
    -------
    InteriorResult
    """
    # --- 1. Regular grid ---
    xv = np.linspace(x_range[0], x_range[1], n_grid)
    yv = np.linspace(y_range[0], y_range[1], n_grid)
    Xg, Yg = np.meshgrid(xv, yv)             # (n_grid, n_grid) each

    # --- 2. Interior mask ---
    xy_all = np.column_stack([Xg.ravel(), Yg.ravel()])   # (n_grid², 2)
    mask_flat = points_inside_polygon(xy_all, P)          # (n_grid²,)
    mask = mask_flat.reshape(n_grid, n_grid)              # (n_grid, n_grid)

    Xlist = Xg[mask]    # (M,)
    Ylist = Yg[mask]    # (M,)
    M = len(Xlist)

    # --- 3. Single-layer potential at interior points ---
    # Yq convention inside this module is (Nq, 2)
    Xpts = np.column_stack([Xlist, Ylist])                # (M, 2)
    K    = _log_kernel_matrix(Xpts, Yq)                   # (M, Nq)

    sigma_wq = sigma * wq                                 # (Nq,)
    Uvals    = K @ sigma_wq                               # (M,)  vectorised

    # --- 4. Assemble grids ---
    Ugrid = np.full((n_grid, n_grid), np.nan)
    Ugrid[mask] = Uvals

    # --- 5. Error metrics ---
    Uex     = None
    Uexgrid = None
    Egrid   = None
    rel_L2  = None
    linf    = None

    if u_exact is not None:
        Uex_vals = u_exact(Xpts)
        if Uex_vals.ndim != 1 or len(Uex_vals) != M:
            raise ValueError(
                f"u_exact must return shape ({M},); got {Uex_vals.shape}"
            )
        Uex = Uex_vals

        Uexgrid = np.full((n_grid, n_grid), np.nan)
        Uexgrid[mask] = Uex

        Egrid = np.full((n_grid, n_grid), np.nan)
        Egrid[mask] = Uvals - Uex

        err    = Uvals - Uex
        norm_ex = max(float(np.linalg.norm(Uex)), _R_MIN)
        rel_L2  = float(np.linalg.norm(err)) / norm_ex
        linf    = float(np.max(np.abs(err)))

    return InteriorResult(
        xv=xv,
        yv=yv,
        Ugrid=Ugrid,
        Uexgrid=Uexgrid,
        Egrid=Egrid,
        Uvals=Uvals,
        Uex=Uex,
        rel_L2=rel_L2,
        linf=linf,
        n_interior=M,
    )
