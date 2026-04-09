"""
Nystrom BEM matrix assembly and reference solve.

MATLAB reference
----------------
assemble_nystrom_matrix   lines 480-548
GMRES solve               lines 123-138

Mathematical background
-----------------------
The single-layer potential representation of the Dirichlet problem is

    u(x) = integral_{partial Omega} G(x, y) sigma(y) ds(y),

with G(x, y) = -(1/2pi) * log|x - y|  (2D Laplace fundamental solution).

Enforcing u(x) = g(x) at x in partial Omega (Nystrom collocation at the
quadrature nodes) gives the linear system

    V * sigma = f,

where  V[i, j] = G(x_i, y_j) * w_j  (off-diagonal, i != j)
             and V[i, i]  is replaced by the analytic self-panel correction.

Assembly strategy (following MATLAB exactly)
--------------------------------------------
1. Compute the full off-diagonal matrix  A[i, j] = G(x_i, y_j) * w_j,
   setting A[i, i] = 0 (singularity skipped by GL nodes).
2. Optionally replace adjacent-panel blocks with refined-rule blocks
   projected back to standard nodes via the Lagrange matrix T.
3. Add the analytic self-panel diagonal correction as  diag(corr).

The BEM solve is GMRES with an optional direct-fallback.

Design notes
------------
- NystromMatrix is a plain dataclass holding V and corr separately so the
  training operator can reuse the same assembly path for non-square systems
  (collocation != quadrature nodes).
- The GMRES wrapper returns a BEMSolution struct that downstream modules
  (reconstruction, error diagnostics) can consume without re-solving.
- scipy.sparse.linalg.gmres is used; interface matches MATLAB gmres closely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scipy.sparse.linalg as spla

from .panel_quad import QuadratureData, RefinedQuadratureData
from .self_correction import self_panel_log_correction
from .projection import refined_to_standard_projection


_LOG_KERNEL_SCALE = -1.0 / (2.0 * np.pi)
_R_MIN = 1e-14   # floor for |x - y| to avoid log(0)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class NystromMatrix:
    """
    Nystrom discretization of the single-layer operator V.

    Attributes
    ----------
    V : ndarray, shape (Nq, Nq)
        Full Nystrom matrix: V = A + diag(corr).
        MATLAB: Vmat = A + diag(corr).
    corr : ndarray, shape (Nq,)
        Diagonal analytic self-panel corrections.
        MATLAB: corr.
    """
    V: np.ndarray     # (Nq, Nq)
    corr: np.ndarray  # (Nq,)


@dataclass
class BEMSolution:
    """
    Output of the BEM/Nystrom reference solve.

    Attributes
    ----------
    sigma : ndarray, shape (Nq,)
        Computed boundary density at quadrature nodes.
    flag : int
        GMRES convergence flag (0 = converged).
    rel_res : float
        Final relative residual.
    n_iter : int
        GMRES iteration count.
    used_direct : bool
        True if GMRES did not converge and the direct fallback was used.
    """
    sigma: np.ndarray
    flag: int
    rel_res: float
    n_iter: int
    used_direct: bool


# ---------------------------------------------------------------------------
# Matrix assembly
# ---------------------------------------------------------------------------

def assemble_nystrom_matrix(
    qdata: QuadratureData,
    rdata: Optional[RefinedQuadratureData] = None,
) -> NystromMatrix:
    """
    Assemble the Nystrom matrix for the single-layer operator.

    MATLAB: assemble_nystrom_matrix (lines 480-548).

    Parameters
    ----------
    qdata : QuadratureData
        Standard panel quadrature (Yq, wq, pan_id, s_on_panel, L_panel, idx_std).
    rdata : RefinedQuadratureData or None
        If provided, adjacent-panel blocks are replaced with refined-rule
        projections.  MATLAB: useNearPanelRefine.

    Returns
    -------
    nmat : NystromMatrix
    """
    Yq = qdata.Yq          # (2, Nq)
    wq = qdata.wq          # (Nq,)
    Nq = qdata.n_quad
    Npan = qdata.n_panels

    # ------------------------------------------------------------------
    # Step 1: off-diagonal kernel matrix  A[i, j] = G(xi, yj) * wj
    # ------------------------------------------------------------------
    # Vectorised: for each row i, compute distances to all nodes, skip i=j.
    A = np.zeros((Nq, Nq))

    for i in range(Nq):
        xi = Yq[:, i]                             # (2,)
        diff = Yq - xi[:, None]                   # (2, Nq)
        r = np.linalg.norm(diff, axis=0)          # (Nq,)
        r = np.maximum(r, _R_MIN)
        r[i] = 1.0                                # avoid log(0); diagonal reset below

        G = _LOG_KERNEL_SCALE * np.log(r)         # (Nq,)
        row = G * wq
        row[i] = 0.0
        A[i, :] = row

    # ------------------------------------------------------------------
    # Step 2: adjacent-panel refinement (optional)
    # MATLAB lines 502-528
    # ------------------------------------------------------------------
    if rdata is not None:
        T = refined_to_standard_projection(qdata.p, rdata.n_sub)  # (n_sub*p, p)
        YqR = rdata.YqR    # (2, NqR)
        wqR = rdata.wqR    # (NqR,)

        for i in range(Nq):
            pid = int(qdata.pan_id[i])
            im1 = (pid - 1) % Npan
            ip1 = (pid + 1) % Npan

            for padj in (im1, ip1):
                js = qdata.idx_std[padj]   # standard indices for adjacent panel
                jr = rdata.idx_ref[padj]   # refined indices for adjacent panel

                # Refined kernel values at xi
                y_ref = YqR[:, jr]                      # (2, n_sub*p)
                diff_r = y_ref - Yq[:, i:i+1]           # (2, n_sub*p)
                rR = np.linalg.norm(diff_r, axis=0)     # (n_sub*p,)
                rR = np.maximum(rR, _R_MIN)
                GR = _LOG_KERNEL_SCALE * np.log(rR)     # (n_sub*p,)
                k_ref = GR * wqR[jr]                    # (n_sub*p,)

                # Project refined contribution onto standard node positions
                # MATLAB: ArefToStd = kRef.' * TrefStd   (1 x p)
                A_ref_to_std = k_ref @ T                 # (p,)

                # Replace: subtract old standard block, add projected refined
                A[i, js] = A[i, js] - A[i, js] + A_ref_to_std

    # ------------------------------------------------------------------
    # Step 3: analytic self-panel diagonal correction
    # MATLAB lines 530-541
    # ------------------------------------------------------------------
    corr = np.empty(Nq)
    for i in range(Nq):
        pid = int(qdata.pan_id[i])
        js = qdata.idx_std[pid]

        I2 = self_panel_log_correction(
            float(qdata.L_panel[pid]),
            float(qdata.s_on_panel[i]),
        )
        sum_self = A[i, js].sum()
        corr[i] = I2 - sum_self

    V = A + np.diag(corr)
    return NystromMatrix(V=V, corr=corr)


# ---------------------------------------------------------------------------
# Reference solve
# ---------------------------------------------------------------------------

def solve_bem(
    nmat: NystromMatrix,
    f: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 300,
    restart: Optional[int] = None,
    use_direct_fallback: bool = True,
) -> BEMSolution:
    """
    Solve V * sigma = f by GMRES with optional direct fallback.

    MATLAB: lines 123-138.

    Parameters
    ----------
    nmat : NystromMatrix
    f : ndarray, shape (Nq,)
        Right-hand side (Dirichlet boundary data at quadrature nodes).
    tol : float
        GMRES relative tolerance.  MATLAB: cfg.gmresTol = 1e-12.
    max_iter : int
        Maximum GMRES iterations.  MATLAB: cfg.gmresMaxit = 300.
    restart : int or None
        GMRES restart parameter.  MATLAB: cfg.gmresRestart = [] (None = no restart).
    use_direct_fallback : bool
        If True and GMRES fails, solve with numpy.linalg.solve.
        MATLAB: cfg.useDirectFallback = true.

    Returns
    -------
    sol : BEMSolution
    """
    V = nmat.V

    # scipy gmres: restart defaults to min(20, n) when None
    sigma, flag = spla.gmres(
        V, f,
        rtol=tol,
        maxiter=max_iter,
        restart=restart,
    )

    # Compute actual relative residual
    res_norm = float(np.linalg.norm(V @ sigma - f))
    f_norm = float(np.linalg.norm(f))
    rel_res = res_norm / max(f_norm, 1e-14)

    # Approximate iteration count (scipy does not expose this cleanly)
    n_iter = max_iter if flag != 0 else -1  # -1 = unknown but converged

    used_direct = False
    if flag != 0 and use_direct_fallback:
        sigma = np.linalg.solve(V, f)
        used_direct = True
        res_norm = float(np.linalg.norm(V @ sigma - f))
        rel_res = res_norm / max(f_norm, 1e-14)

    return BEMSolution(
        sigma=sigma,
        flag=flag,
        rel_res=rel_res,
        n_iter=n_iter,
        used_direct=used_direct,
    )
