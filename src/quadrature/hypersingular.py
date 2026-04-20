"""
Hypersingular operator W_h via the Maue identity, and nullspace regularisation.

Mathematical background
-----------------------
The hypersingular operator W for the 2D Laplace equation is defined by

    (W φ)(x) = -d/ds_x ∫_Γ G(x,y) (dφ/ds_y)(y) ds_y,

which via the Maue regularisation identity equals

    W = -D_s V D_s,

where V is the single-layer operator and D_s is the arc-length derivative.

At the discrete level:

    W_h = -D_h^T  V_h  D_h    (Nq × Nq)

where:
  - V_h is the assembled Nyström matrix (including self-panel corrections),
  - D_h is the block-diagonal tangential derivative matrix (one block per panel).

The self-panel singularity of the log kernel is already handled in V_h;
D_h is local (block-diagonal) and does not interact with the singularity
treatment, so no additional diagonal corrections are needed.

Nullspace regularisation
------------------------
The continuous operator W has a rank-1 nullspace: W · 1 = 0 (constant
functions are harmonic and have zero normal derivative, so W kills them).

The discrete W_h inherits this: W_h @ ones ≈ 0.  To make W̃_h invertible
we add the mean-value operator M:

    W̃_h = W_h + M,    M[i,j] = w_j / Σ_k w_k.

M projects onto the constants, fixing the nullspace without altering W on
the orthogonal complement.

Calderón identity
-----------------
For the single-layer operator on a smooth closed curve,

    -W V = (I/2 - K)(I/2 + K) ≈ I/4 - K²,

where K is the double-layer operator (compact).  Hence the eigenvalues of
W̃V cluster near a constant, giving a well-conditioned preconditioned system.

On polygonal domains (Koch), K is not compact due to the corners, but
-WV still provides substantial spectral improvement over V alone.
"""

from __future__ import annotations

import numpy as np

from .panel_quad import QuadratureData
from .nystrom import NystromMatrix
from .tangential_derivative import build_tangential_derivative_matrix


# ---------------------------------------------------------------------------
# Hypersingular matrix
# ---------------------------------------------------------------------------

def assemble_hypersingular_matrix(
    qdata: QuadratureData,
    nmat: NystromMatrix,
) -> np.ndarray:
    """
    Assemble the hypersingular operator W_h via the Maue identity.

    The Nystrom matrix V_h[i,j] = G(x_i,y_j)*w_j has quadrature weights
    absorbed on the right.  The operator V in L^2 is self-adjoint, which
    translates to diag(w) V_h being symmetric (not V_h itself).

    The correct Maue discretization for the Nystrom V_h is therefore:

        W_h = -D_h^T  diag(wq)  V_h  D_h

    This is the Galerkin bilinear form of W evaluated at the GL nodes:

        W_h[i,j] ≈ ∫∫ G(x,y) (dφ_j/ds)(y) (dψ_i/ds)(x) ds(y) ds(x)

    with Nystrom test functions ψ_i = δ-like at x_i and ψ'_i = D_h[:,i].

    This W_h is exactly symmetric (machine precision) and has a 144-dimensional
    nullspace on Koch(1) — one piecewise constant per panel, since D_h kills
    all panel-wise constants.

    Notes on Calderon identity
    --------------------------
    The continuous Calderon identity -W V = I/4 - K^2 uses V and W as
    operators on L^2(∂Ω).  At the MATRIX level this translates to:

        -W_G V_G ≈ M^2 / 4  (Galerkin: V_G = diag(w) V_h, M = mass matrix)

    This identity holds for smooth boundaries but fails on polygonal domains
    (Koch) where the double-layer operator K is not compact.  The matrix
    cond(W̃V_h) measures how well the Calderon identity holds: for Koch it
    is O(10^4), indicating the identity fails significantly.

    Parameters
    ----------
    qdata : QuadratureData
    nmat  : NystromMatrix  (has nmat.V of shape (Nq, Nq))

    Returns
    -------
    W_h : ndarray (Nq, Nq)
        Hypersingular matrix.  Exactly symmetric.
        W_h · v_p = 0 for any piecewise-constant v_p (144-dim nullspace).
    """
    D_h = build_tangential_derivative_matrix(qdata)   # (Nq, Nq)
    V_h = nmat.V                                        # (Nq, Nq)
    wq  = qdata.wq                                      # (Nq,)

    # Weighted Maue identity: W_h = -D^T diag(w) V D
    # diag(w) V_h is the symmetric Galerkin form of V
    W_h = -D_h.T @ (wq[:, None] * V_h) @ D_h           # (Nq, Nq)

    return W_h


# ---------------------------------------------------------------------------
# Nullspace regularisation
# ---------------------------------------------------------------------------

def regularise_hypersingular(
    W_h: np.ndarray,
    qdata: "QuadratureData",
) -> np.ndarray:
    """
    Symmetrise W_h and fix its full 144-dimensional nullspace.

    Root cause
    ----------
    D_h is block-diagonal (one p_GL × p_GL block per panel).  Each block
    kills the constant function on its panel, giving

        null(D_h) = span{ indicator of panel k : k = 0 … Npan-1 }

    a space of dimension Npan = 144 for Koch(1).  Therefore

        null(W_h) = null(-D_h^T V_h D_h) ⊇ null(D_h),   dim ≥ 144.

    Adding a single global mean-value rank-1 operator M fixes only 1 of
    these 144 null directions and leaves W̃_h nearly singular.

    Fix — block-diagonal panel mean
    ---------------------------------
    Replace M with M_panel: the block-diagonal matrix whose k-th block is
    the rank-1 mean-value operator on panel k:

        M_panel[i,j] = w_j / Σ_{l ∈ panel(i)} w_l    if panel(i) = panel(j)
                     = 0                               otherwise.

    M_panel maps the indicator of panel k to itself (fixes the k-th null
    direction) and maps any zero-mean-per-panel function to zero (so it
    does not interfere with the active subspace of W_h).

    We also symmetrise W_h before adding M_panel, since W_h = -D^T V D
    is approximately but not exactly symmetric (V_h carries quadrature
    weights only on the right).

    Parameters
    ----------
    W_h   : ndarray (Nq, Nq)
        Output of assemble_hypersingular_matrix.
    qdata : QuadratureData
        Needed for idx_std (panel→node mapping) and wq.

    Returns
    -------
    W_tilde : ndarray (Nq, Nq)
        Regularised, symmetrised hypersingular matrix.  Invertible.
    """
    # Step 1: symmetrise
    W_sym = 0.5 * (W_h + W_h.T)

    # Step 2: block-diagonal panel mean operator
    Nq      = len(qdata.wq)
    M_panel = np.zeros((Nq, Nq))
    for pid in range(qdata.n_panels):
        js   = qdata.idx_std[pid]           # node indices on panel pid
        wk   = qdata.wq[js]                 # weights on this panel
        W_k  = float(wk.sum())              # total weight on panel
        # Outer product: M_panel[i,j] = w_j / W_k  for i,j ∈ panel pid
        M_panel[np.ix_(js, js)] = np.outer(np.ones(len(js)), wk) / W_k

    W_tilde = W_sym + M_panel
    return W_tilde
