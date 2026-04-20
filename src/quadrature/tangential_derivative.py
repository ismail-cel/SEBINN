"""
Tangential (arc-length) differentiation on the panelised boundary.

Mathematical background
-----------------------
The Maue identity for the hypersingular operator W reads

    W φ(x) = -d/ds_x ∫_Γ G(x,y) (dφ/ds_y)(y) ds_y
            = -(D_s V D_s) φ,

where D_s is the arc-length derivative on ∂Ω.

On each panel p the density φ is sampled at p_GL Gauss-Legendre nodes
{ξ_1, …, ξ_{p_GL}} ∈ [-1,1].  The tangential derivative is approximated
by Lagrange polynomial differentiation:

    (dφ/ds)|_{y_j} = (2/L_p) Σ_k D̂_{jk} φ(y_k),

where D̂_{jk} = ℓ'_k(ξ_j) is the reference derivative matrix and the
factor 2/L_p maps from the reference [-1,1] to physical arc-length.

The global matrix D_h is block-diagonal: one (p_GL × p_GL) block per
panel.  It is therefore sparse, but stored dense for simplicity (Nq ≤
several thousand in typical experiments).

Stability note
--------------
The barycentric formula for Lagrange interpolation at GL nodes is
numerically stable for p ≤ 30.  We use the explicit loop form (equivalent
to barycentric) which is adequate for p = 16 used throughout this project.
"""

from __future__ import annotations

import numpy as np

from .panel_quad import QuadratureData
from .gauss import gauss_legendre


# ---------------------------------------------------------------------------
# Reference Lagrange derivative matrix
# ---------------------------------------------------------------------------

def lagrange_derivative_matrix(xi: np.ndarray) -> np.ndarray:
    """
    Derivative matrix for Lagrange interpolation at nodes xi.

    D[j, k] = ℓ'_k(ξ_j),  where ℓ_k is the k-th Lagrange basis polynomial
    at the nodes xi.

    The diagonal entry k == j uses the identity
        ℓ'_k(ξ_k) = Σ_{m≠k} 1 / (ξ_k - ξ_m).

    The off-diagonal entry k ≠ j uses
        ℓ'_k(ξ_j) = [Π_{m≠k, m≠j} (ξ_j - ξ_m)] / [Π_{m≠k} (ξ_k - ξ_m)].

    Parameters
    ----------
    xi : ndarray (p,)
        Interpolation nodes on [-1, 1].  For standard use, these are the
        Gauss-Legendre nodes from gauss_legendre(p).

    Returns
    -------
    D : ndarray (p, p)
        Lagrange derivative matrix.
    """
    p = len(xi)
    D = np.zeros((p, p))

    for k in range(p):
        # Denominator: Π_{m≠k} (ξ_k - ξ_m)
        den = 1.0
        for m in range(p):
            if m != k:
                den *= (xi[k] - xi[m])

        for j in range(p):
            if j == k:
                # Diagonal: sum of 1/(ξ_k - ξ_m) for m≠k
                D[j, k] = sum(1.0 / (xi[k] - xi[m]) for m in range(p) if m != k)
            else:
                # Off-diagonal: numerator = Π_{m≠k, m≠j} (ξ_j - ξ_m)
                num = 1.0
                for m in range(p):
                    if m != k and m != j:
                        num *= (xi[j] - xi[m])
                D[j, k] = num / den

    return D


# ---------------------------------------------------------------------------
# Global block-diagonal tangential derivative matrix
# ---------------------------------------------------------------------------

def build_tangential_derivative_matrix(qdata: QuadratureData) -> np.ndarray:
    """
    Build the global tangential differentiation matrix D_h.

    D_h is block-diagonal: one p_GL × p_GL block per panel, each block is
    the reference Lagrange derivative matrix D̂ scaled by 2/L_p.

    The physical derivative at quadrature node j on panel p is
        (dφ/ds)|_{y_j}  =  (2/L_p) Σ_k D̂_{jk} φ(y_k),
    where the factor 2/L_p converts from the reference interval [-1,1]
    (length 2) to physical arc-length (length L_p).

    Parameters
    ----------
    qdata : QuadratureData

    Returns
    -------
    D_h : ndarray (Nq, Nq)
        Block-diagonal tangential derivative matrix.  D_h[i, j] = 0
        whenever nodes i and j lie on different panels.
    """
    p    = qdata.p
    xi, _ = gauss_legendre(p)
    D_ref  = lagrange_derivative_matrix(xi)   # (p, p), reference

    Nq   = qdata.n_quad
    Npan = qdata.n_panels
    D_h  = np.zeros((Nq, Nq))

    for pid in range(Npan):
        js    = qdata.idx_std[pid]     # length-p index array for panel pid
        L_p   = qdata.L_panel[pid]
        scale = 2.0 / float(L_p)      # reference → physical arc-length
        D_h[np.ix_(js, js)] = scale * D_ref

    return D_h
