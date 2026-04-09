"""
Barycentric Lagrange interpolation and refined-to-standard projection matrix.

MATLAB reference
----------------
refined_to_standard_projection_matrix   lines 1508-1526
barycentric_lagrange_matrix             lines 1528-1547
barycentric_weights                     lines 1549-1557

Purpose
-------
When using sub-panel refinement for near-singular rows, the refined quadrature
produces n_sub*p function values.  To subtract the standard panel contribution
and add the refined one coherently, we need to project refined values onto the
p standard GL nodes via Lagrange interpolation.

The projection matrix T has shape (n_sub*p, p) and satisfies:
    refined_vals @ T  ≈  standard_vals          (shape: (1, p))

Used in assemble_nystrom_matrix as:
    A_ref_to_std = (k_ref.T @ T)                 (1 x p)
    A[i, js] = A[i, js] - old_std + A_ref_to_std

Design notes
------------
- Barycentric weights are pre-divided so evaluation is numerically stable
  for high-order GL nodes.
- The coincidence tolerance (1e-14) matches MATLAB exactly.
- Projection matrix is cached by (p, n_sub) since it depends only on the
  GL node positions, not the physical panel geometry.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .gauss import gauss_legendre


# ---------------------------------------------------------------------------
# Barycentric weights
# ---------------------------------------------------------------------------

def barycentric_weights(x: np.ndarray) -> np.ndarray:
    """
    Compute barycentric weights for Lagrange interpolation at nodes x.

    MATLAB: barycentric_weights (lines 1549-1557).

    Parameters
    ----------
    x : ndarray, shape (n,)
        Interpolation nodes (distinct).

    Returns
    -------
    w : ndarray, shape (n,)
        Barycentric weights: w[j] = prod_{k != j} 1/(x[j] - x[k]).
    """
    n = len(x)
    w = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                w[j] /= (x[j] - x[k])
    return w


# ---------------------------------------------------------------------------
# Barycentric Lagrange matrix
# ---------------------------------------------------------------------------

def barycentric_lagrange_matrix(
    x_nodes: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    """
    Evaluate all Lagrange basis polynomials at x_eval points.

    MATLAB: barycentric_lagrange_matrix (lines 1528-1547).

    L[k, j] = l_j(x_eval[k])  where l_j is the j-th Lagrange basis
    polynomial for interpolation nodes x_nodes.

    Parameters
    ----------
    x_nodes : ndarray, shape (n,)
        Interpolation nodes.
    x_eval : ndarray, shape (m,)
        Evaluation points.

    Returns
    -------
    L : ndarray, shape (m, n)
        L[k, j] is the value of the j-th Lagrange basis polynomial at
        x_eval[k].  Each row sums to 1 (partition of unity).
    """
    n = len(x_nodes)
    m = len(x_eval)
    w = barycentric_weights(x_nodes)

    L = np.zeros((m, n))
    tol = 1e-14

    for k in range(m):
        dx = x_eval[k] - x_nodes
        min_idx = int(np.argmin(np.abs(dx)))

        if np.abs(dx[min_idx]) < tol:
            # x_eval[k] coincides with a node: exact cardinal
            L[k, min_idx] = 1.0
        else:
            tmp = w / dx
            denom = tmp.sum()
            L[k, :] = tmp / denom

    return L


# ---------------------------------------------------------------------------
# Refined-to-standard projection matrix
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def refined_to_standard_projection(p: int, n_sub: int) -> np.ndarray:
    """
    Build the (n_sub*p) x p Lagrange projection matrix from refined to
    standard GL nodes on [-1, 1].

    MATLAB: refined_to_standard_projection_matrix (lines 1508-1526).

    The standard GL nodes xi_std are of order p.
    The refined GL nodes xi_ref are n_sub groups of p GL nodes, one group
    per sub-interval [(ss-1)/n_sub, ss/n_sub] mapped to [-1, 1].

    Parameters
    ----------
    p : int
        GL order (points per panel / per sub-panel).
    n_sub : int
        Number of sub-panels per panel.

    Returns
    -------
    T : ndarray, shape (n_sub*p, p)
        T[i, j] = l_j(xi_ref[i])  where l_j uses nodes xi_std.
        Usage: A_ref_to_std = k_ref @ T  (shape: (1, p))

    Notes
    -----
    MATLAB:
        xiRef(k) = 2*tGlobal - 1    (maps sub-interval t to global [-1,1])
        TrefStd = barycentric_lagrange_matrix(xiStd, xiRef)
    """
    xi_std, _ = gauss_legendre(p)
    xi_sub, _ = gauss_legendre(p)

    # Build xi_ref: n_sub*p refined nodes on [-1, 1]
    xi_ref = np.empty(n_sub * p)
    k = 0
    for ss in range(n_sub):
        t_A = ss / n_sub
        t_B = (ss + 1) / n_sub
        for j in range(p):
            t = (xi_sub[j] + 1.0) / 2.0
            t_global = (1.0 - t) * t_A + t * t_B
            xi_ref[k] = 2.0 * t_global - 1.0
            k += 1

    # T[i, j] = l_j(xi_ref[i]) with nodes xi_std
    T = barycentric_lagrange_matrix(xi_std, xi_ref)
    return T
