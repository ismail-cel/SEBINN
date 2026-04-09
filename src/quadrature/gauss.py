"""
Gauss-Legendre quadrature nodes and weights on [-1, 1].

MATLAB reference
----------------
gauss_legendre  lines 1596-1604

The MATLAB implementation uses the Golub-Welsch algorithm: build the
symmetric tridiagonal Jacobi matrix, then diagonalize it.  The weights
follow from the first components of the eigenvectors.  This port is an
exact transcription.

Design notes
------------
- Output is always sorted (nodes increasing, weights matched).
- All downstream quadrature uses this as the single source of GL rules.
- Results are cached via @lru_cache so repeated calls with the same n
  do not recompute the eigendecomposition.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre nodes and weights on [-1, 1].

    Direct port of MATLAB gauss_legendre (lines 1596-1604).

    Parameters
    ----------
    n : int
        Number of quadrature points.

    Returns
    -------
    nodes : ndarray, shape (n,)
        Quadrature nodes in [-1, 1], sorted ascending.
    weights : ndarray, shape (n,)
        Positive quadrature weights; sum(weights) = 2.

    Notes
    -----
    MATLAB:
        beta = 0.5 ./ sqrt(1 - (2*(1:n-1)).^(-2));
        T = diag(beta,1) + diag(beta,-1);
        [V,D] = eig(T);
        x = diag(D);  [x,idx]=sort(x);  V=V(:,idx);
        w = 2*(V(1,:).^2).';
    """
    if n == 1:
        return np.array([0.0]), np.array([2.0])

    # Off-diagonal entries of the symmetric Jacobi matrix
    k = np.arange(1, n, dtype=float)
    beta = 0.5 / np.sqrt(1.0 - (2.0 * k) ** (-2))

    # Symmetric tridiagonal matrix
    T = np.diag(beta, 1) + np.diag(beta, -1)

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(T)

    # Sort ascending
    idx = np.argsort(eigvals)
    nodes = eigvals[idx]
    V = eigvecs[:, idx]

    # Weights from first row of eigenvector matrix
    weights = 2.0 * V[0, :] ** 2

    return nodes, weights
