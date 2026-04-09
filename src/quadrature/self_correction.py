"""
Analytic self-panel correction for the log-kernel single-layer potential.

MATLAB reference
----------------
self_panel_integral_log_kernel  lines 1559-1564

Mathematical background
-----------------------
On a straight panel of length L, the single-layer kernel is

    G(x, y(t)) = -(1/2pi) * log|x - y(t)|

When the collocation point x = y(s0) lies ON the panel, the Gauss
quadrature of G(x, y) * w misses the logarithmic singularity.
The exact value of the integral over the panel is

    I(s0; L) = integral_0^L  -(1/2pi) log|s - s0|  ds
             = -(1/2pi) * [ s0*log(s0) + (L - s0)*log(L - s0) - L ]

This is the DIAGONAL correction used in the Nystrom matrix:

    V[i,i] = I(s0_i; L_i)  (instead of zero from Gauss skipping x=y)

The correction is subtracted from the Gauss-quadrature sum over the
self-panel and replaced by the analytic value.

Design notes
------------
- This function is called once per collocation/quadrature node at assembly
  time; the result is stored in the `corr` vector, not recomputed at runtime.
- The clamp eps0 = 1e-16 prevents log(0) when s0 = 0 or s0 = L, matching
  the MATLAB guard exactly.
"""

from __future__ import annotations

import numpy as np


_EPS0 = 1e-16


def self_panel_log_correction(L: float, s0: float) -> float:
    """
    Analytic value of integral_0^L -(1/2pi) log|s - s0| ds.

    MATLAB: self_panel_integral_log_kernel (lines 1559-1564).

    Parameters
    ----------
    L : float
        Panel length.
    s0 : float
        Position of the evaluation point along the panel (0 <= s0 <= L).

    Returns
    -------
    I : float
        Analytic integral value (negative, since log < 0 for |s-s0| < 1
        when L is small).

    Notes
    -----
    MATLAB:
        eps0 = 1e-16;
        s0 = max(min(s0, L-eps0), eps0);
        Lm = max(L - s0, eps0);
        I2 = -(1/(2*pi)) * ( s0*log(s0) + Lm*log(Lm) - L );
    """
    s0 = max(min(s0, L - _EPS0), _EPS0)
    Lm = max(L - s0, _EPS0)
    return -(1.0 / (2.0 * np.pi)) * (s0 * np.log(s0) + Lm * np.log(Lm) - L)


def self_panel_log_correction_vec(
    L_panel: np.ndarray,
    s0_vec: np.ndarray,
    pan_id: np.ndarray,
) -> np.ndarray:
    """
    Vectorised version: compute the analytic correction for every node.

    Parameters
    ----------
    L_panel : ndarray, shape (Npan,)
        Panel lengths.
    s0_vec : ndarray, shape (Nq,)
        Local arclength position of each node on its panel.
    pan_id : ndarray of int, shape (Nq,)
        0-indexed panel index for each node.

    Returns
    -------
    corr : ndarray, shape (Nq,)
        corr[i] = self_panel_log_correction(L_panel[pan_id[i]], s0_vec[i]).
    """
    corr = np.empty(len(s0_vec))
    for i, (pid, s0) in enumerate(zip(pan_id, s0_vec)):
        corr[i] = self_panel_log_correction(float(L_panel[pid]), float(s0))
    return corr
