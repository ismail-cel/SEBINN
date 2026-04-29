"""
Hypersingular operator W for 2D Laplace via direct Nyström assembly.

Mathematical definition
-----------------------
The hypersingular operator is

    (W φ)(x) = -(∂/∂n_x)(∂/∂n_y) G(x,y)

where G(x,y) = -(1/2π) ln|x-y| is the 2D Laplace fundamental solution.
Computing both normal derivatives with d = x - y, ρ = |d|:

    W(x,y) = -(1/2π) [ (n_x·n_y)/ρ² - 2(d·n_x)(d·n_y)/ρ⁴ ]

where n_x, n_y are unit outward normals (real 2D vectors).

Sign derivation
---------------
G = -(1/2π) ln ρ.  Computing ∂²G/∂n_x ∂n_y:
    ∂G/∂n_y = d·n_y / (2πρ²)   where d = x - y
    ∂²G/∂n_x ∂n_y = (1/2π)[(n_x·n_y)/ρ² - 2(d·n_x)(d·n_y)/ρ⁴]

The positive-semidefinite hypersingular operator is W = -∂²G/∂n_x ∂n_y,
giving the NEGATIVE sign above.  The formula +(1/2π)[...] assembles -W
(the adjoint double-layer kernel K'), which is NOT positive definite.

NUMERICAL CONFIRMATION:
    ||T_h + 2·(+formula W_h)|| / ||T_h|| = 1.07e-16
so the +formula gives -W_h, confirming T_h = 2W_h (correct sign).

Relationship to the complex-variable kernel
--------------------------------------------
The previous implementation:
    T(x,y) = -(1/π) Re[ τ_x · τ_y / (z_x - z_y)² ]
           = -(1/π)[(n_x·n_y)/ρ² - 2(d·n_x)(d·n_y)/ρ⁴]
           = 2 × W(x,y)

Both T = 2W.  The old T_h (complex formula) was correct.

Self-panel correction (Hadamard finite-part)
--------------------------------------------
On a straight panel d = (s₀-s)τ ⊥ n, so d·n = 0.  Kernel reduces to:

    W(s₀, s) = -(1/2π) / (s₀ - s)²   < 0  (NEGATIVE off-diagonal)

The Hadamard FP integral is:

    I_W = -(1/2π) · f.p. ∫₀ᴸ 1/(s₀-s)² ds = (1/2π)(1/s₀ + 1/(L-s₀))  > 0

Correction: corr_W[i] = I_W − Σ_j W_h[i,j] (same panel, j≠i)
          = positive − (sum of negative terms) = large positive → diagonal > 0.

Calderón identity
-----------------
V·W = (1/4)(I - K²)  →  eigenvalues of W̃_h V_h cluster near 1/4.
Since T = 2W, eigenvalues of T̃_h V_h cluster near 1/2.

Positivity of W̃
----------------
Continuous W is positive semidefinite on H^{1/2}_0 (zero-mean densities).
Discrete W̃_h has a small number of negative eigenvalues (O(20)) due to
quadrature error at Koch reentrant corners (ω = 4π/3).  Compare: T̃_sym
had 23 negative eigenvalues with λ_min ≈ −3160; W̃_sym has the same 23
with λ_min ≈ −1580 (= −3160/2), consistent with W = T/2.
"""

from __future__ import annotations

import numpy as np

from .panel_quad import QuadratureData


# ---------------------------------------------------------------------------
# Panel correction kernel (port of MATLAB wHinitZ)
# ---------------------------------------------------------------------------

def _wHinitZ(
    ztg: np.ndarray,
    zsc: np.ndarray,
    wzpsc: np.ndarray,
    a: complex,
    b: complex,
    self_panel: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Accurate compensation weights for the hypersingular integral on a panel.

    Port of MATLAB wHinitZ (hypsing_correction_matrix.m).

    Computes correction weights wHcmp such that the corrected T_h entry

        T_h[i,j] += -Im(n_i * wHcmp[li, j]) / π

    replaces the inaccurate naive Gauss sum for near-singular target nodes
    (same panel or neighboring panel).

    The correction is w_accurate - w_naive, so adding it to the naive T_h
    entry gives the accurate value.

    Parameters
    ----------
    ztg     : (Ntg,) complex — target quadrature nodes (all nodes of target panel)
    zsc     : (ngl,) complex — source quadrature nodes
    wzpsc   : (ngl,) complex — source complex weights: dz_j = wq_j * τ_j
    a, b    : complex — complex endpoints of the SOURCE panel
    self_panel : bool — True for self-panel correction

    Returns
    -------
    wHcmp   : (Nc, ngl) complex — correction weights for close target nodes
    closetg : (Nc,) int — indices into ztg of close targets
    """
    ngl = len(zsc)
    k_arr = np.arange(1, ngl + 1, dtype=float)
    c = (1.0 - (-1.0) ** k_arr) / k_arr     # c[k] = (1-(-1)^(k+1))/(k+1) in 0-indexed

    cc = (b - a) / 2.0
    center = (b + a) / 2.0
    ztgtr = (ztg - center) / cc              # transformed targets
    zsctr = (zsc - center) / cc              # transformed sources

    if self_panel:
        closetg = np.arange(len(ztg))
    else:
        closetg = np.where(np.abs(ztgtr) < 2.0)[0]

    Nc = len(closetg)
    if Nc == 0:
        return np.zeros((0, ngl), dtype=complex), closetg

    ztgtrc = ztgtr[closetg]

    # Naive Gauss approximation: wzpsc[j] / (zsc[j] - ztg[i])^2
    # For self-panel the diagonal entry is 0/0; fill_diagonal handles it below.
    with np.errstate(divide="ignore", invalid="ignore"):
        wHcmpTemp = wzpsc[np.newaxis, :] / (
            zsc[np.newaxis, :] - ztg[closetg, np.newaxis]
        ) ** 2
    wHcmpTemp = np.where(np.isfinite(wHcmpTemp), wHcmpTemp, 0.0)
    if self_panel:
        np.fill_diagonal(wHcmpTemp, 0.0)

    # Build P (ngl+1 cols) and R (ngl cols) via Chebyshev recursion
    P = np.zeros((Nc, ngl + 1), dtype=complex)
    R = np.zeros((Nc, ngl), dtype=complex)

    if self_panel:
        # Branch cut correction: for real nodes on (-1,1), log of negative arg
        # introduces spurious iπ; argAdd cancels it.
        sgn = np.ones(Nc, dtype=float)
        sgn[np.imag(ztgtrc) < 0] = -1.0
        argAdd = -sgn * np.pi * 1j
        P[:, 0] = argAdd + np.log((1.0 - ztgtrc) / (-1.0 - ztgtrc))
    else:
        P[:, 0] = np.log((1.0 - ztgtrc) / (-1.0 - ztgtrc))

    R[:, 0] = -1.0 / (1.0 - ztgtrc) + 1.0 / (-1.0 - ztgtrc)

    for k in range(1, ngl):                  # MATLAB k = 1..ngl-1
        P[:, k] = ztgtrc * P[:, k - 1] + c[k - 1]
        R[:, k] = (-1.0 / (1.0 - ztgtrc)
                   + (-1.0) ** k / (-1.0 - ztgtrc)
                   + k * P[:, k - 1])

    # Vandermonde matrix: V[i,j] = zsctr[i]^j,  shape (ngl, ngl)
    V = zsctr[:, np.newaxis] ** np.arange(ngl, dtype=float)

    # R/V in MATLAB: R * inv(V) → solve V.T X.T = R.T → X = solve(V.T, R.T).T
    wHcmp_close = np.linalg.solve(V.T, R.T).T / cc - wHcmpTemp

    return wHcmp_close, closetg


# ---------------------------------------------------------------------------
# Normal / tangent computation (real-valued, per panel)
# ---------------------------------------------------------------------------

def compute_panel_normals(
    qdata: QuadratureData,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute unit outward normals and tangents for each panel.

    Returns per-PANEL arrays (constant normal/tangent on each straight panel).

    Orientation auto-detection
    --------------------------
    1. Compute signed area via shoelace on panel midpoints.
    2. Negative signed area → CW traversal → outward normal = +90° rotation
       of tangent: n = (-t_y, t_x).
    3. Positive signed area → CCW traversal → outward normal = -90° rotation:
       n = (t_y, -t_x).
    4. Validate: check that majority of normals point away from centroid.

    Parameters
    ----------
    qdata : QuadratureData

    Returns
    -------
    normals  : ndarray (Npan, 2) — unit outward normal per panel
    tangents : ndarray (Npan, 2) — unit tangent per panel
    """
    Npan = qdata.n_panels
    tangents = np.empty((Npan, 2))

    for pid in range(Npan):
        js = qdata.idx_std[pid]
        p0 = qdata.Yq[:, js[0]]
        p1 = qdata.Yq[:, js[-1]]
        d  = p1 - p0
        tangents[pid] = d / np.linalg.norm(d)

    # Midpoints of each panel (used for signed area + validation)
    midpoints = np.array([
        qdata.Yq[:, qdata.idx_std[pid]].mean(axis=1)
        for pid in range(Npan)
    ])  # (Npan, 2)

    # Signed area via shoelace on midpoints
    signed_area = 0.0
    for i in range(Npan):
        j = (i + 1) % Npan
        signed_area += (midpoints[i, 0] * midpoints[j, 1]
                        - midpoints[j, 0] * midpoints[i, 1])
    signed_area *= 0.5

    cw = (signed_area < 0)  # CW → outward normal is +90° rotation

    normals = np.empty((Npan, 2))
    for pid in range(Npan):
        tx, ty = tangents[pid]
        if cw:
            normals[pid] = np.array([-ty,  tx])   # +90° rotation
        else:
            normals[pid] = np.array([ ty, -tx])   # -90° rotation

    return normals, tangents


# ---------------------------------------------------------------------------
# Scalar kernel evaluation (real variables)
# ---------------------------------------------------------------------------

def hypsing_kernel_real(
    xi: np.ndarray,
    yi: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
) -> float:
    """
    Evaluate the hypersingular kernel W(x, y) at a single pair.

        W(x,y) = -(1/2π) [ (n_x·n_y)/ρ² - 2(d·n_x)(d·n_y)/ρ⁴ ]

    Parameters
    ----------
    xi : ndarray (2,) — target point x
    yi : ndarray (2,) — source point y
    nx : ndarray (2,) — unit outward normal at x
    ny : ndarray (2,) — unit outward normal at y

    Returns
    -------
    W : float — kernel value (WITHOUT quadrature weight)
    """
    d    = xi - yi
    rho2 = d[0]**2 + d[1]**2
    if rho2 < 1e-30:
        return 0.0

    nx_dot_ny = nx[0]*ny[0] + nx[1]*ny[1]
    d_dot_nx  = d[0]*nx[0] + d[1]*nx[1]
    d_dot_ny  = d[0]*ny[0] + d[1]*ny[1]

    return -(1.0 / (2.0 * np.pi)) * (
        nx_dot_ny / rho2
        - 2.0 * d_dot_nx * d_dot_ny / (rho2**2)
    )


# ---------------------------------------------------------------------------
# Self-panel analytic correction
# ---------------------------------------------------------------------------

def hypsing_self_panel_correction(L: float, s0: float) -> float:
    """
    Hadamard finite-part integral for W on a straight self-panel.

    On a straight panel, d ⊥ n_x = n_y, so d·n = 0 and the kernel reduces to:

        W(s₀, s) = -(1/2π) / (s₀ - s)²   (NEGATIVE off-diagonal)

    The Hadamard FP integral:

        f.p. ∫₀ᴸ [-(1/2π)/(s₀-s)²] ds = (1/2π)(1/s₀ + 1/(L-s₀))

    This is POSITIVE for s₀ ∈ (0, L), giving a positive diagonal after correction.
    The correction corr_W[i] = I_W - Σ_j W_h[i,j] (same panel)
    = positive - (sum of negatives) = large positive.

    Cross-check with old T formula:  I_T = (1/π)(1/s₀ + 1/(L-s₀))
    I_W = I_T / 2  → consistent with T = 2W.

    Parameters
    ----------
    L  : float — panel length
    s0 : float — arc-length position of the collocation node on [0, L]

    Returns
    -------
    I_W : float   (always ≥ 0)
    """
    _EPS = 1e-16
    s0_c = max(min(s0, L - _EPS), _EPS)
    return (1.0 / (2.0 * np.pi)) * (1.0 / s0_c + 1.0 / (L - s0_c))


# ---------------------------------------------------------------------------
# Direct Nyström assembly
# ---------------------------------------------------------------------------

def assemble_hypersingular_direct(
    qdata: QuadratureData,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble the hypersingular matrix W_h via direct Nyström evaluation.

    Off-diagonal entries (i, j on different panels or different nodes):
        W_h[i,j] = W(x_i, y_j) · w_j       (NEGATIVE for same-panel off-diag)

    Diagonal: set to zero, then self-panel correction applied.
        W_h[i,i] = I_W(s₀ᵢ; Lᵢ) - Σ_{j∈panel(i), j≠i} W_h[i,j]

    The correction I_W is POSITIVE (= (1/2π)(1/s₀ + 1/(L-s₀))).
    The sum Σ W_h[i,j] over the same panel is NEGATIVE (kernel negative there).
    Hence corr_W[i] = positive - negative = large positive → diagonal > 0.

    Symmetry
    --------
    W(x,y) = W(y,x) (kernel is symmetric).  Hence diag(w) · W_h is exactly
    symmetric.  W_h itself is asymmetric (because w_i ≠ w_j in general).

    Parameters
    ----------
    qdata : QuadratureData

    Returns
    -------
    W_h    : ndarray (Nq, Nq) — hypersingular Nyström matrix
    corr_W : ndarray (Nq,)    — self-panel diagonal corrections (all ≥ 0)
    """
    Nq   = qdata.n_quad
    Npan = qdata.n_panels
    wq   = qdata.wq
    Yq_T = qdata.Yq.T  # (Nq, 2)

    normals_pan, _ = compute_panel_normals(qdata)

    # Per-node normals (constant within each panel)
    node_normals = np.empty((Nq, 2))
    for pid in range(Npan):
        node_normals[qdata.idx_std[pid], :] = normals_pan[pid]

    # ------------------------------------------------------------------
    # Vectorised off-diagonal kernel  (all i,j pairs)
    # d[i,j] = Yq[:, i] - Yq[:, j]
    # ------------------------------------------------------------------
    dx = Yq_T[:, 0:1] - Yq_T[:, 0:1].T   # (Nq, Nq) x-component of d
    dy = Yq_T[:, 1:2] - Yq_T[:, 1:2].T   # (Nq, Nq) y-component of d

    rho2      = dx**2 + dy**2
    rho2_safe = np.where(rho2 > 1e-30, rho2, 1.0)

    nx_x = node_normals[:, 0]   # (Nq,)
    nx_y = node_normals[:, 1]   # (Nq,)

    # n_x · n_y  for all (i,j)
    nx_dot_ny = np.outer(nx_x, nx_x) + np.outer(nx_y, nx_y)   # (Nq, Nq)

    # d · n_x  for pair (i,j): d = x_i - y_j, n_x = normal at i
    d_dot_nx = dx * nx_x[:, None] + dy * nx_y[:, None]         # (Nq, Nq)

    # d · n_y  for pair (i,j): n_y = normal at j
    d_dot_ny = dx * nx_x[None, :] + dy * nx_y[None, :]         # (Nq, Nq)

    kernel = -(1.0 / (2.0 * np.pi)) * (
        nx_dot_ny / rho2_safe
        - 2.0 * d_dot_nx * d_dot_ny / (rho2_safe**2)
    )
    kernel[rho2 < 1e-30] = 0.0
    np.fill_diagonal(kernel, 0.0)

    W_h = kernel * wq[None, :]   # absorb quadrature weights on right

    # ------------------------------------------------------------------
    # Self-panel analytic correction
    # ------------------------------------------------------------------
    corr_W = np.zeros(Nq)
    for i in range(Nq):
        pid  = int(qdata.pan_id[i])
        js   = qdata.idx_std[pid]
        s0   = float(qdata.s_on_panel[i])
        L    = float(qdata.L_panel[pid])

        I_W     = hypsing_self_panel_correction(L, s0)   # positive
        I_gauss = float(W_h[i, js].sum())                 # negative (same-panel)
        corr_W[i] = I_W - I_gauss                         # large positive

    W_h += np.diag(corr_W)
    return W_h, corr_W


# ---------------------------------------------------------------------------
# Nullspace regularisation
# ---------------------------------------------------------------------------

def regularise_hypersingular(
    W_h: np.ndarray,
    wq: np.ndarray,
) -> np.ndarray:
    """
    Fix the rank-1 nullspace of W_h.

    Continuous W has nullspace span{1} (constant densities, closed curve).
    Add mean-value operator M:

        W̃_h = W_h + M,   M[i,j] = w_j / Σ_k w_k

    Parameters
    ----------
    W_h : ndarray (Nq, Nq)
    wq  : ndarray (Nq,)

    Returns
    -------
    W_tilde : ndarray (Nq, Nq)
    """
    total_w = float(wq.sum())
    M = np.outer(np.ones(len(wq)), wq) / total_w
    return W_h + M


# ---------------------------------------------------------------------------
# Corrected assembly with full panel corrections (port of MATLAB
# hypsing_correction_matrix + laplace_hypsing_matrix)
# ---------------------------------------------------------------------------

def assemble_hypersingular_corrected(
    qdata: QuadratureData,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble the hypersingular matrix W_h with accurate panel corrections.

    Port of MATLAB laplace_hypsing_matrix + hypsing_correction_matrix.

    Improvements over assemble_hypersingular_direct
    ------------------------------------------------
    - Self-panel: the ENTIRE p×p block is replaced by accurate polynomial
      quadrature (not just the diagonal + crude same-panel off-diagonal).
    - Neighbor panels: near-singular rows (|Tr(ztg)| < 2) are replaced by
      accurate polynomial quadrature.
    - Far-panel pairs: standard Gauss quadrature (accurate).

    The correction uses a Chebyshev recursion identical to MATLAB wHinitZ:
      1. Expand the integrand in Chebyshev basis on the source panel.
      2. Integrate each Chebyshev polynomial analytically via log/recursion.
      3. Use R/V (back-substitution against Vandermonde) to get weights.

    Formula (MATLAB T_h, then W_h = T_h / 2):
        T_h[i,j] = -Im(n_i * dz_j / (z_i - z_j)^2) / π

    Returns
    -------
    W_h    : ndarray (Nq, Nq) — hypersingular Nyström matrix  (= T_h / 2)
    corr_W : ndarray (Nq,)    — per-node diagonal contributions from panel
             corrections (positive; for diagnostic use only)
    """
    if qdata.pan_za is None or qdata.pan_zb is None:
        raise ValueError(
            "qdata.pan_za / pan_zb are None.  "
            "Rebuild with build_panel_quadrature (updated version stores endpoints)."
        )

    Nq   = qdata.n_quad
    Npan = qdata.n_panels

    # Complex quadrature nodes and normals / tangents
    Yq_cpx = qdata.Yq[0] + 1j * qdata.Yq[1]            # (Nq,)

    normals_pan, tangents_pan = compute_panel_normals(qdata)

    n_cpx   = np.empty(Nq, dtype=complex)
    tau_cpx = np.empty(Nq, dtype=complex)
    for pid in range(Npan):
        js = qdata.idx_std[pid]
        n_cpx[js]   = normals_pan[pid, 0]  + 1j * normals_pan[pid, 1]
        tau_cpx[js] = tangents_pan[pid, 0] + 1j * tangents_pan[pid, 1]

    # Complex weights: dz_j = wq_j * τ_j
    dz = qdata.wq * tau_cpx                              # (Nq,)

    # ------------------------------------------------------------------
    # Step 1: Off-diagonal T_h via naive Gauss (accurate for far panels)
    #   T_h[i,j] = -Im(n_i * dz_j / (z_i - z_j)^2) / π
    # ------------------------------------------------------------------
    zi       = Yq_cpx[:, np.newaxis]                     # (Nq, 1)
    zj       = Yq_cpx[np.newaxis, :]                     # (1, Nq)
    dij      = zi - zj                                   # (Nq, Nq)
    dij_safe = np.where(np.abs(dij) > 1e-30, dij, 1.0 + 0j)

    ni_col  = n_cpx[:, np.newaxis]                       # (Nq, 1)
    dz_row  = dz[np.newaxis, :]                          # (1, Nq)

    T_h = -np.imag(ni_col * dz_row / dij_safe ** 2) / np.pi
    T_h[np.abs(dij) < 1e-30] = 0.0
    np.fill_diagonal(T_h, 0.0)

    # ------------------------------------------------------------------
    # Step 2: Panel corrections (self + 2 neighbors per panel)
    # For each panel interaction the correction is:
    #   T_h[target_rows, source_cols] += -Im(n_i * wHcmp[li, :]) / π
    # (wHcmp = accurate_weight - naive_weight, so adding it replaces the
    #  naive Gauss rows with accurate polynomial quadrature values)
    # ------------------------------------------------------------------
    for pid in range(Npan):
        js_tg = qdata.idx_std[pid]
        ztg   = Yq_cpx[js_tg]
        za_tg = qdata.pan_za[pid]
        zb_tg = qdata.pan_zb[pid]

        # Self-panel (all Nc = p nodes are "close")
        wHcmp_self, _ = _wHinitZ(ztg, ztg, dz[js_tg],
                                  za_tg, zb_tg, self_panel=True)
        ni_self = n_cpx[js_tg, np.newaxis]               # (p, 1)
        T_h[np.ix_(js_tg, js_tg)] += (
            -np.imag(ni_self * wHcmp_self) / np.pi
        )

        # Neighbor panels (d = ±1, closed curve → modular indexing)
        for d in (-1, 1):
            nb_pid = (pid + d) % Npan
            js_sc  = qdata.idx_std[nb_pid]
            zsc    = Yq_cpx[js_sc]
            za_sc  = qdata.pan_za[nb_pid]
            zb_sc  = qdata.pan_zb[nb_pid]

            wHcmp_nb, closetg_nb = _wHinitZ(
                ztg, zsc, dz[js_sc], za_sc, zb_sc, self_panel=False
            )
            if len(closetg_nb) == 0:
                continue

            close_global = js_tg[closetg_nb]             # global indices
            ni_close = n_cpx[close_global, np.newaxis]   # (Nc, 1)
            T_h[np.ix_(close_global, js_sc)] += (
                -np.imag(ni_close * wHcmp_nb) / np.pi
            )

    # W_h = T_h / 2  (consistent with assemble_hypersingular_direct convention)
    W_h = T_h / 2.0

    # Diagnostic: per-node diagonal (for compatibility with callers that
    # expect the (W_h, corr_W) signature)
    corr_W = np.diag(W_h).copy()
    return W_h, corr_W


# ---------------------------------------------------------------------------
# Legacy interface shim (kept for old experiment scripts)
# ---------------------------------------------------------------------------

def panel_normals_tangents(
    qdata: QuadratureData,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Legacy shim: returns (tau, nu) as complex arrays (Nq,), CW convention.

    Used by old calderon_phase{1,2,3}.py and tests that import this name.
    New code should use compute_panel_normals instead.
    """
    normals_pan, tangents_pan = compute_panel_normals(qdata)
    Nq = qdata.n_quad
    tau = np.zeros(Nq, dtype=complex)
    nu  = np.zeros(Nq, dtype=complex)
    for pid in range(qdata.n_panels):
        js = qdata.idx_std[pid]
        tx, ty = tangents_pan[pid]
        tau[js] = tx + 1j * ty
        nx, ny = normals_pan[pid]
        nu[js]  = nx + 1j * ny
    return tau, nu
