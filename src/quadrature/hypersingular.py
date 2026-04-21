"""
Hypersingular operator T_h via direct Nyström collocational kernel evaluation.

Mathematical background
-----------------------
The hypersingular operator T for the 2D Laplace equation is:

    (T φ)(x) = ∂/∂n_x ∫_Γ ∂G(x,y)/∂n_y φ(y) ds(y)

where G(x,y) = -(1/2π) log|x-y| is the 2D Laplace fundamental solution.

Via complex-variable representation, the collocational kernel is:

    T(x,y) = -(1/π) Re[ τ(x) · τ(y) / (z(x) - z(y))² ]

where z = x_1 + ix_2 (complex coordinate), τ = unit tangent (complex).

Derivation: the full 2D Laplace hypersingular kernel is
    k(x,y) = -(1/π)(n_x·n_y)/|x-y|² + (2/π)(n_x·(x-y))(n_y·(x-y))/|x-y|⁴
Expressing in complex variables with τ = e^{iθ}, ν = iτ (CW), z-ζ = r·e^{iα}:
    k = -(1/π)cos(θ_x+θ_y-2α)/r² = -(1/π) Re[τ_x·τ_y·conj(z-ζ)²/|z-ζ|⁴]
Using conj(w)/|w|² = 1/w: conj(z-ζ)²/|z-ζ|⁴ = 1/(z-ζ)².  Hence
    k(x,y) = -(1/π) Re[ τ_x · τ_y / (z_x - z_y)² ]

Note: τ_x·τ_y is symmetric → k(x,y) = k(y,x) → diag(w)·T_h is exactly symmetric.
The formula Im[ν·conj(τ)/(z-ζ)²] is INCORRECT (gives -(1/π)cos(θ_x-θ_y-2α)/r²,
which is asymmetric in θ_x, θ_y for θ_x ≠ θ_y).

The Nyström discretisation is:
    T_h[i,j] = T(x_i, y_j) · w_j   (off-diagonal, i≠j or different panels)
    T_h[i,i] = self-panel diagonal correction  (replaces the hypersingular self-panel)

Key difference from the Maue formula
-------------------------------------
The Maue discretisation W_h = -D_h^T diag(w) V_h D_h is a Galerkin formula
that lives in a different discrete space than V_h (which is Nyström/collocational).
The direct T_h above uses the same Nyström framework as V_h, so the product
T̃_h V_h lives in a single consistent discrete space.  This is the prerequisite
for the Calderón identity T̃V ≈ (1/4)I to hold at the matrix level.

Self-panel correction (Hadamard finite-part)
--------------------------------------------
On a straight panel with constant tangent τ (so τ_x = τ_y = τ), the kernel is

    T(s₀, s) = -(1/π) Re[τ²/(s₀-s)²·e^{-2iθ·...}] = -(Re[τ²]/π)/(s₀-s)²

But since both x and y are on the same panel: τ_x = τ_y = τ, so
    T(s₀,s) = -(1/π) Re[τ·τ / (z_x-z_y)²]

The displacement along the panel is z_x - z_y = (s₀-s)·τ (complex), so
    (z_x-z_y)² = (s₀-s)²·τ²
    T(s₀,s) = -(1/π) Re[τ²/((s₀-s)²τ²)] = -(1/π) · 1/(s₀-s)²

The Re[τ²/τ²] = Re[1] = 1, so cos2θ cancels completely on a self-panel!
The Hadamard finite-part integral over [0,L] is:

    I_T = (1/π) · (1/(L − s₀) + 1/s₀)

(note: no cos2θ factor — the same for every panel orientation).

Nullspace regularisation
------------------------
The continuous operator T has a rank-1 nullspace: T · 1 = 0.  Fix by:

    T̃ = T_h + M,    M[i,j] = w_j / Σ_k w_k

Calderón identity
-----------------
For a smooth closed curve, T · V = (I/2 − K)(I/2 + K) ≈ I/4 − K² ≈ I/4
(K compact).  At the Nyström matrix level:
    T_h · V_h ≈ (1/4) · diag(w)
so the eigenvalues of T̃_h V_h cluster near w_avg / 4, giving cond(T̃V) ≈ O(1).

On the Koch snowflake, K is not compact (reentrant corners ω = 4π/3), so
cond(T̃V) will be larger than 1 but should be much smaller than cond(V).
"""

from __future__ import annotations

import numpy as np

from .panel_quad import QuadratureData


# ---------------------------------------------------------------------------
# Tangent / normal computation
# ---------------------------------------------------------------------------

def panel_normals_tangents(
    qdata: QuadratureData,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute unit tangent τ and outward normal ν at each quadrature node.

    For straight (polygonal) panels τ and ν are constant within each panel.
    The Koch snowflake is CW-traversed (see polygon.py), so the outward
    normal is the tangent rotated +90°:

        ν = i · τ    (complex notation)

    i.e. ν = (−τ_y, τ_x) in 2D.

    The tangent direction is computed from the first and last GL node of
    each panel, which lie in the interior of the panel: the direction
    Yq[:,js[-1]] − Yq[:,js[0]] is parallel to the panel edge (exactly so
    for straight panels).

    Parameters
    ----------
    qdata : QuadratureData

    Returns
    -------
    tau : ndarray, shape (Nq,), complex128
        Unit tangent at each node.
    nu  : ndarray, shape (Nq,), complex128
        Unit outward normal at each node (CW convention: ν = i·τ).
    """
    Nq = qdata.n_quad
    tau = np.zeros(Nq, dtype=complex)

    for pid in range(qdata.n_panels):
        js   = qdata.idx_std[pid]
        p0   = qdata.Yq[:, js[0]]
        p1   = qdata.Yq[:, js[-1]]
        d    = p1 - p0
        L_d  = np.linalg.norm(d)
        t    = (d[0] + 1j * d[1]) / L_d   # unit tangent (complex)
        tau[js] = t

    # CW orientation → outward normal = rotate tangent by +90°
    nu = 1j * tau
    return tau, nu


# ---------------------------------------------------------------------------
# Self-panel analytic correction
# ---------------------------------------------------------------------------

def hypsing_self_panel_correction(
    L: float,
    s0: float,
    tau_panel: complex = 1.0 + 0j,
) -> float:
    """
    Hadamard finite-part integral for the hypersingular kernel on a straight panel.

    For any panel orientation, the self-panel kernel reduces to:
        T(s₀, s) = -(1/π) / (s₀ − s)²

    The cos2θ factor cancels because τ_x = τ_y = τ on the same panel:
        Re[τ·τ / ((s₀−s)τ)²] = Re[τ²/((s₀−s)²τ²)] = Re[1/(s₀−s)²] = 1/(s₀−s)²

    The Hadamard finite-part integral of -(1/π)/(s₀-s)² over [0,L] is:
        I_T = (1/π) · (1/(L − s₀) + 1/s₀)

    Parameters
    ----------
    L          : float  — panel length
    s0         : float  — arc-length position of the collocation point on [0,L]
    tau_panel  : complex — unit tangent (not used; kept for API compatibility)

    Returns
    -------
    I_T : float
    """
    _EPS = 1e-16
    s0_c = max(min(s0, L - _EPS), _EPS)
    I_T  = (1.0 / np.pi) * (1.0 / (L - s0_c) + 1.0 / s0_c)
    return I_T


# ---------------------------------------------------------------------------
# Direct Nyström hypersingular matrix
# ---------------------------------------------------------------------------

def assemble_hypersingular_direct(
    qdata: QuadratureData,
) -> np.ndarray:
    """
    Assemble the hypersingular operator T_h via direct Nyström collocational
    evaluation of the complex kernel.

    Off-diagonal entries (i on panel p_i, j on panel p_j, p_i ≠ p_j):
        T_h[i,j] = −(1/π) Im[ ν_i · conj(τ_j) / (z_i − z_j)² ] · w_j

    Self-panel rows: the Gauss sum over the self-panel (j on same panel as i,
    j ≠ i) is augmented by the analytic Hadamard correction:
        T_h[i,i] = I_T(s₀ᵢ; Lᵢ, θᵢ) − Σ_{j∈panel(i)} T_h_off[i,j]

    Parameters
    ----------
    qdata : QuadratureData

    Returns
    -------
    T_h : ndarray, shape (Nq, Nq), float64
        Nyström hypersingular matrix with self-panel correction applied.
        Off-diagonal blocks are exact (no near-panel refinement).
        Self-panel diagonal: analytic Hadamard correction.

    Notes
    -----
    T_h is NOT symmetric as a matrix (it has quadrature weights only on the
    right column).  The symmetric Galerkin form is diag(wq) · T_h.
    The nullspace: T_h · ones ≈ 0 (rank-1 kernel).
    """
    Nq  = qdata.n_quad
    z   = qdata.Yq[0] + 1j * qdata.Yq[1]   # (Nq,) complex
    wq  = qdata.wq                            # (Nq,)

    tau, nu = panel_normals_tangents(qdata)

    # ------------------------------------------------------------------
    # Step 1: Vectorised off-diagonal kernel  (Nq × Nq)
    # Kernel: T(x_i, y_j) = -(1/π) Re[ τ_i · τ_j / (z_i - z_j)² ]
    # ------------------------------------------------------------------
    dz      = z[:, None] - z[None, :]           # (Nq, Nq), z_i − z_j
    dz2     = dz ** 2                             # (Nq, Nq), complex
    # Guard against 0/0 at diagonal; diagonal overwritten by correction anyway
    dz2_safe = np.where(np.abs(dz) > 1e-20, dz2, 1.0 + 0j)

    numer   = tau[:, None] * tau[None, :]        # (Nq, Nq), τ_i · τ_j (both complex)
    kernel  = -(1.0 / np.pi) * np.real(numer / dz2_safe)  # real (Nq, Nq)
    np.fill_diagonal(kernel, 0.0)                # zero diagonal before correction
    T_h = kernel * wq[None, :]                   # absorb weights on right

    # ------------------------------------------------------------------
    # Step 2: Self-panel analytic correction
    # ------------------------------------------------------------------
    corr_T = np.zeros(Nq)
    for i in range(Nq):
        pid = int(qdata.pan_id[i])
        js  = qdata.idx_std[pid]
        s0  = float(qdata.s_on_panel[i])
        L   = float(qdata.L_panel[pid])

        # Panel tangent (same for all nodes on this panel)
        tau_pid   = tau[js[0]]

        # Analytic Hadamard FP value
        I_T = hypsing_self_panel_correction(L, s0, tau_pid)

        # Gauss sum over self-panel (includes T_h[i,i]=0 from fill_diagonal)
        I_gauss = float(T_h[i, js].sum())

        corr_T[i] = I_T - I_gauss

    T_h += np.diag(corr_T)
    return T_h


# ---------------------------------------------------------------------------
# Nullspace regularisation
# ---------------------------------------------------------------------------

def regularise_hypersingular(
    T_h: np.ndarray,
    wq: np.ndarray,
) -> np.ndarray:
    """
    Fix the rank-1 nullspace of T_h by adding a global mean-value operator M.

    The continuous T has nullspace span{1} (constant densities yield zero
    normal derivative for harmonic functions).  The discrete T_h inherits
    this: T_h · ones ≈ 0.

    Adding M fixes the single null direction:
        T̃_h = T_h + M,   M[i,j] = w_j / Σ_k w_k

    M is the orthogonal projector onto constants (in the L²(∂Ω) inner product),
    so it maps any zero-mean density to 0 and constants to themselves.

    This is a rank-1 addition — much simpler than the 144-dimensional panel-mean
    regularisation required by the Maue W_h (whose nullspace = null(D_h) has
    dimension equal to the number of panels).

    Parameters
    ----------
    T_h : ndarray, shape (Nq, Nq)
        Output of assemble_hypersingular_direct.
    wq  : ndarray, shape (Nq,)
        Quadrature weights.

    Returns
    -------
    T_tilde : ndarray, shape (Nq, Nq)
        Regularised hypersingular matrix.  Invertible.
    """
    total_w = float(wq.sum())
    M       = np.outer(np.ones(len(wq)), wq) / total_w
    return T_h + M


# ---------------------------------------------------------------------------
# Legacy Maue-identity version (retained for reference, not used in Phase ≥ 2)
# ---------------------------------------------------------------------------

def _assemble_hypersingular_maue(
    qdata: QuadratureData,
    nmat,
) -> np.ndarray:
    """
    Maue identity: W_h = -D_h^T diag(wq) V_h D_h.

    DEPRECATED: produces cond(W̃V) ≥ cond(V) on Koch due to Galerkin/Nyström
    incompatibility and non-compact K at corners.  Retained for comparison only.
    """
    from .tangential_derivative import build_tangential_derivative_matrix
    D_h = build_tangential_derivative_matrix(qdata)
    V_h = nmat.V
    wq  = qdata.wq
    return -D_h.T @ (wq[:, None] * V_h) @ D_h
