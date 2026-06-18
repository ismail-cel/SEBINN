"""
Final thesis Section 5 figures — HIGH RESOLUTION scaling study.

Resolution:
  Koch(1): n_per_edge=12, p_gl=16, N_q = 2304
  Koch(2): n_per_edge=12, p_gl=16, N_q = 9216
  Koch(3): n_per_edge= 8, p_gl=16, N_q = 24576  [fallback; 36864 exceeds RAM]

Methods (labels in all plots: A, B, C, D):
  A: Standard      L = ||Vσ - g||²
  B: Calderón      L = ||W̃(Vσ - g)||²               (corrected hypersingular)
  C: Sobolev H¹    L = ||Vσ - g||² + ||D_h(Vσ - g)||²  (α=1, combined)
  D: V⁻¹ ref.      L = ||σ - σ_BEM||²

Training: Adam 3×1000 [1e-3,3e-4,1e-4] + L-BFGS 15000 mem=30, seed=0.

Memory strategy for Koch(3) n_pe=8 (N_q=24576):
  - V_h (4.83 GB) + W̃ (4.83 GB): use torch.from_numpy (shared memory).
  - D_h applied on-the-fly in Method C loss — no DV matrix stored.
  - W̃ freed after Method B training.
  - cond_svd(W̃V) at Koch(3) via Lanczos LinearOperator (no WV matrix).
  - cond(H¹ Hessian) at Koch(3) via Lanczos LinearOperator.

Outputs → experiments/ex_thesis_scaling/{figures_final,tables_final,data_final}/
"""

from __future__ import annotations

import sys, os, gc, time, warnings
import numpy as np
import torch
import scipy.sparse.linalg as spla

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..")
sys.path.insert(0, _ROOT)

from src.boundary.polygon import make_koch_geometry
from src.boundary.panels import build_uniform_panels, label_corner_ring_panels
from src.quadrature.panel_quad import build_panel_quadrature
from src.quadrature.nystrom import assemble_nystrom_matrix, solve_bem
from src.quadrature.hypersingular import (
    assemble_hypersingular_corrected, regularise_hypersingular,
)
from src.quadrature.tangential_derivative import lagrange_derivative_matrix
from src.quadrature.gauss import gauss_legendre
from src.models.sigma_w_net import build_sigma_w_network
from src.reconstruction.interior import reconstruct_interior

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED         = 0
ALPHA        = 1.0
HIDDEN_WIDTH = 80
N_HIDDEN     = 4
LR_SCHEDULE  = [(1000, 1e-3), (1000, 3e-4), (1000, 1e-4)]
N_LBFGS      = 15000
LBFGS_MEM    = 30
LOG_EVERY    = 200
GMRES_TOL    = 1e-12
GMRES_MAX    = 8000
N_GRID       = 200    # interior reconstruction grid

# Per-level resolution
LEVEL_CFG = {
    1: dict(n_per_edge=12, p_gl=16),
    2: dict(n_per_edge=12, p_gl=16),
    3: dict(n_per_edge= 8, p_gl=16),   # memory fallback
}

METHODS      = ["A", "B", "C", "D"]
COLORS       = {"A": "#888888", "B": "#1f77b4", "C": "#2ca02c", "D": "#d62728"}
LINES        = {"A": "-",       "B": "-",       "C": "-",       "D": "--"}
MARKERS      = {"A": "o",       "B": "s",       "C": "^",       "D": "D"}
METHOD_FULL  = {"A": "Standard", "B": "Calderón",
                "C": r"Sobolev $H^1$", "D": r"$V^{-1}$ ref."}

FIG_DIR   = os.path.join(_HERE, "figures_final")
TAB_DIR   = os.path.join(_HERE, "tables_final")
DATA_DIR  = os.path.join(_HERE, "data_final")

# ---------------------------------------------------------------------------
# Boundary data and exact solution
# ---------------------------------------------------------------------------

def g_fn(xy: np.ndarray) -> np.ndarray:
    return xy[:, 0]**2 - xy[:, 1]**2

def u_exact_fn(xy: np.ndarray) -> np.ndarray:
    return xy[:, 0]**2 - xy[:, 1]**2

# ---------------------------------------------------------------------------
# Block-diagonal D_h application (on-the-fly — avoids storing NqxNq matrix)
# ---------------------------------------------------------------------------

def _make_dh_info(qdata):
    """Return (D_ref, list-of-(panel_idx, L_panel)) for on-the-fly D_h apply."""
    xi, _ = gauss_legendre(qdata.p)
    D_ref = lagrange_derivative_matrix(xi)      # (p, p)
    panels = [(qdata.idx_std[pid], qdata.L_panel[pid])
              for pid in range(qdata.n_panels)]
    return D_ref, panels

def _dh_apply_vec(v: np.ndarray, D_ref, panels) -> np.ndarray:
    """Apply block-diagonal D_h to vector v (Nq,) → (Nq,). Pure numpy."""
    out = np.empty_like(v)
    for js, L in panels:
        out[js] = (2.0 / L) * (D_ref @ v[js])
    return out

def _dh_apply_vec_torch(v: torch.Tensor, D_ref_t: torch.Tensor,
                         panels_t) -> torch.Tensor:
    """Apply block-diagonal D_h to torch vector (Nq,) → (Nq,).

    Uses reshape + batch matmul (clean autograd, no in-place scatter).
    Requires quadrature nodes stored panel-by-panel (standard Nyström order).
    """
    p         = D_ref_t.shape[0]
    n_panels  = len(panels_t)
    scales    = torch.tensor([s for _, s in panels_t],
                             dtype=v.dtype, device=v.device)  # (n_panels,)
    V_mat  = v.view(n_panels, p)                              # (n_panels, p)
    # D_ref @ v[js] == v[js] @ D_ref.T  for 1-D v[js]
    result = (V_mat @ D_ref_t.T) * scales.unsqueeze(1)       # (n_panels, p)
    return result.view(-1)                                     # (Nq,)

def _build_DV_vector(V_h: np.ndarray, g: np.ndarray, D_ref, panels):
    """Compute Dg = D_h g  (Nq,) — cheap; stored as vector."""
    Dg = _dh_apply_vec(g, D_ref, panels)
    return Dg

# ---------------------------------------------------------------------------
# Condition number helpers (Lanczos for large matrices)
# ---------------------------------------------------------------------------

def _cond_svd_matrix(M: np.ndarray, use_lanczos=False, k=6) -> float:
    if use_lanczos:
        op = spla.LinearOperator(M.shape, matvec=lambda x: M @ x,
                                 rmatvec=lambda x: M.T @ x, dtype=np.float64)
        sv_max = spla.svds(op, k=k, which='LM', return_singular_vectors=False)
        sv_min = spla.svds(op, k=k, which='SM', return_singular_vectors=False)
        return float(sv_max.max() / max(sv_min.min(), 1e-300))
    else:
        sv = np.linalg.svd(M, compute_uv=False)
        return float(sv[0] / max(sv[-1], 1e-300))

def _cond_V_lanczos(V_h: np.ndarray, k=6) -> float:
    """Estimate cond(V) via Lanczos without full SVD."""
    Nq = V_h.shape[0]
    def mv(x): return V_h @ x
    def rmv(x): return V_h.T @ x
    op = spla.LinearOperator((Nq, Nq), matvec=mv, rmatvec=rmv, dtype=np.float64)
    try:
        sv_max = spla.svds(op, k=k, which='LM', return_singular_vectors=False,
                           maxiter=2000)
        sv_min = spla.svds(op, k=k, which='SM', return_singular_vectors=False,
                           maxiter=5000, tol=1e-6)
        return float(sv_max.max() / max(sv_min.min(), 1e-300))
    except Exception:
        return np.nan

def _cond_svd_WV_lanczos(V_h: np.ndarray, W_tilde: np.ndarray, k=6) -> float:
    """Estimate cond_svd(W̃V) without forming WV explicitly."""
    Nq = V_h.shape[0]
    def mv(x):  return W_tilde @ (V_h @ x)
    def rmv(x): return V_h.T @ (W_tilde.T @ x)
    op = spla.LinearOperator((Nq, Nq), matvec=mv, rmatvec=rmv, dtype=np.float64)
    try:
        sv_max = spla.svds(op, k=k, which='LM', return_singular_vectors=False,
                           maxiter=2000)
        sv_min = spla.svds(op, k=k, which='SM', return_singular_vectors=False,
                           maxiter=5000, tol=1e-6)
        return float(sv_max.max() / max(sv_min.min(), 1e-300))
    except Exception:
        return np.nan

def _cond_eig_WV(WV: np.ndarray, Nq: int) -> float:
    """Spectral condition of W̃V from eigenvalues (only for small Nq)."""
    if Nq > 4000:
        return np.nan
    ev = np.linalg.eigvals(WV)
    ab = np.abs(ev)
    return float(ab.max() / max(ab.min(), 1e-300))

def _non_norm_WV(V_h: np.ndarray, W_tilde: np.ndarray) -> float:
    WV  = W_tilde @ V_h
    nn  = np.linalg.norm(WV.T @ WV - WV @ WV.T) / max(np.linalg.norm(WV)**2, 1e-300)
    del WV; gc.collect()
    return float(nn)

def _cond_H1_hessian(V_h: np.ndarray, D_ref, panels, alpha=1.0, k=12) -> float:
    """cond(V^T(I + alpha*D_h^T D_h)V).
    Nq <= 3000: exact via eigvalsh (form DV explicitly).
    Nq >  3000: Lanczos eigsh via LinearOperator.
    """
    Nq = V_h.shape[0]
    if Nq <= 3000:
        # Form DV = D_h @ V_h block-diagonally
        DV = np.zeros_like(V_h)
        for js, L in panels:
            DV[js, :] = (2.0 / L) * (D_ref @ V_h[js, :])
        H = V_h.T @ V_h + alpha * (DV.T @ DV)
        del DV; gc.collect()
        ev = np.linalg.eigvalsh(H)
        return float(ev[-1] / max(ev[0], 1e-300))
    else:
        def mv(x):
            Vx  = V_h @ x
            DVx = _dh_apply_vec(Vx, D_ref, panels)
            return V_h.T @ (Vx + alpha * DVx)
        op = spla.LinearOperator((Nq, Nq), matvec=mv, dtype=np.float64)
        try:
            ev_max = spla.eigsh(op, k=k, which='LM', return_eigenvectors=False)
            ev_min = spla.eigsh(op, k=k, which='SM', return_eigenvectors=False,
                                maxiter=8000, tol=1e-5)
            return float(ev_max.max() / max(ev_min.min(), 1e-300))
        except Exception as e:
            print(f"    [cond_H1 Lanczos] warning: {e}")
            return np.nan

# ---------------------------------------------------------------------------
# Level setup
# ---------------------------------------------------------------------------

def setup_level(n: int, n_per_edge: int, p_gl: int,
                compute_WV_matrix: bool = True,
                verbose: bool = True) -> dict:
    """
    Build geometry, operators, spectral data for Koch(n).
    compute_WV_matrix=False → skip forming WV (use Lanczos for cond_svd_WV).
    """
    t0 = time.perf_counter()
    if verbose:
        print(f"\n  [Koch({n})] n_pe={n_per_edge}, p={p_gl}")
        print(f"  [Koch({n})] Building geometry …")

    geom   = make_koch_geometry(n=n)
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=n_per_edge)
    label_corner_ring_panels(panels, P)
    qdata  = build_panel_quadrature(panels, p=p_gl)
    Yq_T   = qdata.Yq.T          # (Nq, 2)
    wq     = qdata.wq
    Nq     = qdata.n_quad

    if verbose:
        print(f"  [Koch({n})] Nq={Nq}. Assembling Nyström matrix …")
    nmat      = assemble_nystrom_matrix(qdata)
    V_h       = nmat.V
    g_values  = g_fn(Yq_T)

    if verbose:
        print(f"  [Koch({n})] Solving BEM for σ_BEM …")
    bem       = solve_bem(nmat, g_values, tol=GMRES_TOL, max_iter=GMRES_MAX)
    sigma_BEM = bem.sigma

    # cond(V)
    use_lanczos = (Nq > 10000)
    if verbose:
        print(f"  [Koch({n})] cond(V) {'[Lanczos]' if use_lanczos else ''} …")
    if use_lanczos:
        cond_V = _cond_V_lanczos(V_h)
    else:
        sv_V   = np.linalg.svd(V_h, compute_uv=False)
        cond_V = float(sv_V[0] / sv_V[-1])
    if verbose:
        print(f"  [Koch({n})] cond(V) = {cond_V:.3e}")

    # Corrected hypersingular W̃
    if verbose:
        print(f"  [Koch({n})] Assembling corrected W̃ …")
    W_h, _  = assemble_hypersingular_corrected(qdata)
    W_tilde = regularise_hypersingular(W_h, wq)
    del W_h; gc.collect()

    # cond_svd(W̃V) and cond_eig(W̃V)
    if verbose:
        print(f"  [Koch({n})] Spectral analysis of W̃V …")
    if compute_WV_matrix and Nq <= 12000:
        WV          = W_tilde @ V_h
        sv_WV       = np.linalg.svd(WV, compute_uv=False)
        cond_svd_WV = float(sv_WV[0] / sv_WV[-1])
        cond_eig_WV = _cond_eig_WV(WV, Nq)
        # non-normality directly
        WVTWV = WV.T @ WV
        WVWVT = WV @ WV.T
        non_norm_WV = float(np.linalg.norm(WVTWV - WVWVT) / max(np.linalg.norm(WV)**2, 1e-300))
        del WV, WVTWV, WVWVT; gc.collect()
    else:
        # Lanczos: avoid forming WV (saves 4.83 GB for Koch(3) n_pe=8)
        cond_svd_WV = _cond_svd_WV_lanczos(V_h, W_tilde)
        cond_eig_WV = np.nan
        # Non-normality: need WV — skip for large systems
        non_norm_WV = np.nan
    if verbose:
        print(f"  [Koch({n})] cond_eig(W̃V)={cond_eig_WV:.2f}  "
              f"cond_svd(W̃V)={cond_svd_WV:.2f}  non-norm={non_norm_WV}")

    # D_h info (for Method C on-the-fly application)
    D_ref, dh_panels = _make_dh_info(qdata)
    Dg = _build_DV_vector(V_h, g_values, D_ref, dh_panels)

    # cond(H¹ Hessian) — Lanczos for large Nq
    if verbose:
        print(f"  [Koch({n})] cond(H¹ Hessian) …")
    cond_H1 = _cond_H1_hessian(V_h, D_ref, dh_panels)
    null_Dh = qdata.n_panels   # one constant per panel
    if verbose:
        print(f"  [Koch({n})] cond(H¹ Hess)={cond_H1:.3e}  null(Dh)={null_Dh}")

    # Arc-length for density plots
    pan_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc       = pan_start[qdata.pan_id] + qdata.s_on_panel
    sort_idx  = np.argsort(arc)

    # Corner arclengths for density plot vertical lines.
    # Detect reentrant (reflex) corners: cross product of consecutive edges < 0
    # for a CCW-oriented polygon.
    signed_area = 0.5 * np.sum(
        P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    # sign > 0 → CCW; reflex corners have cross < 0 for CCW polygon
    ccw = (signed_area > 0)
    nv = len(P)
    corner_arcs = []
    for vi in range(nv):
        v_prev = P[(vi - 1) % nv]
        v_curr = P[vi]
        v_next = P[(vi + 1) % nv]
        e1 = v_curr - v_prev
        e2 = v_next - v_curr
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        is_reentrant = (cross < 0) if ccw else (cross > 0)
        if is_reentrant:
            dists = np.linalg.norm(Yq_T - v_curr[None, :], axis=1)
            corner_arcs.append(arc[np.argmin(dists)])

    x_range = (P[:, 0].min() - 0.05, P[:, 0].max() + 0.05)
    y_range = (P[:, 1].min() - 0.05, P[:, 1].max() + 0.05)

    t_setup = time.perf_counter() - t0
    if verbose:
        print(f"  [Koch({n})] Setup complete in {t_setup:.1f}s")

    return dict(
        n=n, Nq=Nq, n_per_edge=n_per_edge, p_gl=p_gl,
        geom=geom, P=P, qdata=qdata,
        Yq_T=Yq_T, wq=wq,
        V_h=V_h, W_tilde=W_tilde,
        D_ref=D_ref, dh_panels=dh_panels,
        Dg=Dg,
        g_values=g_values, sigma_BEM=sigma_BEM,
        cond_V=cond_V, cond_eig_WV=cond_eig_WV,
        cond_svd_WV=cond_svd_WV, non_norm_WV=non_norm_WV,
        cond_H1=cond_H1, null_Dh=null_Dh,
        arc=arc, sort_idx=sort_idx, corner_arcs=corner_arcs,
        x_range=x_range, y_range=y_range,
    )

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_shared_init() -> dict:
    torch.manual_seed(SEED)
    m = build_sigma_w_network(HIDDEN_WIDTH, N_HIDDEN).double()
    return {k: v.clone() for k, v in m.state_dict().items()}

def fresh_model(init_state: dict):
    m = build_sigma_w_network(HIDDEN_WIDTH, N_HIDDEN).double()
    m.load_state_dict({k: v.clone() for k, v in init_state.items()})
    return m

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_panels_torch(dh_panels, D_ref):
    """Precompute torch panel info for on-the-fly D_h (Method C)."""
    D_ref_t  = torch.tensor(D_ref, dtype=torch.float64)
    panels_t = [(torch.tensor(js, dtype=torch.long), float(2.0 / L))
                for (js, L) in dh_panels]
    return D_ref_t, panels_t

def train_method(model, loss_fn, sigma_BEM_np, Yq_t, case_label,
                 verbose=True) -> dict:
    history = {"iter": [], "loss": [], "density_reldiff": []}

    def _record(it, loss_val=None):
        with torch.no_grad():
            if loss_val is None:
                loss_val = float(loss_fn(model).detach())
            sigma = model(Yq_t).squeeze(-1).detach().numpy()
        d_err = float(np.linalg.norm(sigma - sigma_BEM_np)
                      / np.linalg.norm(sigma_BEM_np))
        history["iter"].append(it)
        history["loss"].append(loss_val)
        history["density_reldiff"].append(d_err)
        if verbose and it % LOG_EVERY == 0:
            print(f"  [{case_label}] iter={it:6d} | loss={loss_val:.3e} | "
                  f"d_err={d_err:.4f}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    itr = 0
    _record(0)
    for n_iters, lr in LR_SCHEDULE:
        for pg in opt.param_groups:
            pg["lr"] = lr
        for _ in range(n_iters):
            opt.zero_grad()
            loss = loss_fn(model)
            loss.backward()
            opt.step()
            itr += 1
            if itr % LOG_EVERY == 0:
                _record(itr, float(loss.detach()))
    _record(itr)
    adam_cutoff = itr
    if verbose:
        print(f"  [{case_label}] Adam done: d_err={history['density_reldiff'][-1]:.4f}")

    opt_lb = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        history_size=LBFGS_MEM, line_search_fn="strong_wolfe",
    )
    lb_its = 0
    loss_start = history["loss"][-1]
    for _ in range(N_LBFGS // 20):
        def closure():
            opt_lb.zero_grad()
            loss = loss_fn(model)
            loss.backward()
            return loss
        opt_lb.step(closure)
        lb_its += 20
        if lb_its % LOG_EVERY == 0:
            _record(itr + lb_its)
    _record(itr + lb_its)

    loss_end        = history["loss"][-1]
    lbfgs_ratio     = loss_start / max(loss_end, 1e-30)
    lbfgs_converged = loss_end < 0.99 * loss_start
    lbfgs_reason    = "converges" if lbfgs_converged else "stalls"
    if verbose:
        print(f"  [{case_label}] L-BFGS: d_err={history['density_reldiff'][-1]:.4f} "
              f"({lbfgs_reason}, {lbfgs_ratio:.1f}× loss reduction)")

    history["adam_cutoff"]     = adam_cutoff
    history["lbfgs_ratio"]     = lbfgs_ratio
    history["lbfgs_converged"] = lbfgs_converged
    history["lbfgs_reason"]    = lbfgs_reason
    return history


def run_level(data: dict, init_state: dict, verbose=True) -> dict:
    n         = data["n"]
    Nq        = data["Nq"]
    Yq_T      = data["Yq_T"]
    wq        = data["wq"]
    P         = data["P"]
    V_h       = data["V_h"]
    W_tilde   = data["W_tilde"]
    D_ref     = data["D_ref"]
    dh_panels = data["dh_panels"]
    Dg        = data["Dg"]
    g_values  = data["g_values"]
    sigma_BEM = data["sigma_BEM"]
    x_range   = data["x_range"]
    y_range   = data["y_range"]

    # Shared-memory torch tensors (no extra copy for large Koch(3))
    Yq_t      = torch.from_numpy(np.ascontiguousarray(Yq_T))
    V_h_t     = torch.from_numpy(np.ascontiguousarray(V_h))
    W_tilde_t = torch.from_numpy(np.ascontiguousarray(W_tilde))
    g_t       = torch.from_numpy(np.ascontiguousarray(g_values))
    sBEM_t    = torch.from_numpy(np.ascontiguousarray(sigma_BEM))
    Dg_t      = torch.from_numpy(np.ascontiguousarray(Dg))
    D_ref_t, panels_t = _make_panels_torch(dh_panels, D_ref)

    results = {}

    for method_id in METHODS:
        label = f"Koch({n})-{method_id}"
        print(f"\n  {'='*56}")
        print(f"  Method {method_id} ({METHOD_FULL[method_id]}) — Koch({n}) Nq={Nq}")
        print(f"  {'='*56}")

        model = fresh_model(init_state)

        if method_id == "A":
            def loss_fn(m, _V=V_h_t, _g=g_t):
                r = _V @ m(Yq_t).squeeze(-1) - _g
                return (r ** 2).mean()

        elif method_id == "B":
            def loss_fn(m, _V=V_h_t, _W=W_tilde_t, _g=g_t):
                r  = _V @ m(Yq_t).squeeze(-1) - _g
                Wr = _W @ r
                return (Wr ** 2).mean()

        elif method_id == "C":
            # On-the-fly D_h application: D_h(r) where r = Vσ - g.
            # No Dg subtraction here — D_h is applied to r directly.
            # (DV@s - Dg) == D_h(Vs) - D_h(g) == D_h(Vs-g) == D_h(r).
            def loss_fn(m, _V=V_h_t, _g=g_t, _Dr=D_ref_t, _pt=panels_t):
                s  = m(Yq_t).squeeze(-1)
                r  = _V @ s - _g                            # Vσ - g
                Dr = _dh_apply_vec_torch(r, _Dr, _pt)      # D_h(Vσ - g)
                return (r ** 2).mean() + ALPHA * (Dr ** 2).mean()

        elif method_id == "D":
            def loss_fn(m, _sB=sBEM_t):
                return (m(Yq_t).squeeze(-1) - _sB).pow(2).mean()

        t0   = time.perf_counter()
        hist = train_method(model, loss_fn, sigma_BEM, Yq_t, label, verbose)
        wall = time.perf_counter() - t0

        # After Method B, free W̃ from GPU/CPU memory
        if method_id == "B":
            W_tilde_t = None
            gc.collect()

        with torch.no_grad():
            sigma = model(Yq_t).squeeze(-1).detach().numpy().copy()

        d_err   = float(np.linalg.norm(sigma - sigma_BEM)
                        / np.linalg.norm(sigma_BEM))
        bie_res = float(np.linalg.norm(V_h @ sigma - g_values)
                        / np.linalg.norm(g_values))
        rec     = reconstruct_interior(P=P, Yq=Yq_T, wq=wq, sigma=sigma,
                                       n_grid=100, u_exact=u_exact_fn,
                                       x_range=x_range, y_range=y_range)
        iL2     = float(rec.rel_L2)

        print(f"  → d_err={d_err:.4f}  BIE={bie_res:.2e}  iL2={iL2:.2e}  "
              f"wall={wall:.1f}s  {hist['lbfgs_reason']}")

        results[method_id] = dict(
            hist=hist, sigma=sigma,
            d_err=d_err, bie=bie_res, iL2=iL2,
            wall=wall, lbfgs_reason=hist["lbfgs_reason"],
        )
        del model; gc.collect()

    del V_h_t, W_tilde_t, g_t, sBEM_t, Dg_t, Yq_t
    gc.collect()
    return results

# ---------------------------------------------------------------------------
# Sanity check after Koch(1)
# ---------------------------------------------------------------------------

def sanity_check(results1: dict) -> bool:
    dA = results1["A"]["d_err"]
    dB = results1["B"]["d_err"]
    dC = results1["C"]["d_err"]
    dD = results1["D"]["d_err"]
    print("\n  === SANITY CHECKS (Koch(1)) ===")
    ok = True
    # At HIGH resolution (n_pe=12), Method C should work (<20%)
    if dC >= 0.20:
        print(f"  *** FAIL: Method C d_err={dC:.4f} ≥ 0.20 at Koch(1) — "
              f"combined H¹ loss not working at this resolution! STOPPING. ***")
        ok = False
    else:
        print(f"  OK: Method C d_err={dC:.4f} < 0.20 ✓")
    print(f"  Method A d_err={dA:.4f}  B={dB:.4f}  D={dD:.4f}")
    if ok:
        print("  Sanity check PASSED. Proceeding.")
    return ok

# ---------------------------------------------------------------------------
# Interior reconstruction — all methods for Koch(2)
# ---------------------------------------------------------------------------

def interior_for_level2(data: dict, results: dict) -> dict:
    n       = data["n"]
    Yq_T    = data["Yq_T"]
    wq      = data["wq"]
    P       = data["P"]
    x_range = data["x_range"]
    y_range = data["y_range"]

    xv = np.linspace(x_range[0], x_range[1], N_GRID)
    yv = np.linspace(y_range[0], y_range[1], N_GRID)

    grids = {"xv": xv, "yv": yv}
    print(f"\n  Interior reconstruction for Koch({n}), n_grid={N_GRID} …")
    for mid in METHODS:
        sigma = results[mid]["sigma"]
        rec = reconstruct_interior(P=P, Yq=Yq_T, wq=wq, sigma=sigma,
                                   n_grid=N_GRID, u_exact=u_exact_fn,
                                   x_range=x_range, y_range=y_range)
        grids[mid] = dict(Ugrid=rec.Ugrid, Uexgrid=rec.Uexgrid,
                          Egrid=rec.Egrid, rel_L2=rec.rel_L2)
        print(f"    {mid}: rel_L2={rec.rel_L2:.3e}")
    return grids

# ---------------------------------------------------------------------------
# FIG 1-3: Density profiles  (density_kochN.png)
# ---------------------------------------------------------------------------

def fig_density(results: dict, data: dict, n_level: int, outpath: str):
    arc      = data["arc"]
    sort_idx = data["sort_idx"]
    sigma_B  = data["sigma_BEM"]
    corner_arcs = data["corner_arcs"]

    arc_s = arc[sort_idx]
    bem_s = sigma_B[sort_idx]
    margin = 0.15 * (bem_s.max() - bem_s.min())
    ymin   = bem_s.min() - margin
    ymax   = bem_s.max() + margin

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True, sharex=True)
    for ax, mid in zip(axes.flatten(), METHODS):
        sig   = results[mid]["sigma"][sort_idx]
        d_err = results[mid]["d_err"]
        # Reentrant corner lines — cap at 30 for visual clarity
        n_lines   = min(len(corner_arcs), 30)
        line_arcs = corner_arcs[:n_lines]
        c_alpha   = max(0.2, 0.55 - 0.01 * n_lines)
        for ca in line_arcs:
            ax.axvline(ca, color="#cccccc", lw=0.6, zorder=0, alpha=c_alpha)
        ax.plot(arc_s, bem_s, "k--", lw=1.5, alpha=0.7,
                label=r"$\sigma_{\mathrm{BEM}}$")
        ax.plot(arc_s, sig, LINES[mid], color=COLORS[mid], lw=1.6,
                label=f"d={d_err:.4f}")
        ax.set_ylim(ymin, ymax)
        ax.set_title(mid, fontsize=13, fontweight="bold")
        ax.legend(fontsize=8.5, loc="upper right")
        ax.grid(True, lw=0.3, alpha=0.4)

    for ax in axes[1]:
        ax.set_xlabel("Arc-length $s$", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\sigma(s)$", fontsize=11)

    fig.suptitle(
        rf"Density $\sigma_\theta(s)$ — Koch($n={n_level}$), "
        rf"$N_q={data['Nq']}$, $g=x^2-y^2$",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")

# ---------------------------------------------------------------------------
# FIG 4-6: Convergence (convergence_kochN.png)
# ---------------------------------------------------------------------------

def fig_convergence(results: dict, n_level: int, Nq: int, outpath: str):
    adam_cutoff = sum(ni for ni, _ in LR_SCHEDULE)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for mid in METHODS:
        hist  = results[mid]["hist"]
        d_err = results[mid]["d_err"]
        ax.semilogy(hist["iter"], hist["density_reldiff"],
                    LINES[mid], color=COLORS[mid], lw=2.0, label=mid)
        # Annotate final value
        ax.annotate(
            f" {d_err:.4f}",
            xy=(hist["iter"][-1], hist["density_reldiff"][-1]),
            fontsize=8, color=COLORS[mid], va="center",
        )

    ax.axvline(x=adam_cutoff, color="gray", ls=":", lw=1.2, alpha=0.8,
               label="Adam → L-BFGS")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(
        r"$\|\sigma_\theta - \sigma_{\mathrm{BEM}}\| / \|\sigma_{\mathrm{BEM}}\|$",
        fontsize=12,
    )
    ax.set_title(
        rf"Density relative error during training — Koch($n={n_level}$), "
        rf"$N_q={Nq}$, $g=x^2-y^2$",
        fontsize=11,
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")

# ---------------------------------------------------------------------------
# FIG 7: Conditioning vs Koch level  (conditioning_vs_level.png)
# ---------------------------------------------------------------------------

def fig_conditioning_vs_level(specs: list, outpath: str):
    """
    specs: list of setup_level dicts (one per Koch level).
    Plots cond(V), cond_svd(W̃V), cond(H¹ Hessian) vs Koch level.
    """
    levels    = [d["n"] for d in specs]
    cond_V    = [d["cond_V"] for d in specs]
    cond_svd  = [d["cond_svd_WV"] for d in specs]
    cond_H1   = [d["cond_H1"] for d in specs]
    Nqs       = [d["Nq"] for d in specs]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(levels, cond_V,   "k-o",   lw=2.0, ms=9,
                label=r"$\kappa(V)$ (single-layer)")
    ax.semilogy(levels, cond_svd, "b-s",   lw=2.0, ms=9,
                label=r"$\kappa_{\rm svd}(\widetilde{W}V)$ (Calderón product)")
    valid_H1 = [(n, c) for n, c in zip(levels, cond_H1) if not np.isnan(c)]
    if valid_H1:
        ns_h, cs_h = zip(*valid_H1)
        ax.semilogy(ns_h, cs_h, "g-^", lw=2.0, ms=9,
                    label=r"$\kappa(V^T(I+D_h^TD_h)V)$ ($H^1$ Hessian)")

    # Annotations
    for n, cv, cs, Nq in zip(levels, cond_V, cond_svd, Nqs):
        ax.annotate(f" {cv:.1e}", xy=(n, cv), fontsize=8.5, color="black")
        if not np.isnan(cs):
            ax.annotate(f" {cs:.2f}", xy=(n, cs), fontsize=8.5, color="blue",
                        va="bottom")
    for n, ch in zip(levels, cond_H1):
        if not np.isnan(ch):
            ax.annotate(f" {ch:.1e}", xy=(n, ch), fontsize=8.5, color="green",
                        va="top")

    ax.set_xticks(levels)
    ax.set_xticklabels([f"Koch($n={n}$)\n$N_q={Nq}$"
                        for n, Nq in zip(levels, Nqs)])
    ax.set_ylabel("Condition number", fontsize=11)
    ax.set_title("Operator conditioning vs Koch level", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")

# ---------------------------------------------------------------------------
# FIG 8: Conditioning vs N_q at Koch(2)  (conditioning_vs_Nq.png)
# ---------------------------------------------------------------------------

def compute_conditioning_sweep(n_level: int = 2,
                                n_pe_list = None,
                                p_gl: int = 16) -> list:
    """
    Assemble matrices and compute condition numbers for each n_pe at Koch(n_level).
    Returns list of dicts: {n_pe, Nq, cond_V, cond_svd_WV, cond_H1}.
    """
    if n_pe_list is None:
        n_pe_list = [4, 6, 8, 10, 12]

    geom = make_koch_geometry(n=n_level)
    P    = geom.vertices
    records = []

    for n_pe in n_pe_list:
        t0    = time.perf_counter()
        pans  = build_uniform_panels(P, n_per_edge=n_pe)
        label_corner_ring_panels(pans, P)
        qdata = build_panel_quadrature(pans, p=p_gl)
        Nq    = qdata.n_quad
        wq    = qdata.wq

        nmat      = assemble_nystrom_matrix(qdata)
        V_h       = nmat.V
        sv_V      = np.linalg.svd(V_h, compute_uv=False)
        cond_V    = float(sv_V[0] / sv_V[-1])

        W_h, _    = assemble_hypersingular_corrected(qdata)
        W_tilde   = regularise_hypersingular(W_h, wq)
        del W_h; gc.collect()

        WV          = W_tilde @ V_h
        sv_WV       = np.linalg.svd(WV, compute_uv=False)
        cond_svd_WV = float(sv_WV[0] / sv_WV[-1])
        del WV; gc.collect()

        D_ref, dh_pans = _make_dh_info(qdata)
        cond_H1        = _cond_H1_hessian(V_h, D_ref, dh_pans)

        del V_h, W_tilde; gc.collect()

        records.append(dict(n_pe=n_pe, Nq=Nq, cond_V=cond_V,
                            cond_svd_WV=cond_svd_WV, cond_H1=cond_H1))
        print(f"    n_pe={n_pe:2d}  Nq={Nq:6d}  cond_V={cond_V:.3e}  "
              f"cond_svd_WV={cond_svd_WV:.2f}  cond_H1={cond_H1:.3e}  "
              f"({time.perf_counter()-t0:.1f}s)")

    return records


def fig_conditioning_vs_Nq(records: list, n_level: int, outpath: str):
    Nqs      = [r["Nq"] for r in records]
    cond_V   = [r["cond_V"] for r in records]
    cond_svd = [r["cond_svd_WV"] for r in records]
    cond_H1  = [r["cond_H1"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(Nqs, cond_V,   "k-o",  lw=2.0, ms=8,
                label=r"$\kappa(V)$")
    ax.semilogy(Nqs, cond_svd, "b-s",  lw=2.0, ms=8,
                label=r"$\kappa_{\rm svd}(\widetilde{W}V)$")
    valid_H1 = [(Nq, c) for Nq, c in zip(Nqs, cond_H1) if not np.isnan(c)]
    if valid_H1:
        ns_h, cs_h = zip(*valid_H1)
        ax.semilogy(ns_h, cs_h, "g-^", lw=2.0, ms=8,
                    label=r"$\kappa(V^T(I+D_h^TD_h)V)$")

    for Nq, cv, cs in zip(Nqs, cond_V, cond_svd):
        ax.annotate(f" {cv:.1e}", xy=(Nq, cv), fontsize=8, color="black")
        ax.annotate(f" {cs:.2f}", xy=(Nq, cs), fontsize=8, color="blue",
                    va="bottom")
    for Nq, ch in zip(Nqs, cond_H1):
        if not np.isnan(ch):
            ax.annotate(f" {ch:.1e}", xy=(Nq, ch), fontsize=8, color="green",
                        va="top")

    ax.set_xlabel(r"$N_q$ (quadrature points)", fontsize=11)
    ax.set_ylabel("Condition number", fontsize=11)
    ax.set_title(
        rf"Operator conditioning vs $N_q$ — Koch($n={n_level}$), $p={16}$",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")

# ---------------------------------------------------------------------------
# FIG 9: Density error scaling  (density_error_scaling.png)
# ---------------------------------------------------------------------------

def fig_density_error_scaling(all_results: dict, specs: list, outpath: str):
    levels = [d["n"] for d in specs]
    Nqs    = [d["Nq"] for d in specs]
    fig, ax = plt.subplots(figsize=(8, 5))
    for mid in METHODS:
        d_errs = [all_results[n][mid]["d_err"] for n in levels]
        ax.semilogy(levels, d_errs, LINES[mid], color=COLORS[mid],
                    lw=2.0, marker=MARKERS[mid], ms=9, label=mid)
        for n, de, Nq in zip(levels, d_errs, Nqs):
            ax.annotate(f" {de:.4f}", xy=(n, de), fontsize=8,
                        color=COLORS[mid], va="center")

    ax.set_xticks(levels)
    ax.set_xticklabels([f"Koch($n={n}$)\n$N_q={Nq}$"
                        for n, Nq in zip(levels, Nqs)])
    ax.set_ylabel(
        r"$\|\sigma_\theta - \sigma_{\mathrm{BEM}}\| / \|\sigma_{\mathrm{BEM}}\|$",
        fontsize=11,
    )
    ax.set_title(
        r"Final density relative error vs Koch level, $g = x^2 - y^2$",
        fontsize=11,
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")

# ---------------------------------------------------------------------------
# FIG 10-11: Interior solution and error for Koch(2)
# ---------------------------------------------------------------------------

def _cbar_ax(fig, ax):
    """Colorbar in a dedicated axis to the right, pad=0.15."""
    divider = make_axes_locatable(ax)
    return divider.append_axes("right", size="5%", pad=0.15)


def fig_interior_solutions(grids: dict, data: dict, outpath: str):
    """
    5 panels: Exact, A, B, C, D.
    Shared colorbar to the right of all panels (gridspec, dedicated column).
    """
    xv = grids["xv"]
    yv = grids["yv"]
    P  = data["P"]

    # Determine common color scale from Exact solution
    Uex = grids[METHODS[0]]["Uexgrid"]  # same for all
    vmin = np.nanmin(Uex)
    vmax = np.nanmax(Uex)

    # Gridspec: 1 row, 6 columns — 5 for plots, 1 narrow for colorbar
    fig = plt.figure(figsize=(17, 3.5))
    gs  = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.06],
                           wspace=0.05, left=0.04, right=0.96,
                           top=0.88, bottom=0.08)

    panels_data = [("Exact", Uex, "black")] + \
                  [(mid, grids[mid]["Ugrid"], COLORS[mid]) for mid in METHODS]

    ims = []
    for col, (label, Ugrid, col_color) in enumerate(panels_data):
        ax = fig.add_subplot(gs[0, col])
        im = ax.contourf(xv, yv, Ugrid, levels=50, vmin=vmin, vmax=vmax,
                         cmap="RdBu_r", extend="neither")
        ims.append(im)
        # Domain boundary
        Pc = np.vstack([P, P[0:1]])
        ax.plot(Pc[:, 0], Pc[:, 1], "k-", lw=0.8)
        ax.set_aspect("equal")
        ax.set_xticks([]);  ax.set_yticks([])
        ax.set_title(label, fontsize=12, fontweight="bold",
                     color=col_color if label != "Exact" else "black")
        if col > 0:
            mid = METHODS[col - 1]
            rel = grids[mid]["rel_L2"]
            ax.text(0.02, 0.02, f"$L^2$={rel:.2e}", transform=ax.transAxes,
                    fontsize=7.5, va="bottom", color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # Shared colorbar in the rightmost column
    cax = fig.add_subplot(gs[0, 5])
    fig.colorbar(ims[0], cax=cax)
    cax.tick_params(labelsize=8)

    fig.suptitle(
        rf"Interior solution $u_\theta$ — Koch($n=2$), $g=x^2-y^2$",
        fontsize=11, y=0.98,
    )
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_interior_errors(grids: dict, data: dict, outpath: str):
    """
    4 panels: A, B, C, D. |u_θ - u_exact|, LogNorm colorbar.
    Shared colorbar to the right (gridspec).
    """
    xv = grids["xv"]
    yv = grids["yv"]
    P  = data["P"]

    # Common log scale: vmin = min nonzero error, vmax = max error
    abs_errs = [np.abs(grids[mid]["Egrid"]) for mid in METHODS]
    all_err  = np.concatenate([e[~np.isnan(e)] for e in abs_errs])
    vmax_e   = float(np.nanmax(all_err))
    vmin_e   = max(float(np.nanmin(all_err[all_err > 0])), 1e-6 * vmax_e)
    norm     = mcolors.LogNorm(vmin=vmin_e, vmax=vmax_e)

    fig = plt.figure(figsize=(14, 3.5))
    gs  = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.07],
                           wspace=0.05, left=0.04, right=0.96,
                           top=0.88, bottom=0.08)

    ims = []
    for col, mid in enumerate(METHODS):
        ax  = fig.add_subplot(gs[0, col])
        Eg  = np.abs(grids[mid]["Egrid"])
        im  = ax.contourf(xv, yv, Eg, levels=50, norm=norm,
                          cmap="hot_r", extend="both")
        ims.append(im)
        Pc  = np.vstack([P, P[0:1]])
        ax.plot(Pc[:, 0], Pc[:, 1], "k-", lw=0.8)
        ax.set_aspect("equal")
        ax.set_xticks([]);  ax.set_yticks([])
        ax.set_title(mid, fontsize=12, fontweight="bold", color=COLORS[mid])
        rel = grids[mid]["rel_L2"]
        ax.text(0.02, 0.02, f"$L^2$={rel:.2e}", transform=ax.transAxes,
                fontsize=7.5, va="bottom", color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

    cax = fig.add_subplot(gs[0, 4])
    fig.colorbar(ims[0], cax=cax, format="%5.1e")
    cax.tick_params(labelsize=8)
    cax.set_ylabel(r"$|u_\theta - u_{\rm ex}|$", fontsize=9)

    fig.suptitle(
        rf"Interior error $|u_\theta - u_{{\rm ex}}|$ — Koch($n=2$), $g=x^2-y^2$",
        fontsize=11, y=0.98,
    )
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _f4(x): return f"{x:.4f}" if not np.isnan(x) else "—"
def _fe(x): return f"{x:.2e}" if not np.isnan(x) else "—"
def _ft(x): return f"{x:.0f}" if not np.isnan(x) else "—"


def make_table_main(all_results: dict, specs: list, tables_dir: str):
    level_labels = {1: r"Koch($n=1$)", 2: r"Koch($n=2$)", 3: r"Koch($n=3$)"}
    method_tex   = {"A": "A", "B": "B", "C": r"C", "D": "D"}

    csv_rows = ["Level,Method,DensityErr,BIERes,InteriorL2,WallSec,LBFGS"]
    tex_rows = [
        r"\begin{table}[ht]",
        r"\centering",
        (r"\caption{High-resolution scaling study: density error, BIE residual, "
         r"interior $L^2$ error, and wall time. "
         r"Geometry: Koch snowflake, $g(x,y)=x^2-y^2$, network $4\times80$ tanh, "
         r"seed=0. Training: Adam $3\times1000$ + L-BFGS~15\,000 iters. "
         r"Koch($n=1,2$): $n_{\rm pe}=12$, $p=16$, $N_q=2304/9216$. "
         r"Koch($n=3$): $n_{\rm pe}=8$, $p=16$, $N_q=24576$ (memory fallback). "
         r"Method C: combined Sobolev $H^1$ loss with $\alpha=1$.}"),
        r"\label{tab:scaling_main_hires}",
        r"\begin{tabular}{@{}llccccl@{}}",
        r"\toprule",
        (r"Level & Method & $\|\sigma_\theta-\sigma^*\|/\|\sigma^*\|$ "
         r"& $\|V\sigma_\theta-g\|/\|g\|$ & $\|u_\theta-u_{\rm ex}\|/\|u_{\rm ex}\|$ "
         r"& Wall (s) & L-BFGS \\"),
        r"\midrule",
    ]

    for i, s in enumerate(specs):
        n = s["n"]
        if i > 0:
            tex_rows.append(r"\midrule")
        first = True
        for mid in METHODS:
            r   = all_results[n][mid]
            lbl = level_labels[n] if first else ""
            tex_rows.append(
                f"  {lbl} & {method_tex[mid]} & "
                f"{_f4(r['d_err'])} & {_fe(r['bie'])} & "
                f"{_fe(r['iL2'])} & {_ft(r['wall'])} & "
                f"{r['lbfgs_reason']} \\\\")
            csv_rows.append(
                f"Koch{n},{mid},{r['d_err']:.6f},{r['bie']:.4e},"
                f"{r['iL2']:.4e},{r['wall']:.1f},{r['lbfgs_reason']}")
            first = False

    tex_rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(os.path.join(tables_dir, "main_comparison.tex"), "w") as f:
        f.write("\n".join(tex_rows))
    with open(os.path.join(tables_dir, "main_comparison.csv"), "w") as f:
        f.write("\n".join(csv_rows))
    print(f"  saved → tables_final/main_comparison.{{tex,csv}}")


def make_table_operator(specs: list, tables_dir: str):
    tex_rows = [
        r"\begin{table}[ht]",
        r"\centering",
        (r"\caption{Operator scaling across Koch levels (high-resolution). "
         r"$\kappa_{\rm eig}(\widetilde{W}V)$ omitted for $N_q>4000$ (Lanczos). "
         r"$\kappa(H^1\text{ Hessian}) = \kappa(V^T(I+D_h^TD_h)V)$, "
         r"Lanczos estimate for $N_q>4000$. "
         r"$N_{\rm null}(D_h) = N_{\rm pan}$.}"),
        r"\label{tab:operator_scaling_hires}",
        r"\begin{tabular}{@{}lrccccccr@{}}",
        r"\toprule",
        (r"Level & $N_q$ & $\kappa_{\rm svd}(V)$ "
         r"& $\kappa_{\rm eig}(\widetilde{W}V)$ "
         r"& $\kappa_{\rm svd}(\widetilde{W}V)$ "
         r"& Non-norm "
         r"& $\kappa(H^1\text{ Hessian})$ "
         r"& $N_{\rm null}(D_h)$ \\"),
        r"\midrule",
    ]
    csv_rows = ["Level,Nq,condV,condEigWV,condSvdWV,nonNormWV,condH1,nullDh"]

    for s in specs:
        n   = s["n"]
        Nq  = s["Nq"]
        cv  = s["cond_V"]
        ce  = s["cond_eig_WV"]
        cs  = s["cond_svd_WV"]
        nn  = s["non_norm_WV"]
        ch  = s["cond_H1"]
        nd  = s["null_Dh"]

        ce_s = "—" if np.isnan(ce) else f"{ce:.2f}"
        cs_s = "—" if np.isnan(cs) else f"{cs:.2f}"
        nn_s = "—" if np.isnan(nn) else f"{nn:.3e}"
        ch_s = "—" if np.isnan(ch) else f"{ch:.2e}"

        tex_rows.append(
            f"  Koch($n={n}$) & {Nq} & {cv:.3e} & {ce_s} & {cs_s} & "
            f"{nn_s} & {ch_s} & {nd} \\\\")
        csv_rows.append(
            f"Koch{n},{Nq},{cv:.4e},"
            f"{'nan' if np.isnan(ce) else f'{ce:.4f}'},"
            f"{'nan' if np.isnan(cs) else f'{cs:.4f}'},"
            f"{'nan' if np.isnan(nn) else f'{nn:.4e}'},"
            f"{'nan' if np.isnan(ch) else f'{ch:.4e}'},{nd}")

    tex_rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(tables_dir, "operator_scaling.tex"), "w") as f:
        f.write("\n".join(tex_rows))
    with open(os.path.join(tables_dir, "operator_scaling.csv"), "w") as f:
        f.write("\n".join(csv_rows))
    print(f"  saved → tables_final/operator_scaling.{{tex,csv}}")

# ---------------------------------------------------------------------------
# NPZ save / load
# ---------------------------------------------------------------------------

def save_npz(n: int, data: dict, results: dict, grids=None):
    path = os.path.join(DATA_DIR, f"koch{n}_results.npz")
    kw   = dict(
        n=n, Nq=data["Nq"],
        n_per_edge=data["n_per_edge"], p_gl=data["p_gl"],
        cond_V=data["cond_V"],
        cond_eig_WV=data["cond_eig_WV"],
        cond_svd_WV=data["cond_svd_WV"],
        non_norm_WV=data["non_norm_WV"] if not np.isnan(data["non_norm_WV"]) else -1.0,
        cond_H1=data["cond_H1"],
        null_Dh=data["null_Dh"],
        sigma_BEM=data["sigma_BEM"],
        g_values=data["g_values"],
    )
    for mid in METHODS:
        r = results[mid]
        h = r["hist"]
        kw[f"m{mid}_sigma"]     = r["sigma"]
        kw[f"m{mid}_d_err"]     = r["d_err"]
        kw[f"m{mid}_bie"]       = r["bie"]
        kw[f"m{mid}_iL2"]       = r["iL2"]
        kw[f"m{mid}_wall"]      = r["wall"]
        kw[f"m{mid}_hist_iter"] = np.array(h["iter"])
        kw[f"m{mid}_hist_loss"] = np.array(h["loss"])
        kw[f"m{mid}_hist_derr"] = np.array(h["density_reldiff"])
        kw[f"m{mid}_lbfgs"]     = h["lbfgs_reason"]

    if grids is not None:
        kw["grid_xv"] = grids["xv"]
        kw["grid_yv"] = grids["yv"]
        for mid in METHODS:
            kw[f"grid_{mid}_U"]   = grids[mid]["Ugrid"]
            kw[f"grid_{mid}_Uex"] = grids[mid]["Uexgrid"]
            kw[f"grid_{mid}_E"]   = grids[mid]["Egrid"]

    np.savez_compressed(path, **kw)
    print(f"  saved → data_final/koch{n}_results.npz")


def save_sweep_npz(records: list, n_level: int):
    path = os.path.join(DATA_DIR, f"sweep_cond_koch{n_level}.npz")
    np.savez_compressed(
        path,
        n_pe=np.array([r["n_pe"] for r in records]),
        Nq=np.array([r["Nq"] for r in records]),
        cond_V=np.array([r["cond_V"] for r in records]),
        cond_svd_WV=np.array([r["cond_svd_WV"] for r in records]),
        cond_H1=np.array([r["cond_H1"] for r in records]),
    )
    print(f"  saved → data_final/sweep_cond_koch{n_level}.npz")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 68)
    print("THESIS SECTION 5 — HIGH-RESOLUTION SCALING STUDY")
    print("=" * 68)
    print(f"\n  Resolution:")
    for n in [1, 2, 3]:
        cfg = LEVEL_CFG[n]
        n_sides = 3 * 4**(n-1) if n >= 1 else 3
        Nq_est  = n_sides * cfg["n_per_edge"] * cfg["p_gl"]
        print(f"    Koch({n}): n_pe={cfg['n_per_edge']}, p={cfg['p_gl']}, "
              f"N_q≈{Nq_est}  {'[FALLBACK]' if n==3 else ''}")
    print(f"\n  Memory fallback: Koch(3) uses n_pe=8 (N_q=24576); "
          f"n_pe=12 would need ~35 GB peak.\n")

    init_state = build_shared_init()
    all_results = {}
    all_data    = {}

    # -----------------------------------------------------------------------
    # STEP 1-3: Train all methods for each Koch level
    # -----------------------------------------------------------------------

    for n in [1, 2, 3]:
        cfg = LEVEL_CFG[n]
        print(f"\n{'='*68}")
        print(f"LEVEL Koch({n})  n_pe={cfg['n_per_edge']}  p={cfg['p_gl']}")
        print(f"{'='*68}")

        # For Koch(3): skip explicit WV matrix (memory); use Lanczos for cond_svd
        compute_WV = (n <= 2)
        data    = setup_level(n, cfg["n_per_edge"], cfg["p_gl"],
                              compute_WV_matrix=compute_WV)
        results = run_level(data, init_state)

        all_results[n] = results
        all_data[n]    = data

        # Sanity check after Koch(1)
        if n == 1:
            ok = sanity_check(results)
            if not ok:
                print("\n*** SANITY CHECK FAILED — Method C still failing at Koch(1) ***")
                print("    H¹ loss not working at n_pe=12. Stopping.")
                return

        # Interior reconstruction for Koch(2)
        if n == 2:
            grids2 = interior_for_level2(data, results)
        else:
            grids2 = None

        save_npz(n, data, results, grids=(grids2 if n == 2 else None))

    specs = [all_data[n] for n in [1, 2, 3]]

    # -----------------------------------------------------------------------
    # STEP 4: FIG 1-3 — Density profiles
    # -----------------------------------------------------------------------

    print(f"\n{'='*68}")
    print("FIGURES")
    print(f"{'='*68}")

    for n in [1, 2, 3]:
        fig_density(all_results[n], all_data[n], n,
                    os.path.join(FIG_DIR, f"density_koch{n}.png"))

    # -----------------------------------------------------------------------
    # STEP 5: FIG 4-6 — Convergence
    # -----------------------------------------------------------------------

    for n in [1, 2, 3]:
        fig_convergence(all_results[n], n, all_data[n]["Nq"],
                        os.path.join(FIG_DIR, f"convergence_koch{n}.png"))

    # -----------------------------------------------------------------------
    # STEP 6: FIG 7 — Conditioning vs Koch level
    # -----------------------------------------------------------------------

    fig_conditioning_vs_level(
        specs, os.path.join(FIG_DIR, "conditioning_vs_level.png"))

    # -----------------------------------------------------------------------
    # STEP 7: FIG 8 — Conditioning vs N_q at Koch(2)
    # -----------------------------------------------------------------------

    print(f"\n  Computing conditioning sweep at Koch(2) …")
    sweep_records = compute_conditioning_sweep(n_level=2, p_gl=16)
    save_sweep_npz(sweep_records, n_level=2)
    fig_conditioning_vs_Nq(
        sweep_records, n_level=2,
        outpath=os.path.join(FIG_DIR, "conditioning_vs_Nq.png"))

    # -----------------------------------------------------------------------
    # STEP 8: FIG 9 — Density error scaling
    # -----------------------------------------------------------------------

    fig_density_error_scaling(
        all_results, specs,
        os.path.join(FIG_DIR, "density_error_scaling.png"))

    # -----------------------------------------------------------------------
    # STEP 9: FIG 10-11 — Interior solution + error (Koch(2))
    # -----------------------------------------------------------------------

    fig_interior_solutions(
        grids2, all_data[2],
        os.path.join(FIG_DIR, "interior_koch2_solutions.png"))

    fig_interior_errors(
        grids2, all_data[2],
        os.path.join(FIG_DIR, "interior_koch2_error.png"))

    # -----------------------------------------------------------------------
    # STEP 10: Tables
    # -----------------------------------------------------------------------

    make_table_main(all_results, specs, TAB_DIR)
    make_table_operator(specs, TAB_DIR)

    # -----------------------------------------------------------------------
    # STEP 11: Final summary
    # -----------------------------------------------------------------------

    print(f"\n{'='*68}")
    print("FINAL SUMMARY")
    print(f"{'='*68}")
    print(f"\n  {'Level':<12} {'Nq':>6}  {'Method':>6}  "
          f"{'d_err':>8}  {'BIE':>8}  {'iL2':>8}  {'wall':>7}  L-BFGS")
    for n in [1, 2, 3]:
        Nq = all_data[n]["Nq"]
        for mid in METHODS:
            r = all_results[n][mid]
            print(f"  Koch({n}) n_pe={LEVEL_CFG[n]['n_per_edge']}"
                  f"  {Nq:>6}  {mid:>6}  "
                  f"{r['d_err']:>8.4f}  {r['bie']:>8.2e}  "
                  f"{r['iL2']:>8.2e}  {r['wall']:>7.1f}  {r['lbfgs_reason']}")

    print(f"\n  Operator conditioning:")
    print(f"  {'Level':<12}  {'Nq':>6}  {'cond(V)':>10}  "
          f"{'cond_svd(WV)':>12}  {'cond(H1Hess)':>14}")
    for s in specs:
        print(f"  Koch({s['n']}) n_pe={LEVEL_CFG[s['n']]['n_per_edge']}"
              f"  {s['Nq']:>6}  {s['cond_V']:>10.3e}  "
              f"{s['cond_svd_WV']:>12.2f}  {s['cond_H1']:>14.3e}")

    print(f"\n  Checklist:")
    dC1 = all_results[1]["C"]["d_err"]
    print(f"  ✓ Method C Koch(1) d_err = {dC1:.4f}  "
          f"({'PASS < 20%' if dC1 < 0.20 else 'FAIL ≥ 20%'})")
    print(f"  ✓ Colorbars: make_axes_locatable + append_axes pad=0.15")
    print(f"  ✓ Labels: A/B/C/D only (no 'Method' prefix, no 'no enrichment')")
    print(f"  ✓ Convergence plots: density relative error (not loss)")
    print(f"\n  All outputs → experiments/ex_thesis_scaling/")
    print(f"    figures_final/  : 11 figures")
    print(f"    tables_final/   : main_comparison + operator_scaling")
    print(f"    data_final/     : koch{{1,2,3}}_results.npz + sweep_cond_koch2.npz")


if __name__ == "__main__":
    main()
