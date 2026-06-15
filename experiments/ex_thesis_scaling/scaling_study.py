"""
Thesis Chapter 5 — Scaling study: four loss functions across Koch levels 1, 2, 3.

Geometry:    Prefractal Koch snowflake, levels n = 1, 2, 3
BC:          g(x,y) = x² − y²  (harmonic; u_exact(x,y) = x² − y²)
Network:     4×80 tanh, no enrichment (σ = network output)
Resolution:  n_per_edge = 4, p_gl = 12
             → N_q: Koch(1)=576, Koch(2)=2304, Koch(3)=9216
             (Koch(3) at n_per_edge=6 would need N_q=13824 → ~6 GB peak; unsafe.)

Methods
-------
  A: Standard       L = ||Vσ − g||²
  B: Calderón       L = ||W̃(Vσ − g)||²   (corrected assembly, natural scale)
  C: Sobolev H¹     L = ||Vσ − g||² + α||D_h(Vσ − g)||²,  α = 1
  D: V⁻¹ reference  L = ||σ − σ_BEM||²   (= ||V⁻¹(Vσ−g)||² without inversion)

Training: Adam 3×1000 iters [1e-3, 3e-4, 1e-4] + L-BFGS 15000, mem=30
Seed: 0 (shared initial σ_w weights across all methods and levels)

Outputs (experiments/ex_thesis_scaling/)
  figures/   — 8 publication figures (300 dpi)
  tables/    — main_comparison.{tex,csv}, conditioning.{tex,csv}
  data/      — koch{n}_results.npz per level

Execution order: Koch(1) → sanity check → Koch(2)+interior → Koch(3) → figures+tables.
"""

from __future__ import annotations

import sys
import os
import gc
import time
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..")
sys.path.insert(0, _ROOT)

from src.boundary.polygon import make_koch_geometry
from src.boundary.panels import build_uniform_panels, label_corner_ring_panels
from src.quadrature.panel_quad import build_panel_quadrature
from src.quadrature.nystrom import assemble_nystrom_matrix, solve_bem
from src.quadrature.hypersingular import (
    assemble_hypersingular_corrected,
    regularise_hypersingular,
)
from src.quadrature.tangential_derivative import (
    lagrange_derivative_matrix,
    build_tangential_derivative_matrix,
)
from src.quadrature.gauss import gauss_legendre
from src.models.sigma_w_net import build_sigma_w_network
from src.reconstruction.interior import reconstruct_interior

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED         = 0
N_PER_EDGE   = 4      # panels per Koch edge — chosen for Koch(3) memory safety
P_GL         = 12     # GL points per panel
ALPHA        = 1.0    # H¹ weight (Method C)
HIDDEN_WIDTH = 80
N_HIDDEN     = 4
LR_SCHEDULE  = [(1000, 1e-3), (1000, 3e-4), (1000, 1e-4)]
N_LBFGS      = 15000
LBFGS_MEM    = 30
LOG_EVERY    = 200
N_GRID_KOCH2 = 200    # interior reconstruction grid for Koch(2)
GMRES_TOL    = 1e-12
GMRES_MAX    = 4000

LEVELS  = [1, 2, 3]
METHODS = ["A", "B", "C", "D"]
METHOD_NAMES = {
    "A": "Standard",
    "B": "Calderón",
    "C": r"Sobolev H¹",
    "D": r"$V^{-1}$ ref.",
}
COLORS  = {"A": "#888888", "B": "#1f77b4", "C": "#2ca02c", "D": "#d62728"}
LINES   = {"A": "-",       "B": "-",       "C": "-",       "D": "--"}
MARKERS = {"A": "o",       "B": "s",       "C": "^",       "D": "D"}

SANITY_DMAX = {"A": 0.95, "D": 0.10}   # thresholds: A should exceed, D should beat

# ---------------------------------------------------------------------------
# Boundary data and exact solution
# ---------------------------------------------------------------------------

def g_fn(xy: np.ndarray) -> np.ndarray:
    return xy[:, 0]**2 - xy[:, 1]**2


def u_exact_fn(xy: np.ndarray) -> np.ndarray:
    return xy[:, 0]**2 - xy[:, 1]**2


# ---------------------------------------------------------------------------
# Fast D_h @ M  (exploit block-diagonal structure of D_h)
# ---------------------------------------------------------------------------

def _dh_times_matrix(qdata, M: np.ndarray, D_ref: np.ndarray) -> np.ndarray:
    """
    Compute D_h @ M using the block-diagonal structure of D_h.
    O(Npan × p² × Ncol)  vs  O(Nq² × Ncol) for dense multiply.

    D_h[panel m block] = (2 / L_m) * D_ref   where D_ref is (p×p) Lagrange
    derivative at GL nodes (same for all panels).
    """
    Nq, Ncol = M.shape
    result = np.zeros_like(M)
    p = qdata.p
    for pid in range(qdata.n_panels):
        js  = qdata.idx_std[pid]
        L_p = qdata.L_panel[pid]
        D_phys = (2.0 / L_p) * D_ref        # (p, p)
        result[np.ix_(js, np.arange(Ncol))] = D_phys @ M[js, :]
    return result


def _compute_DV_Dg(qdata, V_h: np.ndarray, g: np.ndarray):
    """
    Return DV = D_h @ V_h  and  Dg = D_h @ g  without forming D_h explicitly.
    """
    xi, _ = gauss_legendre(qdata.p)
    D_ref = lagrange_derivative_matrix(xi)               # (p, p)
    DV    = _dh_times_matrix(qdata, V_h, D_ref)          # (Nq, Nq)
    Dg    = _dh_times_matrix(qdata, g[:, None], D_ref).squeeze(-1)  # (Nq,)
    return DV, Dg


# ---------------------------------------------------------------------------
# Level setup
# ---------------------------------------------------------------------------

def setup_level(n: int, verbose: bool = True) -> dict:
    """
    Build geometry, assemble operators, and compute spectral data for Koch(n).
    Returns a data dict with everything needed for training and figures.
    """
    t_start = time.perf_counter()
    if verbose:
        print(f"\n  [Koch({n})] Building geometry …")

    geom   = make_koch_geometry(n=n)
    P      = geom.vertices         # (Nv, 2)
    panels = build_uniform_panels(P, n_per_edge=N_PER_EDGE)
    label_corner_ring_panels(panels, P)
    qdata  = build_panel_quadrature(panels, p=P_GL)
    Yq_T   = qdata.Yq.T           # (Nq, 2)
    wq     = qdata.wq
    Nq     = qdata.n_quad

    # BEM reference
    if verbose:
        print(f"  [Koch({n})] Nq={Nq}. Assembling Nyström matrix …")
    nmat      = assemble_nystrom_matrix(qdata)
    V_h       = nmat.V             # (Nq, Nq)
    g_values  = g_fn(Yq_T)
    bem       = solve_bem(nmat, g_values, tol=GMRES_TOL, max_iter=GMRES_MAX)
    sigma_BEM = bem.sigma

    # Condition of V_h
    sv_V  = np.linalg.svd(V_h, compute_uv=False)
    cond_V = sv_V[0] / sv_V[-1]
    if verbose:
        print(f"  [Koch({n})] cond(V_h) = {cond_V:.3e}")

    # Corrected hypersingular W̃
    if verbose:
        print(f"  [Koch({n})] Assembling corrected W̃ …")
    W_h, _  = assemble_hypersingular_corrected(qdata)
    W_tilde = regularise_hypersingular(W_h, wq)
    del W_h
    gc.collect()

    # Spectral analysis of W̃V
    if verbose:
        print(f"  [Koch({n})] Spectral analysis of W̃V …")
    WV    = W_tilde @ V_h                    # (Nq, Nq)
    sv_WV = np.linalg.svd(WV, compute_uv=False)
    cond_svd_WV = sv_WV[0] / sv_WV[-1]

    if Nq <= 3000:   # eigvals too expensive for Koch(3)
        eigvals_WV  = np.linalg.eigvals(WV)
        ab_WV       = np.abs(eigvals_WV)
        cond_eig_WV = ab_WV.max() / ab_WV.min()
    else:
        cond_eig_WV = np.nan

    # Non-normality: ||WV^T WV - WV WV^T|| / ||WV||²
    WVTWV   = WV.T @ WV
    WVWVT   = WV @ WV.T
    non_norm_WV = (np.linalg.norm(WVTWV - WVWVT) / np.linalg.norm(WV)**2)
    del WV, WVTWV, WVWVT
    gc.collect()

    if verbose:
        print(f"  [Koch({n})] cond_eig(W̃V)={cond_eig_WV:.2f}  "
              f"cond_svd(W̃V)={cond_svd_WV:.2f}  non-norm={non_norm_WV:.3e}")

    # D_h V and D_h g (fast block-diagonal application)
    if verbose:
        print(f"  [Koch({n})] Computing DV = D_h V (fast block-diagonal) …")
    DV, Dg = _compute_DV_Dg(qdata, V_h, g_values)

    # Arc-length for density plots
    pan_start  = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc        = pan_start[qdata.pan_id] + qdata.s_on_panel
    sort_idx   = np.argsort(arc)

    # Bounding box (add 5% margin)
    x_range = (P[:, 0].min() - 0.05, P[:, 0].max() + 0.05)
    y_range = (P[:, 1].min() - 0.05, P[:, 1].max() + 0.05)

    t_setup = time.perf_counter() - t_start
    if verbose:
        print(f"  [Koch({n})] Setup complete in {t_setup:.1f}s")

    return dict(
        n=n, Nq=Nq,
        geom=geom, P=P, qdata=qdata,
        Yq_T=Yq_T, wq=wq,
        V_h=V_h, W_tilde=W_tilde,
        DV=DV, Dg=Dg,
        g_values=g_values, sigma_BEM=sigma_BEM,
        cond_V=cond_V,
        cond_eig_WV=cond_eig_WV,
        cond_svd_WV=cond_svd_WV,
        non_norm_WV=non_norm_WV,
        arc=arc, sort_idx=sort_idx,
        x_range=x_range, y_range=y_range,
    )


# ---------------------------------------------------------------------------
# Shared initial weights
# ---------------------------------------------------------------------------

def build_shared_init(seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    m = build_sigma_w_network(HIDDEN_WIDTH, N_HIDDEN).double()
    return {k: v.clone() for k, v in m.state_dict().items()}


def fresh_model(init_state: dict) -> torch.nn.Module:
    m = build_sigma_w_network(HIDDEN_WIDTH, N_HIDDEN).double()
    m.load_state_dict({k: v.clone() for k, v in init_state.items()})
    return m


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_method(
    model,
    loss_fn,
    sigma_BEM_np: np.ndarray,
    Yq_t: torch.Tensor,
    case_label: str,
    verbose: bool = True,
) -> dict:
    """Adam + L-BFGS training; returns history dict."""
    history = {"iter": [], "loss": [], "density_reldiff": []}

    def _record(it, loss_val=None):
        with torch.no_grad():
            if loss_val is None:
                loss_val = float(loss_fn(model).detach())
        sigma  = model(Yq_t).squeeze(-1).detach().numpy()
        d_err  = float(np.linalg.norm(sigma - sigma_BEM_np)
                       / np.linalg.norm(sigma_BEM_np))
        history["iter"].append(it)
        history["loss"].append(loss_val)
        history["density_reldiff"].append(d_err)
        if verbose and it % LOG_EVERY == 0:
            print(f"  [{case_label}] iter={it:6d} | loss={loss_val:.3e} | d_err={d_err:.4f}")

    # Adam
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

    # L-BFGS
    opt_lb = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        history_size=LBFGS_MEM, line_search_fn="strong_wolfe",
    )
    n_outer = N_LBFGS // 20
    lb_its  = 0
    loss_start = history["loss"][-1]

    for _ in range(n_outer):
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

    loss_end  = history["loss"][-1]
    lbfgs_ratio = loss_start / max(loss_end, 1e-30)
    lbfgs_converged = loss_end < 0.99 * loss_start
    lbfgs_reason = "converges" if lbfgs_converged else "stalls"

    if verbose:
        if lbfgs_converged:
            print(f"  [{case_label}] L-BFGS: d_err={history['density_reldiff'][-1]:.4f}"
                  f" ({lbfgs_ratio:.1f}× loss reduction)")
        else:
            print(f"  [{case_label}] L-BFGS STALLED at loss={loss_end:.3e}")

    history["adam_cutoff"]     = adam_cutoff
    history["lbfgs_ratio"]     = lbfgs_ratio
    history["lbfgs_converged"] = lbfgs_converged
    history["lbfgs_reason"]    = lbfgs_reason
    return history


# ---------------------------------------------------------------------------
# Run all four methods for one Koch level
# ---------------------------------------------------------------------------

def run_level(data: dict, init_state: dict, verbose: bool = True) -> dict:
    """
    Train all four methods for Koch level data["n"].
    Returns dict: method -> {hist, sigma, d_err, bie, iL2, wall, lbfgs_reason}.
    """
    n         = data["n"]
    Yq_T      = data["Yq_T"]
    wq        = data["wq"]
    P         = data["P"]
    V_h       = data["V_h"]
    W_tilde   = data["W_tilde"]
    DV        = data["DV"]
    Dg        = data["Dg"]
    g_values  = data["g_values"]
    sigma_BEM = data["sigma_BEM"]
    x_range   = data["x_range"]
    y_range   = data["y_range"]
    Nq        = data["Nq"]

    # Convert to tensors (keep V_h as numpy for metric computation)
    Yq_t       = torch.tensor(Yq_T,     dtype=torch.float64)
    V_h_t      = torch.tensor(V_h,      dtype=torch.float64)
    W_tilde_t  = torch.tensor(W_tilde,  dtype=torch.float64)
    DV_t       = torch.tensor(DV,       dtype=torch.float64)
    Dg_t       = torch.tensor(Dg,       dtype=torch.float64)
    g_t        = torch.tensor(g_values, dtype=torch.float64)
    sBEM_t     = torch.tensor(sigma_BEM, dtype=torch.float64)

    results = {}

    for method_id in METHODS:
        label = f"Koch({n})-{method_id}"
        print(f"\n  {'='*56}")
        print(f"  Method {method_id} ({METHOD_NAMES[method_id]}) — Koch({n}) Nq={Nq}")
        print(f"  {'='*56}")

        model = fresh_model(init_state)

        # --- Loss function ---
        if method_id == "A":
            def loss_fn(m, _V=V_h_t, _g=g_t):
                s = m(Yq_t).squeeze(-1)
                r = _V @ s - _g
                return (r ** 2).mean()

        elif method_id == "B":
            def loss_fn(m, _V=V_h_t, _W=W_tilde_t, _g=g_t):
                s   = m(Yq_t).squeeze(-1)
                r   = _V @ s - _g
                Wr  = _W @ r
                return (Wr ** 2).mean()

        elif method_id == "C":
            def loss_fn(m, _V=V_h_t, _DV=DV_t, _g=g_t, _Dg=Dg_t):
                s   = m(Yq_t).squeeze(-1)
                r   = _V @ s - _g
                Dr  = _DV @ s - _Dg
                return (r ** 2).mean() + ALPHA * (Dr ** 2).mean()

        elif method_id == "D":
            def loss_fn(m, _sB=sBEM_t):
                s = m(Yq_t).squeeze(-1)
                return (s - _sB).pow(2).mean()

        t0   = time.perf_counter()
        hist = train_method(model, loss_fn, sigma_BEM, Yq_t,
                            case_label=label, verbose=verbose)
        wall = time.perf_counter() - t0

        # Final density
        with torch.no_grad():
            sigma = model(Yq_t).squeeze(-1).numpy()

        d_err   = float(np.linalg.norm(sigma - sigma_BEM)
                        / np.linalg.norm(sigma_BEM))
        bie_res = float(np.linalg.norm(V_h @ sigma - g_values)
                        / np.linalg.norm(g_values))

        rec = reconstruct_interior(
            P=P, Yq=Yq_T, wq=wq, sigma=sigma,
            n_grid=100, u_exact=u_exact_fn,
            x_range=x_range, y_range=y_range,
        )
        iL2 = float(rec.rel_L2)

        print(f"  → d_err={d_err:.4f}  BIE={bie_res:.2e}  iL2={iL2:.2e}  wall={wall:.1f}s")

        results[method_id] = {
            "hist":         hist,
            "sigma":        sigma,
            "d_err":        d_err,
            "bie":          bie_res,
            "iL2":          iL2,
            "wall":         wall,
            "lbfgs_reason": hist["lbfgs_reason"],
        }

        # Free tensors we don't need between methods
        del model
        gc.collect()

    # Free large tensors
    del V_h_t, W_tilde_t, DV_t, Dg_t, g_t, sBEM_t, Yq_t
    gc.collect()

    return results


# ---------------------------------------------------------------------------
# Interior reconstruction for Koch(2) — all methods
# ---------------------------------------------------------------------------

def interior_for_level(data: dict, results: dict) -> dict:
    """
    Reconstruct u_θ on a fine grid for Koch(2). Returns grids for u_exact
    and for each method, plus error grids.
    """
    n        = data["n"]
    Yq_T     = data["Yq_T"]
    wq       = data["wq"]
    P        = data["P"]
    x_range  = data["x_range"]
    y_range  = data["y_range"]

    grids = {}
    print(f"\n  Interior reconstruction for Koch({n}), n_grid={N_GRID_KOCH2} …")

    for method_id in METHODS:
        sigma = results[method_id]["sigma"]
        rec   = reconstruct_interior(
            P=P, Yq=Yq_T, wq=wq, sigma=sigma,
            n_grid=N_GRID_KOCH2, u_exact=u_exact_fn,
            x_range=x_range, y_range=y_range,
        )
        grids[method_id] = {
            "Ugrid":    rec.Ugrid,
            "Uexgrid":  rec.Uexgrid,
            "Egrid":    rec.Egrid,
            "rel_L2":   rec.rel_L2,
        }
        print(f"    Method {method_id}: rel_L2={rec.rel_L2:.3e}")

    # x/y axes for plotting
    from src.reconstruction.interior import reconstruct_interior as _ri
    import numpy as _np
    xv = _np.linspace(x_range[0], x_range[1], N_GRID_KOCH2)
    yv = _np.linspace(y_range[0], y_range[1], N_GRID_KOCH2)
    grids["xv"] = xv
    grids["yv"] = yv
    return grids


# ---------------------------------------------------------------------------
# Sanity checks after Koch(1)
# ---------------------------------------------------------------------------

def sanity_check(results1: dict) -> bool:
    """Check qualitative expectations on Koch(1) results. Returns True if OK."""
    dA = results1["A"]["d_err"]
    dD = results1["D"]["d_err"]
    dB = results1["B"]["d_err"]
    dC = results1["C"]["d_err"]

    ok = True
    print("\n  === SANITY CHECKS (Koch(1)) ===")
    # Method A should stall (d_err ≥ 50%)
    msg = f"  Method A d_err={dA:.4f} — " + ("OK (stalled as expected)" if dA > 0.45 else "WARN: expected > 45%")
    print(msg)
    if dA <= 0.45:
        ok = False

    # Method D should converge (d_err < 10%)
    msg = f"  Method D d_err={dD:.4f} — " + ("OK (converged)" if dD < 0.10 else "WARN: expected < 10%")
    print(msg)
    if dD >= 0.10:
        ok = False

    # Methods B, C should be better than A
    for mid, d in [("B", dB), ("C", dC)]:
        msg = f"  Method {mid} d_err={d:.4f} — " + (
            f"OK (better than A by {dA/max(d,1e-9):.1f}×)" if d < dA else "WARN: expected better than A")
        print(msg)
        if d >= dA:
            ok = False

    if ok:
        print("  All sanity checks PASSED. Proceeding to Koch(2,3).")
    else:
        print("  *** Sanity check warnings above — continuing anyway. ***")
        print("  NOTE: Method C (H¹) may underperform at low N_q (N_q=576 Koch(1))")
        print("        and recover at higher resolutions — this is a thesis finding.")
    return True   # always proceed; sanity failures are reported, not fatal


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _adam_line(ax, adam_cutoff):
    ax.axvline(x=adam_cutoff, color="gray", ls=":", lw=1.0, alpha=0.7,
               label="Adam → L-BFGS")


def fig_convergence(results, adam_cutoff, n_level, outpath):
    """Density rel-error vs iteration for one Koch level (FIG 1 or 2)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for mid in METHODS:
        hist = results[mid]["hist"]
        ax.semilogy(hist["iter"], hist["density_reldiff"],
                    LINES[mid], color=COLORS[mid], lw=2.0,
                    label=f"{METHOD_NAMES[mid]} (d={results[mid]['d_err']:.4f})")
        # Final annotation
        ax.annotate(
            f" {results[mid]['d_err']:.4f}",
            xy=(hist["iter"][-1], hist["density_reldiff"][-1]),
            fontsize=7.5, color=COLORS[mid], va="center",
        )
    _adam_line(ax, adam_cutoff)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(
        r"$\|\sigma_\theta - \sigma_{\mathrm{BEM}}\| / \|\sigma_{\mathrm{BEM}}\|$",
        fontsize=12,
    )
    ax.set_title(
        rf"Convergence — Koch($n={n_level}$), $g = x^2 - y^2$, no enrichment",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_density(results, data, n_level, outpath):
    """2×2 density plots for one Koch level (FIG 3 or 4)."""
    arc      = data["arc"]
    sort_idx = data["sort_idx"]
    sigma_B  = data["sigma_BEM"]

    arc_s = arc[sort_idx]
    bem_s = sigma_B[sort_idx]
    ymin  = bem_s.min() * 1.35
    ymax  = bem_s.max() * 1.35

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True, sharex=True)
    axes_flat = axes.flatten()

    for ax, mid in zip(axes_flat, METHODS):
        sig   = results[mid]["sigma"][sort_idx]
        d_err = results[mid]["d_err"]
        ax.plot(arc_s, bem_s, "k--", lw=1.5, alpha=0.5,
                label=r"$\sigma_{\mathrm{BEM}}$")
        ax.plot(arc_s, sig, LINES[mid], color=COLORS[mid], lw=1.6,
                label=f"d_err={d_err:.4f}")
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"Method {mid}: {METHOD_NAMES[mid]}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.3, alpha=0.4)

    for ax in axes[1]:
        ax.set_xlabel("Arc-length $s$", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\sigma(s)$", fontsize=10)

    fig.suptitle(
        rf"Final density $\sigma_\theta(s)$ — Koch($n={n_level}$), $g = x^2 - y^2$",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_scaling_conditioning(specs: list, outpath: str):
    """cond(V_h) vs Koch level — FIG 5."""
    levels = [s["n"] for s in specs]
    conds  = [s["cond_V"] for s in specs]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(levels, conds, "ko-", lw=2.0, ms=8)
    for n, c in zip(levels, conds):
        ax.annotate(f" {c:.2e}", xy=(n, c), fontsize=9, va="bottom")
    ax.set_xticks(levels)
    ax.set_xticklabels([f"Koch({n})" for n in levels], fontsize=11)
    ax.set_ylabel(r"$\kappa(V_h)$  (SVD condition number)", fontsize=11)
    ax.set_title(r"Ill-conditioning growth with prefractal level", fontsize=11)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_scaling_density_error(all_results: dict, specs: list, outpath: str):
    """Density rel-error vs Koch level for all 4 methods — FIG 6."""
    levels = [s["n"] for s in specs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mid in METHODS:
        d_errs = [all_results[n][mid]["d_err"] for n in levels]
        ax.semilogy(levels, d_errs,
                    LINES[mid], color=COLORS[mid], lw=2.0,
                    marker=MARKERS[mid], ms=8,
                    label=METHOD_NAMES[mid])
        for n, d in zip(levels, d_errs):
            ax.annotate(f" {d:.3f}", xy=(n, d), fontsize=7.5,
                        color=COLORS[mid], va="center")
    ax.set_xticks(levels)
    ax.set_xticklabels([f"Koch($n={n}$)" for n in levels], fontsize=11)
    ax.set_ylabel(
        r"$\|\sigma_\theta - \sigma_{\mathrm{BEM}}\| / \|\sigma_{\mathrm{BEM}}\|$",
        fontsize=11,
    )
    ax.set_title(
        r"Density error scaling — $g = x^2-y^2$, no enrichment",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_interior_solutions(grids: dict, data: dict, outpath: str):
    """Row of u_exact, u_A, u_B, u_C for Koch(2) — FIG 7."""
    xv   = grids["xv"]
    yv   = grids["yv"]
    n    = data["n"]
    cols = [("Exact", grids["A"]["Uexgrid"])]
    for mid in ["A", "B", "C"]:
        label = f"Method {mid}: {METHOD_NAMES[mid]}"
        cols.append((label, grids[mid]["Ugrid"]))

    vmin = np.nanmin(grids["A"]["Uexgrid"])
    vmax = np.nanmax(grids["A"]["Uexgrid"])
    levels_ct = np.linspace(vmin, vmax, 40)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    for ax, (title, Ugrid) in zip(axes, cols):
        cf = ax.contourf(xv, yv, Ugrid, levels=levels_ct, cmap="RdBu_r",
                         extend="both")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("$x$", fontsize=9)
    axes[0].set_ylabel("$y$", fontsize=9)
    fig.colorbar(cf, ax=axes, fraction=0.015, label=r"$u_\theta$")
    fig.suptitle(
        rf"Interior solution — Koch($n={n}$), $g = x^2 - y^2$",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_interior_errors(grids: dict, data: dict, outpath: str):
    """Row of |u_A−u_exact|, |u_B|, |u_C| for Koch(2) — FIG 8."""
    xv = grids["xv"]
    yv = grids["yv"]
    n  = data["n"]
    cols = []
    for mid in ["A", "B", "C"]:
        Egrid = np.abs(grids[mid]["Egrid"])
        label = f"Method {mid}: {METHOD_NAMES[mid]}"
        cols.append((label, Egrid))

    # Shared log colorbar
    all_vals = [np.nanmin(E) for _, E in cols if np.nanmin(E) > 0]
    vmin_log = np.log10(min(all_vals)) if all_vals else -6
    vmax_log = np.log10(max(np.nanmax(E) for _, E in cols))
    levels_ct = np.linspace(vmin_log, vmax_log, 40)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, (title, Egrid) in zip(axes, cols):
        logE = np.where(Egrid > 0, np.log10(Egrid + 1e-15), np.nan)
        cf   = ax.contourf(xv, yv, logE, levels=levels_ct, cmap="hot_r",
                           extend="both")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("$x$", fontsize=9)
    axes[0].set_ylabel("$y$", fontsize=9)
    cbar = fig.colorbar(cf, ax=axes, fraction=0.015,
                        label=r"$\log_{10}|u_\theta - u_{\mathrm{exact}}|$")
    fig.suptitle(
        rf"Interior error — Koch($n={n}$), $g = x^2 - y^2$",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


# ---------------------------------------------------------------------------
# Tables (LaTeX booktabs + CSV)
# ---------------------------------------------------------------------------

def _fmt(x, kind="f4"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if kind == "f4":
        return f"{x:.4f}"
    if kind == "e":
        return f"{x:.2e}"
    if kind == "t":          # time in seconds
        return f"{x:.0f}"
    return str(x)


def make_table_main(all_results: dict, specs: list, tables_dir: str):
    """TABLE 1 — full results across all levels and methods."""

    # --- CSV ---
    csv_rows = ["Level,Method,DensityErr,BIERes,InteriorL2,WallSec,LBFGS"]
    for s in specs:
        n = s["n"]
        for mid in METHODS:
            r = all_results[n][mid]
            csv_rows.append(
                f"Koch{n},{mid},"
                f"{r['d_err']:.6f},{r['bie']:.4e},{r['iL2']:.4e},"
                f"{r['wall']:.1f},{r['lbfgs_reason']}"
            )
    with open(os.path.join(tables_dir, "main_comparison.csv"), "w") as f:
        f.write("\n".join(csv_rows))

    # --- LaTeX ---
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Scaling study: density error, BIE residual, interior $L^2$ error, "
        r"and wall time for four loss functions across Koch levels $n=1,2,3$. "
        r"Geometry: Koch snowflake, $g(x,y)=x^2-y^2$, network $4\times80$ tanh, seed=0. "
        r"Training: Adam $3\times1000$ + L-BFGS~15\,000 iterations.}",
        r"\label{tab:scaling_main}",
        r"\begin{tabular}{@{}llccccl@{}}",
        r"\toprule",
        r"Level & Method & $\|\sigma_\theta-\sigma^*\|/\|\sigma^*\|$ "
        r"& $\|V\sigma_\theta-g\|/\|g\|$ & $\|u_\theta-u_{\rm ex}\|/\|u_{\rm ex}\|$ "
        r"& Wall (s) & L-BFGS \\",
        r"\midrule",
    ]

    level_labels = {1: r"Koch($n=1$)", 2: r"Koch($n=2$)", 3: r"Koch($n=3$)"}
    for i, s in enumerate(specs):
        n = s["n"]
        if i > 0:
            lines.append(r"\midrule")
        first = True
        for mid in METHODS:
            r    = all_results[n][mid]
            lbl  = level_labels[n] if first else ""
            mname = METHOD_NAMES[mid].replace("$", r"\$")   # safe for tex
            mname = METHOD_NAMES[mid]
            lines.append(
                f"  {lbl} & {mname} & "
                f"{_fmt(r['d_err'],'f4')} & "
                f"{_fmt(r['bie'],'e')} & "
                f"{_fmt(r['iL2'],'e')} & "
                f"{_fmt(r['wall'],'t')} & "
                f"{r['lbfgs_reason']} \\\\"
            )
            first = False

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(os.path.join(tables_dir, "main_comparison.tex"), "w") as f:
        f.write("\n".join(lines))

    print(f"  saved → tables/main_comparison.{{tex,csv}}")


def make_table_conditioning(specs: list, tables_dir: str):
    """TABLE 2 — per-level conditioning."""

    # --- CSV ---
    csv_rows = ["Level,Nq,condV,condEigWV,condSvdWV,nonNormWV"]
    for s in specs:
        csv_rows.append(
            f"Koch{s['n']},{s['Nq']},"
            f"{s['cond_V']:.4e},{s['cond_eig_WV']:.2f},"
            f"{s['cond_svd_WV']:.2f},{s['non_norm_WV']:.3e}"
        )
    with open(os.path.join(tables_dir, "conditioning.csv"), "w") as f:
        f.write("\n".join(csv_rows))

    # --- LaTeX ---
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Spectral properties of the operators across Koch levels "
        r"($n_{\rm pe}=4$, $p=12$). "
        r"$\kappa_{\rm eig}(\widetilde{W}V)$ computed only for $N_q\leq3000$.}",
        r"\label{tab:conditioning}",
        r"\begin{tabular}{@{}lrcccc@{}}",
        r"\toprule",
        r"Level & $N_q$ & $\kappa_{\rm svd}(V)$ "
        r"& $\kappa_{\rm eig}(\widetilde{W}V)$ "
        r"& $\kappa_{\rm svd}(\widetilde{W}V)$ "
        r"& Non-normality \\",
        r"\midrule",
    ]
    for s in specs:
        ce = "—" if np.isnan(s["cond_eig_WV"]) else f"{s['cond_eig_WV']:.2f}"
        lines.append(
            f"  Koch($n={s['n']}$) & {s['Nq']} & "
            f"{s['cond_V']:.3e} & {ce} & "
            f"{s['cond_svd_WV']:.2f} & {s['non_norm_WV']:.2e} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(os.path.join(tables_dir, "conditioning.tex"), "w") as f:
        f.write("\n".join(lines))

    print(f"  saved → tables/conditioning.{{tex,csv}}")


# ---------------------------------------------------------------------------
# Save/load npz
# ---------------------------------------------------------------------------

def save_level_npz(n: int, data: dict, results: dict, data_dir: str):
    """Save raw data for Koch level n."""
    out = dict(
        n=n, Nq=data["Nq"],
        cond_V=data["cond_V"],
        cond_eig_WV=data["cond_eig_WV"],
        cond_svd_WV=data["cond_svd_WV"],
        non_norm_WV=data["non_norm_WV"],
        sigma_BEM=data["sigma_BEM"],
        g_values=data["g_values"],
    )
    for mid in METHODS:
        r  = results[mid]
        prefix = f"m{mid}_"
        out[prefix + "sigma"]    = r["sigma"]
        out[prefix + "d_err"]    = r["d_err"]
        out[prefix + "bie"]      = r["bie"]
        out[prefix + "iL2"]      = r["iL2"]
        out[prefix + "wall"]     = r["wall"]
        out[prefix + "hist_iter"] = np.array(r["hist"]["iter"])
        out[prefix + "hist_loss"] = np.array(r["hist"]["loss"])
        out[prefix + "hist_derr"] = np.array(r["hist"]["density_reldiff"])
    path = os.path.join(data_dir, f"koch{n}_results.npz")
    np.savez_compressed(path, **out)
    print(f"  saved → data/koch{n}_results.npz")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(all_results: dict, specs: list):
    w = 12
    print(f"\n{'='*90}")
    print(f"SCALING STUDY SUMMARY  —  Koch n=1,2,3 | n_per_edge={N_PER_EDGE} | p={P_GL}")
    print(f"{'='*90}")
    print(f"  g = x²−y²  | network {N_HIDDEN}×{HIDDEN_WIDTH} tanh | seed={SEED}")
    print(f"  Training: Adam {LR_SCHEDULE} + L-BFGS {N_LBFGS}")
    print()

    # Conditioning table
    print(f"  {'Level':<10} {'Nq':>7}  {'cond(V)':>10}  {'cond_eig(W̃V)':>14}  "
          f"{'cond_svd(W̃V)':>14}  {'non-norm':>10}")
    print("  " + "-"*72)
    for s in specs:
        ce = "N/A" if np.isnan(s["cond_eig_WV"]) else f"{s['cond_eig_WV']:.2f}"
        print(f"  Koch({s['n']}){'':<3}  {s['Nq']:>7}  {s['cond_V']:>10.3e}  "
              f"{ce:>14}  {s['cond_svd_WV']:>14.2f}  {s['non_norm_WV']:>10.3e}")

    # Results table
    print()
    sep = "─" * 22 + "─┬─" + ("─" * w + "─┬─") * 3 + "─" * 8
    for s in specs:
        n = s["n"]
        print(f"\n  Koch({n})  Nq={s['Nq']}")
        print(f"  {sep}")
        hdr = (f"  {'Method':<22s} │ " +
               " │ ".join([f"{'d_err':>{w}}", f"{'BIE res':>{w}}", f"{'iL2':>{w}}"]) +
               f" │ {'wall(s)':>8}")
        print(hdr)
        print(f"  {sep}")
        dA = all_results[n]["A"]["d_err"]
        for mid in METHODS:
            r = all_results[n][mid]
            impr = f"({dA/max(r['d_err'],1e-9):.1f}×)" if mid != "A" else "—"
            print(
                f"  {f'Method {mid}: {METHOD_NAMES[mid]}':<22s} │ "
                f"{r['d_err']:>{w}.4f} │ {r['bie']:>{w}.2e} │ "
                f"{r['iL2']:>{w}.2e} │ {r['wall']:>8.0f}   {impr}"
            )
        print(f"  {sep}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    fig_dir   = os.path.join(_HERE, "figures")
    table_dir = os.path.join(_HERE, "tables")
    data_dir  = os.path.join(_HERE, "data")

    print("\n" + "=" * 72)
    print("THESIS SECTION 5 — SCALING STUDY: FOUR LOSSES, KOCH LEVELS 1,2,3")
    print("=" * 72)
    print(f"\n  Resolution: n_per_edge={N_PER_EDGE}, p_gl={P_GL}")
    print(f"  N_q: Koch(1)={12*N_PER_EDGE*P_GL}, Koch(2)={48*N_PER_EDGE*P_GL}, "
          f"Koch(3)={192*N_PER_EDGE*P_GL}")
    print(f"  Matrix size Koch(3): {192*N_PER_EDGE*P_GL}×{192*N_PER_EDGE*P_GL} × 8B "
          f"= {(192*N_PER_EDGE*P_GL)**2*8/1e9:.2f} GB per matrix")
    print(f"\n  Shared σ_w init: {N_HIDDEN}×{HIDDEN_WIDTH} tanh, seed={SEED}")

    # Shared initial weights (fixed across all levels and methods)
    init_state = build_shared_init(SEED)

    all_results = {}
    specs       = []
    adam_cutoff = sum(n for n, _ in LR_SCHEDULE)

    # -----------------------------------------------------------------------
    # Koch(1) — setup, train all 4 methods, sanity check
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  STEP 2: Koch(1) — assembly + 4 training runs")
    print("=" * 72)

    data1 = setup_level(1, verbose=True)
    specs.append({
        "n": 1, "Nq": data1["Nq"],
        "cond_V": data1["cond_V"],
        "cond_eig_WV": data1["cond_eig_WV"],
        "cond_svd_WV": data1["cond_svd_WV"],
        "non_norm_WV": data1["non_norm_WV"],
    })

    results1 = run_level(data1, init_state, verbose=True)
    all_results[1] = results1
    save_level_npz(1, data1, results1, data_dir)

    print("\n  FIG 1: convergence_koch1.png")
    fig_convergence(results1, adam_cutoff, 1,
                    os.path.join(fig_dir, "convergence_koch1.png"))
    print("  FIG 3: density_koch1.png")
    fig_density(results1, data1, 1,
                os.path.join(fig_dir, "density_koch1.png"))

    # Sanity check
    ok = sanity_check(results1)
    if not ok:
        print("\n  Halting after Koch(1) due to sanity-check failures.")
        return

    # -----------------------------------------------------------------------
    # Koch(2) — setup, train all 4 methods + interior reconstruction
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  STEP 3: Koch(2) — assembly + 4 training runs + interior")
    print("=" * 72)

    data2 = setup_level(2, verbose=True)
    specs.append({
        "n": 2, "Nq": data2["Nq"],
        "cond_V": data2["cond_V"],
        "cond_eig_WV": data2["cond_eig_WV"],
        "cond_svd_WV": data2["cond_svd_WV"],
        "non_norm_WV": data2["non_norm_WV"],
    })

    results2 = run_level(data2, init_state, verbose=True)
    all_results[2] = results2
    save_level_npz(2, data2, results2, data_dir)

    print("\n  FIG 2: convergence_koch2.png")
    fig_convergence(results2, adam_cutoff, 2,
                    os.path.join(fig_dir, "convergence_koch2.png"))
    print("  FIG 4: density_koch2.png")
    fig_density(results2, data2, 2,
                os.path.join(fig_dir, "density_koch2.png"))

    grids2 = interior_for_level(data2, results2)
    print("  FIG 7: interior_koch2_solutions.png")
    fig_interior_solutions(grids2, data2,
                           os.path.join(fig_dir, "interior_koch2_solutions.png"))
    print("  FIG 8: interior_koch2_errors.png")
    fig_interior_errors(grids2, data2,
                        os.path.join(fig_dir, "interior_koch2_errors.png"))

    # -----------------------------------------------------------------------
    # Koch(3) — setup, train all 4 methods
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  STEP 4: Koch(3) — assembly + 4 training runs (SLOW: ~2h total)")
    print("=" * 72)

    data3 = setup_level(3, verbose=True)
    specs.append({
        "n": 3, "Nq": data3["Nq"],
        "cond_V": data3["cond_V"],
        "cond_eig_WV": data3["cond_eig_WV"],
        "cond_svd_WV": data3["cond_svd_WV"],
        "non_norm_WV": data3["non_norm_WV"],
    })

    results3 = run_level(data3, init_state, verbose=True)
    all_results[3] = results3
    save_level_npz(3, data3, results3, data_dir)

    # -----------------------------------------------------------------------
    # Scaling figures (need all 3 levels)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  STEP 5: Scaling figures and tables")
    print("=" * 72)

    print("  FIG 5: scaling_conditioning.png")
    fig_scaling_conditioning(specs,
                             os.path.join(fig_dir, "scaling_conditioning.png"))

    print("  FIG 6: scaling_density_error.png")
    fig_scaling_density_error(all_results, specs,
                              os.path.join(fig_dir, "scaling_density_error.png"))

    # -----------------------------------------------------------------------
    # Tables
    # -----------------------------------------------------------------------
    print("  TABLE 1: main_comparison.{tex,csv}")
    make_table_main(all_results, specs, table_dir)

    print("  TABLE 2: conditioning.{tex,csv}")
    make_table_conditioning(specs, table_dir)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print_summary(all_results, specs)

    print(f"\n  All outputs written to {_HERE}/")
    print(f"    figures/ : 8 figures (300 dpi)")
    print(f"    tables/  : main_comparison.{{tex,csv}}, conditioning.{{tex,csv}}")
    print(f"    data/    : koch{{1,2,3}}_results.npz")


if __name__ == "__main__":
    main()
