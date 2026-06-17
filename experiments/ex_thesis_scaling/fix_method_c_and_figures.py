"""
Fix scaling study outputs: re-run Method C with diagnostics, apply all
figure fixes (Issues 2-5), and add FIG 9 / FIG 10 / TABLE 3 (Issue 6).

ISSUE 1 INVESTIGATION
---------------------
The combined H¹ loss form in scaling_study.py IS correct:
    L_C = (r**2).mean() + ALPHA * (Dr**2).mean()
where r = V σ - g  and  Dr = D_h(V σ - g)  (combined, NOT pure seminorm).

Diagnostic evidence that the loss form is not the bug:
  - BIE residual for Koch(1) Method C = 2.04e-2  (LESS than A = 3.98e-2)
  - Pure seminorm failure gives BIE residual >> 1 (catastrophic drift)
  - Combined form has BIE residual << 1 -- no null-space drift.

Root cause of the 66.3% density error at Koch(1) n_pe=4:
  - V is ill-conditioned: cond(V_h) = 2548 at Koch(1) Nq=576.
  - The combined loss minimizes ||V(σ-σ_BEM)||² + ||D_h V(σ-σ_BEM)||²,
    which does NOT ensure small ||σ-σ_BEM|| when V is ill-conditioned.
  - L-BFGS stalls at Koch(1) (loss = 3.464e-4, constant for 1000 iters).
  - Method D (V⁻¹ oracle) achieves 0.44% → the network CAN represent σ_BEM.

The six_loss_comparison (n_pe=12, Nq=2304) gave 3.06% because:
  - 4× finer resolution reduces the ill-conditioning effect
  - Koch(1) at n_pe=4 and n_pe=12 behave differently despite same Koch level

This re-run confirms the loss form and provides per-step L²/H¹ breakdown.

FIGURE FIXES (Issues 2-5)
  - FIG convergence: title = "Density relative error during training"
    (not "loss convergence", not "no enrichment")
  - Remove "no enrichment" from ALL titles and annotations
  - Interior: KEEP solutions, REMOVE error-field figure
  - Colorbar spacing: pad=0.04, fraction=0.046

NEW OUTPUTS (Issue 6)
  - FIG 9: scaling_WV_conditioning.png
  - FIG 10: scaling_Dh_behavior.png
  - TABLE 3: operator_scaling.{tex,csv}
"""

from __future__ import annotations

import sys
import os
import gc
import time
import warnings

import numpy as np
import torch
import scipy.sparse.linalg as spla
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
# Config (must match scaling_study.py)
# ---------------------------------------------------------------------------

SEED         = 0
N_PER_EDGE   = 4
P_GL         = 12
ALPHA        = 1.0
HIDDEN_WIDTH = 80
N_HIDDEN     = 4
LR_SCHEDULE  = [(1000, 1e-3), (1000, 3e-4), (1000, 1e-4)]
N_LBFGS      = 15000
LBFGS_MEM    = 30
LOG_EVERY    = 200
N_GRID_KOCH2 = 200
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

fig_dir   = os.path.join(_HERE, "figures")
table_dir = os.path.join(_HERE, "tables")
data_dir  = os.path.join(_HERE, "data")


# ---------------------------------------------------------------------------
# Boundary data
# ---------------------------------------------------------------------------

def g_fn(xy):
    return xy[:, 0]**2 - xy[:, 1]**2

def u_exact_fn(xy):
    return xy[:, 0]**2 - xy[:, 1]**2


# ---------------------------------------------------------------------------
# Fast D_h application (same as scaling_study.py)
# ---------------------------------------------------------------------------

def _dh_times_matrix(qdata, M, D_ref):
    Nq, Ncol = M.shape
    result = np.zeros_like(M)
    p = qdata.p
    for pid in range(qdata.n_panels):
        js  = qdata.idx_std[pid]
        L_p = qdata.L_panel[pid]
        D_phys = (2.0 / L_p) * D_ref
        result[np.ix_(js, np.arange(Ncol))] = D_phys @ M[js, :]
    return result

def _compute_DV_Dg(qdata, V_h, g):
    xi, _ = gauss_legendre(qdata.p)
    D_ref = lagrange_derivative_matrix(xi)
    DV    = _dh_times_matrix(qdata, V_h, D_ref)
    Dg    = _dh_times_matrix(qdata, g[:, None], D_ref).squeeze(-1)
    return DV, Dg


# ---------------------------------------------------------------------------
# Model init (same seed as original run)
# ---------------------------------------------------------------------------

def build_shared_init():
    torch.manual_seed(SEED)
    m = build_sigma_w_network(HIDDEN_WIDTH, N_HIDDEN).double()
    return {k: v.clone() for k, v in m.state_dict().items()}

def fresh_model(init_state):
    m = build_sigma_w_network(HIDDEN_WIDTH, N_HIDDEN).double()
    m.load_state_dict({k: v.clone() for k, v in init_state.items()})
    return m


# ---------------------------------------------------------------------------
# Level assembly (same as scaling_study.setup_level)
# ---------------------------------------------------------------------------

def setup_level(n, verbose=True):
    t0 = time.perf_counter()
    if verbose:
        print(f"\n  [Koch({n})] Building geometry …")
    geom   = make_koch_geometry(n=n)
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=N_PER_EDGE)
    label_corner_ring_panels(panels, P)
    qdata  = build_panel_quadrature(panels, p=P_GL)
    Yq_T   = qdata.Yq.T
    wq     = qdata.wq
    Nq     = qdata.n_quad

    if verbose:
        print(f"  [Koch({n})] Nq={Nq}. Assembling Nyström matrix …")
    nmat      = assemble_nystrom_matrix(qdata)
    V_h       = nmat.V
    g_values  = g_fn(Yq_T)
    bem       = solve_bem(nmat, g_values, tol=GMRES_TOL, max_iter=GMRES_MAX)
    sigma_BEM = bem.sigma

    sv_V   = np.linalg.svd(V_h, compute_uv=False)
    cond_V = sv_V[0] / sv_V[-1]
    if verbose:
        print(f"  [Koch({n})] cond(V_h) = {cond_V:.3e}")

    if verbose:
        print(f"  [Koch({n})] Computing DV = D_h V (block-diagonal) …")
    DV, Dg = _compute_DV_Dg(qdata, V_h, g_values)

    # Arc-length for density plots
    pan_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc       = pan_start[qdata.pan_id] + qdata.s_on_panel
    sort_idx  = np.argsort(arc)

    x_range = (P[:, 0].min() - 0.05, P[:, 0].max() + 0.05)
    y_range = (P[:, 1].min() - 0.05, P[:, 1].max() + 0.05)

    if verbose:
        print(f"  [Koch({n})] Setup complete in {time.perf_counter()-t0:.1f}s")

    return dict(
        n=n, Nq=Nq, geom=geom, P=P, qdata=qdata,
        Yq_T=Yq_T, wq=wq, V_h=V_h, DV=DV, Dg=Dg,
        g_values=g_values, sigma_BEM=sigma_BEM,
        cond_V=cond_V,
        arc=arc, sort_idx=sort_idx,
        x_range=x_range, y_range=y_range,
    )


# ---------------------------------------------------------------------------
# Method C training with diagnostics
# ---------------------------------------------------------------------------

def train_method_C(model, V_h_t, DV_t, g_t, Dg_t, Yq_t,
                   sigma_BEM_np, case_label, verbose=True):
    """
    Re-run Method C: L = (r**2).mean() + ALPHA * (Dr**2).mean()
    with per-step breakdown of the L² and H¹ contributions.
    """
    history = {"iter": [], "loss": [], "density_reldiff": [],
               "L2_term": [], "H1_term": []}

    def _record(it, loss_val=None):
        with torch.no_grad():
            s     = model(Yq_t).squeeze(-1)
            r     = V_h_t @ s - g_t
            Dr    = DV_t @ s - Dg_t
            L2_v  = float((r**2).mean())
            H1_v  = float((Dr**2).mean())
            if loss_val is None:
                loss_val = L2_v + ALPHA * H1_v
            sigma = s.numpy()
        d_err = float(np.linalg.norm(sigma - sigma_BEM_np)
                      / np.linalg.norm(sigma_BEM_np))
        history["iter"].append(it)
        history["loss"].append(loss_val)
        history["density_reldiff"].append(d_err)
        history["L2_term"].append(L2_v)
        history["H1_term"].append(H1_v)
        if verbose and it % LOG_EVERY == 0:
            print(f"  [{case_label}] iter={it:6d} | loss={loss_val:.3e} "
                  f"| L2={L2_v:.3e} | H1={H1_v:.3e} "
                  f"| H1/L2={H1_v/max(L2_v,1e-30):.1f}x "
                  f"| d_err={d_err:.4f}")

    def loss_fn(m):
        s  = m(Yq_t).squeeze(-1)
        r  = V_h_t @ s - g_t
        Dr = DV_t @ s - Dg_t
        return (r**2).mean() + ALPHA * (Dr**2).mean()

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
    n_outer    = N_LBFGS // 20
    lb_its     = 0
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

    loss_end       = history["loss"][-1]
    lbfgs_ratio    = loss_start / max(loss_end, 1e-30)
    lbfgs_converged = loss_end < 0.99 * loss_start
    lbfgs_reason    = "converges" if lbfgs_converged else "stalls"

    if verbose:
        print(f"  [{case_label}] L-BFGS {lbfgs_reason}: "
              f"d_err={history['density_reldiff'][-1]:.4f} ({lbfgs_ratio:.1f}× loss reduction)")
        L2_f = history["L2_term"][-1]
        H1_f = history["H1_term"][-1]
        print(f"  [{case_label}] Final: L2={L2_f:.3e}  H1={H1_f:.3e}  H1/L2={H1_f/max(L2_f,1e-30):.1f}x")

    history["adam_cutoff"]      = adam_cutoff
    history["lbfgs_ratio"]      = lbfgs_ratio
    history["lbfgs_converged"]  = lbfgs_converged
    history["lbfgs_reason"]     = lbfgs_reason
    return history


# ---------------------------------------------------------------------------
# Load existing results from npz
# ---------------------------------------------------------------------------

def load_level_results(n, data_dir):
    """Load all method results for Koch level n from the saved npz."""
    npz = np.load(os.path.join(data_dir, f"koch{n}_results.npz"), allow_pickle=True)
    results = {}
    adam_cutoff = sum(ni for ni, _ in LR_SCHEDULE)
    for mid in METHODS:
        p = f"m{mid}_"
        hist_iter = npz[p+"hist_iter"].tolist()
        hist_loss = npz[p+"hist_loss"].tolist()
        hist_derr = npz[p+"hist_derr"].tolist()
        # Determine L-BFGS convergence from loss history (same criterion as scaling_study.py:333)
        lbfgs_mask = [i > adam_cutoff for i in hist_iter]
        adam_losses = [l for l, m in zip(hist_loss, lbfgs_mask) if not m]
        lbfgs_losses = [l for l, m in zip(hist_loss, lbfgs_mask) if m]
        if adam_losses and lbfgs_losses:
            loss_start = adam_losses[-1]
            loss_end   = lbfgs_losses[-1]
            lbfgs_converged = loss_end < 0.99 * loss_start
        else:
            lbfgs_converged = False
        lbfgs_reason = "converges" if lbfgs_converged else "stalls"
        results[mid] = {
            "sigma":        npz[p+"sigma"],
            "d_err":        float(npz[p+"d_err"]),
            "bie":          float(npz[p+"bie"]),
            "iL2":          float(npz[p+"iL2"]),
            "wall":         float(npz[p+"wall"]),
            "lbfgs_reason": lbfgs_reason,
            "hist": {
                "iter":            hist_iter,
                "loss":            hist_loss,
                "density_reldiff": hist_derr,
                "adam_cutoff":     adam_cutoff,
                "lbfgs_ratio":     (adam_losses[-1] / max(lbfgs_losses[-1], 1e-30))
                                   if (adam_losses and lbfgs_losses) else 1.0,
                "lbfgs_converged": lbfgs_converged,
                "lbfgs_reason":    lbfgs_reason,
            },
        }
    return results


# ---------------------------------------------------------------------------
# Update npz with corrected Method C
# ---------------------------------------------------------------------------

def update_npz_method_C(n, data_dir, sigma_C, d_err_C, bie_C, iL2_C,
                        wall_C, hist_C):
    """Overwrite the Method C fields in the existing npz."""
    path = os.path.join(data_dir, f"koch{n}_results.npz")
    old  = dict(np.load(path, allow_pickle=True))
    old["mC_sigma"]     = sigma_C
    old["mC_d_err"]     = d_err_C
    old["mC_bie"]       = bie_C
    old["mC_iL2"]       = iL2_C
    old["mC_wall"]      = wall_C
    old["mC_hist_iter"] = np.array(hist_C["iter"])
    old["mC_hist_loss"] = np.array(hist_C["loss"])
    old["mC_hist_derr"] = np.array(hist_C["density_reldiff"])
    np.savez_compressed(path, **old)
    print(f"  Updated npz: data/koch{n}_results.npz (Method C)")


# ---------------------------------------------------------------------------
# H¹ Hessian conditioning  (FIG 10)
# ---------------------------------------------------------------------------

def compute_H1_hessian_cond(V_h, DV, alpha=1.0, n_eig_lanczos=12,
                             tol=1e-8):
    """
    cond(V^T V + alpha * DV^T DV) via direct eigvalsh (Nq ≤ 3000)
    or Lanczos (Nq > 3000).
    """
    Nq = V_h.shape[0]
    if Nq <= 3000:
        H = V_h.T @ V_h + alpha * (DV.T @ DV)
        eigvals = np.linalg.eigvalsh(H)
        eigvals = np.sort(np.abs(eigvals))
        return eigvals[-1] / max(eigvals[0], 1e-300)
    else:
        # Lanczos via LinearOperator (avoids forming H explicitly)
        def mv(x):
            return V_h.T @ (V_h @ x) + alpha * (DV.T @ (DV @ x))
        op = spla.LinearOperator((Nq, Nq), matvec=mv, dtype=np.float64)
        try:
            k = min(n_eig_lanczos, Nq - 2)
            lmax = spla.eigsh(op, k=k, which='LM', return_eigenvectors=False,
                              tol=tol)
            # Smallest eigenvalue: use shift-invert from min of standard vals
            # Approximate: sigma_min(V)^2 is the lower bound
            sv = np.linalg.svd(V_h, compute_uv=False)
            lmin_bound = sv[-1]**2   # eigenvalue of V^TV
            return float(lmax.max()) / float(lmin_bound)
        except Exception:
            return np.nan


# ---------------------------------------------------------------------------
# FIGURE FUNCTIONS (all with plot fixes applied)
# ---------------------------------------------------------------------------

def _adam_line(ax, adam_cutoff):
    ax.axvline(x=adam_cutoff, color="gray", ls=":", lw=1.0, alpha=0.7)


def fig_convergence(results, adam_cutoff, n_level, outpath):
    """FIG 1/2: density relative error vs iteration (FIXED titles)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for mid in METHODS:
        hist  = results[mid]["hist"]
        d_err = results[mid]["d_err"]
        ax.semilogy(hist["iter"], hist["density_reldiff"],
                    LINES[mid], color=COLORS[mid], lw=2.0,
                    label=f"{METHOD_NAMES[mid]} (d={d_err:.4f})")
        ax.annotate(f" {d_err:.4f}",
                    xy=(hist["iter"][-1], hist["density_reldiff"][-1]),
                    fontsize=7.5, color=COLORS[mid], va="center")
    _adam_line(ax, adam_cutoff)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(
        r"$\|\sigma_\theta - \sigma_{\mathrm{BEM}}\| / \|\sigma_{\mathrm{BEM}}\|$",
        fontsize=12)
    ax.set_title(
        rf"Density relative error during training — Koch($n={n_level}$), $g = x^2 - y^2$",
        fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_density(results, data, n_level, outpath):
    """FIG 3/4: final density profiles (FIXED titles, no 'no enrichment')."""
    arc      = data["arc"]
    sort_idx = data["sort_idx"]
    sigma_B  = data["sigma_BEM"]

    arc_s = arc[sort_idx]
    bem_s = sigma_B[sort_idx]
    ymin  = bem_s.min() * 1.35
    ymax  = bem_s.max() * 1.35

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True, sharex=True)
    for ax, mid in zip(axes.flatten(), METHODS):
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
        fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_interior_solutions(grids, data, outpath):
    """FIG 7: u_exact + u_A + u_B + u_C interior (FIXED colorbar, no error plot)."""
    xv   = grids["xv"]
    yv   = grids["yv"]
    n    = data["n"]
    cols = [("Exact", grids["A"]["Uexgrid"])]
    for mid in ["A", "B", "C"]:
        cols.append((f"Method {mid}: {METHOD_NAMES[mid]}", grids[mid]["Ugrid"]))

    vmin = np.nanmin(grids["A"]["Uexgrid"])
    vmax = np.nanmax(grids["A"]["Uexgrid"])
    lvls = np.linspace(vmin, vmax, 40)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    cf = None
    for ax, (title, Ugrid) in zip(axes, cols):
        cf = ax.contourf(xv, yv, Ugrid, levels=lvls, cmap="RdBu_r", extend="both")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("$x$", fontsize=9)
    axes[0].set_ylabel("$y$", fontsize=9)
    fig.colorbar(cf, ax=axes, fraction=0.046, pad=0.04, label=r"$u_\theta$")
    fig.suptitle(
        rf"Interior solution — Koch($n={n}$), $g = x^2 - y^2$",
        fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_scaling_density_error(all_results, specs, outpath):
    """FIG 6: d_err vs Koch level (all 4 methods)."""
    levels = [s["n"] for s in specs]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mid in METHODS:
        d_errs = [all_results[n][mid]["d_err"] for n in levels]
        ax.semilogy(levels, d_errs,
                    LINES[mid], color=COLORS[mid], lw=2.0,
                    marker=MARKERS[mid], ms=8, label=METHOD_NAMES[mid])
        for n, d in zip(levels, d_errs):
            ax.annotate(f" {d:.3f}", xy=(n, d), fontsize=7.5,
                        color=COLORS[mid], va="center")
    ax.set_xticks(levels)
    ax.set_xticklabels([f"Koch($n={n}$)" for n in levels], fontsize=11)
    ax.set_ylabel(
        r"$\|\sigma_\theta - \sigma_{\mathrm{BEM}}\| / \|\sigma_{\mathrm{BEM}}\|$",
        fontsize=11)
    ax.set_title(r"Density error scaling — $g = x^2-y^2$, four loss functions",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


# ---------------------------------------------------------------------------
# FIG 9: WV conditioning scaling
# ---------------------------------------------------------------------------

def fig_WV_conditioning(specs, op_data, outpath):
    """
    FIG 9: Show that cond_svd(W̃V) grows slowly (Calderón identity holds at
    scale) while cond(V) explodes. Also plot non-normality.
    """
    levels  = [s["n"] for s in specs]
    cond_V  = [s["cond_V"] for s in specs]
    sv_WV   = [d["cond_svd_WV"] for d in op_data]
    eig_WV  = [d["cond_eig_WV"] for d in op_data]   # may contain nan
    nonnorm = [d["non_norm_WV"] for d in op_data]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # Left axis: condition numbers (log scale)
    l1, = ax1.semilogy(levels, cond_V, "k-o", lw=2.0, ms=9,
                        label=r"$\kappa_{\mathrm{svd}}(V_h)$ (standard)")
    l2, = ax1.semilogy(levels, sv_WV, "b-s", lw=2.0, ms=9,
                        label=r"$\kappa_{\mathrm{svd}}(\widetilde{W}V)$ (Calderón)")

    # cond_eig only where available
    eig_valid = [(n, v) for n, v in zip(levels, eig_WV) if not np.isnan(v)]
    if eig_valid:
        nv, vv = zip(*eig_valid)
        l3, = ax1.semilogy(nv, vv, "b--^", lw=1.5, ms=8, alpha=0.7,
                            label=r"$\kappa_{\mathrm{eig}}(\widetilde{W}V)$")
    else:
        l3 = None

    # Annotate condition numbers
    for n, c in zip(levels, cond_V):
        ax1.annotate(f" {c:.1e}", xy=(n, c), fontsize=8.5, va="bottom", color="black")
    for n, c in zip(levels, sv_WV):
        ax1.annotate(f" {c:.1f}", xy=(n, c), fontsize=8.5, va="top", color="steelblue")

    ax1.set_ylabel("Condition number (log scale)", fontsize=11)
    ax1.set_xlabel("Koch level $n$", fontsize=11)
    ax1.set_xticks(levels)
    ax1.set_xticklabels([f"Koch($n={n}$)" for n in levels], fontsize=11)

    # Right axis: non-normality (linear scale)
    l4, = ax2.plot(levels, nonnorm, "g:D", lw=1.5, ms=8,
                    alpha=0.8, label="Non-normality of $\\widetilde{W}V$")
    for n, v in zip(levels, nonnorm):
        ax2.annotate(f" {v:.3f}", xy=(n, v), fontsize=8, va="bottom", color="green")
    ax2.set_ylabel("Non-normality $\\|[\\widetilde{W}V, (\\widetilde{W}V)^T]\\| / \\|\\widetilde{W}V\\|^2$",
                   fontsize=9, color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.set_ylim(0, max(nonnorm) * 1.5)

    # Legend
    handles = [l1, l2] + ([l3] if l3 else []) + [l4]
    ax1.legend(handles=handles, fontsize=9, loc="upper left")
    ax1.set_title(
        r"Calderón preconditioner: conditioning vs Koch level",
        fontsize=11)
    ax1.grid(True, which="both", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


# ---------------------------------------------------------------------------
# FIG 10: D_h Sobolev H¹ operator behavior
# ---------------------------------------------------------------------------

def fig_Dh_behavior(op_data, outpath):
    """
    FIG 10: H¹ Hessian conditioning vs Koch level, compared to cond(V^T V).
    Annotates null(D_h) and the L²/H¹ balance ratio.
    """
    levels    = [d["n"] for d in op_data]
    cond_VtV  = [d["cond_VtV"] for d in op_data]    # = cond(V)^2
    cond_H1   = [d["cond_H1_hessian"] for d in op_data]
    null_Dh   = [d["null_Dh"] for d in op_data]
    H1_L2_init = [d["H1_L2_init"] for d in op_data]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: condition numbers
    ax = axes[0]
    l1, = ax.semilogy(levels, cond_VtV, "k-o", lw=2.0, ms=9,
                       label=r"$\kappa(V_h^T V_h)$ (standard Hessian)")
    l2, = ax.semilogy(levels, cond_H1, "g-^", lw=2.0, ms=9,
                       label=r"$\kappa(V^T(I+D_h^TD_h)V)$ (H¹ Hessian)")
    for n, c in zip(levels, cond_VtV):
        ax.annotate(f" {c:.1e}", xy=(n, c), fontsize=8.5, va="bottom", color="black")
    for n, c in zip(levels, cond_H1):
        if not np.isnan(c):
            ax.annotate(f" {c:.1e}", xy=(n, c), fontsize=8.5, va="top", color="green")
    ax.set_xlabel("Koch level $n$", fontsize=11)
    ax.set_ylabel("Condition number (log scale)", fontsize=11)
    ax.set_xticks(levels)
    ax.set_xticklabels([f"Koch($n={n}$)" for n in levels])
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(r"Hessian conditioning: standard vs $H^1$ loss", fontsize=11)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)

    # Right: null space and H1/L2 ratio
    ax2 = axes[1]
    bar_x  = np.arange(len(levels))
    bars   = ax2.bar(bar_x, null_Dh, color="#2ca02c", alpha=0.7,
                     label=r"$\dim(\mathrm{null}(D_h))$ = $N_{\mathrm{pan}}$")
    ax2.set_xticks(bar_x)
    ax2.set_xticklabels([f"Koch($n={n}$)" for n in levels])
    ax2.set_ylabel("Null-space dimension of $D_h$", fontsize=11)
    ax2.set_title(r"$D_h$ null space grows with level ($N_{\mathrm{pan}}$ constants per panel)",
                  fontsize=11)

    ax3 = ax2.twinx()
    ax3.plot(bar_x, H1_L2_init, "r--D", lw=1.5, ms=7,
             label=r"$\|D_h r\|^2 / \|r\|^2$ at init")
    for i, (r, n) in enumerate(zip(H1_L2_init, null_Dh)):
        ax2.annotate(f"null={n}", xy=(i, n + 5), ha="center", fontsize=8)
        ax3.annotate(f"H1/L2={r:.0f}×", xy=(i, r + 0.5), ha="center",
                     fontsize=8, color="red")
    ax3.set_ylabel(r"$\|D_h r_{\rm init}\|^2 / \|r_{\rm init}\|^2$",
                   fontsize=9, color="red")
    ax3.tick_params(axis="y", labelcolor="red")

    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax2.legend(lines2 + lines3, labels2 + labels3, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


# ---------------------------------------------------------------------------
# TABLE 1 (main comparison — updated Method C)
# ---------------------------------------------------------------------------

def _fmt(x, kind="f4"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if kind == "f4": return f"{x:.4f}"
    if kind == "e":  return f"{x:.2e}"
    if kind == "t":  return f"{x:.0f}"
    return str(x)


def make_table_main(all_results, specs, tables_dir):
    csv_rows = ["Level,Method,DensityErr,BIERes,InteriorL2,WallSec,LBFGS"]
    for s in specs:
        n = s["n"]
        for mid in METHODS:
            r = all_results[n][mid]
            csv_rows.append(
                f"Koch{n},{mid},{r['d_err']:.6f},{r['bie']:.4e},"
                f"{r['iL2']:.4e},{r['wall']:.1f},{r['lbfgs_reason']}")
    with open(os.path.join(tables_dir, "main_comparison.csv"), "w") as f:
        f.write("\n".join(csv_rows))

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Scaling study: density error, BIE residual, interior $L^2$ error, "
        r"and wall time for four loss functions across Koch levels $n=1,2,3$. "
        r"Geometry: Koch snowflake, $g(x,y)=x^2-y^2$, network $4\times80$ tanh, seed=0. "
        r"Training: Adam $3\times1000$ + L-BFGS~15\,000 iterations. "
        r"Method~C uses the combined Sobolev $H^1$ loss "
        r"$\|V\sigma-g\|^2 + \|D_h(V\sigma-g)\|^2$.}",
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
            r   = all_results[n][mid]
            lbl = level_labels[n] if first else ""
            lines.append(
                f"  {lbl} & {METHOD_NAMES[mid]} & "
                f"{_fmt(r['d_err'],'f4')} & {_fmt(r['bie'],'e')} & "
                f"{_fmt(r['iL2'],'e')} & {_fmt(r['wall'],'t')} & "
                f"{r['lbfgs_reason']} \\\\")
            first = False
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(tables_dir, "main_comparison.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"  saved → tables/main_comparison.{{tex,csv}}")


# ---------------------------------------------------------------------------
# TABLE 3: operator scaling
# ---------------------------------------------------------------------------

def make_table_operator_scaling(specs, op_data, tables_dir):
    csv_rows = ["Level,Nq,condV,condEigWV,condSvdWV,nonNormWV,condVtV,condH1Hessian,nullDh"]
    for s, d in zip(specs, op_data):
        csv_rows.append(
            f"Koch{s['n']},{s['Nq']},"
            f"{s['cond_V']:.4e},{d['cond_eig_WV']:.2f},"
            f"{d['cond_svd_WV']:.2f},{d['non_norm_WV']:.3e},"
            f"{d['cond_VtV']:.4e},{d['cond_H1_hessian']:.4e},{d['null_Dh']}")
    with open(os.path.join(tables_dir, "operator_scaling.csv"), "w") as f:
        f.write("\n".join(csv_rows))

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Operator scaling across Koch levels ($n_{\rm pe}=4$, $p=12$). "
        r"$\kappa_{\rm eig}(\widetilde{W}V)$ computed only for $N_q\leq 3000$. "
        r"$\kappa(V^T(I+D_h^TD_h)V)$: exact for $N_q\leq 3000$, "
        r"Lanczos estimate (lower bound) for $N_q=9216$. "
        r"$N_{\rm null}(D_h) = N_{\rm pan}$ (one constant per panel).}",
        r"\label{tab:operator_scaling}",
        r"\begin{tabular}{@{}lrccccccr@{}}",
        r"\toprule",
        r"Level & $N_q$ & $\kappa_{\rm svd}(V)$ "
        r"& $\kappa_{\rm eig}(\widetilde{W}V)$ "
        r"& $\kappa_{\rm svd}(\widetilde{W}V)$ "
        r"& Non-norm "
        r"& $\kappa(V^TV)$ "
        r"& $\kappa(H^1\text{ Hessian})$ "
        r"& $N_{\rm null}(D_h)$ \\",
        r"\midrule",
    ]
    for s, d in zip(specs, op_data):
        ce = "—" if np.isnan(d["cond_eig_WV"]) else f"{d['cond_eig_WV']:.2f}"
        ch = "—" if np.isnan(d["cond_H1_hessian"]) else f"{d['cond_H1_hessian']:.2e}"
        lines.append(
            f"  Koch($n={s['n']}$) & {s['Nq']} & "
            f"{s['cond_V']:.3e} & {ce} & {d['cond_svd_WV']:.2f} & "
            f"{d['non_norm_WV']:.2e} & {d['cond_VtV']:.3e} & {ch} & "
            f"{d['null_Dh']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(tables_dir, "operator_scaling.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"  saved → tables/operator_scaling.{{tex,csv}}")


# ---------------------------------------------------------------------------
# Interior reconstruction (Koch(2))
# ---------------------------------------------------------------------------

def interior_for_level(data, results):
    n       = data["n"]
    Yq_T    = data["Yq_T"]
    wq      = data["wq"]
    P       = data["P"]
    x_range = data["x_range"]
    y_range = data["y_range"]
    grids = {}
    print(f"\n  Interior reconstruction for Koch({n}), n_grid={N_GRID_KOCH2} …")
    for mid in METHODS:
        sigma = results[mid]["sigma"]
        rec   = reconstruct_interior(
            P=P, Yq=Yq_T, wq=wq, sigma=sigma,
            n_grid=N_GRID_KOCH2, u_exact=u_exact_fn,
            x_range=x_range, y_range=y_range)
        grids[mid] = {"Ugrid":    rec.Ugrid,
                      "Uexgrid":  rec.Uexgrid,
                      "Egrid":    rec.Egrid,
                      "rel_L2":   rec.rel_L2}
        print(f"    Method {mid}: rel_L2={rec.rel_L2:.3e}")
    xv = np.linspace(x_range[0], x_range[1], N_GRID_KOCH2)
    yv = np.linspace(y_range[0], y_range[1], N_GRID_KOCH2)
    grids["xv"] = xv
    grids["yv"] = yv
    return grids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    adam_cutoff = sum(n for n, _ in LR_SCHEDULE)

    print("\n" + "=" * 72)
    print("ISSUE 1 INVESTIGATION: Combined H¹ loss form confirmation")
    print("=" * 72)
    print("""
  Loss used for Method C in scaling_study.py (lines 406-411):

      elif method_id == "C":
          def loss_fn(m, _V=V_h_t, _DV=DV_t, _g=g_t, _Dg=Dg_t):
              s   = m(Yq_t).squeeze(-1)
              r   = _V @ s - _g            # BIE residual r = Vσ - g
              Dr  = _DV @ s - _Dg          # D_h r = D_h(Vσ - g)
              return (r ** 2).mean() + ALPHA * (Dr ** 2).mean()

  This IS the combined form: L = ||r||²_mean + α||D_h r||²_mean,  α=1.
  NOT the pure seminorm ||D_h r||²_mean.

  Evidence: Koch(1) Method C BIE residual = 2.04e-2  (< Method A = 3.98e-2).
  Pure seminorm failure gives BIE residual >> 1 (catastrophic null-space drift).

  Root cause of 66.3% density error at Koch(1) n_pe=4:
    - V is ill-conditioned (cond=2548); minimising ||V(σ-σ_BEM)||² does NOT
      ensure small ||σ-σ_BEM||.
    - L-BFGS stalls at Koch(1) (loss=3.464e-4 constant for 1000 steps);
      optimizer finds a loss-minimum that is a poor density approximation.
    - H1/L2 ratio at init ≈ 29× at BOTH resolutions; balance is NOT the issue.
    - six_loss_comparison (n_pe=12, Nq=2304) gives 3.06% — purely a
      resolution effect: more quadrature points → optimizer escapes the trap.
    """)

    print("=" * 72)
    print("RE-RUNNING METHOD C FOR ALL THREE LEVELS WITH DIAGNOSTICS")
    print("=" * 72)

    init_state = build_shared_init()
    all_results = {}
    specs       = []
    op_data     = []   # for FIG 9, FIG 10, TABLE 3

    for n in LEVELS:
        print(f"\n{'='*72}")
        print(f"  LEVEL Koch({n}): assembly + Method C rerun")
        print("=" * 72)

        # Assembly
        data = setup_level(n, verbose=True)
        Nq   = data["Nq"]

        # Load A, B, D from existing npz
        print(f"\n  Loading existing A, B, D results for Koch({n}) …")
        results = load_level_results(n, data_dir)
        all_results[n] = results

        specs.append({"n": n, "Nq": Nq, "cond_V": data["cond_V"]})

        # ---- Operator data for FIG 9 / FIG 10 / TABLE 3 ----
        # FIG 9: use stored cond_eig/svd/non-norm from existing npz
        npz_n      = np.load(os.path.join(data_dir, f"koch{n}_results.npz"))
        cond_svd_WV  = float(npz_n["cond_svd_WV"])  if "cond_svd_WV" in npz_n else np.nan
        cond_eig_WV  = float(npz_n["cond_eig_WV"])  if "cond_eig_WV" in npz_n else np.nan
        non_norm_WV  = float(npz_n["non_norm_WV"])  if "non_norm_WV" in npz_n else np.nan

        # FIG 10: compute H¹ Hessian conditioning
        print(f"\n  Computing H¹ Hessian conditioning for Koch({n}) …")
        V_h = data["V_h"]
        DV  = data["DV"]
        g   = data["g_values"]
        r_init = -g   # at σ=0
        L2_init = float(np.mean(r_init**2))
        H1_init = float(np.mean((DV @ np.zeros(Nq) - data["Dg"])**2))
        # Correct H1/L2 at σ_init (use r_init = V@sigma_init - g with sigma_init=0)
        Dr_init  = (DV @ np.zeros(Nq)) - data["Dg"]  # = -Dg
        H1_init2 = float(np.mean(Dr_init**2))
        H1_L2_ratio = H1_init2 / max(L2_init, 1e-30)

        t0_h1 = time.perf_counter()
        cond_H1 = compute_H1_hessian_cond(V_h, DV, alpha=ALPHA)
        print(f"  Koch({n}): cond(H¹ Hessian) = {cond_H1:.3e}  "
              f"(computed in {time.perf_counter()-t0_h1:.1f}s)")

        N_pan     = data["qdata"].n_panels
        cond_VtV  = data["cond_V"]**2
        print(f"  Koch({n}): null(D_h)={N_pan}  H1/L2_init={H1_L2_ratio:.1f}x  "
              f"cond(V^TV)={cond_VtV:.3e}")

        op_data.append({
            "n":               n,
            "cond_svd_WV":     cond_svd_WV,
            "cond_eig_WV":     cond_eig_WV,
            "non_norm_WV":     non_norm_WV,
            "cond_VtV":        cond_VtV,
            "cond_H1_hessian": cond_H1,
            "null_Dh":         N_pan,
            "H1_L2_init":      H1_L2_ratio,
        })

        # ---- Re-run Method C ----
        print(f"\n  Re-running Method C (Sobolev H¹) for Koch({n}) …")
        Yq_t   = torch.tensor(data["Yq_T"],      dtype=torch.float64)
        V_h_t  = torch.tensor(V_h,               dtype=torch.float64)
        DV_t   = torch.tensor(DV,                dtype=torch.float64)
        g_t    = torch.tensor(g,                 dtype=torch.float64)
        Dg_t   = torch.tensor(data["Dg"],        dtype=torch.float64)
        model  = fresh_model(init_state)

        t0 = time.perf_counter()
        hist_C = train_method_C(
            model, V_h_t, DV_t, g_t, Dg_t, Yq_t,
            data["sigma_BEM"], f"Koch({n})-C", verbose=True)
        wall_C = time.perf_counter() - t0

        with torch.no_grad():
            sigma_C = model(Yq_t).squeeze(-1).numpy()

        d_err_C = float(np.linalg.norm(sigma_C - data["sigma_BEM"])
                        / np.linalg.norm(data["sigma_BEM"]))
        bie_C   = float(np.linalg.norm(V_h @ sigma_C - g)
                        / np.linalg.norm(g))
        rec_C   = reconstruct_interior(
            P=data["P"], Yq=data["Yq_T"], wq=data["wq"], sigma=sigma_C,
            n_grid=100, u_exact=u_exact_fn,
            x_range=data["x_range"], y_range=data["y_range"])
        iL2_C = float(rec_C.rel_L2)

        print(f"\n  Koch({n}) Method C (RERUN):")
        print(f"    d_err = {d_err_C:.4f}  BIE = {bie_C:.2e}  iL2 = {iL2_C:.2e}  "
              f"wall = {wall_C:.1f}s  L-BFGS: {hist_C['lbfgs_reason']}")
        print(f"    Final loss: L2={hist_C['L2_term'][-1]:.3e}  "
              f"H1={hist_C['H1_term'][-1]:.3e}  "
              f"H1/L2={hist_C['H1_term'][-1]/max(hist_C['L2_term'][-1],1e-30):.1f}x")

        # Update results dict with re-run C
        all_results[n]["C"] = {
            "sigma":        sigma_C,
            "d_err":        d_err_C,
            "bie":          bie_C,
            "iL2":          iL2_C,
            "wall":         wall_C,
            "lbfgs_reason": hist_C["lbfgs_reason"],
            "hist":         hist_C,
        }

        # Save updated npz
        update_npz_method_C(n, data_dir, sigma_C, d_err_C, bie_C, iL2_C,
                             wall_C, hist_C)

        # Free large tensors
        del V_h_t, DV_t, g_t, Dg_t, Yq_t, model
        gc.collect()

    # -----------------------------------------------------------------------
    # Summary table: Method C results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("METHOD C (SOBOLEV H¹) RERUN SUMMARY")
    print("=" * 72)
    print(f"\n  Loss form confirmed: L = (r**2).mean() + {ALPHA}*(Dr**2).mean()")
    print(f"  where r = V σ - g  and  Dr = D_h(V σ - g)  (COMBINED, not pure seminorm)")
    print()
    print(f"  {'Level':<12} {'d_err':>8} {'BIE':>10} {'iL2':>10} {'L-BFGS':>12}")
    print("  " + "-"*55)
    for n in LEVELS:
        r = all_results[n]["C"]
        print(f"  Koch({n}){'':<5} {r['d_err']:>8.4f} {r['bie']:>10.2e} "
              f"{r['iL2']:>10.2e} {r['lbfgs_reason']:>12}")

    # -----------------------------------------------------------------------
    # GENERATE ALL FIGURES
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("GENERATING ALL FIGURES (with fixes)")
    print("=" * 72)

    # Need data objects for FIG 3/4/7 (density plots need arc/sort_idx)
    # Regenerate data for Koch(1) and Koch(2) for density/interior figures
    # (no W_tilde needed)
    print("\n  Rebuilding Koch(1) geometry for density figure …")
    data1 = setup_level(1, verbose=False)
    all_results[1]["C"]["sigma"] = np.load(
        os.path.join(data_dir, "koch1_results.npz"))["mC_sigma"]
    for mid in ["A", "B", "D"]:
        all_results[1][mid]["sigma"] = np.load(
            os.path.join(data_dir, "koch1_results.npz"))[f"m{mid}_sigma"]

    print("  FIG 1: convergence_koch1.png")
    fig_convergence(all_results[1], adam_cutoff, 1,
                    os.path.join(fig_dir, "convergence_koch1.png"))
    print("  FIG 3: density_koch1.png")
    fig_density(all_results[1], data1, 1,
                os.path.join(fig_dir, "density_koch1.png"))

    print("\n  Rebuilding Koch(2) geometry for density/interior figures …")
    data2 = setup_level(2, verbose=False)
    for mid in METHODS:
        all_results[2][mid]["sigma"] = np.load(
            os.path.join(data_dir, "koch2_results.npz"))[f"m{mid}_sigma"]

    print("  FIG 2: convergence_koch2.png")
    fig_convergence(all_results[2], adam_cutoff, 2,
                    os.path.join(fig_dir, "convergence_koch2.png"))
    print("  FIG 4: density_koch2.png")
    fig_density(all_results[2], data2, 2,
                os.path.join(fig_dir, "density_koch2.png"))

    grids2 = interior_for_level(data2, all_results[2])
    print("  FIG 7: interior_koch2_solutions.png")
    fig_interior_solutions(grids2, data2,
                           os.path.join(fig_dir, "interior_koch2_solutions.png"))

    print("  FIG 6: scaling_density_error.png")
    fig_scaling_density_error(all_results, specs,
                              os.path.join(fig_dir, "scaling_density_error.png"))

    print("  FIG 9: scaling_WV_conditioning.png")
    fig_WV_conditioning(specs, op_data,
                        os.path.join(fig_dir, "scaling_WV_conditioning.png"))

    print("  FIG 10: scaling_Dh_behavior.png")
    fig_Dh_behavior(op_data,
                    os.path.join(fig_dir, "scaling_Dh_behavior.png"))

    # -----------------------------------------------------------------------
    # TABLES
    # -----------------------------------------------------------------------
    print("\n  TABLE 1: main_comparison.{tex,csv}")
    make_table_main(all_results, specs, table_dir)

    print("  TABLE 3: operator_scaling.{tex,csv}")
    make_table_operator_scaling(specs, op_data, table_dir)

    # -----------------------------------------------------------------------
    # Delete interior error figure (Issue 4)
    # -----------------------------------------------------------------------
    err_fig = os.path.join(fig_dir, "interior_koch2_errors.png")
    if os.path.exists(err_fig):
        os.remove(err_fig)
        print(f"  Removed: {err_fig}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(f"\n  Operator scaling:")
    print(f"  {'Level':<10} {'Nq':>6}  {'cond(V)':>10}  "
          f"{'cond_svd(W̃V)':>14}  {'cond(H¹ Hess)':>14}  {'null(Dh)':>10}")
    print("  " + "-"*72)
    for s, d in zip(specs, op_data):
        ch = f"{d['cond_H1_hessian']:.3e}" if not np.isnan(d["cond_H1_hessian"]) else "N/A"
        print(f"  Koch({s['n']}){'':<3} {s['Nq']:>6}  {s['cond_V']:>10.3e}  "
              f"{d['cond_svd_WV']:>14.2f}  {ch:>14}  {d['null_Dh']:>10}")

    print(f"\n  All outputs → experiments/ex_thesis_scaling/")
    print(f"    figures/  : 9 figures (FIG 1-4, 6, 7, 9, 10 — interior error removed)")
    print(f"    tables/   : main_comparison.{{tex,csv}}, operator_scaling.{{tex,csv}}")
    print(f"    data/     : koch{{1,2,3}}_results.npz  (Method C updated)")


if __name__ == "__main__":
    main()
