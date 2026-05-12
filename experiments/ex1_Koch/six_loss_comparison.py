"""
Six-loss comparison for BIE-based PINN training on Koch(1).

This is the cornerstone preconditioning experiment for the SEBINN paper.
All six cases use identical architecture and initial weights; ONLY the
loss function differs.

Cases
-----
  1. Standard              L = ||Vσ − g||²
  2. Left V⁻¹              L = ||V⁻¹(Vσ−g)||² = ||σ − σ*||²         (gold)
  3. Pure Calderón         L = ||W̃(Vσ − g)||²
  4. Combined Calderón     L = ||Vσ−g||² + β||W̃(Vσ−g)||²   β=10
  5. Pure H¹ Sobolev       L = ||D_h(Vσ − g)||²
  6. Combined H¹ Sobolev   L = ||Vσ−g||² + α||D_h(Vσ−g)||²  α=1

Geometry:  Koch(1), g(x,y) = x²−y², n_per_edge=12, p=16 (Nq=2304)
Network:   4 hidden layers × 80 neurons, tanh, shared initialisation
Training:  Adam 3×1000 at [1e-3, 3e-4, 1e-4] + L-BFGS 15000 iters
"""

from __future__ import annotations

import sys
import os
import time
import copy

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from src.boundary.polygon import make_koch_geometry
from src.boundary.panels import build_uniform_panels, label_corner_ring_panels
from src.quadrature.panel_quad import build_panel_quadrature
from src.quadrature.nystrom import assemble_nystrom_matrix, solve_bem
from src.quadrature.hypersingular import (
    assemble_hypersingular_corrected,
    regularise_hypersingular,
)
from src.quadrature.tangential_derivative import build_tangential_derivative_matrix
from src.models.sigma_w_net import build_sigma_w_network
from src.reconstruction.interior import reconstruct_interior

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED         = 0
N_PER_EDGE   = 12
P_GL         = 16
HIDDEN_WIDTH = 80
N_HIDDEN     = 4
LR_SCHEDULE  = [(1000, 1e-3), (1000, 3e-4), (1000, 1e-4)]
N_LBFGS      = 15000
LBFGS_MEMORY = 30
LOG_EVERY    = 200
N_GRID_FINAL = 201

BETA   = 10.0   # Combined Calderón weight
ALPHA  = 1.0    # Combined H¹ weight

CASES = ["1", "2", "3", "4", "5", "6"]
CASE_LABELS = {
    "1": r"1: Standard $\|Vσ−g\|^2$",
    "2": r"2: Left $V^{-1}$",
    "3": r"3: Pure Calderón",
    "4": r"4: Comb. Calderón ($\beta=10$)",
    "5": r"5: Pure H¹",
    "6": r"6: Comb. H¹ ($\alpha=1$)",
}
COLORS = {
    "1": "#1f77b4",
    "2": "#2ca02c",
    "3": "#d62728",
    "4": "#ff7f0e",
    "5": "#9467bd",
    "6": "#8c564b",
    "BEM": "black",
}
MARKERS = {"1": "o", "2": "s", "3": "^", "4": "D", "5": "v", "6": "P"}
SUCCESS_THRESHOLD = 0.15   # d_err < 15% counts as "works"


# ---------------------------------------------------------------------------
# Boundary data
# ---------------------------------------------------------------------------

def g_fn(xy: np.ndarray) -> np.ndarray:
    return xy[:, 0] ** 2 - xy[:, 1] ** 2


def u_exact_fn(xy: np.ndarray) -> np.ndarray:
    return xy[:, 0] ** 2 - xy[:, 1] ** 2


# ---------------------------------------------------------------------------
# Step 1: Setup
# ---------------------------------------------------------------------------

def setup():
    print("  Building geometry…")
    geom   = make_koch_geometry(n=1)
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=N_PER_EDGE)
    label_corner_ring_panels(panels, P)
    qdata  = build_panel_quadrature(panels, p=P_GL)
    nmat   = assemble_nystrom_matrix(qdata)

    Yq_T      = qdata.Yq.T          # (Nq, 2)
    wq        = qdata.wq
    g_values  = g_fn(Yq_T)
    sigma_BEM = solve_bem(nmat, g_values).sigma
    V_h       = nmat.V
    V_inv     = np.linalg.inv(V_h)
    Nq        = len(sigma_BEM)

    print("  Assembling corrected W̃…")
    W_h, _    = assemble_hypersingular_corrected(qdata)
    W_tilde   = regularise_hypersingular(W_h, wq)

    print("  Building D_h…")
    D_h = build_tangential_derivative_matrix(qdata)
    DV  = D_h @ V_h        # (Nq, Nq)
    Dg  = D_h @ g_values   # (Nq,)

    print(f"  Koch(1): N_panels={qdata.n_panels}, N_quad={Nq}")

    # Arc-length and corner positions
    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc         = panel_start[qdata.pan_id] + qdata.s_on_panel
    sort_idx    = np.argsort(arc)
    corner_arcs = []
    for vi in range(1, len(P), 2):
        c     = P[vi]
        dists = np.linalg.norm(Yq_T - c[None, :], axis=1)
        corner_arcs.append(arc[np.argmin(dists)])
    corner_arcs = sorted(corner_arcs)

    return dict(
        P=P, Yq_T=Yq_T, wq=wq, qdata=qdata,
        g_values=g_values, sigma_BEM=sigma_BEM,
        V_h=V_h, V_inv=V_inv,
        W_tilde=W_tilde,
        D_h=D_h, DV=DV, Dg=Dg,
        Nq=Nq,
        arc=arc, sort_idx=sort_idx, corner_arcs=corner_arcs,
    )


# ---------------------------------------------------------------------------
# Step 2: Spectral analysis
# ---------------------------------------------------------------------------

def spectral_analysis(data: dict) -> dict:
    V_h     = data["V_h"]
    W_tilde = data["W_tilde"]
    DV      = data["DV"]
    Nq      = data["Nq"]

    print("=" * 70)
    print("SPECTRAL ANALYSIS OF EACH LOSS")
    print("=" * 70)

    # Case 1: standard
    sv_V   = np.linalg.svd(V_h, compute_uv=False)
    cond_V = sv_V[0] / sv_V[-1]
    print(f"\nCase 1 (Standard ||Vσ−g||²):")
    print(f"  Effective operator : V")
    print(f"  cond_svd(V)        = {cond_V:.4e}")
    print(f"  Hessian cond       = cond²  = {cond_V**2:.4e}")

    # Case 2: left V⁻¹
    print(f"\nCase 2 (Left V⁻¹ ||σ − σ*||²):")
    print(f"  Effective operator : I")
    print(f"  Hessian cond       = 1.0  (exact)")

    # Case 3: pure Calderón
    WV             = W_tilde @ V_h
    sv_WV          = np.linalg.svd(WV, compute_uv=False)
    eigvals_WV     = np.linalg.eigvals(WV)
    ab_WV          = np.abs(eigvals_WV)
    cond_eig_WV    = ab_WV.max() / ab_WV.min()
    cond_svd_WV    = sv_WV[0] / sv_WV[-1]
    non_norm_WV    = (np.linalg.norm(WV.T @ WV - WV @ WV.T)
                     / np.linalg.norm(WV) ** 2)
    print(f"\nCase 3 (Pure Calderón ||W̃(Vσ−g)||²):")
    print(f"  Effective operator : W̃V")
    print(f"  cond_eig(W̃V)       = {cond_eig_WV:.2f}  [Calderón identity prediction]")
    print(f"  cond_svd(W̃V)       = {cond_svd_WV:.2f}  [actual Hessian conditioning]")
    print(f"  Hessian cond       = cond_svd² = {cond_svd_WV**2:.4e}")
    print(f"  Non-normality      = {non_norm_WV:.3e}  (0=normal, 1=maximally non-normal)")
    print(f"  ★ cond_eig ≈ cond_svd → nearly normal → Hessian cond ≈ 74")

    # Case 4: combined Calderón
    stacked_C4  = np.vstack([V_h, np.sqrt(BETA) * WV])
    sv_C4       = np.linalg.svd(stacked_C4, compute_uv=False)
    cond_C4     = sv_C4[0] / sv_C4[-1]
    print(f"\nCase 4 (Combined Calderón ||r||² + {BETA}||W̃r||²):")
    print(f"  Effective operator : [V; √β·W̃V]")
    print(f"  cond_svd([V;√β·W̃V]) = {cond_C4:.4e}")
    print(f"  Hessian cond         = {cond_C4**2:.4e}")

    # Case 5: pure H¹
    sv_DV      = np.linalg.svd(DV, compute_uv=False)
    cond_DV    = sv_DV[0] / sv_DV[-1]
    rank_DV    = int((sv_DV > 1e-10 * sv_DV[0]).sum())
    null_dim   = Nq - rank_DV
    print(f"\nCase 5 (Pure H¹ ||D_h(Vσ−g)||²):")
    print(f"  Effective operator : D_h V")
    print(f"  cond_svd(D_h V)    = {cond_DV:.4e}")
    print(f"  rank(D_h V)        = {rank_DV} / {Nq}")
    print(f"  nullspace dim      = {null_dim}  (= N_panels: 1 constant per panel)")
    print(f"  Hessian cond       = {cond_DV**2:.4e}  (rank-deficient → +∞ in theory)")
    print(f"  ★ Null space makes loss degenerate: ||D_h r||² = 0 ∀ piecewise-const r")

    # Case 6: combined H¹
    stacked_C6  = np.vstack([V_h, np.sqrt(ALPHA) * DV])
    sv_C6       = np.linalg.svd(stacked_C6, compute_uv=False)
    cond_C6     = sv_C6[0] / sv_C6[-1]
    print(f"\nCase 6 (Combined H¹ ||r||² + {ALPHA}||D_h r||²):")
    print(f"  Effective operator : [V; √α·D_h V]")
    print(f"  cond_svd([V;√α·D_hV]) = {cond_C6:.4e}")
    print(f"  Hessian cond           = {cond_C6**2:.4e}")
    print(f"  ★ V regularises the null space of D_h; full-rank stacked operator")

    return dict(
        cond_V=cond_V, cond_eig_WV=cond_eig_WV, cond_svd_WV=cond_svd_WV,
        cond_C4=cond_C4, cond_DV=cond_DV, rank_DV=rank_DV,
        null_dim=null_dim, cond_C6=cond_C6,
        sv_V=sv_V, sv_WV=sv_WV, sv_DV=sv_DV,
        non_norm_WV=non_norm_WV,
        WV=WV,
    )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _new_model():
    return build_sigma_w_network(HIDDEN_WIDTH, N_HIDDEN).double()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, loss_fn, sigma_BEM_np, Yq_t,
          lr_schedule, n_lbfgs, case_label="?",
          recover_fn=None, verbose=True):
    if recover_fn is None:
        def recover_fn(m):
            with torch.no_grad():
                return m(Yq_t).squeeze(-1).numpy()

    history = {
        "iter": [], "loss": [], "density_reldiff": [], "gnorm": [],
    }

    def _record(it, loss_val=None, gnorm_val=None):
        with torch.no_grad():
            if loss_val is None:
                loss_val = float(loss_fn(model).detach())
        sigma  = recover_fn(model)
        d_err  = float(np.linalg.norm(sigma - sigma_BEM_np)
                       / np.linalg.norm(sigma_BEM_np))
        history["iter"].append(it)
        history["loss"].append(loss_val)
        history["density_reldiff"].append(d_err)
        history["gnorm"].append(gnorm_val)
        if verbose and it % LOG_EVERY == 0:
            gstr = f" | gnorm={gnorm_val:.2e}" if gnorm_val is not None else ""
            print(f"  [{case_label}] iter={it:6d} | loss={loss_val:.3e}"
                  f" | d_err={d_err:.4f}{gstr}")

    # --- Adam ---
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    itr = 0
    _record(0)
    for n_iters, lr in lr_schedule:
        for pg in opt.param_groups:
            pg["lr"] = lr
        for _ in range(n_iters):
            opt.zero_grad()
            loss = loss_fn(model)
            loss.backward()
            gnorm = float(torch.cat([
                p.grad.flatten() for p in model.parameters()
                if p.grad is not None
            ]).norm().item())
            opt.step()
            itr += 1
            if itr % LOG_EVERY == 0:
                _record(itr, float(loss.detach()), gnorm)
    _record(itr)
    if verbose:
        print(f"  [{case_label}] Adam done:  d_err={history['density_reldiff'][-1]:.4f}")

    # --- L-BFGS ---
    opt_lb  = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        history_size=LBFGS_MEMORY, line_search_fn="strong_wolfe",
    )
    n_outer = n_lbfgs // 20
    lb_its  = 0
    loss_at_lbfgs_start = history["loss"][-1]

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

    loss_at_lbfgs_end = history["loss"][-1]
    lbfgs_ratio = loss_at_lbfgs_start / max(loss_at_lbfgs_end, 1e-30)
    lbfgs_made_progress = (loss_at_lbfgs_end < 0.99 * loss_at_lbfgs_start)

    if verbose:
        if lbfgs_made_progress:
            print(f"  [{case_label}] LBFGS done: d_err={history['density_reldiff'][-1]:.4f}"
                  f"  (loss reduced {lbfgs_ratio:.1f}×)")
        else:
            print(f"  [{case_label}] LBFGS STALLED: loss unchanged"
                  f" at {loss_at_lbfgs_end:.3e}")

    history["lbfgs_made_progress"] = lbfgs_made_progress
    history["lbfgs_ratio"]         = lbfgs_ratio
    history["adam_cutoff"]         = itr
    return history


# ---------------------------------------------------------------------------
# Diagnostic notes
# ---------------------------------------------------------------------------

def print_diagnostics(case_id, data, results, spec):
    """Print per-case paper-relevant diagnostics."""
    hist    = results[case_id]["hist"]
    sigma   = results[case_id]["sigma"]
    V_h     = data["V_h"]
    W_tilde = data["W_tilde"]
    DV      = data["DV"]
    Dg      = data["Dg"]
    g       = data["g_values"]
    sigma_BEM = data["sigma_BEM"]
    Nq      = data["Nq"]
    qdata   = data["qdata"]

    res = V_h @ sigma - g

    if case_id == "3":
        # Pure Calderón
        init_loss_std  = (V_h @ np.zeros(Nq) - g)
        init_std_val   = float(np.mean(init_loss_std**2))
        Wres_init      = W_tilde @ (-g)
        init_cal_val   = float(np.mean(Wres_init**2))
        final_gnorm    = hist["gnorm"][-1]
        adam_final_d   = hist["density_reldiff"][hist["adam_cutoff"] // LOG_EVERY]
        stall_iter     = None
        d_errs = hist["density_reldiff"]
        for i in range(len(d_errs) - 1):
            if abs(d_errs[i + 1] - d_errs[i]) < 1e-6:
                stall_iter = hist["iter"][i]
                break

        print(f"\n  [Case 3] Pure Calderón diagnostics:")
        print(f"    init standard loss  = {init_std_val:.3e}")
        print(f"    init Calderón loss  = {init_cal_val:.3e}  "
              f"(ratio = {init_cal_val/max(init_std_val,1e-30):.2f}×)")
        print(f"    final gradient norm = {final_gnorm}")
        print(f"    L-BFGS progress?    = {hist['lbfgs_made_progress']}"
              f"  (ratio {hist['lbfgs_ratio']:.2f}×)")
        if stall_iter is not None:
            print(f"    d_err stalled at iter ≈ {stall_iter}")
        Wres_final = W_tilde @ res
        print(f"    ||W̃r||_final / ||r||_final = "
              f"{np.linalg.norm(Wres_final) / max(np.linalg.norm(res), 1e-30):.2e}")

    elif case_id == "5":
        # Pure H¹
        Dres        = DV @ sigma - Dg
        mean_sigma  = float(np.mean(sigma))
        mean_sigma_BEM = float(np.mean(sigma_BEM))
        print(f"\n  [Case 5] Pure H¹ diagnostics:")
        print(f"    mean(σ_θ)         = {mean_sigma:.4f}")
        print(f"    mean(σ_BEM)       = {mean_sigma_BEM:.4f}")
        print(f"    |Δmean|           = {abs(mean_sigma - mean_sigma_BEM):.4f}"
              f"  ({'drifted' if abs(mean_sigma - mean_sigma_BEM) > 0.05 else 'OK'})")
        # Per-panel mean error
        panel_const_errs = []
        for pid in range(qdata.n_panels):
            js = qdata.idx_std[pid]
            panel_const_errs.append(abs(sigma[js].mean() - sigma_BEM[js].mean()))
        print(f"    per-panel constant error: "
              f"mean={np.mean(panel_const_errs):.3e}, "
              f"max={np.max(panel_const_errs):.3e}")
        print(f"    ||D_h r||_final / Nq = {np.mean(Dres**2):.3e}")
        print(f"    ||r||_final (BIE)    = {float(np.linalg.norm(res)):.3e}")

    elif case_id in ("4", "6"):
        # Combined losses: decompose
        if case_id == "4":
            Wres = W_tilde @ res
            l2_part   = float(np.mean(res**2))
            prec_part = float(BETA * np.mean(Wres**2))
            print(f"\n  [Case 4] Combined Calderón decomposition at convergence:")
            print(f"    L2 part  = {l2_part:.3e}  (||Vσ−g||² / Nq)")
            print(f"    β·W̃ part = {prec_part:.3e}  (β||W̃r||² / Nq)")
            print(f"    ratio    = {prec_part / max(l2_part, 1e-30):.2f}×")
            print(f"    L-BFGS progress? {hist['lbfgs_made_progress']}"
                  f"  ({hist['lbfgs_ratio']:.1f}× loss reduction)")
        else:
            Dres      = DV @ sigma - Dg
            l2_part   = float(np.mean(res**2))
            prec_part = float(ALPHA * np.mean(Dres**2))
            print(f"\n  [Case 6] Combined H¹ decomposition at convergence:")
            print(f"    L2 part  = {l2_part:.3e}  (||Vσ−g||² / Nq)")
            print(f"    α·D_h part = {prec_part:.3e}  (α||D_h r||² / Nq)")
            print(f"    ratio      = {prec_part / max(l2_part, 1e-30):.2f}×")
            print(f"    L-BFGS progress? {hist['lbfgs_made_progress']}"
                  f"  ({hist['lbfgs_ratio']:.1f}× loss reduction)")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _corner_lines(ax, corner_arcs, alpha=0.3, color="lightgray"):
    for ca in corner_arcs:
        ax.axvline(x=ca, color=color, lw=1.2, ls="--", alpha=alpha, zorder=0)


def fig_convergence(results, adam_cutoff, corner_arcs, outpath):
    fig, ax = plt.subplots(figsize=(12, 5))
    for cid in CASES:
        hist = results[cid]["hist"]
        ax.semilogy(hist["iter"], hist["density_reldiff"],
                    "-", color=COLORS[cid], lw=1.8, label=CASE_LABELS[cid])
        ax.annotate(f" {hist['density_reldiff'][-1]:.4f}",
                    xy=(hist["iter"][-1], hist["density_reldiff"][-1]),
                    fontsize=7.5, color=COLORS[cid], va="center")
    ax.axvline(x=adam_cutoff, color="gray", ls=":", lw=1.0, alpha=0.7,
               label="Adam → L-BFGS")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(r"$\|\sigma_\theta - \sigma_\mathrm{BEM}\| / \|\sigma_\mathrm{BEM}\|$",
                  fontsize=12)
    ax.set_title(
        r"Convergence of six loss functions — Koch(1), $g = x^2 - y^2$, no enrichment",
        fontsize=12,
    )
    ax.legend(fontsize=8.5, loc="upper right", ncol=2)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_density(results, sigma_BEM, arc, sort_idx, corner_arcs, outpath):
    arc_s = arc[sort_idx]
    bem_s = sigma_BEM[sort_idx]
    ymin  = bem_s.min() * 1.35
    ymax  = bem_s.max() * 1.35

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True, sharex=True)
    titles = {
        "1": "1: Standard",
        "2": r"2: Left $V^{-1}$",
        "3": r"3: Pure Calderón",
        "4": r"4: Comb. Calderón ($\beta=10$)",
        "5": r"5: Pure H¹",
        "6": r"6: Comb. H¹ ($\alpha=1$)",
    }
    axes_flat = axes.flatten()
    for ax, cid in zip(axes_flat, CASES):
        sig   = results[cid]["sigma"][sort_idx]
        d_err = results[cid]["d_err"]
        ax.plot(arc_s, bem_s, "k--", lw=1.8, alpha=0.55,
                label=r"$\sigma_\mathrm{BEM}$")
        ax.plot(arc_s, sig, "-", lw=1.6, color=COLORS[cid],
                label=f"d={d_err:.4f}")
        _corner_lines(ax, corner_arcs)
        ax.set_ylim(ymin, ymax)
        ax.set_title(titles[cid], fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, lw=0.3, alpha=0.4)
    for ax in axes[1]:
        ax.set_xlabel("Arc-length $s$", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\sigma(s)$", fontsize=10)
    fig.suptitle(
        r"Final density $\sigma_\theta(s)$ — Koch(1), $g = x^2 - y^2$, no enrichment",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_density_error(results, sigma_BEM, arc, sort_idx, corner_arcs, outpath):
    arc_s = arc[sort_idx]
    bem_s = sigma_BEM[sort_idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    for cid in CASES:
        sig = results[cid]["sigma"][sort_idx]
        err = np.abs(sig - bem_s)
        ax.semilogy(arc_s, err + 1e-15, "-", color=COLORS[cid],
                    lw=1.6, label=CASE_LABELS[cid])
    _corner_lines(ax, corner_arcs, alpha=0.5)
    ax.set_xlabel("Arc-length $s$", fontsize=12)
    ax.set_ylabel(r"$|\sigma_\theta(s) - \sigma_\mathrm{BEM}(s)|$", fontsize=12)
    ax.set_title(
        r"Pointwise density error — Koch(1), $g = x^2 - y^2$, no enrichment",
        fontsize=12,
    )
    ax.legend(fontsize=8.5, loc="upper right", ncol=2)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_singular_values(spec, Nq, outpath):
    sv_V   = spec["sv_V"]
    sv_WV  = spec["sv_WV"]
    sv_DV  = spec["sv_DV"]
    idx    = np.arange(1, Nq + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, sv, label, color, cond_str in [
        (axes[0], sv_V,  r"$\sigma_k(V)$ — Cases 1,2",
         COLORS["1"], f"cond={sv_V[0]/sv_V[-1]:.0f}"),
        (axes[1], sv_WV, r"$\sigma_k(\tilde{W}V)$ — Cases 3,4",
         COLORS["3"], f"cond={sv_WV[0]/sv_WV[-1]:.1f}"),
        (axes[2], sv_DV, r"$\sigma_k(D_h V)$ — Cases 5,6",
         COLORS["5"], f"cond={sv_DV[0]/sv_DV[-1]:.2e}"),
    ]:
        ax.semilogy(idx, sv, "-", color=color, lw=1.8, label=label)
        ax.text(0.97, 0.95, cond_str, transform=ax.transAxes,
                ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
        ax.set_xlabel(r"Index $k$", fontsize=11)
        ax.set_ylabel("Singular value", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", lw=0.3, alpha=0.5)

    # Mark the null space of DV
    rank_DV = spec["rank_DV"]
    if rank_DV < Nq:
        axes[2].axvline(x=rank_DV, color="red", ls="--", lw=1.5, alpha=0.7,
                        label=f"rank={rank_DV}")
        axes[2].legend(fontsize=9)

    fig.suptitle(
        r"Singular values of effective operators — Koch(1), $N_q=%d$" % Nq,
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_summary_bar(results, outpath):
    d_errs = [results[cid]["d_err"] for cid in CASES]
    colors = [
        COLORS[cid] if d < SUCCESS_THRESHOLD else "#cccccc"
        for cid, d in zip(CASES, d_errs)
    ]
    edge_colors = [
        "green" if d < SUCCESS_THRESHOLD else "red"
        for d in d_errs
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(CASES))
    bars = ax.bar(x, d_errs, color=colors, edgecolor=edge_colors,
                  linewidth=1.5, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CASE_LABELS[cid].replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
         for cid in CASES],
        fontsize=9, rotation=15, ha="right",
    )
    ax.set_ylabel("Density rel-diff (log scale)", fontsize=12)
    ax.set_title(
        r"Six-loss comparison — Koch(1), $g = x^2 - y^2$, no enrichment"
        f"\nSame network (4×{HIDDEN_WIDTH} tanh), same init, same training schedule",
        fontsize=11,
    )
    ax.axhline(y=SUCCESS_THRESHOLD, color="green", ls="--", lw=1.5, alpha=0.7,
               label=f"Success threshold ({int(SUCCESS_THRESHOLD*100)}%)")
    for bar, d in zip(bars, d_errs):
        ax.text(bar.get_x() + bar.get_width() / 2, d * 1.3,
                f"{d:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", which="both", lw=0.3, alpha=0.5, zorder=0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    fig_dir = os.path.join(_HERE, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("SIX-LOSS COMPARISON — Koch(1), g=x²−y², no enrichment")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Setup
    # ------------------------------------------------------------------
    print("\n--- Step 1: Setup ---")
    data      = setup()
    Yq_T      = data["Yq_T"]
    wq        = data["wq"]
    P         = data["P"]
    qdata     = data["qdata"]
    g_values  = data["g_values"]
    sigma_BEM = data["sigma_BEM"]
    V_h       = data["V_h"]
    V_inv     = data["V_inv"]
    W_tilde   = data["W_tilde"]
    DV        = data["DV"]
    Dg        = data["Dg"]
    Nq        = data["Nq"]
    arc       = data["arc"]
    sort_idx  = data["sort_idx"]
    corner_arcs = data["corner_arcs"]

    # ------------------------------------------------------------------
    # Step 2: Spectral analysis
    # ------------------------------------------------------------------
    print("\n--- Step 2: Spectral analysis ---")
    spec = spectral_analysis(data)

    # ------------------------------------------------------------------
    # Step 3: Torch tensors
    # ------------------------------------------------------------------
    Yq_t        = torch.tensor(Yq_T,      dtype=torch.float64)
    g_t         = torch.tensor(g_values,  dtype=torch.float64)
    V_h_t       = torch.tensor(V_h,       dtype=torch.float64)
    V_inv_t     = torch.tensor(V_inv,     dtype=torch.float64)
    W_tilde_t   = torch.tensor(W_tilde,   dtype=torch.float64)
    DV_t        = torch.tensor(DV,        dtype=torch.float64)
    Dg_t        = torch.tensor(Dg,        dtype=torch.float64)
    sigma_BEM_t = torch.tensor(sigma_BEM, dtype=torch.float64)

    # ------------------------------------------------------------------
    # Step 4: Loss functions
    # ------------------------------------------------------------------
    def loss_1(model):
        s   = model(Yq_t).squeeze(-1)
        res = V_h_t @ s - g_t
        return (res ** 2).mean()

    def loss_2(model):
        s   = model(Yq_t).squeeze(-1)
        res = V_h_t @ s - g_t
        return (V_inv_t @ res).pow(2).mean()

    def loss_3(model):
        s    = model(Yq_t).squeeze(-1)
        res  = V_h_t @ s - g_t
        Wres = W_tilde_t @ res
        return (Wres ** 2).mean()

    def loss_4(model, beta=BETA):
        s    = model(Yq_t).squeeze(-1)
        res  = V_h_t @ s - g_t
        Wres = W_tilde_t @ res
        return (res ** 2).mean() + beta * (Wres ** 2).mean()

    def loss_5(model):
        s    = model(Yq_t).squeeze(-1)
        Dres = DV_t @ s - Dg_t
        return (Dres ** 2).mean()

    def loss_6(model, alpha=ALPHA):
        s    = model(Yq_t).squeeze(-1)
        res  = V_h_t @ s - g_t
        Dres = DV_t @ s - Dg_t
        return (res ** 2).mean() + alpha * (Dres ** 2).mean()

    loss_fns = {
        "1": loss_1, "2": loss_2, "3": loss_3,
        "4": loss_4, "5": loss_5, "6": loss_6,
    }

    # ------------------------------------------------------------------
    # Step 5: Shared initial weights
    # ------------------------------------------------------------------
    print("\n--- Step 5: Initialising models ---")
    torch.manual_seed(SEED)
    base_model = _new_model()
    init_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    def _fresh():
        m = _new_model()
        m.load_state_dict({k: v.clone() for k, v in init_state.items()})
        return m

    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Parameters per model: {n_params}")

    # Print initial loss ratios
    print(f"\n  Initial loss values (σ_θ ≈ 0):")
    with torch.no_grad():
        m0 = _fresh()
        il = {cid: float(loss_fns[cid](m0).detach()) for cid in CASES}
    ref = il["1"]
    for cid in CASES:
        print(f"    Case {cid}: {il[cid]:.3e}  (ratio vs Case 1: {il[cid]/ref:.2f}×)")

    # ------------------------------------------------------------------
    # Step 6: Training
    # ------------------------------------------------------------------
    adam_cutoff = sum(n for n, _ in LR_SCHEDULE)
    results     = {}

    case_titles = {
        "1": "Case 1: Standard ||Vσ−g||²",
        "2": "Case 2: Left V⁻¹  ||σ−σ*||²",
        "3": f"Case 3: Pure Calderón ||W̃(Vσ−g)||²",
        "4": f"Case 4: Combined Calderón  β={BETA}",
        "5": "Case 5: Pure H¹  ||D_h(Vσ−g)||²",
        "6": f"Case 6: Combined H¹  α={ALPHA}",
    }

    for cid in CASES:
        print("\n" + "=" * 60)
        print(case_titles[cid])
        print("=" * 60)
        model = _fresh()
        t0    = time.perf_counter()
        hist  = train(model, loss_fns[cid], sigma_BEM, Yq_t,
                      lr_schedule=LR_SCHEDULE, n_lbfgs=N_LBFGS,
                      case_label=cid, verbose=True)
        wall  = time.perf_counter() - t0

        with torch.no_grad():
            sigma = model(Yq_t).squeeze(-1).numpy()

        d_err = float(np.linalg.norm(sigma - sigma_BEM)
                      / np.linalg.norm(sigma_BEM))
        bie   = float(np.linalg.norm(V_h @ sigma - g_values)
                      / np.linalg.norm(g_values))
        iL2   = float(reconstruct_interior(
            P=P, Yq=Yq_T, wq=wq, sigma=sigma,
            n_grid=N_GRID_FINAL, u_exact=u_exact_fn,
        ).rel_L2)

        results[cid] = {
            "hist": hist, "sigma": sigma,
            "d_err": d_err, "bie": bie, "iL2": iL2,
            "wall": wall,
        }

    # ------------------------------------------------------------------
    # Step 7: Diagnostics
    # ------------------------------------------------------------------
    print("\n--- Step 7: Per-case diagnostics ---")
    for cid in ["3", "4", "5", "6"]:
        print_diagnostics(cid, data, results, spec)

    # ------------------------------------------------------------------
    # Step 8: Summary table
    # ------------------------------------------------------------------
    hess_conds = {
        "1": f"{spec['cond_V']**2:.2e}",
        "2": "1",
        "3": f"{spec['cond_svd_WV']**2:.2e}",
        "4": f"{spec['cond_C4']**2:.2e}",
        "5": "∞ (rank-def.)",
        "6": f"{spec['cond_C6']**2:.2e}",
    }
    lbfgs_reasons = {
        "1": "stalls (ill-cond.)",
        "2": "converges (cond=1)",
        "3": "stalls (Wolfe fails)",
        "4": "?" ,
        "5": "stalls (null space)",
        "6": "converges (PSD, full-rank)",
    }
    for cid in ("3", "4", "5", "6"):
        lbfgs_reasons[cid] = (
            "converges" if results[cid]["hist"]["lbfgs_made_progress"]
            else "stalls"
        )

    print(f"\n{'='*90}")
    print(f"SIX-LOSS COMPARISON — Koch(1), g=x²−y², no enrichment")
    print(f"{'='*90}")
    print(f"  Nq={Nq}  |  network={N_HIDDEN}×{HIDDEN_WIDTH} tanh  |  seed={SEED}")
    print(f"  Training: Adam {LR_SCHEDULE} + L-BFGS {N_LBFGS}")

    w = 14
    sep   = "─" * 28 + "─┬─" + ("─" * w + "─┬─") * 5 + "─" * w
    hdr   = (f"{'Metric':<28s} │ " +
             " │ ".join(f"{'Case '+c:>{w}s}" for c in CASES))
    print(f"\n{sep}\n{hdr}\n{sep}")

    rows = [
        ("Density rel-diff",
         [f"{results[c]['d_err']:.4f}" for c in CASES]),
        ("BIE residual",
         [f"{results[c]['bie']:.2e}" for c in CASES]),
        ("Interior rel L2",
         [f"{results[c]['iL2']:.2e}" for c in CASES]),
        ("Final loss",
         [f"{results[c]['hist']['loss'][-1]:.2e}" for c in CASES]),
        ("Hessian cond (theory)",
         [hess_conds[c] for c in CASES]),
        ("Wall time (s)",
         [f"{results[c]['wall']:.1f}" for c in CASES]),
        ("L-BFGS",
         [lbfgs_reasons[c] for c in CASES]),
    ]
    for name, vals in rows:
        print(f"{name:<28s} │ " +
              " │ ".join(f"{v:>{w}s}" for v in vals))
    print(sep)

    d_ref = results["1"]["d_err"]
    impr = [f"{d_ref/max(results[c]['d_err'],1e-9):.1f}×" for c in CASES]
    impr[0] = "—"
    print(f"{'Improvement vs Case 1':<28s} │ " +
          " │ ".join(f"{v:>{w}s}" for v in impr))
    print(sep)

    # Summary verdict
    print(f"\n  Works (d_err < {int(SUCCESS_THRESHOLD*100)}%): "
          + ", ".join(f"Case {c}"
                      for c in CASES
                      if results[c]["d_err"] < SUCCESS_THRESHOLD))
    print(f"  Fails: "
          + ", ".join(f"Case {c}"
                      for c in CASES
                      if results[c]["d_err"] >= SUCCESS_THRESHOLD))

    # ------------------------------------------------------------------
    # Step 9: Figures
    # ------------------------------------------------------------------
    print("\n--- Step 9: Figures ---")

    fig_convergence(
        results, adam_cutoff, corner_arcs,
        os.path.join(fig_dir, "six_loss_convergence.png"),
    )
    fig_density(
        results, sigma_BEM, arc, sort_idx, corner_arcs,
        os.path.join(fig_dir, "six_loss_density.png"),
    )
    fig_density_error(
        results, sigma_BEM, arc, sort_idx, corner_arcs,
        os.path.join(fig_dir, "six_loss_density_error.png"),
    )
    fig_singular_values(
        spec, Nq,
        os.path.join(fig_dir, "six_loss_singular_values.png"),
    )
    fig_summary_bar(
        results,
        os.path.join(fig_dir, "six_loss_summary.png"),
    )

    print(f"\n  All figures saved to {fig_dir}/")
    return results, spec


if __name__ == "__main__":
    main()
