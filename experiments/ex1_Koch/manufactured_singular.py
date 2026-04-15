"""
Experiment: Koch(1) with manufactured singular density.

All previous Koch experiments used g = x²−y², which has zero enrichment
energy (γ*_lstsq = 0).  This experiment constructs a boundary density with
known per-corner singularity amplitudes, providing a true benchmark for
SE-BINN on the Koch geometry.

Manufactured density
--------------------
  σ_mfg = σ_smooth + Σ_c γ_c^true · σ_s^(c)

where:
  σ_smooth   = BEM solution for g_smooth = x² − y²  (no spikes)
  σ_s^(c)    = singular basis function at corner c  (with cutoff)
  γ_c^true   = [+1.0, −0.5, +1.0, −0.5, +1.0, −0.5]
               (3-fold symmetric: corners 1,5,9 → +1.0; corners 3,7,11 → −0.5)

Boundary data:  g_mfg = V · σ_mfg  (Nyström matrix applied to σ_mfg)

This guarantees:
  - V σ_BEM = g_mfg  ⟹  σ_BEM = σ_mfg  (exact BEM recovery)
  - Enrichment energy = ‖S @ γ_true‖² / ‖σ_mfg‖² >> 2.8%
  - γ*_lstsq = γ_true  (exact per-corner recovery by lstsq projection)

Training cases
--------------
  Case A — BINN:       γ frozen at 0
  Case B — SE-BINN:    γ trainable, init = 0.0
  Case C — SE-BINN★:  γ trainable, init = γ_lstsq

Configuration
-------------
  n_per_edge = 12, p_gl = 16, m_col_base = 16 (Xc = Yq, Nb = Nq = 2304)
  cutoff_radius = 0.5 × min_corner_separation
  per_corner_gamma = True  (6 independent γ_c)

Key hypothesis
--------------
  The A/B density improvement on Koch(1) with genuine singularities should
  be >> the 1.12× gap seen with smooth g = x²−y², matching the L-shape's
  14× improvement.

Figures saved to experiments/ex1_Koch/figures/:
  manufactured_density.png       — σ_mfg and σ_smooth vs arclength
  manufactured_ab.png            — loss / density / density error (A, B, C)
  manufactured_gamma.png         — γ_c trajectories vs iteration
  manufactured_corner_zoom.png   — zoom into one reentrant corner
"""

import sys
import os
import copy
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from src.boundary.polygon import make_koch_geometry
from src.boundary.panels import (
    build_uniform_panels,
    label_corner_ring_panels,
    panel_loss_weights,
)
from src.quadrature.panel_quad import build_panel_quadrature
from src.quadrature.nystrom import assemble_nystrom_matrix, solve_bem
from src.singular.enrichment import SingularEnrichment
from src.models.sebinn import SEBINNModel
from src.training.collocation import build_collocation_points
from src.training.operator import build_operator_state
from src.training.adam_phase import AdamConfig, run_adam_phases
from src.training.lbfgs import LBFGSConfig, run_lbfgs
from src.reconstruction.interior import reconstruct_interior, _log_kernel_matrix


# ===========================================================================
# Configuration
# ===========================================================================

CFG = dict(
    seed           = 0,
    n_per_edge     = 12,
    p_gl           = 16,
    m_col_base     = 16,    # = p_gl  →  Xc = Yq, Nb = Nq = 2304
    w_base         = 1.0,
    w_corner       = 1.0,
    w_ring         = 1.0,
    eq_scale_mode  = "fixed",
    eq_scale_fixed = 10.0,
    gmres_tol      = 1e-12,
    gmres_maxiter  = 300,
    hidden_width   = 80,
    n_hidden       = 4,
    gamma_init     = 0.0,
    # Manufactured density
    gamma_true     = np.array([+1.0, -0.5, +1.0, -0.5, +1.0, -0.5]),
    # Training
    adam_iters     = [1000, 1000, 1000],
    adam_lrs       = [1e-3, 3e-4, 1e-4],
    log_every      = 200,
    lbfgs_max_iters  = 15000,
    lbfgs_grad_tol   = 1e-10,
    lbfgs_step_tol   = 1e-12,
    lbfgs_memory     = 30,
    lbfgs_log_every  = 200,
    lbfgs_alpha0     = 1e-1,
    lbfgs_alpha_fb   = [1e-2, 1e-3],
    lbfgs_armijo_c1  = 1e-4,
    lbfgs_beta       = 0.5,
    lbfgs_max_bt     = 20,
    n_grid_coarse    = 51,
    n_grid_final     = 201,
)


# ===========================================================================
# Arc-length helper
# ===========================================================================

def _boundary_arclength(qdata, n_per_edge):
    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc = panel_start[qdata.pan_id] + qdata.s_on_panel
    Npan  = qdata.n_panels
    total = float(qdata.L_panel.sum())
    Nv    = Npan // n_per_edge
    v_panel_idx = np.arange(Nv) * n_per_edge
    panel_start_full = np.concatenate([[0.0], np.cumsum(qdata.L_panel)])
    vertex_arcs = np.append(panel_start_full[v_panel_idx], total)
    return arc, vertex_arcs, total


# ===========================================================================
# u_exact from manufactured density (single-layer potential)
# ===========================================================================

def make_u_exact_fn(Yq_T, wq, sigma_mfg):
    """u_mfg(x) = Σ_j G(x, Yq_j) σ_mfg_j wq_j."""
    sigma_wq = sigma_mfg * wq

    def _u(xy):
        K = _log_kernel_matrix(xy, Yq_T)   # (M, Nq)
        return K @ sigma_wq                 # (M,)

    return _u


# ===========================================================================
# L-BFGS config builder
# ===========================================================================

def _make_lbfgs_cfg():
    return LBFGSConfig(
        max_iters      = CFG["lbfgs_max_iters"],
        grad_tol       = CFG["lbfgs_grad_tol"],
        step_tol       = CFG["lbfgs_step_tol"],
        memory         = CFG["lbfgs_memory"],
        log_every      = CFG["lbfgs_log_every"],
        alpha0         = CFG["lbfgs_alpha0"],
        alpha_fallback = CFG["lbfgs_alpha_fb"],
        armijo_c1      = CFG["lbfgs_armijo_c1"],
        backtrack_beta = CFG["lbfgs_beta"],
        max_backtrack  = CFG["lbfgs_max_bt"],
    )


# ===========================================================================
# One full training run
# ===========================================================================

def _train_one(
    label: str,
    freeze_gamma: bool,
    gamma_override,
    init_state: dict,
    shared: dict,
    verbose: bool = True,
) -> dict:
    if verbose:
        fz = " [γ frozen=0]" if freeze_gamma else " [γ trainable]"
        print(f"\n{'='*62}")
        print(f"  Case {label}{fz}")
        print(f"{'='*62}")

    t0 = time.perf_counter()

    op          = shared["op"]
    Yq_T        = shared["Yq_T"]
    wq          = shared["wq"]
    P           = shared["P"]
    sigma_bem   = shared["sigma_bem"]
    sigma_s_Yq  = shared["sigma_s_Yq"]   # (Nq, 6) per-corner
    sort_idx    = shared["sort_idx"]
    u_exact     = shared["u_exact"]
    n_gamma     = sigma_s_Yq.shape[1] if sigma_s_Yq.ndim == 2 else 1

    model = SEBINNModel(
        hidden_width = CFG["hidden_width"],
        n_hidden     = CFG["n_hidden"],
        n_gamma      = n_gamma,
        gamma_init   = CFG["gamma_init"],
        dtype        = torch.float64,
    )
    model.load_state_dict(copy.deepcopy(init_state))

    if gamma_override is not None:
        g_ov = np.asarray(gamma_override)
        model.gamma_module.gamma.data.copy_(
            torch.tensor(g_ov, dtype=torch.float64)
        )
        if verbose:
            print(f"  γ warm-start: {list(np.round(g_ov, 5))}")

    if freeze_gamma:
        model.gamma_module.gamma.requires_grad_(False)
        if verbose:
            print(f"  γ frozen at 0")

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  trainable params: {n_tr} | n_gamma={n_gamma}")

    Yq_t      = torch.tensor(Yq_T,      dtype=torch.float64)
    sigma_s_t = torch.tensor(sigma_s_Yq, dtype=torch.float64)   # (Nq, 6)

    # Gamma trajectory recording (for plot)
    gamma_traj = []   # list of (iter, gamma_array)

    stage_checkpoints = []

    def _record_checkpoint(stage, n_iter, loss):
        with torch.no_grad():
            s = model(Yq_t, sigma_s_t).numpy()
        d_err = float(np.linalg.norm(s - sigma_bem)
                      / max(np.linalg.norm(sigma_bem), 1e-14))
        i_out = reconstruct_interior(
            P=P, Yq=Yq_T, wq=wq, sigma=s,
            n_grid=CFG["n_grid_coarse"], u_exact=u_exact,
        )
        g = model.gamma_value()
        g_arr = np.array(g if isinstance(g, list) else [float(g)])
        gamma_traj.append((n_iter, g_arr.copy()))
        stage_checkpoints.append(dict(
            stage=stage, iter=n_iter, loss=loss,
            gamma=g_arr.copy(),
            density_rel_diff=d_err,
            interior_L2=i_out.rel_L2,
        ))
        if verbose:
            gfmt = "[" + ",".join(f"{v:+.4f}" for v in g_arr) + "]"
            print(f"  [{label}] {stage}: loss={loss:.3e} | d_err={d_err:.4f} | "
                  f"iL2={i_out.rel_L2:.3e} | γ={gfmt}")

    # Adam phases
    all_adam_loss = []
    global_it = 0
    for ph_idx, (n_it, lr) in enumerate(zip(CFG["adam_iters"], CFG["adam_lrs"])):
        ph_cfg = AdamConfig(
            phase_iters=[n_it], phase_lrs=[lr], log_every=CFG["log_every"],
        )
        ph_res = run_adam_phases(model, op, ph_cfg, verbose=verbose)
        all_adam_loss.extend(ph_res.loss_hist)
        global_it += ph_res.n_iters
        _record_checkpoint(f"Adam-ph{ph_idx+1}", global_it, ph_res.final_loss)

    # L-BFGS
    lbfgs_cfg = _make_lbfgs_cfg()
    if verbose:
        print(f"\n  [{label}] L-BFGS: max={CFG['lbfgs_max_iters']} | "
              f"mem={CFG['lbfgs_memory']} | grad_tol={CFG['lbfgs_grad_tol']:.0e}")
    lbfgs_res = run_lbfgs(model, op, lbfgs_cfg, verbose=verbose)
    _record_checkpoint(
        "LBFGS", global_it + lbfgs_res.n_iters,
        lbfgs_res.loss_hist[-1] if lbfgs_res.loss_hist else float("nan"),
    )

    # Final density
    with torch.no_grad():
        sigma_final = model(Yq_t, sigma_s_t).numpy()

    final_out = reconstruct_interior(
        P=P, Yq=Yq_T, wq=wq, sigma=sigma_final,
        n_grid=CFG["n_grid_final"], u_exact=u_exact,
    )
    density_rel_diff = float(
        np.linalg.norm(sigma_final - sigma_bem)
        / max(np.linalg.norm(sigma_bem), 1e-14)
    )
    t_total = time.perf_counter() - t0

    if verbose:
        g    = model.gamma_value()
        garr = np.array(g if isinstance(g, list) else [float(g)])
        gfmt = "[" + ",".join(f"{v:+.5f}" for v in garr) + "]"
        print(f"\n  {label} final:")
        print(f"    Interior rel L2  : {final_out.rel_L2:.3e}")
        print(f"    Interior L∞      : {final_out.linf:.3e}")
        print(f"    Density rel-diff : {density_rel_diff:.4f}")
        print(f"    γ final          : {gfmt}")
        print(f"    γ_true           : {list(np.round(CFG['gamma_true'], 4))}")
        g_err = float(np.linalg.norm(garr - CFG["gamma_true"])
                      / max(np.linalg.norm(CFG["gamma_true"]), 1e-14))
        print(f"    ‖γ−γ_true‖/‖γ_true‖ : {g_err:.4f}")
        print(f"    LBFGS reason     : {lbfgs_res.reason}")
        print(f"    Wall time        : {t_total:.1f}s")

    gamma_final = np.array(
        model.gamma_value() if isinstance(model.gamma_value(), list)
        else [float(model.gamma_value())]
    )
    gamma_err = float(
        np.linalg.norm(gamma_final - CFG["gamma_true"])
        / max(np.linalg.norm(CFG["gamma_true"]), 1e-14)
    )

    return dict(
        label             = label,
        freeze_gamma      = freeze_gamma,
        final_rel_L2      = final_out.rel_L2,
        final_linf        = final_out.linf,
        density_rel_diff  = density_rel_diff,
        gamma_final       = gamma_final,
        gamma_err         = gamma_err,
        lbfgs_reason      = lbfgs_res.reason,
        wall_time         = t_total,
        loss_hist_adam    = all_adam_loss,
        loss_hist_lbfgs   = list(lbfgs_res.loss_hist),
        adam_n_iters      = global_it,
        sigma_final       = sigma_final[sort_idx],
        stage_checkpoints = stage_checkpoints,
        gamma_traj        = gamma_traj,
        final_out         = final_out,
    )


# ===========================================================================
# Figures
# ===========================================================================

def _fig_manufactured_density(arc_s, sigma_mfg_s, sigma_smooth_s,
                               vertex_arcs, sing_idx, gamma_true, outpath):
    """(a) σ_mfg and σ_smooth vs arclength."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fig.subplots_adjust(hspace=0.42)

    def _vlines(ax):
        for ci in sing_idx:
            ax.axvline(vertex_arcs[ci], color="#aaaaaa", lw=0.8, ls="--", alpha=0.7)

    ax = axes[0]
    _vlines(ax)
    ax.plot(arc_s, sigma_smooth_s, color="#1f77b4", lw=1.2,
            label=r"$\sigma_\mathrm{smooth}$ (from $g=x^2-y^2$ BEM)")
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$\sigma$", fontsize=11)
    ax.set_title(r"(a) Smooth component $\sigma_\mathrm{smooth}$ — no corner spikes", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, lw=0.3, alpha=0.5)

    ax = axes[1]
    _vlines(ax)
    ax.plot(arc_s, sigma_mfg_s, color="#d62728", lw=1.2,
            label=r"$\sigma_\mathrm{mfg}=\sigma_\mathrm{smooth}+\sum_c\gamma_c^\mathrm{true}\sigma_s^{(c)}$")
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$\sigma$", fontsize=11)
    ax.set_title(r"(b) Manufactured density $\sigma_\mathrm{mfg}$ — 6 reentrant corner spikes", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, lw=0.3, alpha=0.5)

    fig.suptitle(
        r"Koch(1) — Manufactured singular density" + "\n"
        + r"$\gamma^\mathrm{true}=[+1.0,-0.5,+1.0,-0.5,+1.0,-0.5]$",
        fontsize=12, y=1.01,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved manufactured_density → {outpath}")


def _fig_ab(cases, sigma_bem_sorted, arc_s, vertex_arcs, sing_idx, outpath):
    """(b) Loss / density / density error for A, B, C."""
    COLORS = {
        "BINN":     "#1f77b4",
        "SE-BINN":  "#d62728",
        "SE-BINN★": "#2ca02c",
    }
    LS = {
        "BINN":     "-",
        "SE-BINN":  "--",
        "SE-BINN★": "-.",
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.subplots_adjust(hspace=0.42)

    def _vlines(ax):
        for ci in sing_idx:
            ax.axvline(vertex_arcs[ci], color="#aaaaaa", lw=0.7, ls="--", alpha=0.7)

    ax = axes[0]
    for c in cases:
        col  = COLORS.get(c["label"], "gray")
        ha, hl = c["loss_hist_adam"], c["loss_hist_lbfgs"]
        n_a, n_l = len(ha), len(hl)
        ax.semilogy(np.arange(1, n_a+1), ha, color=col, lw=1.4, ls="-",
                    alpha=0.85, label=f"{c['label']} Adam")
        if n_l:
            ax.semilogy(np.arange(n_a+1, n_a+n_l+1), hl, color=col, lw=1.4,
                        ls="--", alpha=0.85, label=f"{c['label']} L-BFGS")
    n_adam = cases[0]["adam_n_iters"]
    ax.axvline(n_adam, color="gray", lw=0.9, ls=":", alpha=0.6,
               label=f"Adam→L-BFGS ({n_adam})")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("(a) Loss history", fontsize=12)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    ax = axes[1]
    _vlines(ax)
    ax.plot(arc_s, sigma_bem_sorted, color="black", lw=1.4, alpha=0.9,
            label=r"$\sigma_\mathrm{mfg}$")
    for c in cases:
        col = COLORS.get(c["label"], "gray")
        ls  = LS.get(c["label"], "-")
        ax.plot(arc_s, c["sigma_final"], color=col, lw=1.1, ls=ls,
                alpha=0.85, label=c["label"])
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$\sigma(s)$", fontsize=11)
    ax.set_title(r"(b) Boundary density", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)

    ax = axes[2]
    _vlines(ax)
    for c in cases:
        col = COLORS.get(c["label"], "gray")
        ls  = LS.get(c["label"], "-")
        err = np.abs(c["sigma_final"] - sigma_bem_sorted)
        ax.semilogy(arc_s, err + 1e-16, color=col, lw=1.1, ls=ls,
                    alpha=0.85,
                    label=f"{c['label']}  (d_err={c['density_rel_diff']:.4f})")
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$|\sigma_\theta - \sigma_\mathrm{mfg}|$", fontsize=11)
    ax.set_title("(c) Pointwise density error", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(
        r"A/B/C Comparison — Koch(1), manufactured $\sigma_\mathrm{mfg}$",
        fontsize=12, y=1.01,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved manufactured_ab → {outpath}")


def _fig_gamma_traj(cases, gamma_true, outpath):
    """(c) γ_c trajectories for trainable cases."""
    n_corners = len(gamma_true)
    CASE_COLORS = {"SE-BINN": plt.cm.Reds, "SE-BINN★": plt.cm.Greens}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.38)

    trainable = [c for c in cases if not c["freeze_gamma"]]
    for ax, case in zip(axes, trainable):
        cmap = CASE_COLORS.get(case["label"], plt.cm.Blues)
        traj = case["gamma_traj"]   # list of (iter, gamma_array)
        if not traj:
            continue
        iters  = np.array([t[0] for t in traj])
        gammas = np.array([t[1] for t in traj])   # (n_checkpoints, 6)

        for c_idx in range(n_corners):
            color = cmap(0.3 + 0.7 * c_idx / max(n_corners - 1, 1))
            ax.plot(iters, gammas[:, c_idx], color=color, lw=1.4,
                    label=f"γ_{c_idx+1} (true={gamma_true[c_idx]:+.1f})")
            ax.axhline(gamma_true[c_idx], color=color, lw=0.8, ls=":",
                       alpha=0.6)

        ax.set_xlabel("Iteration (end of stage)", fontsize=11)
        ax.set_ylabel(r"$\gamma_c$", fontsize=11)
        ax.set_title(f"γ_c trajectories — {case['label']}", fontsize=12)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, lw=0.3, alpha=0.5)

    fig.suptitle(
        r"Per-corner $\gamma_c$ convergence — Koch(1) manufactured density" + "\n"
        r"Dotted lines: $\gamma^\mathrm{true}$",
        fontsize=11,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved manufactured_gamma → {outpath}")


def _fig_corner_zoom(cases, sigma_bem_sorted, arc_s,
                     vertex_arcs, sing_idx, outpath):
    """(d) Zoom into corner 1 (first reentrant corner at index sing_idx[0])."""
    COLORS = {
        "BINN":     "#1f77b4",
        "SE-BINN":  "#d62728",
        "SE-BINN★": "#2ca02c",
    }
    LS = {
        "BINN":     "-",
        "SE-BINN":  "--",
        "SE-BINN★": "-.",
    }

    if len(sing_idx) == 0:
        return
    s_corner = vertex_arcs[sing_idx[0]]
    half_win  = 0.15
    s_lo, s_hi = s_corner - half_win, s_corner + half_win
    mask = (arc_s >= s_lo) & (arc_s <= s_hi)
    if mask.sum() < 10:
        mask = np.ones(len(arc_s), dtype=bool)

    arc_z  = arc_s[mask]
    sigma_z = sigma_bem_sorted[mask]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axvline(s_corner, color="gray", lw=0.9, ls="--", alpha=0.7,
               label=f"corner s={s_corner:.3f}")
    ax.plot(arc_z, sigma_z, color="black", lw=1.6, alpha=0.9,
            label=r"$\sigma_\mathrm{mfg}$")
    for c in cases:
        col = COLORS.get(c["label"], "gray")
        ls  = LS.get(c["label"], "-")
        ax.plot(arc_z, c["sigma_final"][mask], color=col, lw=1.3, ls=ls,
                alpha=0.85, label=c["label"])

    ax.set_xlim(s_lo, s_hi)
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$\sigma(s)$", fontsize=11)
    ax.set_title(
        r"Reentrant corner zoom — $r^{-1/4}$ spike at corner 1" + "\n"
        r"BINN: smooth approximation; SE-BINN: spike via $\gamma\sigma_s$",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved manufactured_corner_zoom → {outpath}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("=" * 72)
    print("  Koch(1) — Manufactured singular density experiment")
    print("=" * 72)

    t_global = time.perf_counter()
    figures_dir = os.path.join(_HERE, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Geometry and quadrature
    # ------------------------------------------------------------------
    print("\n--- Setup ---")
    geom   = make_koch_geometry(n=1)
    P      = geom.vertices           # (12, 2)
    n_v    = len(P)
    panels = build_uniform_panels(P, n_per_edge=CFG["n_per_edge"])
    label_corner_ring_panels(panels, P)

    alpha_koch = np.pi / geom.corner_angles[geom.singular_corner_indices[0]] - 1
    print(f"  Koch(1): {n_v} vertices | {len(panels)} panels")
    print(f"  ω = {geom.corner_angles[geom.singular_corner_indices[0]]/np.pi:.4f}π | "
          f"α = {alpha_koch:.4f}")

    qdata  = build_panel_quadrature(panels, p=CFG["p_gl"])
    Yq_T   = qdata.Yq.T        # (Nq, 2)
    wq     = qdata.wq
    Nq     = qdata.n_quad

    arc, vertex_arcs, total_arc = _boundary_arclength(qdata, CFG["n_per_edge"])
    sort_idx = np.argsort(arc)
    arc_s    = arc[sort_idx]

    # Min corner separation → cutoff radius
    sing_idx_arr = geom.singular_corner_indices
    sing_verts   = P[sing_idx_arr]
    corner_dists = [
        float(np.linalg.norm(sing_verts[i] - sing_verts[j]))
        for i in range(len(sing_verts))
        for j in range(i + 1, len(sing_verts))
    ]
    d_min = float(min(corner_dists))
    R_cut = 0.5 * d_min
    print(f"  d_min = {d_min:.6f}  |  cutoff R = {R_cut:.6f}")
    print(f"  Nq = {Nq} | arc = {total_arc:.4f}")

    # ------------------------------------------------------------------
    # 2. Singular enrichment with cutoff
    # ------------------------------------------------------------------
    enrichment = SingularEnrichment(
        geom=geom,
        use_cutoff=True,
        cutoff_radius=R_cut,
        per_corner_gamma=True,
    )
    n_sing = enrichment.n_singular

    # sigma_s per corner at quadrature nodes
    sigma_s_Yq = enrichment.precompute(Yq_T)   # (Nq, n_sing)  (n_sing=6)
    print(f"  n_sing = {n_sing} | sigma_s_Yq shape = {sigma_s_Yq.shape}")

    # ------------------------------------------------------------------
    # 3. Smooth density from g = x²−y² (no spikes)
    # ------------------------------------------------------------------
    print("\n--- Smooth reference density ---")
    nmat = assemble_nystrom_matrix(qdata)

    f_smooth  = Yq_T[:, 0] ** 2 - Yq_T[:, 1] ** 2
    bem_smooth = solve_bem(nmat, f_smooth,
                           tol=CFG["gmres_tol"], max_iter=CFG["gmres_maxiter"])
    sigma_smooth = bem_smooth.sigma
    print(f"  GMRES (smooth): flag={bem_smooth.flag} | rel_res={bem_smooth.rel_res:.3e}")
    print(f"  σ_smooth: max|σ|={np.max(np.abs(sigma_smooth)):.4f}  "
          f"rms={np.sqrt(np.mean(sigma_smooth**2)):.4f}")

    # ------------------------------------------------------------------
    # 4. Manufactured density σ_mfg = σ_smooth + S @ γ_true
    # ------------------------------------------------------------------
    print("\n--- Manufactured density ---")
    gamma_true = CFG["gamma_true"].copy()       # (6,)
    sigma_sing = sigma_s_Yq @ gamma_true        # (Nq,)  singular contribution
    sigma_mfg  = sigma_smooth + sigma_sing      # (Nq,)

    print(f"  γ_true = {list(np.round(gamma_true, 3))}")
    print(f"  ‖σ_smooth‖ = {np.linalg.norm(sigma_smooth):.4f}")
    print(f"  ‖σ_sing‖   = {np.linalg.norm(sigma_sing):.4f}")
    print(f"  ‖σ_mfg‖    = {np.linalg.norm(sigma_mfg):.4f}")
    print(f"  max|σ_mfg| = {np.max(np.abs(sigma_mfg)):.4f} "
          f"(max|σ_smooth| = {np.max(np.abs(sigma_smooth)):.4f})")

    energy_true = float(np.linalg.norm(sigma_sing) ** 2
                        / max(np.linalg.norm(sigma_mfg) ** 2, 1e-14))
    print(f"  Enrichment energy (‖σ_sing‖²/‖σ_mfg‖²) = {energy_true*100:.2f}%")

    # ------------------------------------------------------------------
    # 5. Manufactured boundary data g_mfg = V · σ_mfg
    # ------------------------------------------------------------------
    print("\n--- Manufactured boundary data ---")
    g_mfg = nmat.V @ sigma_mfg    # (Nq,) = (Nb,) since Xc=Yq

    # BEM recovery check
    bem_mfg = solve_bem(nmat, g_mfg,
                        tol=CFG["gmres_tol"], max_iter=CFG["gmres_maxiter"])
    sigma_bem = bem_mfg.sigma
    print(f"  GMRES (manufactured): flag={bem_mfg.flag} | rel_res={bem_mfg.rel_res:.3e}")
    sigma_rec_err = float(np.linalg.norm(sigma_bem - sigma_mfg)
                          / max(np.linalg.norm(sigma_mfg), 1e-14))
    print(f"  ‖σ_BEM − σ_mfg‖/‖σ_mfg‖ = {sigma_rec_err:.3e}  (≈ machine ε)")

    # ------------------------------------------------------------------
    # 6. Enrichment diagnostic: lstsq projection
    # ------------------------------------------------------------------
    print("\n--- Enrichment diagnostic ---")
    gamma_lstsq, _, _, _ = np.linalg.lstsq(sigma_s_Yq, sigma_bem, rcond=None)
    sigma_proj  = sigma_s_Yq @ gamma_lstsq
    res_norm    = np.linalg.norm(sigma_bem - sigma_proj)
    bem_norm    = np.linalg.norm(sigma_bem)
    energy_lstsq = 1.0 - (res_norm / max(bem_norm, 1e-14)) ** 2

    print(f"  γ*_lstsq = {list(np.round(gamma_lstsq, 5))}")
    print(f"  γ_true   = {list(np.round(gamma_true,  5))}")
    print(f"  ‖γ*−γ_true‖/‖γ_true‖ = "
          f"{np.linalg.norm(gamma_lstsq - gamma_true)/max(np.linalg.norm(gamma_true),1e-14):.4f}")
    print(f"  Enrichment energy (lstsq) = {energy_lstsq*100:.2f}%")

    # ------------------------------------------------------------------
    # 7. Exact interior solution from σ_mfg
    # ------------------------------------------------------------------
    u_exact = make_u_exact_fn(Yq_T, wq, sigma_mfg)

    # Self-check: reconstruct from σ_mfg itself
    mfg_out = reconstruct_interior(
        P=P, Yq=Yq_T, wq=wq, sigma=sigma_mfg,
        n_grid=CFG["n_grid_final"], u_exact=u_exact,
    )
    print(f"\n  u_mfg self-check: rel_L2={mfg_out.rel_L2:.3e}  (should ≈ 0)")

    # ------------------------------------------------------------------
    # 8. Operator state (m_col_base=16 → Xc=Yq)
    # ------------------------------------------------------------------
    print("\n--- Operator setup ---")
    colloc = build_collocation_points(panels, m_col_panel=CFG["m_col_base"])
    w_panel = panel_loss_weights(panels, w_base=CFG["w_base"],
                                 w_corner=CFG["w_corner"], w_ring=CFG["w_ring"])

    # Pass g_smooth as dummy; override op.f after getting eq_scale
    g_dummy = lambda xy: xy[:, 0] ** 2 - xy[:, 1] ** 2
    op, op_diag = build_operator_state(
        colloc=colloc, qdata=qdata, enrichment=enrichment, g=g_dummy,
        panel_weights=w_panel,
        eq_scale_mode  = CFG["eq_scale_mode"],
        eq_scale_fixed = CFG["eq_scale_fixed"],
        dtype=torch.float64, device="cpu",
    )

    # Override op.f with manufactured boundary data (same nodes: Xc=Yq)
    eq_scale = op_diag["eq_scale"]
    op.f = torch.tensor(eq_scale * g_mfg, dtype=torch.float64)

    print(f"  Nb={colloc.n_colloc} | Nq={Nq} | eq_scale={eq_scale:.2e}")
    print(f"  mean|A| = {op_diag['mean_abs_A_before']:.3e}")
    print(f"  op.f overridden with eq_scale×g_mfg at Xc=Yq")

    # ------------------------------------------------------------------
    # 9. Shared initial model state
    # ------------------------------------------------------------------
    torch.manual_seed(CFG["seed"])
    init_model = SEBINNModel(
        hidden_width = CFG["hidden_width"],
        n_hidden     = CFG["n_hidden"],
        n_gamma      = n_sing,
        gamma_init   = CFG["gamma_init"],
        dtype        = torch.float64,
    )
    init_state = copy.deepcopy(init_model.state_dict())
    print(f"  n_params={init_model.n_params()} | n_gamma={n_sing}")
    print(f"  Shared initial state saved.")

    shared = dict(
        op=op, Yq_T=Yq_T, wq=wq, P=P,
        sigma_bem=sigma_bem, sigma_s_Yq=sigma_s_Yq,
        sort_idx=sort_idx, u_exact=u_exact,
    )

    # ------------------------------------------------------------------
    # 10. Training: Cases A, B, C
    # ------------------------------------------------------------------
    res_a = _train_one("BINN",     freeze_gamma=True,  gamma_override=None,
                       init_state=init_state, shared=shared)
    res_b = _train_one("SE-BINN",  freeze_gamma=False, gamma_override=None,
                       init_state=init_state, shared=shared)
    res_c = _train_one("SE-BINN★", freeze_gamma=False, gamma_override=gamma_lstsq,
                       init_state=init_state, shared=shared)

    cases = [res_a, res_b, res_c]

    # ------------------------------------------------------------------
    # 11. Summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("  RESULTS — Koch(1) manufactured singular density")
    print("=" * 80)
    print(f"  γ_true   = {list(np.round(gamma_true, 3))}")
    print(f"  γ*_lstsq = {list(np.round(gamma_lstsq, 4))}")
    print(f"  Enrichment energy (lstsq) = {energy_lstsq*100:.2f}%")
    print()

    col_labels = [c["label"] for c in cases]
    W  = 20
    hdr = f"  {'Metric':<28} | " + " | ".join(f"{l:>{W}}" for l in col_labels)
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr); print(sep)

    def row(name, key, fmt=".3e"):
        vals = []
        for c in cases:
            v = c[key]
            vals.append(f"{v:{fmt}}" if isinstance(v, float) else str(v))
        print(f"  {name:<28} | " + " | ".join(f"{v:>{W}}" for v in vals))

    row("Interior rel L2",   "final_rel_L2")
    row("Interior L∞",       "final_linf")
    row("Density rel-diff",   "density_rel_diff")
    row("γ error ‖γ−γ_true‖/‖γ_true‖", "gamma_err")
    print(sep)

    gvs = []
    for c in cases:
        if c["freeze_gamma"]:
            gvs.append("(frozen=0)")
        else:
            garr = c["gamma_final"]
            gvs.append("[" + ",".join(f"{v:+.3f}" for v in garr) + "]")
    print(f"  {'γ final':<28} | " + " | ".join(f"{v:>{W}}" for v in gvs))

    rvs = [c["lbfgs_reason"] for c in cases]
    print(f"  {'LBFGS reason':<28} | " + " | ".join(f"{v:>{W}}" for v in rvs))
    wvs = [f"{c['wall_time']:.1f}s" for c in cases]
    print(f"  {'Wall time':<28} | " + " | ".join(f"{v:>{W}}" for v in wvs))
    print(sep)

    de  = [c["density_rel_diff"] for c in cases]
    iL2 = [c["final_rel_L2"]     for c in cases]
    if de[0] > 0 and de[1] > 0:
        ab = de[0] / de[1]
        print(f"\n  A/B density improvement (BINN→SE-BINN)  : {ab:.2f}×")
    if de[0] > 0 and de[2] > 0:
        ac = de[0] / de[2]
        print(f"  A/C density improvement (BINN→SE-BINN★) : {ac:.2f}×")
    if iL2[0] > 0 and iL2[1] > 0:
        print(f"  A/B interior improvement                 : {iL2[0]/iL2[1]:.2f}×")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 12. Hypothesis check
    # ------------------------------------------------------------------
    print("\n--- Hypothesis check ---")
    print(f"  H1. Enrichment energy >> 2.8%: {energy_lstsq*100:.2f}%  "
          f"{'✓' if energy_lstsq > 0.1 else '— (low, check cutoff)'}")
    ab_ratio = de[0] / de[1] if de[1] > 0 else float("nan")
    print(f"  H2. BINN plateau: d_err_A={de[0]:.4f}  "
          f"{'✓ (>10%)' if de[0] > 0.1 else f'({de[0]*100:.1f}%)'}")
    print(f"  H3. SE-BINN << 5% density error: d_err_B={de[1]:.4f}  "
          f"{'✓' if de[1] < 0.05 else '✗'}")
    print(f"  H4. A/B gap >> 1.12× (smooth Koch gap): {ab_ratio:.2f}×  "
          f"{'✓' if ab_ratio > 2 else '✗'}")
    g_err_b = res_b["gamma_err"]
    print(f"  H5. γ_c convergence ‖γ−γ_true‖/‖γ_true‖: SE-BINN={g_err_b:.4f}  "
          f"{'✓ (<20%)' if g_err_b < 0.2 else '✗'}")

    total_wall = time.perf_counter() - t_global
    print(f"\n  Total wall time: {total_wall:.1f}s")

    # ------------------------------------------------------------------
    # 13. Figures
    # ------------------------------------------------------------------
    print("\n--- Generating figures ---")
    sing_idx_list = list(sing_idx_arr)

    _fig_manufactured_density(
        arc_s          = arc_s,
        sigma_mfg_s    = sigma_mfg[sort_idx],
        sigma_smooth_s = sigma_smooth[sort_idx],
        vertex_arcs    = vertex_arcs,
        sing_idx       = sing_idx_list,
        gamma_true     = gamma_true,
        outpath        = os.path.join(figures_dir, "manufactured_density.png"),
    )
    _fig_ab(
        cases           = cases,
        sigma_bem_sorted = sigma_bem[sort_idx],
        arc_s           = arc_s,
        vertex_arcs     = vertex_arcs,
        sing_idx        = sing_idx_list,
        outpath         = os.path.join(figures_dir, "manufactured_ab.png"),
    )
    _fig_gamma_traj(
        cases      = cases,
        gamma_true = gamma_true,
        outpath    = os.path.join(figures_dir, "manufactured_gamma.png"),
    )
    _fig_corner_zoom(
        cases           = cases,
        sigma_bem_sorted = sigma_bem[sort_idx],
        arc_s           = arc_s,
        vertex_arcs     = vertex_arcs,
        sing_idx        = sing_idx_list,
        outpath         = os.path.join(figures_dir, "manufactured_corner_zoom.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
