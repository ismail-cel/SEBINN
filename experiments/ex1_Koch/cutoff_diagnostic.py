"""
Cutoff diagnostic experiment: Koch(1), g = x² − y².

The SingularEnrichment class has use_cutoff=False by default.  Without a
cutoff, sigma_s^(c)(y) = -(π/ω) r_c^{-1/4} is evaluated at EVERY boundary
point for ALL 6 corners, including points far from the corner where the
contribution is a smooth, slowly-decaying tail of magnitude ~3.5.

This creates three problems:
  1. σ_w must counteract the smooth bias everywhere (wasted capacity).
  2. lstsq projection fits the tails, not the spikes (enrichment energy
     appears small).
  3. The 6 γ_c values couple through the overlapping smooth tails.

This experiment:
  Part 1  — Geometry reference quantities.
  Part 2  — Cutoff sweep: enrichment energy, condition number, background
             magnitude for R ∈ {0.3,0.5,0.7,1.0} × d_min  and R=inf.
  Part 3  — Visualisation of σ_s with/without cutoff vs arclength.
  Part 4  — Full A/B/C training comparison (no cutoff vs best cutoff).

Figures saved to experiments/ex1_Koch/figures/:
  cutoff_diagnostic.png
  cutoff_enrichment_energy.png
  cutoff_ab_comparison.png
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
from src.reconstruction.interior import reconstruct_interior


# ===========================================================================
# Configuration
# ===========================================================================

CFG = dict(
    seed           = 0,
    n_per_edge     = 12,
    p_gl           = 16,
    m_col_base     = 4,
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
    # Training schedule (heavier than run.py to give fair L-BFGS comparison)
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
    n_grid           = 201,
)


def u_exact(xy: np.ndarray) -> np.ndarray:
    return xy[:, 0] ** 2 - xy[:, 1] ** 2


# ===========================================================================
# Arc-length helper
# ===========================================================================

def _boundary_arclength(qdata, n_per_edge):
    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc = panel_start[qdata.pan_id] + qdata.s_on_panel
    Npan         = qdata.n_panels
    total_length = float(qdata.L_panel.sum())
    Nv           = Npan // n_per_edge
    v_panel_idx  = np.arange(Nv) * n_per_edge
    panel_start_full = np.concatenate([[0.0], np.cumsum(qdata.L_panel)])
    vertex_arcs  = np.append(panel_start_full[v_panel_idx], total_length)
    return arc, vertex_arcs, total_length


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
# Part 2 helper: enrichment diagnostics for a given enrichment object
# ===========================================================================

def _enrichment_diagnostics(
    enrichment: SingularEnrichment,
    Yq_T: np.ndarray,
    sigma_bem: np.ndarray,
    arc: np.ndarray,
    total_arc: float,
) -> dict:
    """
    Compute all enrichment diagnostic quantities for a given enrichment.

    Returns a dict with keys:
      S_per_corner   : (Nq, n_sing)
      s_shared       : (Nq,)
      gamma_pc       : (n_sing,)   lstsq per-corner coefficients
      gamma_shared   : float
      energy_pc      : float       enrichment energy (per-corner)
      energy_shared  : float       enrichment energy (shared)
      cond_STS       : float       condition number of S^T S
      S_rms          : float       rms of S_per_corner
      bg_magnitude   : float       mean|σ_s| at midpoints far from corners
    """
    n_sing = enrichment.n_singular

    # --- per-corner and shared σ_s fields ---
    S = enrichment.evaluate_per_corner(Yq_T)    # (Nq, n_sing)
    s = enrichment.evaluate(Yq_T)               # (Nq,)

    # --- per-corner lstsq ---
    gamma_pc, _, _, _ = np.linalg.lstsq(S, sigma_bem, rcond=None)
    sigma_proj_pc     = S @ gamma_pc
    res_norm_pc       = np.linalg.norm(sigma_bem - sigma_proj_pc)
    bem_norm          = np.linalg.norm(sigma_bem)
    energy_pc         = 1.0 - (res_norm_pc / max(bem_norm, 1e-14)) ** 2

    # --- shared lstsq ---
    ss = float(s @ s)
    gamma_shared   = float(s @ sigma_bem) / max(ss, 1e-14)
    sigma_proj_sh  = gamma_shared * s
    res_norm_sh    = np.linalg.norm(sigma_bem - sigma_proj_sh)
    energy_shared  = 1.0 - (res_norm_sh / max(bem_norm, 1e-14)) ** 2

    # --- condition number of S^T S ---
    STS = S.T @ S
    cond_STS = float(np.linalg.cond(STS))

    # --- rms of S ---
    S_rms = float(np.sqrt(np.mean(S ** 2)))

    # --- smooth background: mean|σ_s| at arc midpoints (far from all corners) ---
    # Use quadrature nodes near arc/4, arc/2, 3arc/4 as "far from corners"
    bg_arcs = np.array([0.25, 0.50, 0.75]) * total_arc
    bg_mag_vals = []
    for s_target in bg_arcs:
        idx_near = np.argmin(np.abs(arc - s_target))
        bg_mag_vals.append(abs(s[idx_near]))
    bg_magnitude = float(np.mean(bg_mag_vals))

    return dict(
        S_per_corner  = S,
        s_shared      = s,
        gamma_pc      = gamma_pc,
        gamma_shared  = gamma_shared,
        energy_pc     = energy_pc,
        energy_shared = energy_shared,
        cond_STS      = cond_STS,
        S_rms         = S_rms,
        bg_magnitude  = bg_magnitude,
        sigma_proj_pc = sigma_proj_pc,
    )


# ===========================================================================
# Part 4 helper: one full training run
# ===========================================================================

def _train_one(
    label: str,
    freeze_gamma: bool,
    init_state: dict,
    shared: dict,
    enrichment: SingularEnrichment,
    verbose: bool = True,
) -> dict:
    """
    One full training run (3 Adam phases + L-BFGS).

    The enrichment determines both the operator state (via sigma_s tensors)
    and the model n_gamma.  For Case A (BINN) we still pass the enrichment
    to build the operator but freeze gamma at zero.
    """
    if verbose:
        fz = " [γ frozen=0]" if freeze_gamma else " [γ trainable]"
        print(f"\n{'='*62}")
        print(f"  Case {label}{fz}")
        print(f"{'='*62}")

    t0 = time.perf_counter()

    op        = shared["op_map"][label]   # pre-built per case
    Yq_T      = shared["Yq_T"]
    wq        = shared["wq"]
    P         = shared["P"]
    sigma_bem = shared["sigma_bem"]
    sort_idx  = shared["sort_idx"]

    # The sigma_s values for density evaluation at inference time
    sigma_s_Yq = enrichment.precompute(Yq_T)       # (Nq,) or (Nq, n_sing)
    n_gamma    = enrichment.n_gamma

    model = SEBINNModel(
        hidden_width = CFG["hidden_width"],
        n_hidden     = CFG["n_hidden"],
        n_gamma      = n_gamma,
        gamma_init   = CFG["gamma_init"],
        dtype        = torch.float64,
    )
    model.load_state_dict(copy.deepcopy(init_state))

    if freeze_gamma:
        model.gamma_module.gamma.requires_grad_(False)
        if verbose:
            print(f"  γ frozen at 0")

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  trainable params: {n_tr} | n_gamma={n_gamma}")

    Yq_t      = torch.tensor(Yq_T,      dtype=torch.float64)
    sigma_s_t = torch.tensor(sigma_s_Yq, dtype=torch.float64)

    stage_checkpoints = []

    def _record_checkpoint(stage, n_iter, loss):
        with torch.no_grad():
            s = model(Yq_t, sigma_s_t).numpy()
        d_err = float(np.linalg.norm(s - sigma_bem)
                      / max(np.linalg.norm(sigma_bem), 1e-14))
        i_out = reconstruct_interior(
            P=P, Yq=Yq_T, wq=wq, sigma=s,
            n_grid=51, u_exact=u_exact,
        )
        g = model.gamma_value()
        stage_checkpoints.append(dict(
            stage=stage, iter=n_iter, loss=loss,
            gamma=g if isinstance(g, list) else float(g),
            density_rel_diff=d_err, interior_L2=i_out.rel_L2,
        ))
        if verbose:
            gfmt = (f"[{','.join(f'{v:.4f}' for v in g)}]"
                    if isinstance(g, list) else f"{float(g):.5f}")
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
        n_grid=CFG["n_grid"], u_exact=u_exact,
    )
    density_rel_diff = float(
        np.linalg.norm(sigma_final - sigma_bem)
        / max(np.linalg.norm(sigma_bem), 1e-14)
    )
    t_total = time.perf_counter() - t0

    if verbose:
        g = model.gamma_value()
        gfmt = (f"[{','.join(f'{v:.4f}' for v in g)}]"
                if isinstance(g, list) else f"{float(g):.6f}")
        print(f"\n  {label} final:")
        print(f"    Interior rel L2  : {final_out.rel_L2:.3e}")
        print(f"    Interior L∞      : {final_out.linf:.3e}")
        print(f"    Density rel-diff : {density_rel_diff:.4f}")
        print(f"    γ final          : {gfmt}")
        print(f"    LBFGS reason     : {lbfgs_res.reason}")
        print(f"    Wall time        : {t_total:.1f}s")

    return dict(
        label            = label,
        freeze_gamma     = freeze_gamma,
        final_rel_L2     = final_out.rel_L2,
        final_linf       = final_out.linf,
        density_rel_diff = density_rel_diff,
        gamma_vals       = model.gamma_value(),
        lbfgs_reason     = lbfgs_res.reason,
        wall_time        = t_total,
        loss_hist_adam   = all_adam_loss,
        loss_hist_lbfgs  = list(lbfgs_res.loss_hist),
        adam_n_iters     = global_it,
        sigma_final      = sigma_final[sort_idx],
        stage_checkpoints = stage_checkpoints,
        final_out        = final_out,
    )


# ===========================================================================
# Figures
# ===========================================================================

def _fig_diagnostic(
    arc, arc_sorted, sort_idx, sigma_bem,
    diag_noc, diag_cut, best_R, d_min, vertex_arcs, sing_idx, outpath
):
    """Part 3: σ_s with/without cutoff vs arclength, residual comparison."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 16))
    fig.subplots_adjust(hspace=0.45)

    s_noc = diag_noc["s_shared"][sort_idx]
    s_cut = diag_cut["s_shared"][sort_idx]
    sb    = sigma_bem[sort_idx]

    def _vlines(ax):
        for ci in sing_idx:
            ax.axvline(vertex_arcs[ci], color="#aaaaaa", lw=0.7, ls="--", alpha=0.7)

    # (a) σ_s without cutoff
    ax = axes[0]
    _vlines(ax)
    ax.plot(arc_sorted, s_noc, color="#1f77b4", lw=1.2)
    ax.set_xlabel("Arc-length $s$"); ax.set_ylabel(r"$\sigma_s$")
    ax.set_title(r"(a) $\sigma_s$ — no cutoff  (smooth background ≈ −3.5 everywhere)",
                 fontsize=11)
    ax.grid(True, lw=0.3, alpha=0.5)

    # (b) σ_s with best cutoff
    ax = axes[1]
    _vlines(ax)
    ax.plot(arc_sorted, s_cut, color="#d62728", lw=1.2,
            label=f"cutoff R={best_R:.3f}")
    ax.set_xlabel("Arc-length $s$"); ax.set_ylabel(r"$\sigma_s$")
    ax.set_title(r"(b) $\sigma_s$ — with cutoff  (isolated spikes, zero elsewhere)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)

    # (c) σ_BEM vs γ*·σ_s (both cases)
    ax = axes[2]
    _vlines(ax)
    sigma_proj_noc = diag_noc["sigma_proj_pc"][sort_idx]
    sigma_proj_cut = diag_cut["sigma_proj_pc"][sort_idx]
    ax.plot(arc_sorted, sb,              color="black",   lw=1.4, label=r"$\sigma_\mathrm{BEM}$")
    ax.plot(arc_sorted, sigma_proj_noc,  color="#1f77b4", lw=1.1, ls="--",
            label=r"$\gamma^*\sigma_s$ (no cutoff)")
    ax.plot(arc_sorted, sigma_proj_cut,  color="#d62728", lw=1.1, ls="-.",
            label=r"$\gamma^*\sigma_s$ (cutoff)")
    ax.set_xlabel("Arc-length $s$"); ax.set_ylabel(r"$\sigma$")
    ax.set_title(r"(c) $\sigma_\mathrm{BEM}$ vs lstsq projection $\gamma^*\sigma_s$", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)

    # (d) residual |σ_BEM - γ*σ_s|
    ax = axes[3]
    _vlines(ax)
    res_noc = np.abs(sb - sigma_proj_noc)
    res_cut = np.abs(sb - sigma_proj_cut)
    ax.semilogy(arc_sorted, res_noc + 1e-16, color="#1f77b4", lw=1.1, ls="--",
                label=f"no cutoff  (energy={diag_noc['energy_pc']*100:.1f}%)")
    ax.semilogy(arc_sorted, res_cut + 1e-16, color="#d62728", lw=1.1, ls="-.",
                label=f"cutoff  (energy={diag_cut['energy_pc']*100:.1f}%)")
    ax.set_xlabel("Arc-length $s$")
    ax.set_ylabel(r"$|\sigma_\mathrm{BEM} - \gamma^*\sigma_s|$")
    ax.set_title("(d) Projection residual", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(
        r"Koch(1) — $\sigma_s$ cutoff diagnostic: no-cutoff vs R=" + f"{best_R:.3f}"
        + f"\n$d_\\min={d_min:.4f}$",
        fontsize=12, y=1.01,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved cutoff_diagnostic → {outpath}")


def _fig_enrichment_energy(sweep_data, d_min, outpath):
    """Part 5b: enrichment energy vs R/d_min."""
    labels   = [d["label"]      for d in sweep_data]
    x_vals   = [d["R_over_dmin"] for d in sweep_data]
    e_pc     = [d["energy_pc"]  * 100 for d in sweep_data]
    e_sh     = [d["energy_shared"] * 100 for d in sweep_data]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.36)

    # Build x positions: finite ones as float, inf separately
    x_finite   = [x for x in x_vals if np.isfinite(x)]
    e_pc_fin   = [e for x, e in zip(x_vals, e_pc)   if np.isfinite(x)]
    e_sh_fin   = [e for x, e in zip(x_vals, e_sh)   if np.isfinite(x)]
    e_pc_inf   = e_pc[-1]
    e_sh_inf   = e_sh[-1]

    ax = axes[0]
    ax.plot(x_finite, e_pc_fin, "o-", color="#d62728", lw=1.6, ms=8,
            label="per-corner (6 γ)")
    ax.plot(x_finite, e_sh_fin, "s--", color="#1f77b4", lw=1.4, ms=7,
            label="shared (1 γ)")
    ax.axhline(e_pc_inf, color="#d62728", lw=1.0, ls=":", alpha=0.7,
               label=f"no cutoff (per-corner) = {e_pc_inf:.1f}%")
    ax.axhline(e_sh_inf, color="#1f77b4", lw=1.0, ls=":", alpha=0.7,
               label=f"no cutoff (shared) = {e_sh_inf:.1f}%")
    ax.set_xlabel(r"$R / d_{\min}$", fontsize=12)
    ax.set_ylabel("Enrichment energy (%)", fontsize=11)
    ax.set_title("(a) Enrichment energy vs cutoff radius", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)

    ax = axes[1]
    # Condition number
    conds    = [d["cond_STS"]   for d in sweep_data]
    c_fin    = [c for x, c in zip(x_vals, conds)   if np.isfinite(x)]
    c_inf    = conds[-1]

    ax.semilogy(x_finite, c_fin, "^-", color="#2ca02c", lw=1.6, ms=8,
                label="cond(S'S) with cutoff")
    ax.axhline(c_inf, color="#2ca02c", lw=1.0, ls=":", alpha=0.7,
               label=f"no cutoff = {c_inf:.2e}")
    ax.set_xlabel(r"$R / d_{\min}$", fontsize=12)
    ax.set_ylabel(r"cond($S^T S$)", fontsize=11)
    ax.set_title(r"(b) Condition number of $S^T S$ vs cutoff radius", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(
        r"Koch(1): enrichment energy and $S^T S$ conditioning vs cutoff radius",
        fontsize=12,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved cutoff_enrichment_energy → {outpath}")


def _fig_ab_comparison(
    cases, sigma_bem_sorted, arc_sorted, vertex_arcs, sing_idx, outpath
):
    """Part 5c: loss / density / density error for A, B, C."""
    COLORS = {
        "BINN":              "#1f77b4",
        "SE-BINN (no cutoff)": "#d62728",
        "SE-BINN (cutoff)":  "#2ca02c",
    }
    LS = {
        "BINN":              "-",
        "SE-BINN (no cutoff)": "--",
        "SE-BINN (cutoff)":  "-.",
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.subplots_adjust(hspace=0.42)

    ax = axes[0]
    for c in cases:
        col  = COLORS.get(c["label"], "gray")
        ls   = LS.get(c["label"], "-")
        ha, hl = c["loss_hist_adam"], c["loss_hist_lbfgs"]
        n_a, n_l = len(ha), len(hl)
        ax.semilogy(np.arange(1, n_a+1), ha, color=col, lw=1.4, ls="-",
                    alpha=0.85, label=f"{c['label']} (Adam)")
        if n_l:
            ax.semilogy(np.arange(n_a+1, n_a+n_l+1), hl, color=col, lw=1.4,
                        ls="--", alpha=0.85, label=f"{c['label']} (L-BFGS)")
    n_adam = cases[0]["adam_n_iters"]
    ax.axvline(n_adam, color="gray", lw=0.9, ls=":", alpha=0.6,
               label=f"Adam→L-BFGS ({n_adam})")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("(a) Loss history", fontsize=12)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    ax = axes[1]
    for ci in sing_idx:
        ax.axvline(vertex_arcs[ci], color="#aaaaaa", lw=0.7, ls="--", alpha=0.7)
    ax.plot(arc_sorted, sigma_bem_sorted, color="black", lw=1.3, alpha=0.9,
            label=r"$\sigma_\mathrm{BEM}$")
    for c in cases:
        col = COLORS.get(c["label"], "gray")
        ls  = LS.get(c["label"], "-")
        ax.plot(arc_sorted, c["sigma_final"], color=col, lw=1.1, ls=ls,
                alpha=0.85, label=c["label"])
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$\sigma(s)$", fontsize=11)
    ax.set_title(r"(b) Boundary density", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)

    ax = axes[2]
    for ci in sing_idx:
        ax.axvline(vertex_arcs[ci], color="#aaaaaa", lw=0.7, ls="--", alpha=0.7)
    for c in cases:
        col = COLORS.get(c["label"], "gray")
        ls  = LS.get(c["label"], "-")
        err = np.abs(c["sigma_final"] - sigma_bem_sorted)
        ax.semilogy(arc_sorted, err + 1e-16, color=col, lw=1.1, ls=ls,
                    alpha=0.85,
                    label=f"{c['label']}  (d_err={c['density_rel_diff']:.4f})")
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$|\sigma_\theta - \sigma_\mathrm{BEM}|$", fontsize=11)
    ax.set_title("(c) Pointwise density error", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(
        "A/B/C Comparison — Koch(1), g=x²−y²\n"
        "BINN vs SE-BINN (no cutoff) vs SE-BINN (cutoff)",
        fontsize=12, y=1.01,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved cutoff_ab_comparison → {outpath}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("=" * 72)
    print("  Cutoff diagnostic: Koch(1), g = x² − y²")
    print("=" * 72)

    t_global = time.perf_counter()
    figures_dir = os.path.join(_HERE, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Geometry and quadrature
    # ------------------------------------------------------------------
    print("\n--- Part 1: Geometry ---")
    geom   = make_koch_geometry(n=1)
    P      = geom.vertices        # (12, 2)
    n_v    = len(P)

    panels = build_uniform_panels(P, n_per_edge=CFG["n_per_edge"])
    label_corner_ring_panels(panels, P)

    qdata  = build_panel_quadrature(panels, p=CFG["p_gl"])
    Yq_T   = qdata.Yq.T
    wq     = qdata.wq
    Nq     = qdata.n_quad

    arc, vertex_arcs, total_arc = _boundary_arclength(qdata, CFG["n_per_edge"])
    sort_idx = np.argsort(arc)
    arc_sorted = arc[sort_idx]

    # Geometry reference quantities
    edge_lengths = np.array([
        float(np.linalg.norm(P[(i+1) % n_v] - P[i]))
        for i in range(n_v)
    ])
    mean_edge = float(edge_lengths.mean())
    min_edge  = float(edge_lengths.min())

    sing_idx = geom.singular_corner_indices          # e.g. [1,3,5,7,9,11]
    sing_verts = P[sing_idx]                         # (6, 2)
    # Minimum distance between any two distinct reentrant corners
    corner_dists = []
    for i in range(len(sing_verts)):
        for j in range(i+1, len(sing_verts)):
            corner_dists.append(float(np.linalg.norm(sing_verts[i] - sing_verts[j])))
    d_min = float(min(corner_dists))

    print(f"  Koch(1): {n_v} vertices | {len(panels)} panels | {Nq} quad nodes")
    print(f"  mean_edge_length   = {mean_edge:.6f}")
    print(f"  min_edge_length    = {min_edge:.6f}")
    print(f"  min_corner_separation (d_min) = {d_min:.6f}")
    print(f"  singular corners: indices {list(sing_idx)}")
    print(f"  Constraint: R < d_min/2 = {d_min/2:.6f}")
    print(f"  Natural choice: R = 0.5 × d_min = {0.5*d_min:.6f}")

    # ------------------------------------------------------------------
    # 2. BEM reference (shared across all sweep points)
    # ------------------------------------------------------------------
    print("\n--- BEM reference ---")
    nmat     = assemble_nystrom_matrix(qdata)
    f_bnd    = u_exact(Yq_T)
    bem_sol  = solve_bem(nmat, f_bnd,
                         tol=CFG["gmres_tol"], max_iter=CFG["gmres_maxiter"])
    sigma_bem = bem_sol.sigma
    print(f"  GMRES: flag={bem_sol.flag} | rel_res={bem_sol.rel_res:.3e}")

    # ------------------------------------------------------------------
    # 3. Part 2: Cutoff sweep
    # ------------------------------------------------------------------
    print("\n--- Part 2: Cutoff sweep ---")
    R_factors   = [0.3, 0.5, 0.7, 1.0]
    R_values    = [f * d_min for f in R_factors]

    sweep_data  = []

    for R_fac, R in zip(R_factors, R_values):
        enrich = SingularEnrichment(
            geom=geom, use_cutoff=True, cutoff_radius=R,
            per_corner_gamma=True,
        )
        diag = _enrichment_diagnostics(enrich, Yq_T, sigma_bem, arc, total_arc)
        row = dict(
            label        = f"R={R_fac:.1f}×d_min",
            R_fac        = R_fac,
            R            = R,
            R_over_dmin  = R_fac,
            **diag,
        )
        sweep_data.append(row)
        print(f"  R={R_fac:.1f}×d_min ({R:.4f}): "
              f"E_pc={diag['energy_pc']*100:.2f}% | "
              f"E_sh={diag['energy_shared']*100:.2f}% | "
              f"cond={diag['cond_STS']:.2e} | "
              f"bg={diag['bg_magnitude']:.4f}")

    # No-cutoff case (R = inf)
    enrich_noc = SingularEnrichment(
        geom=geom, use_cutoff=False, per_corner_gamma=True,
    )
    diag_noc = _enrichment_diagnostics(enrich_noc, Yq_T, sigma_bem, arc, total_arc)
    sweep_data.append(dict(
        label       = "R=inf (no cutoff)",
        R_fac       = float("inf"),
        R           = float("inf"),
        R_over_dmin = float("inf"),
        **diag_noc,
    ))
    print(f"  R=inf (no cutoff): "
          f"E_pc={diag_noc['energy_pc']*100:.2f}% | "
          f"E_sh={diag_noc['energy_shared']*100:.2f}% | "
          f"cond={diag_noc['cond_STS']:.2e} | "
          f"bg={diag_noc['bg_magnitude']:.4f}")

    # Print full sweep table
    print()
    w = 100
    print("=" * w)
    print("  CUTOFF SWEEP TABLE  —  Koch(1), g=x²−y², per-corner γ (6 corners)")
    print("=" * w)
    hdr = ("  " + f"{'R/d_min':>10}" + " | " + f"{'E_per_corner':>14}" + " | "
           + f"{'E_shared':>10}" + " | " + f"{'bg_mag':>8}" + " | "
           + f"{'S_rms':>8}" + " | " + f"{'cond(STS)':>12}" + " | gamma*_c (6 values)")
    print(hdr)
    print("  " + "-" * (w - 2))
    for d in sweep_data:
        gamma_str = (
            "[" + ",".join(f"{v:+.3f}" for v in d["gamma_pc"]) + "]"
        )
        R_str = f"{d['R_fac']:.1f}" if np.isfinite(d["R_fac"]) else "inf"
        print(f"  {R_str:>10} | {d['energy_pc']*100:>13.2f}% | "
              f"{d['energy_shared']*100:>9.2f}% | "
              f"{d['bg_magnitude']:>8.4f} | {d['S_rms']:>8.4f} | "
              f"{d['cond_STS']:>12.2e} | {gamma_str}")
    print("=" * w)

    # ------------------------------------------------------------------
    # 4. Select best R (highest per-corner enrichment energy, finite)
    # ------------------------------------------------------------------
    finite_sweep = [d for d in sweep_data if np.isfinite(d["R_fac"])]
    best = max(finite_sweep, key=lambda d: d["energy_pc"])
    best_R = best["R"]
    print(f"\n  Best R = {best['R_fac']:.1f}×d_min = {best_R:.6f}  "
          f"(E_pc={best['energy_pc']*100:.2f}%)")

    # ------------------------------------------------------------------
    # 5. Part 3: Figures — diagnostic
    # ------------------------------------------------------------------
    print("\n--- Part 3: Diagnostic figure ---")
    enrich_best = SingularEnrichment(
        geom=geom, use_cutoff=True, cutoff_radius=best_R, per_corner_gamma=True,
    )
    diag_best = _enrichment_diagnostics(
        enrich_best, Yq_T, sigma_bem, arc, total_arc
    )

    _fig_diagnostic(
        arc=arc, arc_sorted=arc_sorted, sort_idx=sort_idx,
        sigma_bem=sigma_bem,
        diag_noc=diag_noc, diag_cut=diag_best,
        best_R=best_R, d_min=d_min,
        vertex_arcs=vertex_arcs, sing_idx=list(sing_idx),
        outpath=os.path.join(figures_dir, "cutoff_diagnostic.png"),
    )
    _fig_enrichment_energy(
        sweep_data=sweep_data, d_min=d_min,
        outpath=os.path.join(figures_dir, "cutoff_enrichment_energy.png"),
    )

    # ------------------------------------------------------------------
    # 6. Part 4: Training comparison A / B / C
    # ------------------------------------------------------------------
    print("\n--- Part 4: Training comparison A/B/C ---")

    # Build operator states for each case
    # Case A: BINN — enrichment doesn't matter (gamma frozen), use no-cutoff
    # Case B: SE-BINN no cutoff
    # Case C: SE-BINN with best cutoff
    # All share the same quadrature / boundary geometry.

    w_panel = panel_loss_weights(panels, w_base=CFG["w_base"],
                                 w_corner=CFG["w_corner"], w_ring=CFG["w_ring"])
    colloc  = build_collocation_points(panels, m_col_panel=CFG["m_col_base"])

    print("  Building operator states ...")

    # Case A / B use no-cutoff enrichment
    op_noc, op_diag_noc = build_operator_state(
        colloc=colloc, qdata=qdata, enrichment=enrich_noc, g=u_exact,
        panel_weights=w_panel,
        eq_scale_mode=CFG["eq_scale_mode"], eq_scale_fixed=CFG["eq_scale_fixed"],
        dtype=torch.float64, device="cpu",
    )
    # Case C uses best-cutoff enrichment
    op_cut, op_diag_cut = build_operator_state(
        colloc=colloc, qdata=qdata, enrichment=enrich_best, g=u_exact,
        panel_weights=w_panel,
        eq_scale_mode=CFG["eq_scale_mode"], eq_scale_fixed=CFG["eq_scale_fixed"],
        dtype=torch.float64, device="cpu",
    )
    print(f"  Nb={colloc.n_colloc} | eq_scale={op_diag_noc['eq_scale']:.2e}")

    # Shared initial model state (n_gamma=6 for all trainable cases)
    torch.manual_seed(CFG["seed"])
    n_gamma = enrich_noc.n_gamma   # = 6

    init_model = SEBINNModel(
        hidden_width = CFG["hidden_width"],
        n_hidden     = CFG["n_hidden"],
        n_gamma      = n_gamma,
        gamma_init   = CFG["gamma_init"],
        dtype        = torch.float64,
    )
    init_state = copy.deepcopy(init_model.state_dict())
    print(f"  n_params={init_model.n_params()} | n_gamma={n_gamma}")
    print(f"  Shared initial state saved.")

    shared = dict(
        op_map   = {
            "BINN":                op_noc,
            "SE-BINN (no cutoff)": op_noc,
            "SE-BINN (cutoff)":    op_cut,
        },
        Yq_T=Yq_T, wq=wq, P=P,
        sigma_bem=sigma_bem, sort_idx=sort_idx,
    )

    # Run cases
    res_a = _train_one("BINN", freeze_gamma=True,  init_state=init_state,
                       shared=shared, enrichment=enrich_noc)
    res_b = _train_one("SE-BINN (no cutoff)", freeze_gamma=False, init_state=init_state,
                       shared=shared, enrichment=enrich_noc)
    res_c = _train_one("SE-BINN (cutoff)", freeze_gamma=False, init_state=init_state,
                       shared=shared, enrichment=enrich_best)

    cases = [res_a, res_b, res_c]

    # ------------------------------------------------------------------
    # 7. Summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("  A/B/C COMPARISON  —  Koch(1), g=x²−y², cutoff diagnostic")
    print("=" * 80)
    col_labels = [c["label"] for c in cases]
    W = 22
    hdr = f"  {'Metric':<24} | " + " | ".join(f"{l:>{W}}" for l in col_labels)
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr); print(sep)

    def row(name, key, fmt=".3e"):
        vals = []
        for c in cases:
            v = c[key]
            vals.append(f"{v:{fmt}}" if isinstance(v, float) else str(v))
        print(f"  {name:<24} | " + " | ".join(f"{v:>{W}}" for v in vals))

    row("Interior rel L2",  "final_rel_L2")
    row("Interior L∞",      "final_linf")
    row("Density rel-diff",  "density_rel_diff")
    print(sep)

    gvs = []
    for c in cases:
        if c["freeze_gamma"]:
            gvs.append("(frozen=0)")
        else:
            g = c["gamma_vals"]
            gvs.append("[" + ",".join(f"{v:+.3f}" for v in g) + "]"
                       if isinstance(g, list) else f"{float(g):.6f}")
    print(f"  {'γ final':<24} | " + " | ".join(f"{v:>{W}}" for v in gvs))

    rvs = [c["lbfgs_reason"] for c in cases]
    print(f"  {'LBFGS reason':<24} | " + " | ".join(f"{v:>{W}}" for v in rvs))
    wvs = [f"{c['wall_time']:.1f}s" for c in cases]
    print(f"  {'Wall time':<24} | " + " | ".join(f"{v:>{W}}" for v in wvs))
    print(sep)

    # A/B and A/C improvements
    de = [c["density_rel_diff"] for c in cases]
    iL2 = [c["final_rel_L2"]   for c in cases]
    if de[0] > 0 and de[1] > 0:
        print(f"\n  A/B density improvement (BINN→no-cutoff)  : {de[0]/de[1]:.2f}×")
    if de[0] > 0 and de[2] > 0:
        print(f"  A/C density improvement (BINN→cutoff)     : {de[0]/de[2]:.2f}×")
    if de[1] > 0 and de[2] > 0:
        print(f"  B/C density improvement (no-cutoff→cutoff): {de[1]/de[2]:.2f}×")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 8. Hypothesis check
    # ------------------------------------------------------------------
    print("\n--- Hypothesis check ---")
    enrich_energies = [d["energy_pc"] for d in sweep_data if np.isfinite(d["R_fac"])]
    best_E = max(enrich_energies)
    E_noc  = diag_noc["energy_pc"]
    cond_best = best["cond_STS"]
    cond_noc  = diag_noc["cond_STS"]
    bg_best   = best["bg_magnitude"]
    bg_noc    = diag_noc["bg_magnitude"]

    print(f"  H1. Enrichment energy increases with cutoff: "
          f"{E_noc*100:.2f}% → {best_E*100:.2f}%  "
          f"{'✓' if best_E > E_noc * 1.5 else '— (small gain)' if best_E > E_noc else '✗'}")
    print(f"  H2. Condition number decreases (improves): "
          f"{cond_noc:.2e} → {cond_best:.2e}  "
          f"{'✓' if cond_best < cond_noc else '✗'}")
    print(f"  H3. Background drops to ~0: "
          f"{bg_noc:.4f} → {bg_best:.4f}  "
          f"{'✓' if bg_best < bg_noc * 0.2 else '✗'}")
    print(f"  H4. γ*_c magnitude increases (spike not diluted): "
          f"max|γ| no-cutoff={np.max(np.abs(diag_noc['gamma_pc'])):.4f}  "
          f"cutoff={np.max(np.abs(diag_best['gamma_pc'])):.4f}  "
          f"{'✓' if np.max(np.abs(diag_best['gamma_pc'])) > np.max(np.abs(diag_noc['gamma_pc'])) else '✗'}")
    if de[1] > 0 and de[2] > 0:
        ab_ratio = de[0] / de[1]
        ac_ratio = de[0] / de[2]
        print(f"  H5. A/B density gap widens with cutoff: "
              f"no-cutoff gap={ab_ratio:.2f}× | cutoff gap={ac_ratio:.2f}×  "
              f"{'✓' if ac_ratio > ab_ratio else '✗'}")

    total_wall = time.perf_counter() - t_global
    print(f"\n  Total wall time: {total_wall:.1f}s")

    # ------------------------------------------------------------------
    # 9. Remaining figure
    # ------------------------------------------------------------------
    print("\n--- Generating final figure ---")
    _fig_ab_comparison(
        cases=cases,
        sigma_bem_sorted=sigma_bem[sort_idx],
        arc_sorted=arc_sorted,
        vertex_arcs=vertex_arcs,
        sing_idx=list(sing_idx),
        outpath=os.path.join(figures_dir, "cutoff_ab_comparison.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
