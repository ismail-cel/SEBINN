"""
Corner-penalty experiment: L-shape, manufactured singular density.

We test whether the corner penalty λ·mean(σ_w(x)²) helps on the L-shape
where the enrichment energy is 61.7% and γ*_true = 1.0.

This is the CONTRASTING case to the Koch(1) g=x²−y² experiment, where the
penalty HURT because enrichment energy was 0% (γ*_lstsq=0).

Here the singular part dominates near (0,0): forcing σ_w→0 at the corner
should correctly redirect the singularity into γ·σ_s.

Boundary data: manufactured density
-------------------------------------
  σ_mfg = γ_true × σ_s + sin(2π s / L)
  g = V σ_mfg   (so σ_BEM = σ_mfg exactly, γ*_lstsq = 1.0)

Domain: L-shape [-1,1]²∖[0,1]×[0,1]
Vertices (CCW): (-1,-1), (1,-1), (1,0), (0,0), (0,1), (-1,1)
Reentrant corner at (0,0): ω = 3π/2, α = π/ω − 1 = −1/3

Cases
------
  Sweep    : SE-BINN, λ ∈ {0.0, 0.01, 0.1, 1.0, 10.0}
  Case A   : BINN (γ frozen at 0, no penalty)
  Case B   : SE-BINN (γ trainable, no penalty)  [= sweep λ=0]
  Case C   : SE-BINN + penalty (γ trainable, best λ)
  Case D   : SE-BINN★ + penalty (γ trainable, best λ, γ_init = γ_lstsq)

Key hypotheses
--------------
  1. Penalty HELPS here (unlike Koch): density rel-diff decreases vs λ=0.
  2. σ_w RMS at corners decreases monotonically with λ.
  3. γ converges closer to 1.0 with penalty than without.
  4. Optimal λ exists: over-penalisation hurts for very large λ.
  5. Interior error stays stable or improves for moderate λ.

Figures saved to experiments/ex2_Lshape/figures/:
  corner_penalty_sweep_singular.png    — density rel-diff + σ_w RMS vs λ
  corner_penalty_ab_singular.png       — loss / density / density error (A,B,C,D)
  corner_penalty_detail_singular.png   — corner zoom (A,B,C,D vs BEM)
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

from src.boundary.lshape import make_lshape_geometry
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
from src.training.operator import build_operator_state, select_corner_points
from src.training.loss import make_loss_fn, sebinn_loss as _bie_only
from src.training.adam_phase import AdamConfig, run_adam_phases
from src.training.lbfgs import LBFGSConfig, run_lbfgs
from src.reconstruction.interior import reconstruct_interior, _log_kernel_matrix


# ===========================================================================
# Configuration
# ===========================================================================

CFG = dict(
    seed           = 0,
    # geometry
    n_per_edge     = 16,          # 6 edges × 16 = 96 panels
    p_gl           = 16,
    m_col_base     = 4,
    w_base         = 1.0,
    w_corner       = 1.0,
    w_ring         = 1.0,
    # equation scaling
    eq_scale_mode  = "fixed",
    eq_scale_fixed = 10.0,
    # BEM
    gmres_tol      = 1e-12,
    gmres_maxiter  = 300,
    # model
    hidden_width   = 80,
    n_hidden       = 4,
    gamma_init     = 0.0,
    # manufactured density
    gamma_true     = 1.0,
    # corner penalty
    radius_factor  = 0.3,
    lambda_sweep   = [0.0, 0.01, 0.1, 1.0, 10.0],
    # Adam — 3 phases
    adam_iters     = [1000, 1000, 1000],
    adam_lrs       = [1e-3, 3e-4, 1e-4],
    log_every      = 200,
    # L-BFGS
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
    # evaluation
    n_grid_coarse  = 51,
    n_grid_final   = 101,
)


# ===========================================================================
# Manufactured density construction (same as run_singular.py)
# ===========================================================================

def make_manufactured_density(sigma_s_Yq, arc, total_arc, gamma_true):
    """σ_mfg = γ_true × σ_s + sin(2π s / L)."""
    sigma_smooth = np.sin(2.0 * np.pi * arc / total_arc)
    return gamma_true * sigma_s_Yq + sigma_smooth


def make_u_exact_fn(Yq_T, wq, sigma_mfg):
    """u_exact(x) = Σ_j G(x, y_j) σ_mfg_j w_j."""
    sigma_wq = sigma_mfg * wq

    def _u_exact(xy):
        K = _log_kernel_matrix(xy, Yq_T)
        return K @ sigma_wq

    return _u_exact


# ===========================================================================
# Arc-length helper
# ===========================================================================

def _boundary_arclength(qdata, n_per_edge):
    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc = panel_start[qdata.pan_id] + qdata.s_on_panel

    Npan         = qdata.n_panels
    total_length = float(qdata.L_panel.sum())
    Nv           = Npan // n_per_edge

    v_panel_idx      = np.arange(Nv) * n_per_edge
    panel_start_full = np.concatenate([[0.0], np.cumsum(qdata.L_panel)])
    vertex_arcs      = np.append(panel_start_full[v_panel_idx], total_length)

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
# Single training run
# ===========================================================================

def _train_one(
    label: str,
    freeze_gamma: bool,
    init_state: dict,
    shared: dict,
    lambda_corner: float = 0.0,
    gamma_override=None,
    verbose: bool = True,
) -> dict:
    """
    One full training run (3 Adam phases + L-BFGS).

    gamma_override : if not None, overwrite γ after loading init_state.
    """
    if verbose:
        fz  = " [γ frozen=0]" if freeze_gamma else " [γ trainable]"
        pen = f", λ={lambda_corner}" if lambda_corner > 0 else ""
        print(f"\n{'='*62}")
        print(f"  Case {label}{fz}{pen}")
        print(f"{'='*62}")

    t0 = time.perf_counter()

    op           = shared["op"]
    enrichment   = shared["enrichment"]
    Yq_T         = shared["Yq_T"]
    wq           = shared["wq"]
    P            = shared["P"]
    sigma_bem    = shared["sigma_bem"]
    sigma_s_Yq   = shared["sigma_s_Yq"]
    sort_idx     = shared["sort_idx"]
    u_exact      = shared["u_exact"]
    corner_pts_t = shared["corner_pts_t"]   # Tensor (Nc, 2)
    corner_ss_t  = shared["corner_ss_t"]    # Tensor (Nc,)

    model = SEBINNModel(
        hidden_width = CFG["hidden_width"],
        n_hidden     = CFG["n_hidden"],
        n_gamma      = enrichment.n_gamma,
        gamma_init   = CFG["gamma_init"],
        dtype        = torch.float64,
    )
    model.load_state_dict(copy.deepcopy(init_state))

    if gamma_override is not None:
        model.gamma_module.gamma.data.fill_(float(gamma_override))
        if verbose:
            print(f"  γ warm-start: {float(gamma_override):.6f}")

    if freeze_gamma:
        model.gamma_module.gamma.requires_grad_(False)
        if verbose:
            print(f"  γ frozen at 0")

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  trainable params: {n_tr} | Nc={len(corner_pts_t)}")

    # Build loss function (plain BIE or penalised)
    loss_fn = make_loss_fn(
        corner_points  = corner_pts_t if lambda_corner > 0 else None,
        corner_sigma_s = corner_ss_t  if lambda_corner > 0 else None,
        lambda_corner  = lambda_corner,
    )

    Yq_t      = torch.tensor(Yq_T,       dtype=torch.float64)
    sigma_s_t = torch.tensor(sigma_s_Yq, dtype=torch.float64)

    stage_checkpoints = []

    def _record_checkpoint(stage, n_iter_so_far, loss):
        with torch.no_grad():
            s = model(Yq_t, sigma_s_t).numpy()
        d_err = float(np.linalg.norm(s - sigma_bem)
                      / max(np.linalg.norm(sigma_bem), 1e-14))
        i_out = reconstruct_interior(
            P=P, Yq=Yq_T, wq=wq, sigma=s,
            n_grid=CFG["n_grid_coarse"], u_exact=u_exact,
            x_range=(-1.0, 1.0), y_range=(-1.0, 1.0),
        )
        g = model.gamma_value()
        stage_checkpoints.append(dict(
            stage            = stage,
            iter             = n_iter_so_far,
            loss             = loss,
            gamma            = g if isinstance(g, list) else float(g),
            density_rel_diff = d_err,
            interior_L2      = i_out.rel_L2,
        ))
        if verbose:
            gfmt = (f"[{','.join(f'{v:.4f}' for v in g)}]"
                    if isinstance(g, list) else f"{float(g):.5f}")
            print(f"  [{label}] {stage}: loss={loss:.3e} | d_err={d_err:.4f} | "
                  f"iL2={i_out.rel_L2:.3e} | γ={gfmt}")

    # Adam phases
    all_adam_loss = []
    global_adam_iter = 0
    for ph_idx, (n_it, lr) in enumerate(zip(CFG["adam_iters"], CFG["adam_lrs"])):
        ph_cfg = AdamConfig(
            phase_iters=[n_it], phase_lrs=[lr], log_every=CFG["log_every"],
        )
        ph_res = run_adam_phases(model, op, ph_cfg, verbose=verbose, loss_fn=loss_fn)
        all_adam_loss.extend(ph_res.loss_hist)
        global_adam_iter += ph_res.n_iters
        _record_checkpoint(f"Adam-ph{ph_idx+1}", global_adam_iter, ph_res.final_loss)

    # L-BFGS
    lbfgs_cfg = _make_lbfgs_cfg()
    if verbose:
        print(f"\n  [{label}] L-BFGS: max={CFG['lbfgs_max_iters']} | "
              f"mem={CFG['lbfgs_memory']} | grad_tol={CFG['lbfgs_grad_tol']:.0e}")
    lbfgs_res = run_lbfgs(model, op, lbfgs_cfg, verbose=verbose, loss_fn=loss_fn)
    _record_checkpoint(
        "LBFGS", global_adam_iter + lbfgs_res.n_iters,
        lbfgs_res.loss_hist[-1] if lbfgs_res.loss_hist else float("nan"),
    )

    # Final density
    with torch.no_grad():
        sigma_final      = model(Yq_t, sigma_s_t).numpy()
        sigma_w_corners  = model.sigma_w(corner_pts_t).numpy()   # (Nc,)

    sigma_w_rms_corners = float(np.sqrt(np.mean(sigma_w_corners ** 2)))

    final_out = reconstruct_interior(
        P=P, Yq=Yq_T, wq=wq, sigma=sigma_final,
        n_grid=CFG["n_grid_final"], u_exact=u_exact,
        x_range=(-1.0, 1.0), y_range=(-1.0, 1.0),
    )
    density_rel_diff = float(
        np.linalg.norm(sigma_final - sigma_bem)
        / max(np.linalg.norm(sigma_bem), 1e-14)
    )

    # BIE loss component only (no penalty — for fair comparison across λ)
    with torch.no_grad():
        _, dbg_bie = _bie_only(model, op)
    bie_loss_final = dbg_bie["loss"]

    t_total = time.perf_counter() - t0

    if verbose:
        g = model.gamma_value()
        gfmt = (f"[{','.join(f'{v:.4f}' for v in g)}]"
                if isinstance(g, list) else f"{float(g):.6f}")
        print(f"\n  {label} final:")
        print(f"    Interior rel L2    : {final_out.rel_L2:.3e}")
        print(f"    Interior L∞        : {final_out.linf:.3e}")
        print(f"    BIE loss (no λ)    : {bie_loss_final:.3e}")
        print(f"    Density rel-diff   : {density_rel_diff:.4f}")
        print(f"    σ_w RMS corners    : {sigma_w_rms_corners:.4f}")
        print(f"    γ final            : {gfmt}")
        print(f"    LBFGS reason       : {lbfgs_res.reason}")
        print(f"    Wall time          : {t_total:.1f}s")

    return dict(
        label               = label,
        lambda_corner       = lambda_corner,
        freeze_gamma        = freeze_gamma,
        final_rel_L2        = final_out.rel_L2,
        final_linf          = final_out.linf,
        bie_loss_final      = bie_loss_final,
        density_rel_diff    = density_rel_diff,
        sigma_w_rms_corners = sigma_w_rms_corners,
        gamma_vals          = model.gamma_value(),
        lbfgs_reason        = lbfgs_res.reason,
        wall_time           = t_total,
        loss_hist_adam      = all_adam_loss,
        loss_hist_lbfgs     = list(lbfgs_res.loss_hist),
        adam_n_iters        = global_adam_iter,
        sigma_final         = sigma_final[sort_idx],
        sigma_w_corners     = sigma_w_corners,
        stage_checkpoints   = stage_checkpoints,
        final_out           = final_out,
    )


# ===========================================================================
# λ sweep (silent)
# ===========================================================================

def _run_sweep(lambdas, init_state, shared):
    results = []
    for lam in lambdas:
        label = f"SE-BINN λ={lam:.2g}"
        print(f"  Sweep: λ = {lam:.3g} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        res = _train_one(
            label         = label,
            freeze_gamma  = False,
            init_state    = init_state,
            shared        = shared,
            lambda_corner = lam,
            verbose       = False,
        )
        print(f"done in {time.perf_counter()-t0:.1f}s | "
              f"d_err={res['density_rel_diff']:.4f} | "
              f"iL2={res['final_rel_L2']:.3e} | "
              f"σ_w_rms={res['sigma_w_rms_corners']:.4f} | "
              f"γ={float(res['gamma_vals']):.4f}")
        results.append(res)
    return results


# ===========================================================================
# Tables
# ===========================================================================

def _print_sweep_table(sweep_results, gamma_lstsq):
    print()
    w = 90
    print("=" * w)
    print("  λ SWEEP RESULTS  —  L-shape, manufactured σ_mfg = γ_true σ_s + sin(2πs/L)")
    print("=" * w)
    hdr = (f"  {'λ':>8} | {'dens rel-diff':>15} | {'int rel L2':>12} | "
           f"{'BIE loss':>12} | {'σ_w RMS':>10} | {'γ final':>10}")
    print(hdr)
    print("  " + "-" * (w - 2))
    for r in sweep_results:
        lam = r["lambda_corner"]
        g   = r["gamma_vals"]
        gv  = float(g) if not isinstance(g, list) else float(g[0])
        print(f"  {lam:>8.3g} | {r['density_rel_diff']:>15.4f} | "
              f"{r['final_rel_L2']:>12.3e} | "
              f"{r['bie_loss_final']:>12.3e} | "
              f"{r['sigma_w_rms_corners']:>10.4f} | "
              f"{gv:>10.4f}")
    print("=" * w)
    print(f"  γ*_lstsq (target) = {gamma_lstsq:.6f}")


def _print_abcd_table(cases, gamma_lstsq, gamma_true):
    print()
    print("=" * 86)
    print("  A/B/C/D COMPARISON  —  L-shape, manufactured singular density")
    print("=" * 86)
    labels = [c["label"] for c in cases]
    w = 15
    hdr = f"  {'Metric':<24} | " + " | ".join(f"{l:>{w}}" for l in labels)
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr); print(sep)

    def row(name, key, fmt=".3e"):
        vals = []
        for c in cases:
            v = c[key]
            if isinstance(v, float):
                vals.append(f"{v:{fmt}}")
            else:
                vals.append(str(v))
        print(f"  {name:<24} | " + " | ".join(f"{v:>{w}}" for v in vals))

    row("Interior rel L2",  "final_rel_L2")
    row("Interior L∞",      "final_linf")
    row("BIE loss",         "bie_loss_final")
    row("Density rel-diff", "density_rel_diff")
    row("σ_w RMS corners",  "sigma_w_rms_corners")
    print(sep)

    gvs = []
    for c in cases:
        if c["freeze_gamma"]:
            gvs.append("(frozen=0)")
        else:
            g = c["gamma_vals"]
            gvs.append(f"{float(g):.6f}" if not isinstance(g, list) else str(g))
    print(f"  {'γ final':<24} | " + " | ".join(f"{v:>{w}}" for v in gvs))

    rvs = [c["lbfgs_reason"] for c in cases]
    print(f"  {'LBFGS reason':<24} | " + " | ".join(f"{v:>{w}}" for v in rvs))

    wvs = [f"{c['wall_time']:.1f}s" for c in cases]
    print(f"  {'Wall time':<24} | " + " | ".join(f"{v:>{w}}" for v in wvs))
    print(sep)
    print(f"  γ*_lstsq = {gamma_lstsq:.6f}  (γ_true = {gamma_true:.6f})")
    print("=" * 86)


# ===========================================================================
# Figures
# ===========================================================================

_COLORS = {
    "BINN":            "#1f77b4",
    "SE-BINN":         "#d62728",
    "SE-BINN+λ":       "#2ca02c",
    "SE-BINN★+λ":      "#ff7f0e",
}
_LS = {
    "BINN":            "-",
    "SE-BINN":         "--",
    "SE-BINN+λ":       "-.",
    "SE-BINN★+λ":      ":",
}


def _fig_sweep(sweep_results, outpath):
    """(a) density rel-diff and σ_w RMS vs λ."""
    lambdas  = [r["lambda_corner"]       for r in sweep_results]
    d_errs   = [r["density_rel_diff"]    for r in sweep_results]
    sw_rms   = [r["sigma_w_rms_corners"] for r in sweep_results]

    best_idx = int(np.argmin(d_errs))
    best_lam = lambdas[best_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.36)

    ax = axes[0]
    ax.plot(lambdas, d_errs, "o-", color="#d62728", lw=1.6, ms=7)
    ax.axvline(best_lam, color="gray", lw=1.0, ls="--", alpha=0.8,
               label=f"best λ={best_lam:.2g}")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel(r"$\lambda$", fontsize=12)
    ax.set_ylabel(r"$\|\sigma_\theta - \sigma_\mathrm{mfg}\|/\|\sigma_\mathrm{mfg}\|$",
                  fontsize=11)
    ax.set_title("(a) Density rel-diff vs λ", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    ax = axes[1]
    ax.plot(lambdas, sw_rms, "s-", color="#2ca02c", lw=1.6, ms=7)
    ax.axvline(best_lam, color="gray", lw=1.0, ls="--", alpha=0.8,
               label=f"best λ={best_lam:.2g}")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel(r"$\lambda$", fontsize=12)
    ax.set_ylabel(r"$\sigma_w$ RMS near corner", fontsize=11)
    ax.set_title(r"(b) $\sigma_w$ RMS at corner points vs $\lambda$", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(
        r"λ sweep — L-shape, manufactured $\sigma_\mathrm{mfg}=\sigma_s+\sin(2\pi s/L)$"
        "\n(SE-BINN with corner penalty, γ trainable)",
        fontsize=11, y=1.01,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved corner_penalty_sweep_singular → {outpath}")


def _fig_abcd(cases, sigma_bem_sorted, arc, vertex_arcs,
              singular_arc_idx, gamma_true, best_lambda, outpath):
    """(b) Loss history, density, and density error for all 4 cases."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.subplots_adjust(hspace=0.42)

    # --- Panel (a): loss history ---
    ax = axes[0]
    for c in cases:
        col  = _COLORS.get(c["label"], "gray")
        ls   = _LS.get(c["label"], "-")
        ha, hl = c["loss_hist_adam"], c["loss_hist_lbfgs"]
        n_a, n_l = len(ha), len(hl)
        ax.semilogy(np.arange(1, n_a+1), ha,
                    color=col, lw=1.4, ls="-", alpha=0.85,
                    label=f"{c['label']} Adam")
        if n_l:
            ax.semilogy(np.arange(n_a+1, n_a+n_l+1), hl,
                        color=col, lw=1.4, ls="--", alpha=0.85,
                        label=f"{c['label']} L-BFGS")
    n_adam = cases[0]["adam_n_iters"]
    ax.axvline(n_adam, color="gray", lw=0.9, ls=":", alpha=0.6,
               label=f"Adam→L-BFGS ({n_adam})")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Loss (BIE component)", fontsize=11)
    ax.set_title("(a) Loss history", fontsize=12)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    # --- Panel (b): density ---
    ax = axes[1]
    for ci in singular_arc_idx:
        ax.axvline(vertex_arcs[ci], color="#999999", lw=0.7, ls="--", alpha=0.7)
    ax.plot(arc, sigma_bem_sorted, color="black", lw=1.4, alpha=0.9,
            label=r"$\sigma_\mathrm{mfg}$ (reference)")
    for c in cases:
        col = _COLORS.get(c["label"], "gray")
        ls  = _LS.get(c["label"], "-")
        ax.plot(arc, c["sigma_final"], color=col, lw=1.1, ls=ls,
                alpha=0.85, label=c["label"])
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$\sigma(s)$", fontsize=11)
    ax.set_title(r"(b) Boundary density", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, lw=0.3, alpha=0.5)

    # --- Panel (c): density error ---
    ax = axes[2]
    for ci in singular_arc_idx:
        ax.axvline(vertex_arcs[ci], color="#999999", lw=0.7, ls="--", alpha=0.7)
    for c in cases:
        col = _COLORS.get(c["label"], "gray")
        ls  = _LS.get(c["label"], "-")
        err = np.abs(c["sigma_final"] - sigma_bem_sorted)
        ax.semilogy(arc, err + 1e-16, color=col, lw=1.1, ls=ls,
                    alpha=0.85, label=c["label"])
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$|\sigma_\theta - \sigma_\mathrm{mfg}|$", fontsize=11)
    ax.set_title("(c) Pointwise density error", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(
        r"A/B/C/D Comparison — L-shape, manufactured $\sigma_\mathrm{mfg}$"
        f"\nγ_true={gamma_true}  |  best λ={best_lambda:.2g}",
        fontsize=11, y=1.01,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved corner_penalty_ab_singular → {outpath}")


def _fig_corner_zoom(cases, sigma_bem_sorted, arc, vertex_arcs,
                     singular_arc_idx, outpath):
    """(c) Zoom into the reentrant corner region."""
    # Identify arclength of the reentrant corner (vertex index 3)
    if len(singular_arc_idx) > 0:
        s_corner = vertex_arcs[singular_arc_idx[0]]
    else:
        s_corner = 0.0

    half_win = 0.6     # arclength window around the corner
    s_lo, s_hi = s_corner - half_win, s_corner + half_win

    total_arc = vertex_arcs[-1]
    # If window wraps around, use the full arc with a tight clip
    mask = (arc >= s_lo) & (arc <= s_hi)
    if mask.sum() < 20:
        mask = np.ones(len(arc), dtype=bool)

    arc_z = arc[mask]
    sem_z = sigma_bem_sorted[mask]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axvline(s_corner, color="gray", lw=0.8, ls="--", alpha=0.7,
               label=f"corner arc s={s_corner:.3f}")
    ax.plot(arc_z, sem_z, color="black", lw=1.5, alpha=0.9,
            label=r"$\sigma_\mathrm{mfg}$ (reference)")

    for c in cases:
        col = _COLORS.get(c["label"], "gray")
        ls  = _LS.get(c["label"], "-")
        ax.plot(arc_z, c["sigma_final"][mask], color=col, lw=1.3, ls=ls,
                alpha=0.85, label=c["label"])

    ax.set_xlim(s_lo, s_hi)
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$\sigma(s)$", fontsize=11)
    ax.set_title(
        r"Corner region zoom — reentrant corner at $(0,0)$" "\n"
        r"σ_BEM has $r^{-1/3}$ spike; SE-BINN+λ should have σ_w≈0 near corner",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved corner_penalty_detail_singular → {outpath}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("=" * 72)
    print("  Corner-penalty experiment: L-shape, manufactured singular density")
    print("=" * 72)

    t_global = time.perf_counter()
    figures_dir = os.path.join(_HERE, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Geometry and quadrature
    # ------------------------------------------------------------------
    print("\n--- Setup ---")
    geom   = make_lshape_geometry()
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=CFG["n_per_edge"])
    label_corner_ring_panels(panels, P)

    alpha_sing = np.pi / geom.corner_angles[3] - 1
    print(f"  L-shape: vertices={geom.n_vertices} | panels={len(panels)}")
    print(f"  Reentrant corner: vertex 3 = {P[3]} | "
          f"ω={geom.corner_angles[3]/np.pi:.4f}π | α={alpha_sing:.4f}")

    qdata = build_panel_quadrature(panels, p=CFG["p_gl"])
    Yq_T  = qdata.Yq.T   # (Nq, 2)
    wq    = qdata.wq
    Nq    = qdata.n_quad

    arc, vertex_arcs, total_arc = _boundary_arclength(qdata, CFG["n_per_edge"])
    sort_idx = np.argsort(arc)
    print(f"  Nq={Nq} | total arc={total_arc:.4f}")

    # ------------------------------------------------------------------
    # 2. Singular enrichment (single corner at vertex 3 = (0,0))
    # ------------------------------------------------------------------
    enrichment  = SingularEnrichment(geom=geom, per_corner_gamma=False)
    sigma_s_Yq  = enrichment.precompute(Yq_T)   # (Nq,)

    # ------------------------------------------------------------------
    # 3. Manufactured density
    # ------------------------------------------------------------------
    print("\n--- Manufactured density ---")
    gamma_true  = float(CFG["gamma_true"])
    sigma_mfg   = make_manufactured_density(sigma_s_Yq, arc, total_arc, gamma_true)

    s_proj     = gamma_true * sigma_s_Yq
    energy_true = float(np.linalg.norm(s_proj)**2
                        / max(np.linalg.norm(sigma_mfg)**2, 1e-14))
    print(f"  γ_true = {gamma_true:.4f}")
    print(f"  ‖σ_s‖    = {np.linalg.norm(sigma_s_Yq):.4f}")
    print(f"  ‖σ_mfg‖  = {np.linalg.norm(sigma_mfg):.4f}")
    print(f"  Enrichment energy (γ_true²‖σ_s‖²/‖σ_mfg‖²) = {energy_true*100:.2f}%")

    # ------------------------------------------------------------------
    # 4. BEM reference: V σ_BEM = g = V σ_mfg
    # ------------------------------------------------------------------
    print("\n--- BEM reference ---")
    nmat     = assemble_nystrom_matrix(qdata)
    g_bnd    = nmat.V @ sigma_mfg
    bem_sol  = solve_bem(nmat, g_bnd,
                         tol=CFG["gmres_tol"], max_iter=CFG["gmres_maxiter"])
    sigma_bem = bem_sol.sigma
    print(f"  GMRES: flag={bem_sol.flag} | rel_res={bem_sol.rel_res:.3e}")

    sigma_recovery_err = float(np.linalg.norm(sigma_bem - sigma_mfg)
                               / max(np.linalg.norm(sigma_mfg), 1e-14))
    print(f"  ‖σ_BEM − σ_mfg‖/‖σ_mfg‖ = {sigma_recovery_err:.3e}  (≈ machine ε)")

    u_exact = make_u_exact_fn(Yq_T, wq, sigma_mfg)

    # ------------------------------------------------------------------
    # 5. Enrichment diagnostic
    # ------------------------------------------------------------------
    print("\n--- Enrichment diagnostic ---")
    gamma_lstsq = float(np.dot(sigma_s_Yq, sigma_bem)
                        / max(np.dot(sigma_s_Yq, sigma_s_Yq), 1e-14))
    sigma_proj  = gamma_lstsq * sigma_s_Yq
    res_norm    = np.linalg.norm(sigma_bem - sigma_proj)
    bem_norm    = np.linalg.norm(sigma_bem)
    energy_frac = 1.0 - (res_norm / max(bem_norm, 1e-14)) ** 2

    print(f"  γ*_lstsq            : {gamma_lstsq:.6f}  (γ_true = {gamma_true:.6f})")
    print(f"  |γ*_lstsq − γ_true| : {abs(gamma_lstsq - gamma_true):.3e}")
    print(f"  Enrichment energy   : {energy_frac*100:.4f}%")
    print(f"  (Koch comparison    : ~2.8%; this should be >> 2.8%)")

    # ------------------------------------------------------------------
    # 6. Corner point selection
    # ------------------------------------------------------------------
    print("\n--- Corner penalty setup ---")
    corner_idx = select_corner_points(qdata, geom, radius_factor=CFG["radius_factor"])
    Nc = len(corner_idx)
    corner_pts_np = Yq_T[corner_idx]           # (Nc, 2)
    corner_ss_np  = sigma_s_Yq[corner_idx]     # (Nc,)
    corner_pts_t  = torch.tensor(corner_pts_np, dtype=torch.float64)
    corner_ss_t   = torch.tensor(corner_ss_np,  dtype=torch.float64)

    # Compute mean edge length for radius diagnostic
    n_edges = len(P)
    edge_lengths = np.array([
        float(np.linalg.norm(P[(i+1) % n_edges] - P[i]))
        for i in range(n_edges)
    ])
    R = CFG["radius_factor"] * float(edge_lengths.mean())
    print(f"  Nc={Nc} nodes within R={CFG['radius_factor']}×mean_edge={R:.4f} "
          f"of reentrant corner")
    print(f"  σ_s at corner points: "
          f"min={corner_ss_np.min():.4f}  max={corner_ss_np.max():.4f}  "
          f"rms={float(np.sqrt(np.mean(corner_ss_np**2))):.4f}")

    # σ_mfg rms at corner points — this is what the penalty competes with
    sigma_mfg_corners_rms = float(np.sqrt(np.mean(sigma_mfg[corner_idx]**2)))
    print(f"  σ_mfg RMS at corner points (total density) = {sigma_mfg_corners_rms:.4f}")
    print(f"  → σ_s fraction at corner: "
          f"{float(np.sqrt(np.mean((gamma_true*corner_ss_np)**2)))/max(sigma_mfg_corners_rms,1e-14)*100:.1f}%")

    # ------------------------------------------------------------------
    # 7. Operator state and initial model
    # ------------------------------------------------------------------
    print("\n--- Operator setup ---")
    w_panel = panel_loss_weights(panels, w_base=CFG["w_base"],
                                 w_corner=CFG["w_corner"], w_ring=CFG["w_ring"])
    colloc = build_collocation_points(panels, m_col_panel=CFG["m_col_base"])
    op, op_diag = build_operator_state(
        colloc=colloc, qdata=qdata, enrichment=enrichment, g=u_exact,
        panel_weights=w_panel,
        eq_scale_mode=CFG["eq_scale_mode"],
        eq_scale_fixed=CFG["eq_scale_fixed"],
        dtype=torch.float64, device="cpu",
    )
    print(f"  Nb={colloc.n_colloc} | eq_scale={op_diag['eq_scale']:.2e} | "
          f"mean|A|={op_diag['mean_abs_A_before']:.3e}")

    torch.manual_seed(CFG["seed"])
    init_model = SEBINNModel(
        hidden_width = CFG["hidden_width"],
        n_hidden     = CFG["n_hidden"],
        n_gamma      = enrichment.n_gamma,
        gamma_init   = CFG["gamma_init"],
        dtype        = torch.float64,
    )
    init_state = copy.deepcopy(init_model.state_dict())
    print(f"  Model: n_params={init_model.n_params()} | n_gamma={enrichment.n_gamma}")
    print(f"  Shared initial state saved.")

    shared = dict(
        op=op, enrichment=enrichment,
        Yq_T=Yq_T, wq=wq, P=P,
        sigma_bem=sigma_bem, sigma_s_Yq=sigma_s_Yq,
        sort_idx=sort_idx, u_exact=u_exact,
        corner_pts_t=corner_pts_t, corner_ss_t=corner_ss_t,
    )

    # ------------------------------------------------------------------
    # 8. λ sweep
    # ------------------------------------------------------------------
    print("\n--- λ sweep ---")
    sweep_results = _run_sweep(CFG["lambda_sweep"], init_state, shared)
    _print_sweep_table(sweep_results, gamma_lstsq)

    # Pick best λ by density rel-diff
    d_errs     = [r["density_rel_diff"] for r in sweep_results]
    best_idx   = int(np.argmin(d_errs))
    best_lambda = CFG["lambda_sweep"][best_idx]
    best_res    = sweep_results[best_idx]
    print(f"\n  Best λ = {best_lambda:.3g}  "
          f"(density rel-diff = {best_res['density_rel_diff']:.4f})")

    # ------------------------------------------------------------------
    # 9. Full A/B/C/D comparison (verbose)
    # ------------------------------------------------------------------
    print("\n--- Full A/B/C/D comparison ---")

    # Case A: BINN (γ frozen)
    res_a = _train_one("BINN", freeze_gamma=True, init_state=init_state,
                       shared=shared, lambda_corner=0.0)

    # Case B: SE-BINN (γ trainable, no penalty) — reuse λ=0 sweep result
    # but run verbose for full diagnostics
    res_b = _train_one("SE-BINN", freeze_gamma=False, init_state=init_state,
                       shared=shared, lambda_corner=0.0)

    # Case C: SE-BINN + best penalty
    res_c = _train_one("SE-BINN+λ", freeze_gamma=False, init_state=init_state,
                       shared=shared, lambda_corner=best_lambda)

    # Case D: SE-BINN★ + best penalty (warm γ_init from lstsq)
    res_d = _train_one("SE-BINN★+λ", freeze_gamma=False, init_state=init_state,
                       shared=shared, lambda_corner=best_lambda,
                       gamma_override=gamma_lstsq)

    cases = [res_a, res_b, res_c, res_d]
    _print_abcd_table(cases, gamma_lstsq, gamma_true)

    # ------------------------------------------------------------------
    # 10. Hypothesis check
    # ------------------------------------------------------------------
    print("\n--- Hypothesis check ---")
    d_b = res_b["density_rel_diff"]
    d_c = res_c["density_rel_diff"]
    sw_b = res_b["sigma_w_rms_corners"]
    sw_c = res_c["sigma_w_rms_corners"]

    g_b = float(res_b["gamma_vals"])
    g_c = float(res_c["gamma_vals"])

    print(f"  H1. Penalty HELPS (d_err decreases B→C):  "
          f"B={d_b:.4f} → C={d_c:.4f}  "
          f"{'✓' if d_c < d_b else '✗ (HURT)'}")
    print(f"  H2. σ_w RMS decreases monotonically in sweep:")
    sw_list = [r["sigma_w_rms_corners"] for r in sweep_results]
    mono = all(sw_list[i] >= sw_list[i+1] for i in range(len(sw_list)-1))
    print(f"      {[f'{v:.4f}' for v in sw_list]}  {'✓' if mono else '✗'}")
    print(f"  H3. γ converges closer to 1 with penalty:  "
          f"B={g_b:.4f} → C={g_c:.4f}  target=1.0  "
          f"{'✓' if abs(g_c-1.0) < abs(g_b-1.0) else '✗'}")
    print(f"  H4. Best λ is moderate (not 0 or 10):  "
          f"best λ={best_lambda:.3g}  "
          f"{'✓' if 0 < best_lambda < 10 else '— (check manually)'}")
    iL2_b = res_b["final_rel_L2"]
    iL2_c = res_c["final_rel_L2"]
    print(f"  H5. Interior error stable:  "
          f"B={iL2_b:.3e} → C={iL2_c:.3e}  "
          f"{'✓' if iL2_c <= iL2_b * 3 else '✗ (degraded)'}")

    # Contrast with Koch
    print(f"\n  Koch(1) g=x²−y² result (for contrast):")
    print(f"    Koch enrichment energy ≈ 0%  →  penalty HURT  (d_err: 0.551→0.585)")
    print(f"  This experiment enrichment energy = {energy_frac*100:.2f}%")
    if d_c < d_b:
        print(f"  → Penalty HELPED as hypothesised: Δ(d_err) = {d_b-d_c:.4f}")
    else:
        print(f"  → Penalty DID NOT help: Δ(d_err) = {d_c-d_b:+.4f} "
              f"(investigate σ_w contribution at corners)")

    total_wall = time.perf_counter() - t_global
    print(f"\n  Total wall time: {total_wall:.1f}s")

    # ------------------------------------------------------------------
    # 11. Figures
    # ------------------------------------------------------------------
    print("\n--- Generating figures ---")
    sing_arc_idx = list(geom.singular_corner_indices)

    _fig_sweep(
        sweep_results=sweep_results,
        outpath=os.path.join(figures_dir, "corner_penalty_sweep_singular.png"),
    )
    _fig_abcd(
        cases=cases,
        sigma_bem_sorted=sigma_bem[sort_idx],
        arc=arc[sort_idx],
        vertex_arcs=vertex_arcs,
        singular_arc_idx=sing_arc_idx,
        gamma_true=gamma_true,
        best_lambda=best_lambda,
        outpath=os.path.join(figures_dir, "corner_penalty_ab_singular.png"),
    )
    _fig_corner_zoom(
        cases=cases,
        sigma_bem_sorted=sigma_bem[sort_idx],
        arc=arc[sort_idx],
        vertex_arcs=vertex_arcs,
        singular_arc_idx=sing_arc_idx,
        outpath=os.path.join(figures_dir, "corner_penalty_detail_singular.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
