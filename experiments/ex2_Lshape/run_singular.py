"""
Experiment 2b: SE-BINN on the L-shaped domain — Manufactured singular density.

Domain
------
[-1,1]² with the upper-right quadrant [0,1]×[0,1] removed.
Vertices (CCW): (-1,-1), (1,-1), (1,0), (0,0), (0,1), (-1,1).
Reentrant corner at (0,0): ω = 3π/2, α = π/ω − 1 = −1/3.

Boundary data strategy: MANUFACTURED DENSITY
---------------------------------------------
Direct use of g = r^{2/3}sin(2θ/3) as Dirichlet data fails because that
function has a jump in its tangential derivative at the notch corner (1,0),
which creates a non-integrable σ ~ r^{-1} singularity there — completely
unrelated to the intended reentrant corner singularity at (0,0).

Instead we use the MANUFACTURED DENSITY approach:

  σ_mfg = γ_true × σ_s + σ_smooth

where:
  σ_s      = −(2/3) r^{−1/3}   (the canonical enrichment density, r=|y−(0,0)|)
  σ_smooth = sin(2π s / L)      (smooth, arc-length-based oscillation, L=8)
  γ_true   = 1.0

Then set the boundary data:
  g_i = Σ_j  V[i,j] σ_mfg_j   (single-layer potential evaluated on the boundary)

and define the exact interior solution:
  u_exact(x) = Σ_j  G(x, y_j) σ_mfg_j w_j

By construction:
  - V σ_BEM = g  ⟹  σ_BEM = σ_mfg  (exactly, to machine ε)
  - u_exact is well-defined (σ_mfg is integrable — σ_s ∈ L¹ on the boundary)
  - γ*_lstsq = γ_true = 1.0  (exact recovery)
  - Enrichment energy = ‖γ_true σ_s‖² / ‖σ_mfg‖²  (controlled, nonzero)

Three training cases — all from the same initial weights
----------------------------------------------------------
  Case A — BINN:    γ frozen at 0 (σ = σ_w only)
  Case B — SE-BINN: γ trainable, γ_init = 0
  Case C — SE-BINN★: γ trainable, γ_init = γ_true = 1.0 (warm start)

Figures saved to experiments/ex2_Lshape/figures/
  ab_comparison_singular.png    — loss, density, density error (all 3 cases)
  convergence_singular.png      — loss curves + stage-level density and γ
  interior_error_singular.png   — u_exact | SE-BINN★ u_θ | log₁₀|error|

Key hypothesis
--------------
  1. BEM error (interior rel L2) ≪ 1%  (by construction)
  2. γ*_lstsq ≈ 1.0  (manufactured to be exact)
  3. Enrichment energy >> Koch's ~2.8%  (stronger singularity, α=−1/3 vs −1/4)
  4. A/B gap (BINN→SE-BINN) >> Koch's 12%  (BINN must fit r^{−1/3} with σ_w)
  5. SE-BINN★ converges faster due to warm start
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
from src.training.operator import build_operator_state
from src.training.adam_phase import AdamConfig, run_adam_phases
from src.training.lbfgs import LBFGSConfig, run_lbfgs
from src.reconstruction.interior import reconstruct_interior, _log_kernel_matrix


# ===========================================================================
# Configuration
# ===========================================================================

CFG = dict(
    seed           = 0,
    # geometry
    n_per_edge     = 16,         # 6 edges × 16 = 96 panels
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
    # manufactured density parameters
    gamma_true     = 1.0,        # true singular coefficient in σ_mfg
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
# Manufactured density construction
# ===========================================================================

def make_manufactured_density(
    sigma_s_Yq: np.ndarray,    # (Nq,) singular basis at quadrature nodes
    arc: np.ndarray,           # (Nq,) global arc-length at each quadrature node
    total_arc: float,          # total perimeter
    gamma_true: float,         # singular amplitude
) -> np.ndarray:
    """
    Build σ_mfg = γ_true × σ_s + σ_smooth.

    σ_smooth = sin(2π s / L)  — smooth, arc-length-based oscillation.

    Returns σ_mfg of shape (Nq,).
    """
    sigma_smooth = np.sin(2.0 * np.pi * arc / total_arc)
    return gamma_true * sigma_s_Yq + sigma_smooth


def make_u_exact_fn(Yq_T: np.ndarray, wq: np.ndarray, sigma_mfg: np.ndarray):
    """
    Return a callable u_exact(xy) that evaluates the single-layer potential
    at arbitrary points using σ_mfg.

      u_exact(x) = Σ_j G(x, y_j) σ_mfg_j w_j

    Parameters
    ----------
    Yq_T     : (Nq, 2) quadrature nodes
    wq       : (Nq,)   weights
    sigma_mfg : (Nq,)  manufactured density
    """
    sigma_wq = sigma_mfg * wq  # (Nq,) precomputed

    def _u_exact(xy: np.ndarray) -> np.ndarray:
        K = _log_kernel_matrix(xy, Yq_T)   # (M, Nq)
        return K @ sigma_wq                 # (M,)

    return _u_exact


# ===========================================================================
# Arc-length helper
# ===========================================================================

def _boundary_arclength(qdata, n_per_edge: int):
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
# One full training run (3 Adam phases + L-BFGS)
# ===========================================================================

def _train_case(
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

    op         = shared["op"]
    enrichment = shared["enrichment"]
    Yq_T       = shared["Yq_T"]
    wq         = shared["wq"]
    P          = shared["P"]
    sigma_bem  = shared["sigma_bem"]
    sigma_s_Yq = shared["sigma_s_Yq"]
    sort_idx   = shared["sort_idx"]
    u_exact    = shared["u_exact"]

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
        print(f"  trainable params: {n_tr}")

    Yq_t      = torch.tensor(Yq_T,       dtype=torch.float64)
    sigma_s_t = torch.tensor(sigma_s_Yq, dtype=torch.float64)

    stage_checkpoints = []

    def _record_checkpoint(stage: str, n_iter_so_far: int, loss: float):
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

    # Adam
    all_adam_loss = []
    global_adam_iter = 0
    for ph_idx, (n_it, lr) in enumerate(zip(CFG["adam_iters"], CFG["adam_lrs"])):
        ph_cfg = AdamConfig(
            phase_iters=[n_it], phase_lrs=[lr], log_every=CFG["log_every"],
        )
        ph_res = run_adam_phases(model, op, ph_cfg, verbose=verbose)
        all_adam_loss.extend(ph_res.loss_hist)
        global_adam_iter += ph_res.n_iters
        _record_checkpoint(f"Adam-ph{ph_idx+1}", global_adam_iter, ph_res.final_loss)

    # L-BFGS
    lbfgs_cfg = LBFGSConfig(
        max_iters=CFG["lbfgs_max_iters"], grad_tol=CFG["lbfgs_grad_tol"],
        step_tol=CFG["lbfgs_step_tol"], memory=CFG["lbfgs_memory"],
        log_every=CFG["lbfgs_log_every"], alpha0=CFG["lbfgs_alpha0"],
        alpha_fallback=CFG["lbfgs_alpha_fb"], armijo_c1=CFG["lbfgs_armijo_c1"],
        backtrack_beta=CFG["lbfgs_beta"], max_backtrack=CFG["lbfgs_max_bt"],
    )
    if verbose:
        print(f"\n  [{label}] L-BFGS: max={CFG['lbfgs_max_iters']} | "
              f"mem={CFG['lbfgs_memory']} | grad_tol={CFG['lbfgs_grad_tol']:.0e}")
    lbfgs_res = run_lbfgs(model, op, lbfgs_cfg, verbose=verbose)
    _record_checkpoint(
        "LBFGS", global_adam_iter + lbfgs_res.n_iters,
        lbfgs_res.loss_hist[-1] if lbfgs_res.loss_hist else float("nan"),
    )

    with torch.no_grad():
        sigma_final = model(Yq_t, sigma_s_t).numpy()

    final_out = reconstruct_interior(
        P=P, Yq=Yq_T, wq=wq, sigma=sigma_final,
        n_grid=CFG["n_grid_final"], u_exact=u_exact,
        x_range=(-1.0, 1.0), y_range=(-1.0, 1.0),
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
        label=label, freeze_gamma=freeze_gamma,
        final_rel_L2=final_out.rel_L2, final_linf=final_out.linf,
        density_rel_diff=density_rel_diff, gamma_vals=model.gamma_value(),
        lbfgs_reason=lbfgs_res.reason, wall_time=t_total,
        stage_checkpoints=stage_checkpoints,
        loss_hist_adam=all_adam_loss,
        loss_hist_lbfgs=list(lbfgs_res.loss_hist),
        adam_n_iters=global_adam_iter,
        sigma_final=sigma_final[sort_idx],
        final_out=final_out,
    )


# ===========================================================================
# Summary table
# ===========================================================================

def _print_table(cases, bem_rel_L2, bem_linf, gamma_lstsq, energy_frac, gamma_true):
    print()
    print("=" * 72)
    print("  L-SHAPE BENCHMARK RESULTS  (manufactured singular density)")
    print("=" * 72)
    print(f"  σ_mfg = γ_true σ_s + sin(2πs/L)  |  γ_true = {gamma_true:.4f}")
    print(f"  BEM reference       : rel_L2={bem_rel_L2:.3e} | linf={bem_linf:.3e}")
    print(f"  γ*_lstsq            : {gamma_lstsq:.6f}  (should equal γ_true={gamma_true:.4f})")
    print(f"  Enrichment energy   : {energy_frac*100:.2f}%  "
          "(fraction of ‖σ_BEM‖² explained by γ*σ_s)")
    print()

    cols = [c["label"] for c in cases]
    w    = 16
    hdr  = f"  {'Metric':<26} | " + " | ".join(f"{c:>{w}}" for c in cols)
    sep  = "  " + "-" * (len(hdr) - 2)
    print(hdr); print(sep)

    def row(name, key, fmt=".3e"):
        vals = [f"{c[key]:{fmt}}" if isinstance(c[key], float) else str(c[key])
                for c in cases]
        print(f"  {name:<26} | " + " | ".join(f"{v:>{w}}" for v in vals))

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
            gvs.append(f"{float(g):.6f}" if not isinstance(g, list) else str(g))
    print(f"  {'γ final':<26} | " + " | ".join(f"{v:>{w}}" for v in gvs))
    rvs = [c["lbfgs_reason"] for c in cases]
    print(f"  {'LBFGS reason':<26} | " + " | ".join(f"{v:>{w}}" for v in rvs))
    wvs = [f"{c['wall_time']:.1f}s" for c in cases]
    print(f"  {'Wall time':<26} | " + " | ".join(f"{v:>{w}}" for v in wvs))
    print(sep)

    rL2 = [c["final_rel_L2"] for c in cases]
    if len(rL2) >= 2 and rL2[0] > 0 and rL2[1] > 0:
        ab = rL2[0] / rL2[1]
        print(f"\n  A/B improvement (BINN → SE-BINN)  : {ab:.2f}×")
    if len(rL2) >= 3 and rL2[0] > 0 and rL2[2] > 0:
        ac = rL2[0] / rL2[2]
        print(f"  A/C improvement (BINN → SE-BINN★) : {ac:.2f}×")
    de = [c["density_rel_diff"] for c in cases]
    if len(de) >= 2 and de[0] > 0 and de[1] > 0:
        print(f"  A/B density improvement            : {de[0]/de[1]:.2f}×")
    print("=" * 72)


# ===========================================================================
# Figures
# ===========================================================================

def _fig_ab_comparison(cases, sigma_bem_sorted, arc, vertex_arcs,
                        singular_idx, gamma_true, outpath):
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.subplots_adjust(hspace=0.42)
    COLORS = {"BINN": "#1f77b4", "SE-BINN": "#d62728", "SE-BINN★": "#2ca02c"}
    LSMAP  = {"BINN": "-",       "SE-BINN": "--",       "SE-BINN★": ":"}

    ax = axes[0]
    for c in cases:
        ha, hl = c["loss_hist_adam"], c["loss_hist_lbfgs"]
        n_a, n_l = len(ha), len(hl)
        col = COLORS[c["label"]]
        ax.semilogy(np.arange(1, n_a+1), ha, color=col, lw=1.4, ls="-",
                    alpha=0.85, label=f"{c['label']} (Adam)")
        if n_l:
            ax.semilogy(np.arange(n_a+1, n_a+n_l+1), hl, color=col, lw=1.4,
                        ls="--", alpha=0.85, label=f"{c['label']} (L-BFGS)")
    n_adam = cases[0]["adam_n_iters"]
    ax.axvline(n_adam, color="gray", lw=0.9, ls=":", alpha=0.7,
               label=f"Adam→L-BFGS (iter {n_adam})")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("(a) Loss history", fontsize=12)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    ax = axes[1]
    for ci in singular_idx:
        ax.axvline(vertex_arcs[ci], color="#999999", lw=0.7, ls="--", alpha=0.7)
    ax.plot(arc, sigma_bem_sorted, color="black", lw=1.2, alpha=0.9,
            label=r"$\sigma_\mathrm{mfg}$ (reference)")
    for c in cases:
        ax.plot(arc, c["sigma_final"], color=COLORS[c["label"]],
                lw=1.0, ls=LSMAP[c["label"]], alpha=0.85, label=c["label"])
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$\sigma(s)$", fontsize=11)
    ax.set_title(r"(b) Boundary density, $\sigma_\mathrm{mfg}=\gamma_\mathrm{true}\sigma_s + \sin(2\pi s/L)$",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, lw=0.3, alpha=0.5)

    ax = axes[2]
    for ci in singular_idx:
        ax.axvline(vertex_arcs[ci], color="#999999", lw=0.7, ls="--", alpha=0.7)
    for c in cases:
        err = np.abs(c["sigma_final"] - sigma_bem_sorted)
        ax.semilogy(arc, err + 1e-16, color=COLORS[c["label"]],
                    lw=1.0, ls=LSMAP[c["label"]], alpha=0.85, label=c["label"])
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$|\sigma_\theta - \sigma_\mathrm{mfg}|$", fontsize=11)
    ax.set_title("(c) Pointwise density error", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(
        r"A/B/C Comparison: L-shape, manufactured $\sigma_\mathrm{mfg}=\gamma_\mathrm{true}\sigma_s + \sin(2\pi s/L)$"
        + f"\n$\\gamma_\\mathrm{{true}}={gamma_true}$  |  "
        r"BINN ($\gamma=0$) vs SE-BINN ($\gamma$ free) vs SE-BINN★ (warm $\gamma$)",
        fontsize=11, y=1.01,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved ab_comparison_singular → {outpath}")


def _fig_convergence(cases, outpath, gamma_true):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.38)
    COLORS = {"BINN": "#1f77b4", "SE-BINN": "#d62728", "SE-BINN★": "#2ca02c"}
    LSMAP  = {"BINN": "-",       "SE-BINN": "--",       "SE-BINN★": ":"}

    ax = axes[0]
    for c in cases:
        ha, hl = c["loss_hist_adam"], c["loss_hist_lbfgs"]
        n_a, n_l = len(ha), len(hl)
        col, ls = COLORS[c["label"]], LSMAP[c["label"]]
        ax.semilogy(np.arange(1, n_a+1), ha, color=col, lw=1.2, ls="-", alpha=0.85)
        if n_l:
            ax.semilogy(np.arange(n_a+1, n_a+n_l+1), hl, color=col, lw=1.2,
                        ls="--", alpha=0.85, label=c["label"])
        else:
            ax.semilogy([], [], color=col, lw=1.2, ls=ls, label=c["label"])
    ax.axvline(cases[0]["adam_n_iters"], color="gray", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
    ax.set_title("(i) Loss curve", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, which="both", lw=0.3, alpha=0.5)

    ax = axes[1]
    for c in cases:
        cps   = c["stage_checkpoints"]
        iters = [p["iter"] for p in cps]
        derr  = [p["density_rel_diff"] for p in cps]
        col, ls = COLORS[c["label"]], LSMAP[c["label"]]
        ax.plot(iters, derr, color=col, lw=1.3, ls=ls, marker="o",
                ms=5, alpha=0.9, label=c["label"])
    ax.set_xlabel("Iteration (end of stage)")
    ax.set_ylabel(r"$\|\sigma_\theta - \sigma_\mathrm{mfg}\|/\|\sigma_\mathrm{mfg}\|$",
                  fontsize=10)
    ax.set_title("(ii) Density rel-diff vs iteration", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, lw=0.3, alpha=0.5)

    ax = axes[2]
    for c in cases:
        if c["freeze_gamma"]:
            continue
        cps   = c["stage_checkpoints"]
        iters = [p["iter"] for p in cps]
        gvals = [p["gamma"] if isinstance(p["gamma"], float) else p["gamma"][0]
                 for p in cps]
        col, ls = COLORS[c["label"]], LSMAP[c["label"]]
        ax.plot(iters, gvals, color=col, lw=1.3, ls=ls, marker="s",
                ms=5, alpha=0.9, label=c["label"])
    ax.axhline(gamma_true, color="black", lw=0.9, ls="--", alpha=0.7,
               label=rf"$\gamma_\mathrm{{true}}={gamma_true}$")
    ax.axhline(0, color="gray", lw=0.5, ls=":", alpha=0.4)
    ax.set_xlabel("Iteration (end of stage)")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title(r"(iii) $\gamma$ trajectory", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, lw=0.3, alpha=0.5)

    fig.suptitle(
        r"Convergence — L-shape, manufactured $\sigma_\mathrm{mfg}=\gamma_\mathrm{true}\sigma_s + \sin(2\pi s/L)$",
        fontsize=11, y=1.02,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved convergence_singular → {outpath}")


def _fig_interior(bem_out, best_case, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.3)
    final_out = best_case["final_out"]

    def _imshow(ax, data, title, cmap="RdBu_r", symm=True):
        vmax = np.nanmax(np.abs(data)) if symm else None
        vmin = -vmax if symm else None
        im = ax.imshow(data, origin="lower", cmap=cmap,
                       extent=(-1, 1, -1, 1), vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
        plt.colorbar(im, ax=ax, shrink=0.8)

    _imshow(axes[0], bem_out.Ugrid,   r"$u_\mathrm{exact}$ (from $\sigma_\mathrm{mfg}$)")
    _imshow(axes[1], final_out.Ugrid, rf"SE-BINN★ $u_\theta$")

    Egrid = np.abs(final_out.Egrid)
    Egrid = np.where(np.isnan(Egrid), np.nan, np.maximum(Egrid, 1e-10))
    im = axes[2].imshow(np.log10(Egrid), origin="lower", cmap="hot_r",
                        extent=(-1, 1, -1, 1))
    axes[2].set_title(r"$\log_{10}|u_\theta - u_\mathrm{exact}|$", fontsize=11)
    axes[2].set_xlabel("$x$"); axes[2].set_ylabel("$y$")
    plt.colorbar(im, ax=axes[2], shrink=0.8)
    for ax in axes:
        ax.plot(0, 0, "w+", ms=12, mew=2)

    fig.suptitle(
        r"Interior solution — L-shape, manufactured $\sigma_\mathrm{mfg}$" + "\n"
        f"SE-BINN★ rel L2 = {final_out.rel_L2:.3e}  |  "
        f"u_exact rel L2 = {bem_out.rel_L2:.3e}",
        fontsize=12,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved interior_error_singular → {outpath}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("=" * 72)
    print("  Experiment 2b: SE-BINN on L-shape — manufactured singular density")
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
    print(f"  singular corners: {list(geom.singular_corner_indices)} "
          f"(ω={geom.corner_angles[3]/np.pi:.4f}π, α={alpha_sing:.4f})")
    print(f"  Koch comparison: α_Koch={np.pi/(4*np.pi/3)-1:.4f}  →  "
          f"L-shape singularity IS stronger ({abs(alpha_sing):.4f} > {abs(np.pi/(4*np.pi/3)-1):.4f})")

    qdata = build_panel_quadrature(panels, p=CFG["p_gl"])
    Yq_T  = qdata.Yq.T   # (Nq, 2)
    wq    = qdata.wq
    Nq    = qdata.n_quad

    arc, vertex_arcs, total_arc = _boundary_arclength(qdata, CFG["n_per_edge"])
    sort_idx = np.argsort(arc)
    print(f"  Quadrature: Nq={Nq} | total arc-length={total_arc:.4f}")

    # ------------------------------------------------------------------
    # 2. Singular enrichment basis
    # ------------------------------------------------------------------
    enrichment  = SingularEnrichment(geom=geom, per_corner_gamma=False)
    sigma_s_Yq  = enrichment.precompute(Yq_T)   # (Nq,)
    print(f"\n  σ_s range: [{sigma_s_Yq.min():.4f}, {sigma_s_Yq.max():.4f}]")
    print(f"  σ_s at reentrant corner vicinity: "
          f"r_min ≈ {np.linalg.norm(Yq_T - P[3], axis=1).min():.5f}")

    # ------------------------------------------------------------------
    # 3. Manufactured density
    # ------------------------------------------------------------------
    print("\n--- Manufactured density ---")
    gamma_true  = CFG["gamma_true"]
    sigma_mfg   = make_manufactured_density(sigma_s_Yq, arc, total_arc, gamma_true)
    sigma_smooth = np.sin(2.0 * np.pi * arc / total_arc)

    # Check enrichment energy fraction of the manufactured density
    s_proj  = gamma_true * sigma_s_Yq
    res_mfg = sigma_mfg - s_proj   # = sigma_smooth
    energy_true = float(np.linalg.norm(s_proj)**2 / max(np.linalg.norm(sigma_mfg)**2, 1e-14))
    print(f"  γ_true = {gamma_true:.4f}")
    print(f"  ‖σ_s‖    = {np.linalg.norm(sigma_s_Yq):.4f}")
    print(f"  ‖σ_smooth‖ = {np.linalg.norm(sigma_smooth):.4f}")
    print(f"  ‖σ_mfg‖  = {np.linalg.norm(sigma_mfg):.4f}")
    print(f"  Enrichment energy (γ_true²‖σ_s‖²/‖σ_mfg‖²) = {energy_true*100:.2f}%")

    # ------------------------------------------------------------------
    # 4. BEM reference: solve V σ_BEM = g where g = V σ_mfg
    # ------------------------------------------------------------------
    print("\n--- BEM reference (manufactured) ---")
    nmat     = assemble_nystrom_matrix(qdata)
    g_bnd    = nmat.V @ sigma_mfg       # g = V σ_mfg (RHS from manufactured density)
    bem_sol  = solve_bem(nmat, g_bnd,
                         tol=CFG["gmres_tol"], max_iter=CFG["gmres_maxiter"])
    sigma_bem = bem_sol.sigma
    print(f"  GMRES: flag={bem_sol.flag} | rel_res={bem_sol.rel_res:.3e} | "
          f"iters={bem_sol.n_iter}")

    # σ_BEM should equal σ_mfg exactly
    sigma_recovery_err = float(np.linalg.norm(sigma_bem - sigma_mfg)
                               / max(np.linalg.norm(sigma_mfg), 1e-14))
    print(f"  ‖σ_BEM − σ_mfg‖/‖σ_mfg‖ = {sigma_recovery_err:.3e}  "
          "(should be ≈ machine ε)")
    print(f"  σ_BEM max|σ| = {np.max(np.abs(sigma_bem)):.4f}")

    # u_exact: single-layer integral with σ_mfg
    u_exact = make_u_exact_fn(Yq_T, wq, sigma_mfg)

    # Check interior reconstruction with the exact σ_mfg
    bem_out = reconstruct_interior(
        P=P, Yq=Yq_T, wq=wq, sigma=sigma_mfg,
        n_grid=CFG["n_grid_final"], u_exact=u_exact,
        x_range=(-1.0, 1.0), y_range=(-1.0, 1.0),
    )
    print(f"  u_exact interior self-check: rel_L2={bem_out.rel_L2:.3e}  "
          "(should be ≈ 0 since u_exact is computed from σ_mfg directly)")

    # σ_BEM near the reentrant corner
    corner_v = P[3]
    dist_to_corner = np.linalg.norm(Yq_T - corner_v[None, :], axis=1)
    near_idx = np.argsort(dist_to_corner)[:5]
    print(f"\n  σ_BEM near (0,0) reentrant corner:")
    for i in near_idx:
        r_i  = dist_to_corner[i]
        s_i  = sigma_bem[i]
        ss_i = sigma_s_Yq[i]
        if abs(ss_i) > 1e-12:
            ratio = s_i / ss_i
            print(f"    r={r_i:.5f}  σ_BEM={s_i:+.4f}  σ_s={ss_i:+.4f}  "
                  f"σ_BEM/σ_s={ratio:+.4f}  (expected ≈ {gamma_true:.4f})")

    # ------------------------------------------------------------------
    # 5. Enrichment diagnostic (lstsq projection onto σ_s)
    # ------------------------------------------------------------------
    print("\n--- Enrichment diagnostic ---")
    gamma_lstsq = float(
        np.dot(sigma_s_Yq, sigma_bem) / max(np.dot(sigma_s_Yq, sigma_s_Yq), 1e-14)
    )
    sigma_proj = gamma_lstsq * sigma_s_Yq
    res_norm   = np.linalg.norm(sigma_bem - sigma_proj)
    bem_norm   = np.linalg.norm(sigma_bem)
    energy_frac = 1.0 - (res_norm / max(bem_norm, 1e-14)) ** 2

    print(f"  γ*_lstsq              : {gamma_lstsq:.6f}  (γ_true = {gamma_true:.6f})")
    print(f"  |γ*_lstsq − γ_true|   : {abs(gamma_lstsq - gamma_true):.3e}  (should be ≈ 0)")
    print(f"  Enrichment energy     : {energy_frac*100:.4f}%  "
          f"(Koch comparison: ~2.8%)")
    print(f"  Residual ‖σ_BEM − γ*σ_s‖/‖σ_BEM‖ : {res_norm/max(bem_norm,1e-14):.4f}")

    if energy_frac > 0.028:
        print(f"  → Enrichment fraction {energy_frac*100:.2f}% > Koch 2.8%  ✓")
    if abs(gamma_lstsq - gamma_true) < 0.1:
        print(f"  → γ*_lstsq ≈ γ_true  ✓")

    # ------------------------------------------------------------------
    # 6. Shared operator state
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

    # ------------------------------------------------------------------
    # 7. Shared initial model state
    # ------------------------------------------------------------------
    torch.manual_seed(CFG["seed"])
    init_model = SEBINNModel(
        hidden_width=CFG["hidden_width"], n_hidden=CFG["n_hidden"],
        n_gamma=enrichment.n_gamma, gamma_init=CFG["gamma_init"],
        dtype=torch.float64,
    )
    init_state = copy.deepcopy(init_model.state_dict())
    print(f"  Model: n_params={init_model.n_params()} | n_gamma={enrichment.n_gamma}")
    print(f"  Shared initial state saved.")

    shared = dict(
        op=op, enrichment=enrichment,
        Yq_T=Yq_T, wq=wq, P=P,
        sigma_bem=sigma_bem, sigma_s_Yq=sigma_s_Yq,
        sort_idx=sort_idx, u_exact=u_exact,
    )

    # ------------------------------------------------------------------
    # 8. Case A: BINN (γ frozen at 0)
    # ------------------------------------------------------------------
    res_a = _train_case("BINN", freeze_gamma=True, gamma_override=None,
                        init_state=init_state, shared=shared, verbose=True)

    # ------------------------------------------------------------------
    # 9. Case B: SE-BINN (γ trainable, init=0)
    # ------------------------------------------------------------------
    res_b = _train_case("SE-BINN", freeze_gamma=False, gamma_override=None,
                        init_state=init_state, shared=shared, verbose=True)

    # ------------------------------------------------------------------
    # 10. Case C: SE-BINN★ (γ trainable, warm start from γ_true)
    # ------------------------------------------------------------------
    res_c = _train_case("SE-BINN★", freeze_gamma=False,
                        gamma_override=gamma_lstsq,
                        init_state=init_state, shared=shared, verbose=True)

    # ------------------------------------------------------------------
    # 11. Summary
    # ------------------------------------------------------------------
    cases = [res_a, res_b, res_c]
    _print_table(cases, bem_out.rel_L2, bem_out.linf, gamma_lstsq, energy_frac, gamma_true)

    best = min(cases, key=lambda c: c["final_rel_L2"])
    print(f"\n  Best case: {best['label']} | final rel_L2={best['final_rel_L2']:.3e}")

    # Hypothesis check
    print("\n--- Hypothesis check ---")
    print(f"  1. BEM error ≪ 1%: rel_L2={sigma_recovery_err:.3e}  "
          f"{'✓' if sigma_recovery_err<0.01 else '✗'}")
    print(f"  2. γ*_lstsq ≈ γ_true: {gamma_lstsq:.4f} vs {gamma_true:.4f}  "
          f"{'✓' if abs(gamma_lstsq-gamma_true)<0.1 else '✗'}")
    print(f"  3. Enrichment energy > Koch 2.8%: {energy_frac*100:.2f}%  "
          f"{'✓' if energy_frac>0.028 else '✗'}")
    rL2 = [c["final_rel_L2"] for c in cases]
    ab_ratio = rL2[0] / rL2[1] if rL2[1] > 0 else float("nan")
    print(f"  4. A/B gap > Koch 12% (i.e. >1.12×): {ab_ratio:.2f}×  "
          f"{'✓' if ab_ratio>1.12 else '✗'}")

    total_wall = time.perf_counter() - t_global
    print(f"\n  Total wall time: {total_wall:.1f}s")

    # ------------------------------------------------------------------
    # 12. Figures
    # ------------------------------------------------------------------
    print("\n--- Generating figures ---")
    sing_idx = list(geom.singular_corner_indices)

    _fig_ab_comparison(
        cases=cases,
        sigma_bem_sorted=sigma_bem[sort_idx],
        arc=arc[sort_idx],
        vertex_arcs=vertex_arcs,
        singular_idx=sing_idx,
        gamma_true=gamma_true,
        outpath=os.path.join(figures_dir, "ab_comparison_singular.png"),
    )
    _fig_convergence(
        cases=cases, gamma_true=gamma_true,
        outpath=os.path.join(figures_dir, "convergence_singular.png"),
    )
    _fig_interior(
        bem_out=reconstruct_interior(
            P=P, Yq=Yq_T, wq=wq, sigma=sigma_mfg,
            n_grid=CFG["n_grid_final"], u_exact=u_exact,
            x_range=(-1.0, 1.0), y_range=(-1.0, 1.0),
        ),
        best_case=best,
        outpath=os.path.join(figures_dir, "interior_error_singular.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
