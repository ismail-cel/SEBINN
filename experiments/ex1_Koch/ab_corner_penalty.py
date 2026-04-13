"""
Corner-penalty experiment: Koch(1), g = x² − y².

We test whether a direct regularisation penalty on σ_w near the
reentrant corners breaks the degeneracy that causes the density plateau.

Setup
-----
Domain: Koch(1) snowflake, 6 reentrant corners (ω = 4π/3, α = −1/4).
BIE:    Single-layer potential, Dirichlet BC g = x² − y².

Three cases (all same initial weights):
  A — BINN       : γ frozen at 0, no penalty
  B — SE-BINN    : γ trainable (per-corner), no penalty
  C — SE-BINN+λ  : γ trainable (per-corner), corner penalty with best λ

λ sweep over {0.01, 0.1, 1.0, 10.0} is run first to find the best λ.

Loss with corner penalty:
    L_total = L_BIE + λ · mean(σ_w(x)²)  for x near corners

The penalty acts on σ_w ONLY, forcing singular behavior into γσ_s.

Figures saved to experiments/ex1_Koch/figures/:
  corner_penalty_sweep.png        — rel-diff / interior L2 / σ_w RMS vs λ
  corner_penalty_ab.png           — loss / density / density error (A, B, C)
  corner_penalty_density_detail.png — corner zoom (A, B, C, BEM)
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
from src.training.operator import build_operator_state, select_corner_points
from src.training.loss import make_loss_fn
from src.training.adam_phase import AdamConfig, run_adam_phases
from src.training.lbfgs import LBFGSConfig, run_lbfgs
from src.reconstruction.interior import reconstruct_interior


# ===========================================================================
# Configuration
# ===========================================================================

CFG = dict(
    seed           = 0,
    # geometry
    n_per_edge     = 12,
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
    n_grid         = 201,
)


def u_exact(xy: np.ndarray) -> np.ndarray:
    return xy[:, 0] ** 2 - xy[:, 1] ** 2


# ===========================================================================
# Arc-length helper
# ===========================================================================

def _boundary_arclength(qdata, n_per_edge: int):
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
# LBFGSConfig builder
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
    verbose: bool = True,
) -> dict:
    """
    One full training run (3 Adam phases + L-BFGS).

    Returns a result dict with loss_hist_adam, loss_hist_lbfgs,
    sigma_final (sorted by arc), final_rel_L2, density_rel_diff, gamma_vals,
    sigma_w_rms_corners, bie_loss_final, etc.
    """
    if verbose:
        fz  = " [γ frozen]" if freeze_gamma else " [γ trainable]"
        pen = f" λ={lambda_corner}" if lambda_corner > 0 else ""
        print(f"\n{'='*62}")
        print(f"  Case {label}{fz}{pen}")
        print(f"{'='*62}")

    t0 = time.perf_counter()

    op          = shared["op"]
    enrichment  = shared["enrichment"]
    Yq_T        = shared["Yq_T"]
    wq          = shared["wq"]
    P           = shared["P"]
    sigma_bem   = shared["sigma_bem"]
    sigma_s_Yq  = shared["sigma_s_Yq"]
    sort_idx    = shared["sort_idx"]
    corner_pts_t = shared["corner_pts_t"]       # Tensor (Nc, 2)
    corner_ss_t  = shared["corner_ss_t"]        # Tensor (Nc,) or (Nc, ng)

    # Build model
    model = SEBINNModel(
        hidden_width = CFG["hidden_width"],
        n_hidden     = CFG["n_hidden"],
        n_gamma      = enrichment.n_gamma,
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
        print(f"  trainable params: {n_tr} | Nc={len(corner_pts_t)}")

    # Build loss function
    loss_fn = make_loss_fn(
        corner_points  = corner_pts_t  if lambda_corner > 0 else None,
        corner_sigma_s = corner_ss_t   if lambda_corner > 0 else None,
        lambda_corner  = lambda_corner,
    )

    Yq_t     = torch.tensor(Yq_T, dtype=torch.float64)
    sigma_s_t = torch.tensor(sigma_s_Yq, dtype=torch.float64)

    # Adam — 3 phases
    all_adam_loss = []
    global_adam_iter = 0
    for ph_idx, (n_it, lr) in enumerate(zip(CFG["adam_iters"], CFG["adam_lrs"])):
        ph_cfg = AdamConfig(
            phase_iters=[n_it], phase_lrs=[lr], log_every=CFG["log_every"],
        )
        ph_res = run_adam_phases(model, op, ph_cfg, verbose=verbose, loss_fn=loss_fn)
        all_adam_loss.extend(ph_res.loss_hist)
        global_adam_iter += ph_res.n_iters

    # L-BFGS
    lbfgs_cfg = _make_lbfgs_cfg()
    if verbose:
        print(f"\n  [{label}] L-BFGS: max={CFG['lbfgs_max_iters']} | "
              f"mem={CFG['lbfgs_memory']} | grad_tol={CFG['lbfgs_grad_tol']:.0e}")
    lbfgs_res = run_lbfgs(model, op, lbfgs_cfg, verbose=verbose, loss_fn=loss_fn)

    # Final density
    with torch.no_grad():
        sigma_final = model(Yq_t, sigma_s_t).numpy()
        sigma_w_corners_np = model.sigma_w(corner_pts_t).numpy()

    sigma_w_rms_corners = float(np.sqrt(np.mean(sigma_w_corners_np ** 2)))

    final_out = reconstruct_interior(
        P=P, Yq=Yq_T, wq=wq, sigma=sigma_final,
        n_grid=CFG["n_grid"], u_exact=u_exact,
    )

    density_rel_diff = float(
        np.linalg.norm(sigma_final - sigma_bem)
        / max(np.linalg.norm(sigma_bem), 1e-14)
    )

    # BIE loss component (without penalty, for fair comparison across λ)
    with torch.no_grad():
        from src.training.loss import sebinn_loss as _bie_only
        _, dbg_bie = _bie_only(model, op)
    bie_loss_final = dbg_bie["loss"]

    t_total = time.perf_counter() - t0

    if verbose:
        g = model.gamma_value()
        gfmt = f"[{','.join(f'{v:.4f}' for v in g)}]" if isinstance(g, list) else f"{float(g):.6f}"
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
        label              = label,
        lambda_corner      = lambda_corner,
        freeze_gamma       = freeze_gamma,
        final_rel_L2       = final_out.rel_L2,
        final_linf         = final_out.linf,
        bie_loss_final     = bie_loss_final,
        density_rel_diff   = density_rel_diff,
        sigma_w_rms_corners= sigma_w_rms_corners,
        gamma_vals         = model.gamma_value(),
        lbfgs_reason       = lbfgs_res.reason,
        wall_time          = t_total,
        loss_hist_adam     = all_adam_loss,
        loss_hist_lbfgs    = list(lbfgs_res.loss_hist),
        adam_n_iters       = global_adam_iter,
        sigma_final        = sigma_final[sort_idx],
        sigma_w_corners    = sigma_w_corners_np,
        final_out          = final_out,
    )


# ===========================================================================
# λ sweep (lightweight: same Adam phases + L-BFGS, silent)
# ===========================================================================

def _run_sweep(lambdas, init_state, shared, verbose=False):
    """
    For each λ in lambdas, train SE-BINN with that penalty weight.
    Returns a list of result dicts (same format as _train_one).
    """
    results = []
    for lam in lambdas:
        label = f"SE-BINN λ={lam:.2g}"
        print(f"  Sweep: λ = {lam:.3g} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        res = _train_one(
            label        = label,
            freeze_gamma = False,
            init_state   = init_state,
            shared       = shared,
            lambda_corner= lam,
            verbose      = verbose,
        )
        print(f"done in {time.perf_counter()-t0:.1f}s | "
              f"d_err={res['density_rel_diff']:.4f} | "
              f"iL2={res['final_rel_L2']:.3e} | "
              f"σ_w_rms={res['sigma_w_rms_corners']:.4f}")
        results.append(res)
    return results


# ===========================================================================
# Tables
# ===========================================================================

def _print_sweep_table(sweep_results):
    print()
    print("=" * 90)
    print("  λ SWEEP RESULTS  —  Koch(1), g = x² − y², SE-BINN with corner penalty")
    print("=" * 90)
    hdr = (f"  {'λ':>8} | {'density rel-diff':>18} | {'interior rel L2':>16} | "
           f"{'BIE loss':>12} | {'σ_w RMS corners':>16} | {'γ_c pattern':>20}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in sweep_results:
        lam = r["lambda_corner"]
        g   = r["gamma_vals"]
        if isinstance(g, list):
            gstr = "[" + ",".join(f"{v:.2f}" for v in g) + "]"
        else:
            gstr = f"{float(g):.4f}"
        print(f"  {lam:>8.3g} | {r['density_rel_diff']:>18.4f} | "
              f"{r['final_rel_L2']:>16.3e} | {r['bie_loss_final']:>12.3e} | "
              f"{r['sigma_w_rms_corners']:>16.4f} | {gstr:>20}")
    print("=" * 90)


def _print_abc_table(res_a, res_b, res_c, bem_rel_L2, best_lambda):
    print()
    print("=" * 80)
    print(f"  A/B/C COMPARISON  Koch(1), g = x²−y²  (best λ = {best_lambda:.3g})")
    print("=" * 80)
    cases = [res_a, res_b, res_c]
    cols  = [c["label"] for c in cases]
    w     = 18

    hdr = f"  {'Metric':<24} | " + " | ".join(f"{c:>{w}}" for c in cols)
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr); print(sep)

    def row(name, key, fmt=".3e"):
        vals = [f"{c[key]:{fmt}}" for c in cases]
        print(f"  {name:<24} | " + " | ".join(f"{v:>{w}}" for v in vals))

    row("Interior rel L2",   "final_rel_L2")
    row("Interior L∞",       "final_linf")
    row("BIE loss",          "bie_loss_final")
    row("Density rel-diff",  "density_rel_diff")
    row("σ_w RMS corners",   "sigma_w_rms_corners")
    print(sep)

    gvs = []
    for c in cases:
        g = c["gamma_vals"]
        if c["freeze_gamma"]:
            gvs.append("(frozen=0)")
        elif isinstance(g, list):
            gvs.append("[" + ",".join(f"{v:.3f}" for v in g) + "]")
        else:
            gvs.append(f"{float(g):.6f}")
    print(f"  {'γ final':<24} | " + " | ".join(f"{v:>{w}}" for v in gvs))

    rvs = [c["lbfgs_reason"] for c in cases]
    print(f"  {'LBFGS reason':<24} | " + " | ".join(f"{v:>{w}}" for v in rvs))

    wvs = [f"{c['wall_time']:.1f}s" for c in cases]
    print(f"  {'Wall time':<24} | " + " | ".join(f"{v:>{w}}" for v in wvs))

    print(sep)
    print(f"  {'BEM ref rel L2':<24} | " + " | ".join(f"{bem_rel_L2:>{w}.3e}" for _ in cases))
    print(sep)

    rL2 = [c["final_rel_L2"] for c in cases]
    if rL2[0] > 0 and rL2[1] > 0:
        print(f"\n  A/B improvement (BINN → SE-BINN)      : {rL2[0]/rL2[1]:.2f}×")
    if rL2[0] > 0 and rL2[2] > 0:
        print(f"  A/C improvement (BINN → SE-BINN+λ)    : {rL2[0]/rL2[2]:.2f}×")
    de = [c["density_rel_diff"] for c in cases]
    if de[0] > 0 and de[1] > 0:
        print(f"  A/B density improvement                 : {de[0]/de[1]:.2f}×")
    if de[0] > 0 and de[2] > 0:
        print(f"  A/C density improvement (with penalty)  : {de[0]/de[2]:.2f}×")
    print("=" * 80)


# ===========================================================================
# Figures
# ===========================================================================

def _fig_sweep(sweep_results, outpath):
    """
    3-panel figure: density rel-diff / interior L2 / σ_w RMS vs λ.
    λ=0 is the no-penalty SE-BINN (dashed reference line).
    """
    lambdas  = [r["lambda_corner"] for r in sweep_results]
    d_errs   = [r["density_rel_diff"]    for r in sweep_results]
    il2s     = [r["final_rel_L2"]        for r in sweep_results]
    sw_rms   = [r["sigma_w_rms_corners"] for r in sweep_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.35)

    # Separate λ=0 (reference) from the rest
    lam_plot  = [l for l in lambdas if l > 0]
    d_plot    = [d for l, d in zip(lambdas, d_errs)  if l > 0]
    il2_plot  = [v for l, v in zip(lambdas, il2s)    if l > 0]
    sw_plot   = [v for l, v in zip(lambdas, sw_rms)  if l > 0]
    d0        = next(d for l, d in zip(lambdas, d_errs)  if l == 0)
    il2_0     = next(v for l, v in zip(lambdas, il2s)    if l == 0)
    sw_0      = next(v for l, v in zip(lambdas, sw_rms)  if l == 0)

    def _panel(ax, ys, y0, ylabel, title, ymin_zero=False):
        ax.semilogx(lam_plot, ys, "o-", color="#d62728", lw=1.5, ms=6)
        ax.axhline(y0, color="gray", lw=1.0, ls="--", alpha=0.7, label="λ=0 (no penalty)")
        ax.set_xlabel("λ (penalty weight)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9); ax.grid(True, which="both", lw=0.3, alpha=0.5)
        if ymin_zero:
            ax.set_ylim(bottom=0)

    _panel(axes[0], d_plot,  d0,   r"$\|\sigma_\theta - \sigma_\mathrm{BEM}\|/\|\sigma_\mathrm{BEM}\|$",
           "(a) Density rel-diff vs λ", ymin_zero=True)
    _panel(axes[1], il2_plot, il2_0, "Interior rel L2",
           "(b) Interior rel L2 vs λ", ymin_zero=True)
    _panel(axes[2], sw_plot, sw_0, r"$\sigma_w$ RMS at corners",
           r"(c) $\sigma_w$ RMS at corners vs λ")

    fig.suptitle(r"Corner-penalty sweep — Koch(1), $g = x^2 - y^2$", fontsize=13)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved corner_penalty_sweep → {outpath}")


def _fig_abc(res_a, res_b, res_c, sigma_bem_sorted, arc, vertex_arcs,
             singular_idx, best_lambda, outpath):
    """
    3-subplot: loss history / density / density error for cases A, B, C.
    """
    COLORS = {"BINN": "#1f77b4", "SE-BINN": "#d62728"}
    def _col(label):
        if "BINN" in label and "SE" not in label:
            return "#1f77b4"
        if "+" in label or "λ" in label.lower():
            return "#2ca02c"
        return "#d62728"

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.subplots_adjust(hspace=0.42)

    # Loss history
    ax = axes[0]
    for c in [res_a, res_b, res_c]:
        ha, hl = c["loss_hist_adam"], c["loss_hist_lbfgs"]
        n_a, n_l = len(ha), len(hl)
        col = _col(c["label"])
        ax.semilogy(np.arange(1, n_a+1), ha, color=col, lw=1.4, ls="-", alpha=0.85,
                    label=f"{c['label']} (Adam)")
        if n_l:
            ax.semilogy(np.arange(n_a+1, n_a+n_l+1), hl, color=col, lw=1.4,
                        ls="--", alpha=0.85, label=f"{c['label']} (L-BFGS)")
    ax.axvline(res_a["adam_n_iters"], color="gray", lw=0.8, ls=":", alpha=0.6,
               label=f"Adam→LBFGS (iter {res_a['adam_n_iters']})")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
    ax.set_title("(a) Loss history (BIE component shown)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    # Density
    ax = axes[1]
    for ci in singular_idx:
        ax.axvline(vertex_arcs[ci], color="#999999", lw=0.7, ls="--", alpha=0.6)
    ax.plot(arc, sigma_bem_sorted, "k-", lw=1.2, alpha=0.9,
            label=r"$\sigma_\mathrm{BEM}$")
    for c, ls in zip([res_a, res_b, res_c], ["-", "--", ":"]):
        ax.plot(arc, c["sigma_final"], color=_col(c["label"]),
                lw=1.0, ls=ls, alpha=0.85, label=c["label"])
    ax.set_xlabel("Arc-length $s$"); ax.set_ylabel(r"$\sigma(s)$")
    ax.set_title("(b) Boundary density", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, lw=0.3, alpha=0.5)

    # Density error
    ax = axes[2]
    for ci in singular_idx:
        ax.axvline(vertex_arcs[ci], color="#999999", lw=0.7, ls="--", alpha=0.6)
    for c, ls in zip([res_a, res_b, res_c], ["-", "--", ":"]):
        err = np.abs(c["sigma_final"] - sigma_bem_sorted)
        ax.semilogy(arc, err + 1e-16, color=_col(c["label"]),
                    lw=1.0, ls=ls, alpha=0.85, label=c["label"])
    ax.set_xlabel("Arc-length $s$")
    ax.set_ylabel(r"$|\sigma_\theta - \sigma_\mathrm{BEM}|$")
    ax.set_title("(c) Pointwise density error", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(
        f"A/B/C Comparison — Koch(1), $g = x^2-y^2$, best λ = {best_lambda:.3g}\n"
        "BINN (γ=0) | SE-BINN | SE-BINN+λ",
        fontsize=12, y=1.01,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved corner_penalty_ab → {outpath}")


def _fig_corner_zoom(res_a, res_b, res_c, sigma_bem_sorted,
                     arc, vertex_arcs, singular_idx, outpath):
    """
    Zoom into a reentrant corner region: arc ± window around each singular vertex.
    Shows all 4 densities (BEM, A, B, C) on a single axis per corner.
    Only the first singular corner is plotted for clarity.
    """
    if len(singular_idx) == 0:
        return

    # Use first singular corner only
    ci     = singular_idx[0]
    s_c    = vertex_arcs[ci]
    total_arc = vertex_arcs[-1]
    # Window: ± half of a vertex-to-vertex spacing
    if len(vertex_arcs) >= 2:
        spacing = (vertex_arcs[-1] - vertex_arcs[0]) / max(len(vertex_arcs) - 1, 1)
        w = spacing * 0.4
    else:
        w = total_arc * 0.1
    lo, hi = s_c - w, s_c + w

    mask = (arc >= lo) & (arc <= hi)
    if mask.sum() < 2:
        # Fallback: slightly wider window
        w   *= 2
        lo   = s_c - w
        hi   = s_c + w
        mask = (arc >= lo) & (arc <= hi)

    fig, ax = plt.subplots(figsize=(9, 5))

    def _col(label):
        if "BINN" in label and "SE" not in label:
            return "#1f77b4"
        if "+" in label or "λ" in label.lower():
            return "#2ca02c"
        return "#d62728"

    ax.plot(arc[mask], sigma_bem_sorted[mask], "k-", lw=1.8, alpha=0.9,
            label=r"$\sigma_\mathrm{BEM}$ (reference)")
    for c, ls in zip([res_a, res_b, res_c], ["-", "--", ":"]):
        ax.plot(arc[mask], c["sigma_final"][mask], color=_col(c["label"]),
                lw=1.3, ls=ls, alpha=0.9, label=c["label"])

    ax.axvline(s_c, color="gray", lw=0.8, ls="--", alpha=0.6, label="corner")
    ax.set_xlabel("Arc-length $s$", fontsize=12)
    ax.set_ylabel(r"$\sigma(s)$", fontsize=12)
    ax.set_title(
        f"Corner zoom — Koch(1) reentrant corner #{ci}\n"
        r"$\sigma_\mathrm{BEM}$ vs BINN / SE-BINN / SE-BINN+λ",
        fontsize=12,
    )
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, lw=0.3, alpha=0.5)

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved corner_penalty_density_detail → {outpath}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("=" * 70)
    print("  Koch(1) Corner-Penalty Experiment")
    print("=" * 70)

    t_global = time.perf_counter()
    figures_dir = os.path.join(_HERE, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Geometry and quadrature
    # ------------------------------------------------------------------
    print("\n--- Setup ---")
    geom    = make_koch_geometry(n=1)
    P       = geom.vertices
    panels  = build_uniform_panels(P, n_per_edge=CFG["n_per_edge"])
    label_corner_ring_panels(panels, P)

    qdata = build_panel_quadrature(panels, p=CFG["p_gl"])
    Yq_T  = qdata.Yq.T
    wq    = qdata.wq
    Nq    = qdata.n_quad
    arc, vertex_arcs, total_arc = _boundary_arclength(qdata, CFG["n_per_edge"])
    sort_idx = np.argsort(arc)

    alpha_sing = np.pi / geom.corner_angles[geom.singular_corner_indices[0]] - 1
    print(f"  Koch(1): vertices={geom.n_vertices} | panels={len(panels)} | "
          f"singular={len(geom.singular_corner_indices)} corners")
    print(f"  α = {alpha_sing:.4f}  |  Nq={Nq} | arc={total_arc:.4f}")

    # ------------------------------------------------------------------
    # 2. BEM reference
    # ------------------------------------------------------------------
    print("\n--- BEM reference ---")
    nmat     = assemble_nystrom_matrix(qdata)
    f_bnd    = u_exact(Yq_T)
    bem_sol  = solve_bem(nmat, f_bnd, tol=CFG["gmres_tol"],
                         max_iter=CFG["gmres_maxiter"])
    sigma_bem = bem_sol.sigma
    print(f"  GMRES: flag={bem_sol.flag} | rel_res={bem_sol.rel_res:.3e}")

    bem_out = reconstruct_interior(
        P=P, Yq=Yq_T, wq=wq, sigma=sigma_bem,
        n_grid=CFG["n_grid"], u_exact=u_exact,
    )
    print(f"  BEM interior: rel_L2={bem_out.rel_L2:.3e} | linf={bem_out.linf:.3e}")
    if bem_out.rel_L2 > 1e-2:
        raise RuntimeError(f"BEM quality too low: {bem_out.rel_L2:.3e}")

    # ------------------------------------------------------------------
    # 3. Enrichment setup
    # ------------------------------------------------------------------
    enrichment = SingularEnrichment(geom=geom, per_corner_gamma=True)
    sigma_s_Yq = enrichment.precompute(Yq_T)   # (Nq,) or (Nq, nc)

    # Lstsq enrichment energy
    ss_flat = sigma_s_Yq if sigma_s_Yq.ndim == 1 else sigma_s_Yq.sum(axis=1)
    gamma_lstsq = float(np.dot(ss_flat, sigma_bem) / max(np.dot(ss_flat, ss_flat), 1e-14))
    proj = gamma_lstsq * ss_flat
    energy_frac = 1.0 - (np.linalg.norm(sigma_bem - proj)
                         / max(np.linalg.norm(sigma_bem), 1e-14)) ** 2
    print(f"\n  γ*_lstsq = {gamma_lstsq:.4f}  |  Enrichment energy = {energy_frac*100:.2f}%")

    # ------------------------------------------------------------------
    # 4. Corner points for penalty
    # ------------------------------------------------------------------
    corner_indices = select_corner_points(qdata, geom, radius_factor=CFG["radius_factor"])
    Nc = len(corner_indices)
    print(f"  Corner penalty: Nc={Nc} nodes within R={CFG['radius_factor']:.2f}×mean_edge "
          f"of any reentrant corner")

    Yq_corner_np = Yq_T[corner_indices]        # (Nc, 2)
    sigma_s_corner_np = enrichment.precompute(Yq_corner_np)  # (Nc,) or (Nc, ng)

    corner_pts_t = torch.tensor(Yq_corner_np,     dtype=torch.float64)
    corner_ss_t  = torch.tensor(sigma_s_corner_np, dtype=torch.float64)

    # ------------------------------------------------------------------
    # 5. Operator state
    # ------------------------------------------------------------------
    print("\n--- Operator setup ---")
    w_panel = panel_loss_weights(panels, w_base=CFG["w_base"],
                                  w_corner=CFG["w_corner"], w_ring=CFG["w_ring"])
    colloc  = build_collocation_points(panels, m_col_panel=CFG["m_col_base"])
    op, op_diag = build_operator_state(
        colloc=colloc, qdata=qdata, enrichment=enrichment, g=u_exact,
        panel_weights=w_panel,
        eq_scale_mode=CFG["eq_scale_mode"], eq_scale_fixed=CFG["eq_scale_fixed"],
        dtype=torch.float64, device="cpu",
    )
    print(f"  Nb={colloc.n_colloc} | eq_scale={op_diag['eq_scale']:.2e} | "
          f"mean|A|={op_diag['mean_abs_A_before']:.3e}")

    # ------------------------------------------------------------------
    # 6. Shared initial model state
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
        sort_idx=sort_idx,
        corner_pts_t=corner_pts_t, corner_ss_t=corner_ss_t,
    )

    # ------------------------------------------------------------------
    # 7. λ sweep
    # ------------------------------------------------------------------
    print("\n--- λ sweep ---")
    sweep_results = _run_sweep(CFG["lambda_sweep"], init_state, shared, verbose=False)
    _print_sweep_table(sweep_results)

    # Pick best λ: lowest density rel-diff (among λ > 0 entries)
    nonzero = [(r, r["density_rel_diff"]) for r in sweep_results if r["lambda_corner"] > 0]
    best_res_sweep, best_d = min(nonzero, key=lambda x: x[1])
    best_lambda = best_res_sweep["lambda_corner"]
    print(f"\n  Best λ = {best_lambda:.3g}  "
          f"(density rel-diff = {best_d:.4f})")

    # ------------------------------------------------------------------
    # 8. Full A/B/C comparison
    # ------------------------------------------------------------------
    print("\n--- Full A/B/C comparison ---")

    res_a = _train_one("BINN",       freeze_gamma=True,  init_state=init_state,
                       shared=shared, lambda_corner=0.0,         verbose=True)
    res_b = _train_one("SE-BINN",    freeze_gamma=False, init_state=init_state,
                       shared=shared, lambda_corner=0.0,         verbose=True)
    res_c = _train_one(f"SE-BINN+λ={best_lambda:.3g}",
                       freeze_gamma=False, init_state=init_state,
                       shared=shared, lambda_corner=best_lambda, verbose=True)

    _print_abc_table(res_a, res_b, res_c, bem_out.rel_L2, best_lambda)

    # Hypothesis check
    print("\n--- Hypothesis check ---")
    print(f"  1. σ_w RMS decreases with λ:  "
          f"{sweep_results[0]['sigma_w_rms_corners']:.4f} (λ=0) → "
          f"{best_res_sweep['sigma_w_rms_corners']:.4f} (λ={best_lambda:.3g})  "
          f"{'✓' if best_res_sweep['sigma_w_rms_corners'] < sweep_results[0]['sigma_w_rms_corners'] else '✗'}")
    de = [res_a["density_rel_diff"], res_b["density_rel_diff"], res_c["density_rel_diff"]]
    print(f"  2. Penalty decreases density error vs SE-BINN:  "
          f"B={de[1]:.4f} → C={de[2]:.4f}  "
          f"{'✓' if de[2] < de[1] else '✗'}")
    print(f"  3. Interior error maintained:  "
          f"A={res_a['final_rel_L2']:.3e} B={res_b['final_rel_L2']:.3e} "
          f"C={res_c['final_rel_L2']:.3e}  "
          f"{'✓' if res_c['final_rel_L2'] <= res_a['final_rel_L2'] * 2 else '? (degraded)' }")
    gc = res_c["gamma_vals"]
    gb = res_b["gamma_vals"]
    gc_mean = float(np.mean(gc)) if isinstance(gc, list) else float(gc)
    gb_mean = float(np.mean(gb)) if isinstance(gb, list) else float(gb)
    print(f"  4. γ pushed toward lstsq value {gamma_lstsq:.4f}:  "
          f"B_mean={gb_mean:.4f} → C_mean={gc_mean:.4f}  "
          f"{'✓' if abs(gc_mean - gamma_lstsq) < abs(gb_mean - gamma_lstsq) else '✗'}")

    total_wall = time.perf_counter() - t_global
    print(f"\n  Total wall time: {total_wall:.1f}s")

    # ------------------------------------------------------------------
    # 9. Figures
    # ------------------------------------------------------------------
    print("\n--- Generating figures ---")
    sing_idx = list(geom.singular_corner_indices)

    _fig_sweep(
        sweep_results=sweep_results,
        outpath=os.path.join(figures_dir, "corner_penalty_sweep.png"),
    )
    _fig_abc(
        res_a=res_a, res_b=res_b, res_c=res_c,
        sigma_bem_sorted=sigma_bem[sort_idx],
        arc=arc[sort_idx], vertex_arcs=vertex_arcs,
        singular_idx=sing_idx, best_lambda=best_lambda,
        outpath=os.path.join(figures_dir, "corner_penalty_ab.png"),
    )
    _fig_corner_zoom(
        res_a=res_a, res_b=res_b, res_c=res_c,
        sigma_bem_sorted=sigma_bem[sort_idx],
        arc=arc[sort_idx], vertex_arcs=vertex_arcs,
        singular_idx=sing_idx,
        outpath=os.path.join(figures_dir, "corner_penalty_density_detail.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
