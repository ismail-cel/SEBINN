"""
Convergence study: extended training of SE-BINN on Koch(1).

Critical question
-----------------
Does the density relative error vs BEM decrease steadily with more
optimization budget, or does it plateau — indicating a fundamental
convergence barrier (loss landscape, architecture, or problem setup)?

Training schedule
-----------------
  Stage 1: Adam  3 phases [1000, 1000, 1000] at lr [1e-3, 3e-4, 1e-4]
  Stage 2: L-BFGS  max_iters=15000, memory=30, grad_tol=1e-10
  Stage 3: (if Stage 2 hits maxIters) Adam 2000 iters at lr=1e-5 to
           perturb saddle, then L-BFGS max_iters=10000

Diagnostics recorded every log_every iterations:
  - loss, gradient norm
  - density rel-diff vs BEM  ||σ_θ(Yq) - σ_BEM|| / ||σ_BEM||   (cheap)
  - γ_c values (all 6)
  - interior rel L2 error on 51×51 grid  (every eval_every iterations)

Geometry / problem: Koch(1), u_exact = x² − y², Dirichlet BVP.
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
from src.training.adam_phase import AdamConfig, run_adam_phases, AdamResult
from src.training.lbfgs import LBFGSConfig, run_lbfgs, LBFGSResult
from src.reconstruction.interior import reconstruct_interior


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CFG = dict(
    seed           = 0,
    n_per_edge     = 12,
    p_gl           = 16,
    m_col_base     = 4,
    eq_scale_mode  = "fixed",
    eq_scale_fixed = 10.0,
    gmres_tol      = 1e-12,
    gmres_maxiter  = 300,
    hidden_width   = 80,
    n_hidden       = 4,
    gamma_init     = 0.0,
    # Stage 1 Adam
    adam_stage1_iters = [1000, 1000, 1000],
    adam_stage1_lrs   = [1e-3,  3e-4,  1e-4],
    adam_log_every    = 100,
    # Stage 2 L-BFGS
    lbfgs1_max_iters  = 15000,
    lbfgs1_memory     = 30,
    lbfgs1_grad_tol   = 1e-10,
    lbfgs1_log_every  = 50,
    # Stage 3 saddle-shake Adam + L-BFGS
    adam_stage3_iters = [2000],
    adam_stage3_lrs   = [1e-5],
    lbfgs2_max_iters  = 10000,
    lbfgs2_memory     = 30,
    lbfgs2_grad_tol   = 1e-10,
    lbfgs2_log_every  = 50,
    # Shared L-BFGS settings
    lbfgs_step_tol    = 1e-12,
    lbfgs_alpha0      = 1e-1,
    lbfgs_alpha_fb    = [1e-2, 1e-3],
    lbfgs_armijo_c1   = 1e-4,
    lbfgs_beta        = 0.5,
    lbfgs_max_bt      = 20,
    # Diagnostics
    eval_every        = 500,   # interior L2 check (expensive)
    n_grid_coarse     = 51,    # grid for cheap interior eval
    n_grid_final      = 201,   # grid for final reconstruction
)


def u_exact(xy: np.ndarray) -> np.ndarray:
    return xy[:, 0] ** 2 - xy[:, 1] ** 2


# ---------------------------------------------------------------------------
# Arc-length helper
# ---------------------------------------------------------------------------

def _boundary_arclength(qdata, n_per_edge):
    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc = panel_start[qdata.pan_id] + qdata.s_on_panel
    Npan  = qdata.n_panels
    Nv    = Npan // n_per_edge
    v_idx = np.arange(Nv) * n_per_edge
    ps    = np.concatenate([[0.0], np.cumsum(qdata.L_panel)])
    vertex_arcs = np.append(ps[v_idx], ps[-1])
    return arc, vertex_arcs


# ---------------------------------------------------------------------------
# Diagnostic tracker
# ---------------------------------------------------------------------------

class Tracker:
    """
    Records training diagnostics at every checkpoint.

    Fields (all lists, one entry per recorded step):
      global_iter : global iteration index
      stage       : stage label string ("Adam-1", "LBFGS-1", etc.)
      loss        : scalar loss
      grad_norm   : gradient norm (NaN for Adam, where it's unavailable)
      density_err : ||σ_θ - σ_BEM|| / ||σ_BEM||
      gamma_vals  : list of γ_c values at this step
      interior_L2 : interior rel L2 (or NaN if not evaluated)
    """

    def __init__(self):
        self.global_iter   = []
        self.stage         = []
        self.loss          = []
        self.grad_norm     = []
        self.density_err   = []
        self.gamma_vals    = []
        self.interior_L2   = []

    def record(
        self,
        it:           int,
        stage:        str,
        loss:         float,
        grad_norm:    float,
        density_err:  float,
        gamma:        object,
        interior_L2:  float = float("nan"),
    ):
        self.global_iter.append(it)
        self.stage.append(stage)
        self.loss.append(loss)
        self.grad_norm.append(grad_norm)
        self.density_err.append(density_err)
        gv = gamma if isinstance(gamma, list) else [float(gamma)]
        self.gamma_vals.append(gv)
        self.interior_L2.append(interior_L2)

    def arrays(self):
        return (
            np.array(self.global_iter),
            np.array(self.loss),
            np.array(self.grad_norm),
            np.array(self.density_err),
            np.array(self.interior_L2),
            np.array(self.gamma_vals),   # (N, n_gamma)
        )


def _density_err(model, sigma_s_Yq_t, Yq_T_t, sigma_bem, wq):
    """Cheap: forward pass at Yq, L2 difference vs BEM."""
    with torch.no_grad():
        sigma_nn = model(Yq_T_t, sigma_s_Yq_t).numpy()
    return float(np.linalg.norm(sigma_nn - sigma_bem)
                 / max(np.linalg.norm(sigma_bem), 1e-14))


def _interior_L2_cheap(model, sigma_s_Yq_t, Yq_T_t, Yq_T_np, wq, P, n_grid):
    """Coarse interior L2 check (51×51 grid)."""
    with torch.no_grad():
        sigma_nn = model(Yq_T_t, sigma_s_Yq_t).numpy()
    res = reconstruct_interior(P=P, Yq=Yq_T_np, wq=wq, sigma=sigma_nn,
                               n_grid=n_grid, u_exact=u_exact)
    return res.rel_L2


# ---------------------------------------------------------------------------
# Custom Adam run with per-iteration density tracking
# ---------------------------------------------------------------------------

def _run_adam_tracked(
    model, op, phase_iters, phase_lrs,
    sigma_s_Yq_t, Yq_T_t, Yq_T_np, sigma_bem, wq, P,
    stage_label, tracker, global_offset,
    log_every, eval_every, n_grid_coarse,
    verbose=True,
):
    """Adam phases with diagnostic recording at every log_every steps."""
    from src.training.loss import sebinn_loss

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=phase_lrs[0], betas=(0.9, 0.999), eps=1e-8,
    )

    global_it = global_offset
    total_inner = 0

    for i_phase, (n_it, lr) in enumerate(zip(phase_iters, phase_lrs)):
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        if verbose:
            print(f"  {stage_label} phase {i_phase+1}/{len(phase_iters)} | "
                  f"iters={n_it} | lr={lr:.1e}")

        for j in range(n_it):
            optimizer.zero_grad()
            loss, dbg = sebinn_loss(model, op)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.detach())
            total_inner += 1
            global_it += 1

            log_this = (j == 0) or ((total_inner) % log_every == 0) or (j == n_it - 1)
            if log_this:
                d_err = _density_err(model, sigma_s_Yq_t, Yq_T_t, sigma_bem, wq)
                gamma = model.gamma_value()
                i_L2  = float("nan")
                if global_it % eval_every == 0:
                    i_L2 = _interior_L2_cheap(
                        model, sigma_s_Yq_t, Yq_T_t, Yq_T_np, wq, P, n_grid_coarse)

                tracker.record(
                    it          = global_it,
                    stage       = stage_label,
                    loss        = loss_val,
                    grad_norm   = float("nan"),   # not tracked in Adam
                    density_err = d_err,
                    gamma       = gamma,
                    interior_L2 = i_L2,
                )

                gv = gamma if isinstance(gamma, list) else [gamma]
                gv_str = "[" + ",".join(f"{v:.4f}" for v in gv) + "]"
                if verbose:
                    print(f"    {stage_label} {global_it:5d} | "
                          f"loss={loss_val:.3e} | d_err={d_err:.3e} | γ={gv_str}")

    return global_it


# ---------------------------------------------------------------------------
# Custom L-BFGS run with per-iteration density tracking
# ---------------------------------------------------------------------------

def _run_lbfgs_tracked(
    model, op, cfg_lbfgs,
    sigma_s_Yq_t, Yq_T_t, Yq_T_np, sigma_bem, wq, P,
    stage_label, tracker, global_offset,
    log_every, eval_every, n_grid_coarse,
    verbose=True,
):
    """
    L-BFGS loop with tracking injected into the existing run_lbfgs code.

    We call run_lbfgs from src/training/lbfgs.py for the actual optimisation,
    then reconstruct the trajectory from its loss_hist/grad_hist.
    But we also need per-iteration density_err and gamma, which run_lbfgs
    doesn't record.

    Strategy: wrap run_lbfgs and sample diagnostics from the loss_hist at
    every log_every step.  The density_err is cheap (just a forward pass at
    Yq) so we re-evaluate it at saved checkpoints by rerunning the forward
    pass — but we don't have the parameter history.

    Instead: we run L-BFGS in chunks of `log_every` iterations, recording
    diagnostics after each chunk.
    """
    from src.training.loss import sebinn_loss
    from src.training.lbfgs import (
        _loss_and_grad, _two_loop, _armijo_line_search,
    )

    max_iters      = cfg_lbfgs.max_iters
    grad_tol       = cfg_lbfgs.grad_tol
    step_tol       = cfg_lbfgs.step_tol
    memory         = cfg_lbfgs.memory
    alpha0         = cfg_lbfgs.alpha0
    alpha_fallback = cfg_lbfgs.alpha_fallback

    theta = model.to_vector()
    f, g  = _loss_and_grad(theta, model, op)

    S, Y, rho_list = [], [], []
    n_ls_fail = 0
    reason    = "maxIters"
    global_it = global_offset

    if verbose:
        print(f"  {stage_label} | max_iters={max_iters} | grad_tol={grad_tol:.1e} "
              f"| memory={memory}")

    for k in range(1, max_iters + 1):
        global_it = global_offset + k

        gnorm = float(g.norm())

        if gnorm < grad_tol:
            reason = "gradTol"
            # record final
            d_err = _density_err(model, sigma_s_Yq_t, Yq_T_t, sigma_bem, wq)
            gamma = model.gamma_value()
            tracker.record(global_it, stage_label, f, gnorm, d_err, gamma)
            if verbose:
                print(f"    {stage_label} {global_it:5d} | loss={f:.3e} | "
                      f"gradNorm={gnorm:.3e} → gradTol reached")
            break

        # descent direction
        if S:
            p_dir = -_two_loop(g, S, Y, rho_list)
        else:
            p_dir = -g

        alpha_starts = [alpha0] + list(alpha_fallback)
        alpha, theta_new, f_new, g_new, accepted = _armijo_line_search(
            theta, f, g, p_dir, model, op, cfg_lbfgs, alpha_starts
        )

        if not accepted:
            n_ls_fail += 1
            # Reset curvature history, retry steepest descent
            S.clear(); Y.clear(); rho_list.clear()
            p_dir = -g
            alpha, theta_new, f_new, g_new, accepted = _armijo_line_search(
                theta, f, g, p_dir, model, op, cfg_lbfgs, alpha_starts
            )
            if not accepted:
                reason = "lsFailure"
                d_err = _density_err(model, sigma_s_Yq_t, Yq_T_t, sigma_bem, wq)
                gamma = model.gamma_value()
                tracker.record(global_it, stage_label, f, gnorm, d_err, gamma)
                if verbose:
                    print(f"    {stage_label} {global_it:5d} | lsFailure — stopping")
                break

        # curvature pair
        s_vec = theta_new - theta
        y_vec = g_new - g
        sy    = float(s_vec.dot(y_vec))
        snorm = float(s_vec.norm())

        if snorm < step_tol:
            reason = "stepTol"
            d_err = _density_err(model, sigma_s_Yq_t, Yq_T_t, sigma_bem, wq)
            gamma = model.gamma_value()
            tracker.record(global_it, stage_label, f_new, gnorm, d_err, gamma)
            if verbose:
                print(f"    {stage_label} {global_it:5d} | stepTol reached")
            break

        if sy > 1e-12 * snorm * float(y_vec.norm()):
            rho = 1.0 / sy
            S.append(s_vec.detach().clone())
            Y.append(y_vec.detach().clone())
            rho_list.append(rho)
            if len(S) > memory:
                S.pop(0); Y.pop(0); rho_list.pop(0)

        theta, f, g = theta_new, f_new, g_new

        # Periodic logging
        log_this = (k == 1) or (k % log_every == 0) or (k == max_iters)
        if log_this:
            d_err = _density_err(model, sigma_s_Yq_t, Yq_T_t, sigma_bem, wq)
            gamma = model.gamma_value()
            i_L2  = float("nan")
            if global_it % eval_every == 0:
                i_L2 = _interior_L2_cheap(
                    model, sigma_s_Yq_t, Yq_T_t, Yq_T_np, wq, P, n_grid_coarse)

            tracker.record(
                it          = global_it,
                stage       = stage_label,
                loss        = f,
                grad_norm   = gnorm,
                density_err = d_err,
                gamma       = gamma,
                interior_L2 = i_L2,
            )

            gv = gamma if isinstance(gamma, list) else [gamma]
            gv_str = "[" + ",".join(f"{v:.4f}" for v in gv) + "]"
            if verbose:
                print(f"    {stage_label} {global_it:5d} | loss={f:.3e} | "
                      f"gNorm={gnorm:.3e} | d_err={d_err:.3e} | γ={gv_str}")

    else:
        reason = "maxIters"

    # Ensure model holds final theta
    model.from_vector(theta)

    if verbose:
        print(f"  {stage_label} done | iters={k} | reason={reason} | "
              f"ls_fail={n_ls_fail}")

    return global_offset + k, reason


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _make_figure(tracker, stage_boundaries, singular_corner_indices,
                 vertex_arcs, arc_sorted, sigma_bem_sorted,
                 sigma_final_sorted, sigma_final_all,
                 outpath):
    its, loss, gnorm, d_err, i_L2, gammas = tracker.arrays()

    # Finite-only for interior L2
    mask_L2 = np.isfinite(i_L2)

    fig, axes = plt.subplots(4, 1, figsize=(13, 18))
    fig.subplots_adjust(hspace=0.40)

    stages    = np.array(tracker.stage)
    stage_colors = {"Adam-1": "#1f77b4", "LBFGS-1": "#2ca02c",
                    "Adam-3": "#ff7f0e", "LBFGS-2": "#d62728"}

    # ---- (a) Loss ----
    ax = axes[0]
    for sl, col in stage_colors.items():
        mask = stages == sl
        if mask.sum() == 0:
            continue
        ax.semilogy(its[mask], loss[mask], color=col, lw=1.3,
                    label=sl, alpha=0.85)
    for it_b in stage_boundaries:
        ax.axvline(it_b, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("Global iteration", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("(a) Loss vs iteration", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    # ---- (b) Density rel-diff ----
    ax = axes[1]
    for sl, col in stage_colors.items():
        mask = stages == sl
        if mask.sum() == 0:
            continue
        ax.semilogy(its[mask], d_err[mask], color=col, lw=1.3,
                    label=sl, alpha=0.85)
    for it_b in stage_boundaries:
        ax.axvline(it_b, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("Global iteration", fontsize=11)
    ax.set_ylabel(r"$\|\sigma_\theta - \sigma_\mathrm{BEM}\| / \|\sigma_\mathrm{BEM}\|$",
                  fontsize=10)
    ax.set_title("(b) Density relative error vs BEM  [KEY: plateau = bottleneck]",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    # ---- (c) Interior rel L2 ----
    ax = axes[2]
    if mask_L2.sum() > 0:
        ax.semilogy(its[mask_L2], i_L2[mask_L2], "o-",
                    color="#9467bd", lw=1.5, ms=4, label="Interior rel L2 (51×51)")
    for it_b in stage_boundaries:
        ax.axvline(it_b, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("Global iteration", fontsize=11)
    ax.set_ylabel("Interior rel L2", fontsize=11)
    ax.set_title("(c) Interior relative L2 error (coarse 51×51 grid)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    # ---- (d) γ_c trajectories ----
    ax = axes[3]
    n_gamma = gammas.shape[1]
    cmap = plt.cm.tab10
    for c in range(n_gamma):
        ax.plot(its, gammas[:, c], color=cmap(c), lw=1.2,
                label=f"$\\gamma_{c+1}$", alpha=0.85)
    for it_b in stage_boundaries:
        ax.axvline(it_b, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)
    ax.set_xlabel("Global iteration", fontsize=11)
    ax.set_ylabel(r"$\gamma_c$", fontsize=11)
    ax.set_title(r"(d) Per-corner $\gamma_c$ trajectories", fontsize=12)
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    ax.grid(True, lw=0.3, alpha=0.5)

    # Stage labels on top axis
    for sl, col in stage_colors.items():
        mask = stages == sl
        if mask.sum() == 0:
            continue
        mid = float(np.median(its[mask]))
        axes[0].text(mid, axes[0].get_ylim()[1],
                     sl, color=col, fontsize=7, ha="center", va="bottom",
                     clip_on=True)

    fig.suptitle(
        "Convergence Study — SE-BINN, Koch(1), $u = x^2 - y^2$\n"
        "Adam(3000) + L-BFGS(15000) + Adam(2000) + L-BFGS(10000)",
        fontsize=13, y=1.01,
    )

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved convergence_study → {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("=" * 65)
    print("  Convergence Study — SE-BINN, Koch(1), u = x² − y²")
    print("=" * 65)

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Geometry, quadrature, BEM
    # ------------------------------------------------------------------
    print("\n--- Setup ---")
    geom   = make_koch_geometry(n=1)
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=CFG["n_per_edge"])
    label_corner_ring_panels(panels, P)

    qdata  = build_panel_quadrature(panels, p=CFG["p_gl"])
    Yq_T   = qdata.Yq.T       # (Nq, 2)
    wq     = qdata.wq
    Nq     = qdata.n_quad

    arc, vertex_arcs = _boundary_arclength(qdata, CFG["n_per_edge"])
    sort_idx = np.argsort(arc)

    print(f"  Koch(1): vertices={geom.n_vertices} | panels={len(panels)} | Nq={Nq}")

    nmat    = assemble_nystrom_matrix(qdata)
    bem_sol = solve_bem(nmat, u_exact(Yq_T),
                        tol=CFG["gmres_tol"], max_iter=CFG["gmres_maxiter"])
    sigma_bem = bem_sol.sigma
    bem_out   = reconstruct_interior(P=P, Yq=Yq_T, wq=wq, sigma=sigma_bem,
                                     n_grid=CFG["n_grid_final"], u_exact=u_exact)
    print(f"  BEM: rel_L2={bem_out.rel_L2:.3e} | linf={bem_out.linf:.3e}")

    # ------------------------------------------------------------------
    # 2. Operator state
    # ------------------------------------------------------------------
    enrichment = SingularEnrichment(geom=geom, per_corner_gamma=True)
    w_panel    = panel_loss_weights(panels, w_base=1.0, w_corner=1.0, w_ring=1.0)
    colloc     = build_collocation_points(panels, m_col_panel=CFG["m_col_base"])
    op, op_diag = build_operator_state(
        colloc=colloc, qdata=qdata, enrichment=enrichment, g=u_exact,
        panel_weights=w_panel,
        eq_scale_mode=CFG["eq_scale_mode"], eq_scale_fixed=CFG["eq_scale_fixed"],
        dtype=torch.float64, device="cpu",
    )
    print(f"  Operator: Nb={colloc.n_colloc} | eq_scale={op_diag['eq_scale']:.2e}")

    # Precomputed tensors for diagnostics
    sigma_s_Yq   = enrichment.precompute(Yq_T)
    Yq_T_t       = torch.tensor(Yq_T,       dtype=torch.float64)
    sigma_s_Yq_t = torch.tensor(sigma_s_Yq, dtype=torch.float64)

    # ------------------------------------------------------------------
    # 3. Model initialisation
    # ------------------------------------------------------------------
    torch.manual_seed(CFG["seed"])
    model = SEBINNModel(
        hidden_width = CFG["hidden_width"],
        n_hidden     = CFG["n_hidden"],
        n_gamma      = enrichment.n_gamma,
        gamma_init   = CFG["gamma_init"],
        dtype        = torch.float64,
    )
    print(f"  Model: n_params={model.n_params()} | n_gamma={enrichment.n_gamma}")

    # L-BFGS config builder
    def _lbfgs_cfg(max_iters, memory, grad_tol, log_every):
        return LBFGSConfig(
            max_iters      = max_iters,
            grad_tol       = grad_tol,
            step_tol       = CFG["lbfgs_step_tol"],
            memory         = memory,
            log_every      = log_every,
            alpha0         = CFG["lbfgs_alpha0"],
            alpha_fallback = CFG["lbfgs_alpha_fb"],
            armijo_c1      = CFG["lbfgs_armijo_c1"],
            backtrack_beta = CFG["lbfgs_beta"],
            max_backtrack  = CFG["lbfgs_max_bt"],
        )

    tracker          = Tracker()
    stage_boundaries = []
    global_it        = 0

    # ------------------------------------------------------------------
    # Stage 1: Adam
    # ------------------------------------------------------------------
    print("\n=== Stage 1: Adam ===")
    global_it = _run_adam_tracked(
        model=model, op=op,
        phase_iters=CFG["adam_stage1_iters"],
        phase_lrs=CFG["adam_stage1_lrs"],
        sigma_s_Yq_t=sigma_s_Yq_t, Yq_T_t=Yq_T_t, Yq_T_np=Yq_T,
        sigma_bem=sigma_bem, wq=wq, P=P,
        stage_label="Adam-1", tracker=tracker, global_offset=0,
        log_every=CFG["adam_log_every"], eval_every=CFG["eval_every"],
        n_grid_coarse=CFG["n_grid_coarse"], verbose=True,
    )
    stage_boundaries.append(global_it)

    # ------------------------------------------------------------------
    # Stage 2: L-BFGS (main)
    # ------------------------------------------------------------------
    print(f"\n=== Stage 2: L-BFGS (max {CFG['lbfgs1_max_iters']} iters) ===")
    global_it, reason_lbfgs1 = _run_lbfgs_tracked(
        model=model, op=op,
        cfg_lbfgs=_lbfgs_cfg(CFG["lbfgs1_max_iters"], CFG["lbfgs1_memory"],
                              CFG["lbfgs1_grad_tol"], CFG["lbfgs1_log_every"]),
        sigma_s_Yq_t=sigma_s_Yq_t, Yq_T_t=Yq_T_t, Yq_T_np=Yq_T,
        sigma_bem=sigma_bem, wq=wq, P=P,
        stage_label="LBFGS-1", tracker=tracker, global_offset=global_it,
        log_every=CFG["lbfgs1_log_every"], eval_every=CFG["eval_every"],
        n_grid_coarse=CFG["n_grid_coarse"], verbose=True,
    )
    stage_boundaries.append(global_it)

    # ------------------------------------------------------------------
    # Stage 3: saddle-shake + second L-BFGS (only if maxIters)
    # ------------------------------------------------------------------
    reason_lbfgs2 = "skipped"
    if reason_lbfgs1 == "maxIters":
        print(f"\n=== Stage 3a: Adam saddle-shake ({CFG['adam_stage3_iters']} iters) ===")
        global_it = _run_adam_tracked(
            model=model, op=op,
            phase_iters=CFG["adam_stage3_iters"],
            phase_lrs=CFG["adam_stage3_lrs"],
            sigma_s_Yq_t=sigma_s_Yq_t, Yq_T_t=Yq_T_t, Yq_T_np=Yq_T,
            sigma_bem=sigma_bem, wq=wq, P=P,
            stage_label="Adam-3", tracker=tracker, global_offset=global_it,
            log_every=CFG["adam_log_every"], eval_every=CFG["eval_every"],
            n_grid_coarse=CFG["n_grid_coarse"], verbose=True,
        )
        stage_boundaries.append(global_it)

        print(f"\n=== Stage 3b: L-BFGS restart (max {CFG['lbfgs2_max_iters']} iters) ===")
        global_it, reason_lbfgs2 = _run_lbfgs_tracked(
            model=model, op=op,
            cfg_lbfgs=_lbfgs_cfg(CFG["lbfgs2_max_iters"], CFG["lbfgs2_memory"],
                                  CFG["lbfgs2_grad_tol"], CFG["lbfgs2_log_every"]),
            sigma_s_Yq_t=sigma_s_Yq_t, Yq_T_t=Yq_T_t, Yq_T_np=Yq_T,
            sigma_bem=sigma_bem, wq=wq, P=P,
            stage_label="LBFGS-2", tracker=tracker, global_offset=global_it,
            log_every=CFG["lbfgs2_log_every"], eval_every=CFG["eval_every"],
            n_grid_coarse=CFG["n_grid_coarse"], verbose=True,
        )
        stage_boundaries.append(global_it)
    else:
        print(f"\n  Stage 2 converged ({reason_lbfgs1}) — Stage 3 skipped.")

    t_total = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    print("\n--- Final evaluation ---")
    with torch.no_grad():
        sigma_final = model(Yq_T_t, sigma_s_Yq_t).numpy()

    final_out = reconstruct_interior(
        P=P, Yq=Yq_T, wq=wq, sigma=sigma_final,
        n_grid=CFG["n_grid_final"], u_exact=u_exact,
    )

    density_rel_diff = float(
        np.linalg.norm(sigma_final - sigma_bem)
        / max(np.linalg.norm(sigma_bem), 1e-14)
    )

    # Get final grad norm
    from src.training.loss import sebinn_loss
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss_final, _ = sebinn_loss(model, op)
    loss_final.backward()
    grad_final = torch.cat([
        p.grad.detach().view(-1) if p.grad is not None
        else torch.zeros(p.numel(), dtype=torch.float64)
        for p in model.parameters()
    ])
    grad_norm_final = float(grad_final.norm())

    gamma_final = model.gamma_value()
    gv = gamma_final if isinstance(gamma_final, list) else [gamma_final]

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("  CONVERGENCE STUDY RESULTS")
    print("=" * 65)
    print(f"  BEM reference         : rel_L2={bem_out.rel_L2:.3e} | "
          f"linf={bem_out.linf:.3e}")
    print(f"  Final interior rel L2 : {final_out.rel_L2:.3e}")
    print(f"  Final interior L∞     : {final_out.linf:.3e}")
    print(f"  Final loss            : {float(loss_final.detach()):.3e}")
    print(f"  Final grad norm       : {grad_norm_final:.3e}")
    print(f"  Density rel-diff      : {density_rel_diff:.3e}")
    print(f"  Total iterations      : {global_it}")
    print(f"  Wall time             : {t_total:.1f}s")
    print(f"  L-BFGS-1 reason       : {reason_lbfgs1}")
    print(f"  L-BFGS-2 reason       : {reason_lbfgs2}")
    gv_str = "[" + ", ".join(f"{v:.4f}" for v in gv) + "]"
    print(f"  γ_c final             : {gv_str}")

    # Key convergence assessment
    its_arr, _, _, d_err_arr, _, _ = tracker.arrays()
    last_500 = d_err_arr[its_arr > max(its_arr) - 500]
    if len(last_500) >= 2:
        rel_change = abs(last_500[-1] - last_500[0]) / max(last_500[0], 1e-14)
        print(f"\n  Density error over last 500 iters: {last_500[0]:.4f} → {last_500[-1]:.4f}")
        print(f"  Relative change: {rel_change:.2%}")
        if rel_change < 0.005:
            print("  *** PLATEAU detected: density error stopped improving. "
                  "Architecture or loss landscape is the bottleneck. ***")
        else:
            print("  *** Still decreasing: more budget would help. ***")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    figures_dir = os.path.join(_HERE, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    outpath = os.path.join(figures_dir, "convergence_study.png")

    _make_figure(
        tracker              = tracker,
        stage_boundaries     = stage_boundaries[:-1],  # skip trailing sentinel
        singular_corner_indices = list(geom.singular_corner_indices),
        vertex_arcs          = vertex_arcs,
        arc_sorted           = arc[sort_idx],
        sigma_bem_sorted     = sigma_bem[sort_idx],
        sigma_final_sorted   = sigma_final[sort_idx],
        sigma_final_all      = sigma_final,
        outpath              = outpath,
    )


if __name__ == "__main__":
    main()
