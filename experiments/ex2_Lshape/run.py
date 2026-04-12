"""
Experiment 2: SE-BINN on L-shaped domain, u_exact = r^(2/3) sin(2θ/3).

Domain
------
[-1,1]² with upper-right quadrant [0,1]×[0,1] removed.
Vertices (CCW): (-1,-1), (1,-1), (1,0), (0,0), (0,1), (-1,1).
Reentrant corner at (0,0): ω = 3π/2, α = π/ω − 1 = −1/3.

Exact solution
--------------
u_exact = r^(2/3) sin(2θ/3)

where (r,θ) are polar coordinates centred at the reentrant corner (0,0)
and θ = (−atan2(y,x)) mod 2π is measured clockwise from the +x edge.

Properties:
  - Δu = 0 everywhere in the interior (singular harmonic eigenmode)
  - u = 0 on both edges meeting at the corner (edges 2 and 3)
  - The corner singularity in u is EXACTLY r^(2/3) (the lowest mode)
  - The density σ = V^{-1}g has a r^{−1/3} spike at (0,0)
  - Enrichment basis σ_s = −(2/3) r^{−1/3} exactly matches the singularity

Why this choice?
----------------
For g = x²−y², the function vanishes at (0,0) and has no leading singularity
there → γ*_lstsq ≈ 0 on the L-shape (verified empirically).

For g = r^(2/3) sin(2θ/3), the singularity is EXACTLY of the type our
enrichment targets → large enrichment energy fraction → wide A/B gap.

Singular exponent comparison
-----------------------------
L-shape: α = −1/3   →   σ_s = −(2/3) r^{−1/3}   (STRONGER singularity)
Koch(1): α = −1/4   →   σ_s = −(3/4) r^{−1/4}

Three training cases (all from same initial weights)
----------------------------------------------------
  Case A — BINN:       γ frozen at 0 (plain BINN, σ = σ_w only)
  Case B — SE-BINN:    γ trainable, γ_init = 0
  Case C — SE-BINN★:   γ trainable, γ_init = γ*_lstsq (warm start)

Output figures (saved to experiments/ex2_Lshape/figures/)
---------------------------------------------------------
  ab_comparison.png         — loss, density, density error (all 3 cases)
  convergence_diagnostics.png — loss, density rel-diff, γ trajectories
  interior_error.png        — BEM | SE-BINN★ | log₁₀|error|
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
from matplotlib.colors import LogNorm

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
from src.training.lbfgs import (
    LBFGSConfig, _loss_and_grad, _two_loop, _armijo_line_search,
)
from src.reconstruction.interior import reconstruct_interior


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CFG = dict(
    seed           = 0,
    n_per_edge     = 16,         # 6 edges × 16 = 96 panels
    p_gl           = 16,
    m_col_base     = 4,          # → Nb = 384 collocation points
    eq_scale_mode  = "fixed",
    eq_scale_fixed = 10.0,
    gmres_tol      = 1e-12,
    gmres_maxiter  = 300,
    hidden_width   = 80,
    n_hidden       = 4,
    gamma_init     = 0.0,
    # Adam — 3 phases
    adam_iters     = [1000, 1000, 1000],
    adam_lrs       = [1e-3,  3e-4,  1e-4],
    adam_log_every = 200,
    # L-BFGS
    lbfgs_max_iters  = 15000,
    lbfgs_memory     = 30,
    lbfgs_grad_tol   = 1e-10,
    lbfgs_log_every  = 200,
    lbfgs_step_tol   = 1e-12,
    lbfgs_alpha0     = 1e-1,
    lbfgs_alpha_fb   = [1e-2, 1e-3],
    lbfgs_armijo_c1  = 1e-4,
    lbfgs_beta       = 0.5,
    lbfgs_max_bt     = 20,
    # Diagnostics
    eval_every       = 500,
    n_grid_coarse    = 51,
    n_grid_final     = 101,
)


def u_exact(xy: np.ndarray) -> np.ndarray:
    """u = r^(2/3) sin(2θ/3), θ measured clockwise from +x at the corner (0,0)."""
    r     = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    theta = (-np.arctan2(xy[:, 1], xy[:, 0])) % (2.0 * np.pi)
    return np.where(r < 1e-15, 0.0, r ** (2.0 / 3.0) * np.sin(2.0 * theta / 3.0))


# ---------------------------------------------------------------------------
# Arc-length helper
# ---------------------------------------------------------------------------

def _boundary_arclength(qdata, n_per_edge):
    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc = panel_start[qdata.pan_id] + qdata.s_on_panel
    Nv  = qdata.n_panels // n_per_edge
    ps  = np.concatenate([[0.0], np.cumsum(qdata.L_panel)])
    vertex_arcs = np.append(ps[np.arange(Nv) * n_per_edge], ps[-1])
    return arc, vertex_arcs


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def _density_err(model, sigma_s_t, Yq_t, sigma_bem):
    with torch.no_grad():
        s = model(Yq_t, sigma_s_t).numpy()
    return float(np.linalg.norm(s - sigma_bem) / max(np.linalg.norm(sigma_bem), 1e-14))


def _interior_L2(model, sigma_s_t, Yq_t, Yq_np, wq, P, n_grid):
    with torch.no_grad():
        s = model(Yq_t, sigma_s_t).numpy()
    res = reconstruct_interior(P=P, Yq=Yq_np, wq=wq, sigma=s,
                               n_grid=n_grid, u_exact=u_exact,
                               x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))
    return res.rel_L2


# ---------------------------------------------------------------------------
# Per-iteration diagnostic tracker
# ---------------------------------------------------------------------------

class Tracker:
    def __init__(self, label):
        self.label      = label
        self.iters      = []
        self.stages     = []
        self.losses     = []
        self.grad_norms = []
        self.d_errs     = []
        self.gammas     = []    # list of float each step
        self.i_L2s      = []   # NaN unless eval_every

    def record(self, it, stage, loss, gnorm, d_err, gamma, i_L2=float("nan")):
        self.iters.append(it)
        self.stages.append(stage)
        self.losses.append(loss)
        self.grad_norms.append(gnorm)
        self.d_errs.append(d_err)
        g = float(gamma) if not isinstance(gamma, list) else gamma[0]
        self.gammas.append(g)
        self.i_L2s.append(i_L2)

    def arrays(self):
        return (np.array(self.iters), np.array(self.losses),
                np.array(self.grad_norms), np.array(self.d_errs),
                np.array(self.gammas), np.array(self.i_L2s))


# ---------------------------------------------------------------------------
# Custom Adam with tracking
# ---------------------------------------------------------------------------

def _run_adam_tracked(model, op, phase_iters, phase_lrs,
                      sigma_s_t, Yq_t, Yq_np, wq, P, sigma_bem,
                      label, tracker, offset,
                      log_every, eval_every, n_grid_coarse, verbose):
    from src.training.loss import sebinn_loss

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=phase_lrs[0], betas=(0.9, 0.999), eps=1e-8,
    )
    global_it   = offset
    total_inner = 0

    for i_ph, (n_it, lr) in enumerate(zip(phase_iters, phase_lrs)):
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        if verbose:
            print(f"  [{label}] Adam ph {i_ph+1}/{len(phase_iters)} | "
                  f"iters={n_it} | lr={lr:.1e}")

        for j in range(n_it):
            optimizer.zero_grad()
            loss, _ = sebinn_loss(model, op)
            loss.backward()
            optimizer.step()

            total_inner += 1
            global_it   += 1
            loss_val     = float(loss.detach())

            if (j == 0) or (total_inner % log_every == 0) or (j == n_it - 1):
                d_err = _density_err(model, sigma_s_t, Yq_t, sigma_bem)
                gamma = model.gamma_value()
                i_L2  = float("nan")
                if global_it % eval_every == 0:
                    i_L2 = _interior_L2(model, sigma_s_t, Yq_t, Yq_np, wq, P, n_grid_coarse)

                tracker.record(global_it, "Adam", loss_val, float("nan"),
                                d_err, gamma, i_L2)

                gv = gamma if isinstance(gamma, list) else float(gamma)
                if verbose:
                    print(f"    [{label}] Adam {global_it:5d} | "
                          f"loss={loss_val:.3e} | d_err={d_err:.3e} | γ={gv:.5f}")

    return global_it


# ---------------------------------------------------------------------------
# Custom L-BFGS with tracking
# ---------------------------------------------------------------------------

def _run_lbfgs_tracked(model, op, cfg_lbfgs,
                        sigma_s_t, Yq_t, Yq_np, wq, P, sigma_bem,
                        label, tracker, offset,
                        log_every, eval_every, n_grid_coarse, verbose):
    max_iters      = cfg_lbfgs.max_iters
    grad_tol       = cfg_lbfgs.grad_tol
    step_tol       = cfg_lbfgs.step_tol
    memory         = cfg_lbfgs.memory

    theta = model.to_vector()
    f, g  = _loss_and_grad(theta, model, op)
    S, Y, rho_list = [], [], []
    n_ls_fail = 0
    reason    = "maxIters"

    if verbose:
        print(f"  [{label}] LBFGS start | max={max_iters} | mem={memory} | "
              f"grad_tol={grad_tol:.1e}")

    for k in range(1, max_iters + 1):
        global_it = offset + k
        gnorm = float(g.norm())

        if gnorm < grad_tol:
            reason = "gradTol"
            d_err  = _density_err(model, sigma_s_t, Yq_t, sigma_bem)
            gamma  = model.gamma_value()
            tracker.record(global_it, "LBFGS", f, gnorm, d_err, gamma)
            if verbose:
                print(f"    [{label}] LBFGS {global_it:6d} | gradTol")
            break

        p_dir = -_two_loop(g, S, Y, rho_list) if S else -g
        alpha_starts = [cfg_lbfgs.alpha0] + list(cfg_lbfgs.alpha_fallback)

        alpha, theta_new, f_new, g_new, accepted = _armijo_line_search(
            theta, f, g, p_dir, model, op, cfg_lbfgs, alpha_starts)

        if not accepted:
            n_ls_fail += 1
            S.clear(); Y.clear(); rho_list.clear()
            alpha, theta_new, f_new, g_new, accepted = _armijo_line_search(
                theta, f, g, -g, model, op, cfg_lbfgs, alpha_starts)
            if not accepted:
                reason = "lsFailure"
                d_err  = _density_err(model, sigma_s_t, Yq_t, sigma_bem)
                gamma  = model.gamma_value()
                tracker.record(global_it, "LBFGS", f, gnorm, d_err, gamma)
                if verbose:
                    print(f"    [{label}] LBFGS {global_it:6d} | lsFailure")
                break

        s_vec = theta_new - theta
        y_vec = g_new - g
        snorm = float(s_vec.norm())

        if snorm < step_tol:
            reason = "stepTol"
            d_err  = _density_err(model, sigma_s_t, Yq_t, sigma_bem)
            gamma  = model.gamma_value()
            tracker.record(global_it, "LBFGS", f, gnorm, d_err, gamma)
            if verbose:
                print(f"    [{label}] LBFGS {global_it:6d} | stepTol")
            break

        sy = float(s_vec.dot(y_vec))
        if sy > 1e-12 * snorm * float(y_vec.norm()):
            rho = 1.0 / sy
            S.append(s_vec.detach().clone())
            Y.append(y_vec.detach().clone())
            rho_list.append(rho)
            if len(S) > memory:
                S.pop(0); Y.pop(0); rho_list.pop(0)

        theta, f, g = theta_new, f_new, g_new

        log_this = (k == 1) or (k % log_every == 0) or (k == max_iters)
        if log_this:
            d_err = _density_err(model, sigma_s_t, Yq_t, sigma_bem)
            gamma = model.gamma_value()
            i_L2  = float("nan")
            if global_it % eval_every == 0:
                i_L2 = _interior_L2(model, sigma_s_t, Yq_t, Yq_np, wq, P, n_grid_coarse)

            tracker.record(global_it, "LBFGS", f, gnorm, d_err, gamma, i_L2)

            gv = gamma if isinstance(gamma, list) else float(gamma)
            if verbose:
                print(f"    [{label}] LBFGS {global_it:6d} | "
                      f"loss={f:.3e} | gNorm={gnorm:.3e} | "
                      f"d_err={d_err:.3e} | γ={gv:.5f}")
    else:
        reason = "maxIters"

    model.from_vector(theta)
    if verbose:
        print(f"  [{label}] LBFGS done | iters={k} | reason={reason} | "
              f"ls_fail={n_ls_fail}")

    return offset + k, reason


# ---------------------------------------------------------------------------
# Train one case
# ---------------------------------------------------------------------------

def _train_case(
    label, init_state, gamma_override,
    freeze_gamma, shared, verbose=True,
):
    """
    Run one training case (A, B, or C) from shared initial state.

    Parameters
    ----------
    label          : str ("BINN", "SE-BINN", "SE-BINN*")
    init_state     : shared initial state_dict
    gamma_override : float or None — if set, overrides gamma after loading state
    freeze_gamma   : bool — freeze gamma parameter
    shared         : dict of precomputed shared objects
    """
    print(f"\n{'='*60}")
    print(f"  Case: {label}")
    print(f"{'='*60}")
    t0 = time.perf_counter()

    op         = shared["op"]
    enrichment = shared["enrichment"]
    Yq_np      = shared["Yq_np"]
    wq         = shared["wq"]
    P          = shared["P"]
    sigma_bem  = shared["sigma_bem"]
    sigma_s_t  = shared["sigma_s_t"]
    Yq_t       = shared["Yq_t"]
    sort_idx   = shared["sort_idx"]

    # Build model from shared init
    model = SEBINNModel(
        hidden_width = CFG["hidden_width"],
        n_hidden     = CFG["n_hidden"],
        n_gamma      = enrichment.n_gamma,
        gamma_init   = CFG["gamma_init"],
        dtype        = torch.float64,
    )
    model.load_state_dict(copy.deepcopy(init_state))

    if gamma_override is not None:
        model.gamma_module.gamma.data.fill_(gamma_override)
        if verbose:
            print(f"  γ warm-start: {gamma_override:.6f}")

    if freeze_gamma:
        model.gamma_module.gamma.requires_grad_(False)
        if verbose:
            print(f"  γ frozen at 0")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  trainable params: {n_trainable}")

    tracker   = Tracker(label)
    global_it = 0

    # --- Adam ---
    global_it = _run_adam_tracked(
        model=model, op=op,
        phase_iters=CFG["adam_iters"], phase_lrs=CFG["adam_lrs"],
        sigma_s_t=sigma_s_t, Yq_t=Yq_t, Yq_np=Yq_np,
        wq=wq, P=P, sigma_bem=sigma_bem,
        label=label, tracker=tracker, offset=0,
        log_every=CFG["adam_log_every"], eval_every=CFG["eval_every"],
        n_grid_coarse=CFG["n_grid_coarse"], verbose=verbose,
    )
    adam_n_iters = global_it
    adam_boundary = global_it

    # --- L-BFGS ---
    lbfgs_cfg = LBFGSConfig(
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
    global_it, lbfgs_reason = _run_lbfgs_tracked(
        model=model, op=op, cfg_lbfgs=lbfgs_cfg,
        sigma_s_t=sigma_s_t, Yq_t=Yq_t, Yq_np=Yq_np,
        wq=wq, P=P, sigma_bem=sigma_bem,
        label=label, tracker=tracker, offset=adam_n_iters,
        log_every=CFG["lbfgs_log_every"], eval_every=CFG["eval_every"],
        n_grid_coarse=CFG["n_grid_coarse"], verbose=verbose,
    )

    # Final density and reconstruction
    with torch.no_grad():
        sigma_final = model(Yq_t, sigma_s_t).numpy()

    final_out = reconstruct_interior(
        P=P, Yq=Yq_np, wq=wq, sigma=sigma_final,
        n_grid=CFG["n_grid_final"], u_exact=u_exact,
        x_range=(-1.0, 1.0), y_range=(-1.0, 1.0),
    )

    density_rel_diff = float(
        np.linalg.norm(sigma_final - sigma_bem)
        / max(np.linalg.norm(sigma_bem), 1e-14)
    )
    t_total = time.perf_counter() - t0

    gamma_val = model.gamma_value()
    gv = float(gamma_val) if not isinstance(gamma_val, list) else gamma_val[0]

    if verbose:
        print(f"\n  {label} final:")
        print(f"    Interior rel L2  : {final_out.rel_L2:.3e}")
        print(f"    Interior L∞      : {final_out.linf:.3e}")
        print(f"    Density rel-diff : {density_rel_diff:.3e}")
        print(f"    γ final          : {gv:.6f}")
        print(f"    LBFGS reason     : {lbfgs_reason}")
        print(f"    Wall time        : {t_total:.1f}s")

    return dict(
        label            = label,
        tracker          = tracker,
        final_rel_L2     = final_out.rel_L2,
        final_linf       = final_out.linf,
        density_rel_diff = density_rel_diff,
        gamma_final      = gv,
        lbfgs_reason     = lbfgs_reason,
        wall_time        = t_total,
        sigma_final      = sigma_final,
        sigma_sorted     = sigma_final[sort_idx],
        adam_boundary    = adam_boundary,
        final_out        = final_out,
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

COLORS = {"BINN": "#1f77b4", "SE-BINN": "#d62728", "SE-BINN*": "#2ca02c"}
LSMAP  = {"BINN": "-",       "SE-BINN": "--",       "SE-BINN*": "-."}


def _fig_ab_comparison(cases, sigma_bem_sorted, arc, vertex_arcs,
                        singular_idx, outpath):
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    fig.subplots_adjust(hspace=0.42)

    ax = axes[0]
    for c in cases:
        t  = c["tracker"]
        it = np.array(t.iters)
        ls = np.array(t.losses)
        col = COLORS[c["label"]]
        lsn = LSMAP[c["label"]]
        adam_b = c["adam_boundary"]
        mask_a = it <= adam_b
        mask_l = it > adam_b
        ax.semilogy(it[mask_a], ls[mask_a], color=col, lw=1.4, ls=lsn,
                    alpha=0.9, label=f"{c['label']} (Adam)")
        if mask_l.sum():
            ax.semilogy(it[mask_l], ls[mask_l], color=col, lw=1.4, ls=lsn,
                        alpha=0.9, label=f"{c['label']} (LBFGS)")
        ax.axvline(adam_b, color=col, lw=0.6, ls=":", alpha=0.4)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("(a) Loss history", fontsize=12)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    ax = axes[1]
    # vertical dashed lines at reentrant corner
    for ci in singular_idx:
        ax.axvline(vertex_arcs[ci], color="#999999", lw=0.8, ls="--", alpha=0.7)
    ax.plot(arc, sigma_bem_sorted, color="black", lw=1.2, label=r"$\sigma_\mathrm{BEM}$")
    for c in cases:
        col = COLORS[c["label"]]
        lsn = LSMAP[c["label"]]
        ax.plot(arc, c["sigma_sorted"], color=col, lw=1.0, ls=lsn, alpha=0.85,
                label=c["label"])
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$\sigma(s)$", fontsize=11)
    ax.set_title("(b) Boundary density comparison", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, lw=0.3, alpha=0.5)

    ax = axes[2]
    for ci in singular_idx:
        ax.axvline(vertex_arcs[ci], color="#999999", lw=0.8, ls="--", alpha=0.7)
    for c in cases:
        col = COLORS[c["label"]]
        lsn = LSMAP[c["label"]]
        err = np.abs(c["sigma_sorted"] - sigma_bem_sorted)
        err = np.maximum(err, 1e-14)
        ax.semilogy(arc, err, color=col, lw=1.0, ls=lsn, alpha=0.85, label=c["label"])
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$|\sigma - \sigma_\mathrm{BEM}|$", fontsize=11)
    ax.set_title("(c) Pointwise density error", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(
        r"A/B/C Comparison: L-shape, $u = r^{2/3}\sin(2\theta/3)$" + "\n"
        r"BINN ($\gamma=0$) vs SE-BINN ($\gamma$ free) vs SE-BINN★ (warm $\gamma$)",
        fontsize=12, y=1.01,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved ab_comparison → {outpath}")


def _fig_convergence(cases, outpath):
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    fig.subplots_adjust(hspace=0.42)

    ax = axes[0]
    for c in cases:
        t   = c["tracker"]
        it  = np.array(t.iters)
        ls  = np.array(t.losses)
        col = COLORS[c["label"]]
        lsn = LSMAP[c["label"]]
        ax.semilogy(it, ls, color=col, lw=1.3, ls=lsn, alpha=0.9, label=c["label"])
        ax.axvline(c["adam_boundary"], color=col, lw=0.6, ls=":", alpha=0.4)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("(a) Loss vs iteration", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    ax = axes[1]
    for c in cases:
        t   = c["tracker"]
        it  = np.array(t.iters)
        de  = np.array(t.d_errs)
        col = COLORS[c["label"]]
        lsn = LSMAP[c["label"]]
        ax.semilogy(it, de, color=col, lw=1.3, ls=lsn, alpha=0.9, label=c["label"])
        ax.axvline(c["adam_boundary"], color=col, lw=0.6, ls=":", alpha=0.4)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(r"$\|\sigma_\theta - \sigma_\mathrm{BEM}\|/\|\sigma_\mathrm{BEM}\|$",
                  fontsize=10)
    ax.set_title("(b) Density rel-diff vs BEM  [KEY: plateau check]", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    ax = axes[2]
    for c in cases:
        if c["label"] == "BINN":
            continue
        t   = c["tracker"]
        it  = np.array(t.iters)
        gam = np.array(t.gammas)
        col = COLORS[c["label"]]
        lsn = LSMAP[c["label"]]
        ax.plot(it, gam, color=col, lw=1.3, ls=lsn, alpha=0.9, label=c["label"])
        ax.axvline(c["adam_boundary"], color=col, lw=0.6, ls=":", alpha=0.4)
    ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(r"$\gamma$", fontsize=11)
    ax.set_title(r"(c) $\gamma$ trajectory (SE-BINN cases)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)

    fig.suptitle(r"Convergence diagnostics — L-shape, $u = r^{2/3}\sin(2\theta/3)$",
                 fontsize=12, y=1.01)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved convergence_diagnostics → {outpath}")


def _fig_interior(bem_out, best_case, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.3)

    final_out = best_case["final_out"]

    def _imshow(ax, data, title, cmap="RdBu_r", symm=True):
        Ugrid = data
        vmax = np.nanmax(np.abs(Ugrid)) if symm else None
        vmin = -vmax if symm else None
        im = ax.imshow(Ugrid, origin="lower", cmap=cmap,
                       extent=(-1, 1, -1, 1), vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
        plt.colorbar(im, ax=ax, shrink=0.8)

    _imshow(axes[0], bem_out.Ugrid, r"BEM reference $u_\mathrm{BEM}$")
    _imshow(axes[1], final_out.Ugrid,
            rf"SE-BINN★ $u_\theta$", cmap="RdBu_r")

    # Error in log10 scale
    Egrid = np.abs(final_out.Egrid)
    Egrid = np.where(np.isnan(Egrid), np.nan, np.maximum(Egrid, 1e-10))
    im = axes[2].imshow(np.log10(Egrid), origin="lower", cmap="hot_r",
                        extent=(-1, 1, -1, 1))
    axes[2].set_title(r"$\log_{10}|u_\theta - u_\mathrm{exact}|$", fontsize=11)
    axes[2].set_xlabel("$x$"); axes[2].set_ylabel("$y$")
    plt.colorbar(im, ax=axes[2], shrink=0.8)

    # Mark reentrant corner
    for ax in axes:
        ax.plot(0, 0, "w+", ms=10, mew=2, label="reentrant corner")

    fig.suptitle(
        r"Interior solution — L-shape, $u = r^{2/3}\sin(2\theta/3)$" + "\n"
        f"SE-BINN★ rel L2 = {final_out.rel_L2:.3e}  |  "
        f"BEM rel L2 = {bem_out.rel_L2:.3e}",
        fontsize=12,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved interior_error → {outpath}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_table(cases, bem_rel_L2, bem_linf, gamma_lstsq, energy_frac):
    print()
    print("=" * 70)
    print("  L-SHAPE BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  BEM reference         : rel_L2={bem_rel_L2:.3e} | linf={bem_linf:.3e}")
    print(f"  γ*_lstsq              : {gamma_lstsq:.6f}")
    print(f"  Enrichment energy     : {energy_frac*100:.2f}%  "
          f"(vs Koch ~2.8%  — hypothesis: should be larger)")
    print()
    cols = ["BINN (A)", "SE-BINN (B)", "SE-BINN★ (C)"]
    header = f"  {'Metric':<26} | " + " | ".join(f"{c:>14}" for c in cols)
    sep    = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)

    def row(name, key):
        vals = [f"{c[key]:.3e}" if isinstance(c[key], float) else str(c[key])
                for c in cases]
        print(f"  {name:<26} | " + " | ".join(f"{v:>14}" for v in vals))

    row("Interior rel L2",  "final_rel_L2")
    row("Interior L∞",      "final_linf")
    row("Density rel-diff",  "density_rel_diff")
    print(sep)

    # gamma
    gv = [f"(frozen)" if c["label"] == "BINN"
          else f"{c['gamma_final']:.6f}" for c in cases]
    print(f"  {'γ final':<26} | " + " | ".join(f"{v:>14}" for v in gv))

    # LBFGS reason
    rv = [c["lbfgs_reason"] for c in cases]
    print(f"  {'LBFGS reason':<26} | " + " | ".join(f"{v:>14}" for v in rv))

    # wall time
    wv = [f"{c['wall_time']:.1f}s" for c in cases]
    print(f"  {'Wall time':<26} | " + " | ".join(f"{v:>14}" for v in wv))

    print(sep)
    rL2 = [c["final_rel_L2"] for c in cases]
    if rL2[0] > 0 and rL2[1] > 0:
        ab = rL2[0] / rL2[1]
        ac = rL2[0] / rL2[2] if rL2[2] > 0 else float("nan")
        print(f"\n  A/B improvement (BINN→SE-BINN)   : {ab:.2f}×")
        print(f"  A/C improvement (BINN→SE-BINN★)  : {ac:.2f}×")
    de = [c["density_rel_diff"] for c in cases]
    if de[0] > 0 and de[1] > 0:
        abde = de[0] / de[1]
        print(f"  A/B density error improvement    : {abde:.2f}×")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("=" * 65)
    print("  Experiment 2: SE-BINN on L-shaped domain, u = r^(2/3) sin(2θ/3)")
    print("=" * 65)

    t_global = time.perf_counter()
    figures_dir = os.path.join(_HERE, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Geometry, quadrature
    # ------------------------------------------------------------------
    print("\n--- Setup ---")
    geom   = make_lshape_geometry()
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=CFG["n_per_edge"])
    label_corner_ring_panels(panels, P)

    print(f"  L-shape: vertices={geom.n_vertices} | panels={len(panels)}")
    print(f"  singular corners: {geom.singular_corner_indices} "
          f"→ ω={geom.corner_angles[3]:.6f} rad = 3π/2")
    alpha_sing = np.pi / geom.corner_angles[3] - 1
    print(f"  α = π/ω − 1 = {alpha_sing:.6f}  (Koch had {np.pi/(4*np.pi/3)-1:.6f})")

    qdata  = build_panel_quadrature(panels, p=CFG["p_gl"])
    Yq_np  = qdata.Yq.T       # (Nq, 2)
    wq     = qdata.wq
    Nq     = qdata.n_quad

    arc, vertex_arcs = _boundary_arclength(qdata, CFG["n_per_edge"])
    sort_idx = np.argsort(arc)

    print(f"  Quadrature: Nq={Nq} | total arc = {float(qdata.L_panel.sum()):.4f}")

    # ------------------------------------------------------------------
    # 2. BEM reference + validation
    # ------------------------------------------------------------------
    print("\n--- BEM reference ---")
    nmat     = assemble_nystrom_matrix(qdata)
    f_bnd    = u_exact(Yq_np)
    bem_sol  = solve_bem(nmat, f_bnd,
                         tol=CFG["gmres_tol"], max_iter=CFG["gmres_maxiter"])
    sigma_bem = bem_sol.sigma
    print(f"  BEM solve: flag={bem_sol.flag} | rel_res={bem_sol.rel_res:.3e} | "
          f"direct={bem_sol.used_direct}")

    bem_out = reconstruct_interior(
        P=P, Yq=Yq_np, wq=wq, sigma=sigma_bem,
        n_grid=CFG["n_grid_final"], u_exact=u_exact,
        x_range=(-1.0, 1.0), y_range=(-1.0, 1.0),
    )
    print(f"  BEM interior: rel_L2={bem_out.rel_L2:.3e} | linf={bem_out.linf:.3e}")
    assert bem_out.rel_L2 < 5e-3, f"BEM interior error too large: {bem_out.rel_L2:.3e}"

    # Check σ_BEM spike
    reentrant_v = geom.vertices[3]
    dist_to_corner = np.linalg.norm(Yq_np - reentrant_v[None, :], axis=1)
    # Closest 5 quadrature nodes
    nearest_idx = np.argsort(dist_to_corner)[:5]
    print(f"  σ_BEM at reentrant corner (0,0): max|σ|={np.max(np.abs(sigma_bem)):.4f}")
    print(f"  Nearest-corner values: {sigma_bem[nearest_idx]}")

    # Validate spike is stronger than Koch
    print(f"  (For reference, Koch(1) has r^{{-1/4}} spike; L-shape has r^{{-1/3}})")

    # ------------------------------------------------------------------
    # 3. Enrichment diagnostic: lstsq projection
    # ------------------------------------------------------------------
    print("\n--- Enrichment diagnostic ---")
    enrichment = SingularEnrichment(geom=geom, per_corner_gamma=False)
    sigma_s_np = enrichment.precompute(Yq_np)   # (Nq,)

    # Unweighted lstsq: γ*_lstsq = (σ_s · σ_BEM) / (σ_s · σ_s)
    # NOTE: γ*_lstsq is the GLOBAL L2 projection coefficient.  It is NOT the
    # physical singular amplitude, which is σ_BEM / σ_s as r→0 (≈ 1.0 here).
    # The large γ*_lstsq ≈ 5.4 arises because ||σ_s|| ≪ ||σ_BEM||; the
    # physical fraction ||σ_s||² / ||σ_BEM||² ≈ 1.3% is much smaller than the
    # lstsq energy fraction below.
    gamma_lstsq     = float(np.dot(sigma_s_np, sigma_bem) / np.dot(sigma_s_np, sigma_s_np))
    sigma_proj      = gamma_lstsq * sigma_s_np
    res_norm        = np.linalg.norm(sigma_bem - sigma_proj)
    bem_norm        = np.linalg.norm(sigma_bem)
    energy_frac     = 1.0 - (res_norm / bem_norm) ** 2  # lstsq R²
    phys_energy_frac = np.dot(sigma_s_np, sigma_s_np) / max(bem_norm**2, 1e-14)

    print(f"  σ_s range: [{sigma_s_np.min():.4f}, {sigma_s_np.max():.4f}]")
    print(f"  γ*_lstsq                        : {gamma_lstsq:.6f}")
    print(f"  Lstsq energy frac (R²)          : {energy_frac*100:.2f}%")
    print(f"  Physical energy frac ||σ_s||²/||σ_BEM||² : {phys_energy_frac*100:.2f}%")
    print(f"  Residual ||σ_BEM - γ*σ_s|| / ||σ_BEM|| : {res_norm/bem_norm:.4f}")

    # ------------------------------------------------------------------
    # 4. Operator state (shared across all cases)
    # ------------------------------------------------------------------
    print("\n--- Operator setup ---")
    w_panel = panel_loss_weights(panels, w_base=1.0, w_corner=1.0, w_ring=1.0)
    colloc  = build_collocation_points(panels, m_col_panel=CFG["m_col_base"])
    op, op_diag = build_operator_state(
        colloc=colloc, qdata=qdata, enrichment=enrichment, g=u_exact,
        panel_weights=w_panel,
        eq_scale_mode=CFG["eq_scale_mode"], eq_scale_fixed=CFG["eq_scale_fixed"],
        dtype=torch.float64, device="cpu",
    )
    print(f"  Nb={colloc.n_colloc} | eq_scale={op_diag['eq_scale']:.2e}")

    sigma_s_t = torch.tensor(sigma_s_np, dtype=torch.float64)
    Yq_t      = torch.tensor(Yq_np,      dtype=torch.float64)

    # ------------------------------------------------------------------
    # 5. Shared initial model
    # ------------------------------------------------------------------
    torch.manual_seed(CFG["seed"])
    init_model = SEBINNModel(
        hidden_width=CFG["hidden_width"], n_hidden=CFG["n_hidden"],
        n_gamma=enrichment.n_gamma, gamma_init=CFG["gamma_init"],
        dtype=torch.float64,
    )
    init_state = copy.deepcopy(init_model.state_dict())
    print(f"  Model: n_params={init_model.n_params()} | n_gamma={enrichment.n_gamma}")
    print(f"  Initial state saved — all 3 cases start from here.")

    shared = dict(
        op=op, enrichment=enrichment,
        Yq_np=Yq_np, wq=wq, P=P, sigma_bem=sigma_bem,
        sigma_s_t=sigma_s_t, Yq_t=Yq_t, sort_idx=sort_idx,
    )

    # ------------------------------------------------------------------
    # 6. Case A: BINN (γ frozen)
    # ------------------------------------------------------------------
    res_a = _train_case("BINN",    init_state,
                         gamma_override=None, freeze_gamma=True,
                         shared=shared, verbose=True)

    # ------------------------------------------------------------------
    # 7. Case B: SE-BINN (γ free, init=0)
    # ------------------------------------------------------------------
    res_b = _train_case("SE-BINN", init_state,
                         gamma_override=None, freeze_gamma=False,
                         shared=shared, verbose=True)

    # ------------------------------------------------------------------
    # 8. Case C: SE-BINN* (γ free, warm-start)
    # ------------------------------------------------------------------
    res_c = _train_case("SE-BINN*", init_state,
                         gamma_override=gamma_lstsq, freeze_gamma=False,
                         shared=shared, verbose=True)

    # ------------------------------------------------------------------
    # 9. Summary table
    # ------------------------------------------------------------------
    cases = [res_a, res_b, res_c]
    _print_table(cases, bem_out.rel_L2, bem_out.linf, gamma_lstsq, energy_frac)

    # Plateau check for best case
    best = min(cases, key=lambda c: c["final_rel_L2"])
    d_errs = np.array(best["tracker"].d_errs)
    iters  = np.array(best["tracker"].iters)
    last_500 = d_errs[iters > iters[-1] - 500]
    if len(last_500) >= 2:
        rel_ch = abs(last_500[-1] - last_500[0]) / max(last_500[0], 1e-14)
        print(f"\n  Best case ({best['label']}) density error "
              f"over last 500 iters: {last_500[0]:.4f} → {last_500[-1]:.4f}")
        print(f"  Relative change: {rel_ch:.2%}")
        if rel_ch < 0.005:
            print("  *** PLATEAU — architecture/landscape bottleneck ***")
        else:
            print("  *** Still decreasing — more budget would help ***")

    total_wall = time.perf_counter() - t_global
    print(f"\n  Total experiment wall time: {total_wall:.1f}s")

    # ------------------------------------------------------------------
    # 10. Figures
    # ------------------------------------------------------------------
    print("\n--- Generating figures ---")
    singular_idx = list(geom.singular_corner_indices)

    _fig_ab_comparison(
        cases=cases,
        sigma_bem_sorted=sigma_bem[sort_idx],
        arc=arc[sort_idx], vertex_arcs=vertex_arcs,
        singular_idx=singular_idx,
        outpath=os.path.join(figures_dir, "ab_comparison.png"),
    )

    _fig_convergence(
        cases=cases,
        outpath=os.path.join(figures_dir, "convergence_diagnostics.png"),
    )

    # Best case for interior plot
    _fig_interior(
        bem_out=bem_out,
        best_case=best,
        outpath=os.path.join(figures_dir, "interior_error.png"),
    )

    print("Done.")


if __name__ == "__main__":
    main()
