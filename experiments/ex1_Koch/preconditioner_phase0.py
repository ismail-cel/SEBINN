"""
Experiment: Koch(1) preconditioner phase 0 — conditioning vs capacity diagnostic.

Question
--------
Is the density plateau (~45–51%) seen in all Koch(1) training runs caused by
(a) ill-conditioning of V  (standard loss gives poor gradients for smooth modes)
(b) limited network capacity  (gradient direction is fine, but the net saturates)

Method — exact preconditioner V_h^{-1}
---------------------------------------
  L_std (σ) = ||V_h σ − g||²
  L_prec(σ) = ||V_h^{-1}(V_h σ − g)||² ≈ ||σ − σ_BEM||²

V_h^{-1} is the dense inverse of the Nyström matrix (Nq×Nq).  It maps the
BIE residual directly onto the density error, so every Fourier-like mode of σ
receives equal gradient weight regardless of the singular values of V_h.

2×2 design (all from identical initial weights, same manufactured density)
-----------
  Case A — BINN    + standard loss      (γ frozen=0, L_std)
  Case B — SE-BINN + standard loss      (γ trainable, L_std)
  Case C — BINN    + preconditioned loss (γ frozen=0, L_prec)
  Case D — SE-BINN + preconditioned loss (γ trainable, L_prec)

All cases use eq_scale_mode='none'  (eq_scale = 1.0).

Hypotheses
----------
  H1 (conditioning): C/D substantially outperform A/B on density rel-diff
                     → V-conditioning is the primary bottleneck.
  H2 (capacity):     All four cases plateau at similar density rel-diff
                     → network capacity / architecture is the bottleneck,
                       not gradient quality.

Manufactured density (same as manufactured_singular.py)
---------------------------------------------------------
  σ_mfg = σ_smooth + S @ γ_true,   γ_true = [+1,−0.5,+1,−0.5,+1,−0.5]
  g_mfg = V · σ_mfg
  eq_scale_mode = 'none'  →  op.f = g_mfg  (no scaling)

Figures saved to experiments/ex1_Koch/figures/:
  phase0_density_convergence.png  — density rel-diff vs iteration  ← KEY PLOT
  phase0_comparison.png           — final density profiles and pointwise errors
  phase0_loss.png                 — loss histories (log scale)
"""

import sys
import os
import copy
import time

import numpy as np
import scipy.linalg as la
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
    seed             = 0,
    n_per_edge       = 12,
    p_gl             = 16,
    m_col_base       = 16,    # = p_gl  →  Xc = Yq, Nb = Nq
    w_base           = 1.0,
    w_corner         = 1.0,
    w_ring           = 1.0,
    eq_scale_mode    = "none",   # IMPORTANT: eq_scale = 1.0 for ALL cases
    gamma_true       = np.array([+1.0, -0.5, +1.0, -0.5, +1.0, -0.5]),
    gamma_init       = 0.0,
    hidden_width     = 80,
    n_hidden         = 4,
    # Adam
    adam_iters       = [1000, 1000, 1000],
    adam_lrs         = [1e-3, 3e-4, 1e-4],
    log_every        = 200,
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
    # Interior reconstruction
    n_grid_coarse    = 51,
    n_grid_final     = 201,
)


# ===========================================================================
# Preconditioned loss
# ===========================================================================

def preconditioned_loss(
    model,
    op,
) -> tuple:
    """
    Preconditioned BIE loss: L_prec = ||V_h^{-1}(V_h σ − g)||²

    Since V_h^{-1}(V_h σ − g) ≈ σ − σ_BEM, this loss measures the density
    error directly, giving uniform gradient weight to all density modes
    regardless of the singular values of V_h.

    Requires op.V_inv  — set externally as torch.Tensor (Nq, Nq).
    """
    # Enriched density at quadrature and collocation nodes
    sigma_std = model(op.Yq, op.sigma_s_q)       # (Nq,)
    sigma_c   = model(op.Xc, op.sigma_s_c)       # (Nb,)

    # BIE residual  r = V_h σ − g  (unscaled, eq_scale=1.0)
    Vstd = op.A @ sigma_std                      # (Nb,)
    Vsig = Vstd + op.corr * sigma_c             # (Nb,)
    res  = Vsig - op.f                           # (Nb,)

    # Apply exact preconditioner: ẑ = V_h^{-1} r  ≈  σ − σ_BEM
    prec_res = op.V_inv @ res                    # (Nb,)

    # Weighted MSE of preconditioned residual
    prec_res2 = prec_res ** 2                    # (Nb,)
    loss = (op.wCol * prec_res2).sum() / op.wCol_sum

    with torch.no_grad():
        dbg = {
            "mean_abs_res":  float(res.detach().abs().mean()),
            "mean_abs_prec": float(prec_res.detach().abs().mean()),
            "mse_scaled":    float(prec_res2.detach().mean()),
            "mse_unscaled":  float(prec_res2.detach().mean()),   # eq_scale=1
            "loss":          float(loss.detach()),
            "gamma":         model.gamma_value(),
        }

    return loss, dbg


def standard_loss(model, op) -> tuple:
    """
    Standard BIE loss: L_std = wCol-weighted mean(res²),  res = V_h σ − g.
    Replicates sebinn_loss without importing it (eq_scale=1.0 here).
    """
    from src.training.loss import sebinn_loss
    return sebinn_loss(model, op)


# ===========================================================================
# Helpers
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


def make_u_exact_fn(Yq_T, wq, sigma_mfg):
    sigma_wq = sigma_mfg * wq

    def _u(xy):
        K = _log_kernel_matrix(xy, Yq_T)
        return K @ sigma_wq

    return _u


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
# Training run
# ===========================================================================

def _train_one(
    label: str,
    freeze_gamma: bool,
    use_precond: bool,
    init_state: dict,
    shared: dict,
    verbose: bool = True,
) -> dict:
    if verbose:
        loss_name = "L_prec" if use_precond else "L_std"
        fz = " [γ frozen=0]" if freeze_gamma else " [γ trainable]"
        print(f"\n{'='*68}")
        print(f"  Case {label} | {loss_name}{fz}")
        print(f"{'='*68}")

    t0 = time.perf_counter()

    op         = shared["op"]
    Yq_T       = shared["Yq_T"]
    wq         = shared["wq"]
    P          = shared["P"]
    sigma_bem  = shared["sigma_bem"]
    sigma_s_Yq = shared["sigma_s_Yq"]   # (Nq, 6)
    sort_idx   = shared["sort_idx"]
    u_exact    = shared["u_exact"]
    n_gamma    = sigma_s_Yq.shape[1]

    loss_fn = preconditioned_loss if use_precond else standard_loss

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
        print(f"  trainable params: {n_tr} | n_gamma={n_gamma} | loss={loss_fn.__name__}")

    Yq_t      = torch.tensor(Yq_T,       dtype=torch.float64)
    sigma_s_t = torch.tensor(sigma_s_Yq,  dtype=torch.float64)

    # Trajectory: (global_iter, density_rel_diff, loss_val, gamma_arr)
    traj = []

    def _density_reldiff():
        with torch.no_grad():
            s = model(Yq_t, sigma_s_t).numpy()
        return float(np.linalg.norm(s - sigma_bem)
                     / max(np.linalg.norm(sigma_bem), 1e-14)), s

    stage_checkpoints = []

    def _record(stage, n_iter, loss_val):
        d_err, s = _density_reldiff()
        i_out = reconstruct_interior(
            P=P, Yq=Yq_T, wq=wq, sigma=s,
            n_grid=CFG["n_grid_coarse"], u_exact=u_exact,
        )
        g = model.gamma_value()
        g_arr = np.array(g if isinstance(g, list) else [float(g)])
        traj.append((n_iter, d_err, loss_val, g_arr.copy()))
        stage_checkpoints.append(dict(
            stage=stage, iter=n_iter, loss=loss_val,
            density_rel_diff=d_err, interior_L2=i_out.rel_L2,
            gamma=g_arr.copy(),
        ))
        if verbose:
            gfmt = "[" + ",".join(f"{v:+.4f}" for v in g_arr) + "]"
            print(f"  [{label}] {stage}: loss={loss_val:.3e} | "
                  f"d_err={d_err:.4f} | iL2={i_out.rel_L2:.3e} | γ={gfmt}")

    # --- Adam phases ---
    all_adam_loss = []
    global_it = 0
    for ph_idx, (n_it, lr) in enumerate(zip(CFG["adam_iters"], CFG["adam_lrs"])):
        ph_cfg = AdamConfig(
            phase_iters=[n_it], phase_lrs=[lr], log_every=CFG["log_every"],
        )
        ph_res = run_adam_phases(model, op, ph_cfg, verbose=verbose,
                                 loss_fn=loss_fn)
        all_adam_loss.extend(ph_res.loss_hist)
        global_it += ph_res.n_iters
        _record(f"Adam-ph{ph_idx+1}", global_it, ph_res.final_loss)

    # --- L-BFGS ---
    lbfgs_cfg = _make_lbfgs_cfg()
    if verbose:
        print(f"\n  [{label}] L-BFGS: max={CFG['lbfgs_max_iters']} | "
              f"mem={CFG['lbfgs_memory']} | grad_tol={CFG['lbfgs_grad_tol']:.0e}")
    lbfgs_res = run_lbfgs(model, op, lbfgs_cfg, verbose=verbose,
                          loss_fn=loss_fn)
    _record(
        "LBFGS", global_it + lbfgs_res.n_iters,
        lbfgs_res.loss_hist[-1] if lbfgs_res.loss_hist else float("nan"),
    )

    # --- Final metrics ---
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
        gamma_true = CFG["gamma_true"]
        g_err = float(np.linalg.norm(garr - gamma_true)
                      / max(np.linalg.norm(gamma_true), 1e-14))
        print(f"\n  {label} final:")
        print(f"    Interior rel L2  : {final_out.rel_L2:.3e}")
        print(f"    Interior L∞      : {final_out.linf:.3e}")
        print(f"    Density rel-diff : {density_rel_diff:.4f}")
        print(f"    γ final          : {gfmt}")
        print(f"    γ_true           : {list(np.round(gamma_true, 3))}")
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
        label            = label,
        use_precond      = use_precond,
        freeze_gamma     = freeze_gamma,
        final_rel_L2     = final_out.rel_L2,
        final_linf       = final_out.linf,
        density_rel_diff = density_rel_diff,
        gamma_final      = gamma_final,
        gamma_err        = gamma_err,
        lbfgs_reason     = lbfgs_res.reason,
        wall_time        = t_total,
        loss_hist_adam   = all_adam_loss,
        loss_hist_lbfgs  = list(lbfgs_res.loss_hist),
        adam_n_iters     = global_it,
        sigma_final      = sigma_final[sort_idx],
        stage_checkpoints= stage_checkpoints,
        traj             = traj,   # (iter, d_err, loss, gamma_arr) at each stage
        final_out        = final_out,
    )


# ===========================================================================
# Figures
# ===========================================================================

COLORS = {
    "A (BINN, std)":     "#1f77b4",   # blue
    "B (SE-BINN, std)":  "#ff7f0e",   # orange
    "C (BINN, prec)":    "#2ca02c",   # green
    "D (SE-BINN, prec)": "#d62728",   # red
}
LS = {
    "A (BINN, std)":     "-",
    "B (SE-BINN, std)":  "--",
    "C (BINN, prec)":    "-.",
    "D (SE-BINN, prec)": ":",
}


def _fig_density_convergence(cases, outpath):
    """
    KEY PLOT — density rel-diff vs training iteration for all 4 cases.

    This directly answers: do C/D (preconditioned) achieve lower plateau
    than A/B (standard)?  If yes → conditioning bottleneck.  If no → capacity.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for c in cases:
        col = COLORS.get(c["label"], "gray")
        ls  = LS.get(c["label"], "-")
        traj = c["traj"]   # list of (iter, d_err, loss, gamma_arr)
        if not traj:
            continue
        iters  = np.array([t[0] for t in traj])
        d_errs = np.array([t[1] for t in traj])
        ax.semilogy(iters, d_errs, color=col, ls=ls, lw=2.0, marker="o",
                    markersize=5, label=c["label"])

    # Mark Adam→L-BFGS transition
    n_adam = cases[0]["adam_n_iters"]
    ax.axvline(n_adam, color="gray", lw=0.9, ls=":", alpha=0.6,
               label=f"Adam→L-BFGS ({n_adam})")

    ax.set_xlabel("Training iteration (end of stage)", fontsize=12)
    ax.set_ylabel(r"Density rel-diff  $\|\sigma_\theta - \sigma_\mathrm{BEM}\|/\|\sigma_\mathrm{BEM}\|$",
                  fontsize=11)
    ax.set_title(
        r"Density convergence — standard vs preconditioned loss" + "\n"
        r"Koch(1) manufactured density, eq\_scale=none",
        fontsize=12,
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved phase0_density_convergence → {outpath}")


def _fig_comparison(cases, sigma_bem_sorted, arc_s, vertex_arcs, sing_idx, outpath):
    """Final density profiles and pointwise errors for all 4 cases."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    fig.subplots_adjust(hspace=0.38)

    def _vlines(ax):
        for ci in sing_idx:
            ax.axvline(vertex_arcs[ci], color="#aaaaaa", lw=0.7, ls="--", alpha=0.7)

    ax = axes[0]
    _vlines(ax)
    ax.plot(arc_s, sigma_bem_sorted, color="black", lw=1.6, alpha=0.9,
            label=r"$\sigma_\mathrm{mfg}$")
    for c in cases:
        col = COLORS.get(c["label"], "gray")
        ls  = LS.get(c["label"], "-")
        ax.plot(arc_s, c["sigma_final"], color=col, lw=1.2, ls=ls,
                alpha=0.85, label=c["label"])
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$\sigma(s)$", fontsize=11)
    ax.set_title(r"(a) Final boundary density", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, lw=0.3, alpha=0.5)

    ax = axes[1]
    _vlines(ax)
    for c in cases:
        col = COLORS.get(c["label"], "gray")
        ls  = LS.get(c["label"], "-")
        err = np.abs(c["sigma_final"] - sigma_bem_sorted)
        ax.semilogy(arc_s, err + 1e-16, color=col, lw=1.2, ls=ls,
                    alpha=0.85,
                    label=f"{c['label']}  d_err={c['density_rel_diff']:.4f}")
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$|\sigma_\theta - \sigma_\mathrm{mfg}|$", fontsize=11)
    ax.set_title("(b) Pointwise density error", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(
        r"Preconditioner phase 0 — Koch(1) manufactured density",
        fontsize=12, y=1.01,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved phase0_comparison → {outpath}")


def _fig_loss(cases, outpath):
    """Loss histories (log scale) for all 4 cases."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.32)

    ax = axes[0]
    for c in cases:
        col = COLORS.get(c["label"], "gray")
        ls  = LS.get(c["label"], "-")
        ha = c["loss_hist_adam"]
        hl = c["loss_hist_lbfgs"]
        n_a, n_l = len(ha), len(hl)
        ax.semilogy(np.arange(1, n_a+1), ha, color=col, lw=1.4, ls="-",
                    alpha=0.85, label=f"{c['label']} Adam")
        if n_l:
            ax.semilogy(np.arange(n_a+1, n_a+n_l+1), hl, color=col, lw=1.4,
                        ls="--", alpha=0.85)
    n_adam = cases[0]["adam_n_iters"]
    ax.axvline(n_adam, color="gray", lw=0.9, ls=":", alpha=0.6)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("(a) All cases", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    # Split: standard vs preconditioned
    ax = axes[1]
    for c in cases:
        col = COLORS.get(c["label"], "gray")
        ls  = LS.get(c["label"], "-")
        traj = c["traj"]
        if not traj:
            continue
        iters  = np.array([t[0] for t in traj])
        losses = np.array([t[2] for t in traj])
        ax.semilogy(iters, np.maximum(losses, 1e-20), color=col, lw=1.8,
                    ls=ls, marker="o", markersize=5, label=c["label"])
    ax.set_xlabel("Iteration (end of stage)", fontsize=11)
    ax.set_ylabel("Loss (stage checkpoints)", fontsize=11)
    ax.set_title("(b) Stage-checkpoint loss", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(
        r"Loss histories — standard ($L_\mathrm{std}$) vs preconditioned ($L_\mathrm{prec}$)",
        fontsize=12,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved phase0_loss → {outpath}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("=" * 72)
    print("  Koch(1) — Preconditioner phase 0 diagnostic")
    print("=" * 72)

    t_global = time.perf_counter()
    figures_dir = os.path.join(_HERE, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Geometry and quadrature
    # ------------------------------------------------------------------
    print("\n--- Setup ---")
    geom   = make_koch_geometry(n=1)
    P      = geom.vertices
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
    sing_idx_arr = geom.singular_corner_indices

    # Min corner separation → cutoff radius
    sing_verts  = P[sing_idx_arr]
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
    # 2. Singular enrichment
    # ------------------------------------------------------------------
    enrichment = SingularEnrichment(
        geom=geom,
        use_cutoff=True,
        cutoff_radius=R_cut,
        per_corner_gamma=True,
    )
    n_sing = enrichment.n_singular
    sigma_s_Yq = enrichment.precompute(Yq_T)   # (Nq, 6)
    print(f"  n_sing = {n_sing} | sigma_s_Yq shape = {sigma_s_Yq.shape}")

    # ------------------------------------------------------------------
    # 3. Nyström matrix (for BEM and V_inv)
    # ------------------------------------------------------------------
    print("\n--- Nyström matrix + V_inv ---")
    nmat = assemble_nystrom_matrix(qdata)

    # Exact preconditioner: V_h^{-1}
    print(f"  Computing V_inv = la.inv(nmat.V)  ({Nq}×{Nq}) ...")
    t_vinv = time.perf_counter()
    V_inv_np = la.inv(nmat.V)
    print(f"  V_inv computed in {time.perf_counter()-t_vinv:.2f}s")
    print(f"  cond(V) = {np.linalg.cond(nmat.V):.3e}")
    print(f"  ‖V_inv · V − I‖_F / ‖I‖_F = "
          f"{np.linalg.norm(V_inv_np @ nmat.V - np.eye(Nq)) / np.sqrt(Nq):.3e}")

    # ------------------------------------------------------------------
    # 4. Smooth density and manufactured density
    # ------------------------------------------------------------------
    print("\n--- Manufactured density ---")
    f_smooth   = Yq_T[:, 0] ** 2 - Yq_T[:, 1] ** 2
    bem_smooth = solve_bem(nmat, f_smooth,
                           tol=1e-12, max_iter=300)
    sigma_smooth = bem_smooth.sigma
    print(f"  GMRES (smooth): flag={bem_smooth.flag} | rel_res={bem_smooth.rel_res:.3e}")

    gamma_true = CFG["gamma_true"].copy()
    sigma_sing = sigma_s_Yq @ gamma_true
    sigma_mfg  = sigma_smooth + sigma_sing

    energy_true = float(np.linalg.norm(sigma_sing) ** 2
                        / max(np.linalg.norm(sigma_mfg) ** 2, 1e-14))
    print(f"  γ_true = {list(np.round(gamma_true, 3))}")
    print(f"  ‖σ_smooth‖ = {np.linalg.norm(sigma_smooth):.4f}")
    print(f"  ‖σ_sing‖   = {np.linalg.norm(sigma_sing):.4f}")
    print(f"  ‖σ_mfg‖    = {np.linalg.norm(sigma_mfg):.4f}")
    print(f"  Enrichment energy = {energy_true*100:.2f}%")

    # ------------------------------------------------------------------
    # 5. Manufactured boundary data g_mfg = V · σ_mfg
    # ------------------------------------------------------------------
    g_mfg = nmat.V @ sigma_mfg    # (Nq,)

    bem_mfg  = solve_bem(nmat, g_mfg, tol=1e-12, max_iter=300)
    sigma_bem = bem_mfg.sigma
    sigma_rec_err = float(np.linalg.norm(sigma_bem - sigma_mfg)
                          / max(np.linalg.norm(sigma_mfg), 1e-14))
    print(f"\n  GMRES (manufactured): flag={bem_mfg.flag} | rel_res={bem_mfg.rel_res:.3e}")
    print(f"  ‖σ_BEM − σ_mfg‖/‖σ_mfg‖ = {sigma_rec_err:.3e}  (≈ machine ε)")

    # lstsq enrichment diagnostic
    gamma_lstsq, _, _, _ = np.linalg.lstsq(sigma_s_Yq, sigma_bem, rcond=None)
    sigma_proj   = sigma_s_Yq @ gamma_lstsq
    res_norm     = np.linalg.norm(sigma_bem - sigma_proj)
    bem_norm     = np.linalg.norm(sigma_bem)
    energy_lstsq = 1.0 - (res_norm / max(bem_norm, 1e-14)) ** 2
    print(f"  γ*_lstsq = {list(np.round(gamma_lstsq, 5))}")
    print(f"  Enrichment energy (lstsq) = {energy_lstsq*100:.2f}%")

    # Exact interior solution
    u_exact = make_u_exact_fn(Yq_T, wq, sigma_mfg)

    # ------------------------------------------------------------------
    # 6. Operator state (eq_scale_mode='none' → eq_scale=1.0)
    # ------------------------------------------------------------------
    print("\n--- Operator setup (eq_scale_mode='none') ---")
    colloc  = build_collocation_points(panels, m_col_panel=CFG["m_col_base"])
    w_panel = panel_loss_weights(panels, w_base=CFG["w_base"],
                                 w_corner=CFG["w_corner"], w_ring=CFG["w_ring"])

    g_dummy = lambda xy: xy[:, 0] ** 2 - xy[:, 1] ** 2
    op, op_diag = build_operator_state(
        colloc=colloc, qdata=qdata, enrichment=enrichment, g=g_dummy,
        panel_weights=w_panel,
        eq_scale_mode  = CFG["eq_scale_mode"],   # "none" → eq_scale=1.0
        eq_scale_fixed = 1.0,
        dtype=torch.float64, device="cpu",
    )

    # Override op.f with manufactured boundary data (eq_scale=1 → no scaling)
    eq_scale = op_diag["eq_scale"]
    assert abs(eq_scale - 1.0) < 1e-12, f"Expected eq_scale=1.0, got {eq_scale}"
    op.f = torch.tensor(g_mfg, dtype=torch.float64)

    # Attach V_inv to the operator state (OperatorState is not frozen)
    op.V_inv = torch.tensor(V_inv_np, dtype=torch.float64)

    print(f"  Nb={colloc.n_colloc} | Nq={Nq} | eq_scale={eq_scale:.2e}")
    print(f"  mean|A| = {op_diag['mean_abs_A_before']:.3e}")
    print(f"  op.f overridden with g_mfg (unscaled)")
    print(f"  op.V_inv attached ({Nq}×{Nq})")

    # Verify: V_inv @ (V @ sigma_bem) ≈ sigma_bem
    with torch.no_grad():
        sigma_bem_t  = torch.tensor(sigma_bem, dtype=torch.float64)
        sigma_s_bem  = torch.tensor(sigma_s_Yq @ np.zeros(n_sing), dtype=torch.float64)
        # Build Vsigma manually: A @ sigma_bem + corr * sigma_bem  (same as V @ sigma)
        Vs    = op.A @ sigma_bem_t + op.corr * sigma_bem_t
        prec  = op.V_inv @ (Vs - op.f)
        print(f"  Sanity: ‖V_inv(V·σ_BEM − g)‖/‖σ_BEM‖ = "
              f"{float(prec.norm()) / max(float(sigma_bem_t.norm()), 1e-14):.3e}  (should ≈ 0)")

    # ------------------------------------------------------------------
    # 7. Shared initial model state
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
    print(f"\n  n_params={init_model.n_params()} | n_gamma={n_sing}")
    print(f"  Shared initial state saved.")

    shared = dict(
        op=op, Yq_T=Yq_T, wq=wq, P=P,
        sigma_bem=sigma_bem, sigma_s_Yq=sigma_s_Yq,
        sort_idx=sort_idx, u_exact=u_exact,
    )

    # ------------------------------------------------------------------
    # 8. Training: four cases
    # ------------------------------------------------------------------
    res_a = _train_one("A (BINN, std)",     freeze_gamma=True,  use_precond=False,
                       init_state=init_state, shared=shared)
    res_b = _train_one("B (SE-BINN, std)",  freeze_gamma=False, use_precond=False,
                       init_state=init_state, shared=shared)
    res_c = _train_one("C (BINN, prec)",    freeze_gamma=True,  use_precond=True,
                       init_state=init_state, shared=shared)
    res_d = _train_one("D (SE-BINN, prec)", freeze_gamma=False, use_precond=True,
                       init_state=init_state, shared=shared)

    cases = [res_a, res_b, res_c, res_d]

    # ------------------------------------------------------------------
    # 9. Summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("  RESULTS — Preconditioner phase 0")
    print("=" * 80)
    print(f"  γ_true   = {list(np.round(gamma_true, 3))}")
    print(f"  γ*_lstsq = {list(np.round(gamma_lstsq, 4))}")
    print(f"  Enrichment energy (lstsq) = {energy_lstsq*100:.2f}%")
    print(f"  cond(V)  = {np.linalg.cond(nmat.V):.3e}")
    print()

    col_labels = [c["label"] for c in cases]
    W   = 22
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

    wvs = [f"{c['wall_time']:.1f}s" for c in cases]
    print(f"  {'Wall time':<28} | " + " | ".join(f"{v:>{W}}" for v in wvs))
    rvs = [c["lbfgs_reason"] for c in cases]
    print(f"  {'LBFGS reason':<28} | " + " | ".join(f"{v:>{W}}" for v in rvs))
    print(sep)

    # Interpretation
    de = {c["label"]: c["density_rel_diff"] for c in cases}
    std_best  = min(de["A (BINN, std)"], de["B (SE-BINN, std)"])
    prec_best = min(de["C (BINN, prec)"], de["D (SE-BINN, prec)"])

    print()
    print(f"  Best std  (A/B) density rel-diff : {std_best:.4f}")
    print(f"  Best prec (C/D) density rel-diff : {prec_best:.4f}")
    if std_best > 1e-6:
        ratio = std_best / max(prec_best, 1e-10)
        print(f"  Preconditioning improvement      : {ratio:.2f}×")
        if ratio > 2.0:
            print("  → Hypothesis H1 SUPPORTED: conditioning is a major bottleneck.")
        else:
            print("  → Hypothesis H2 SUPPORTED: capacity is the primary bottleneck.")

    # ------------------------------------------------------------------
    # 10. Figures
    # ------------------------------------------------------------------
    print("\n--- Figures ---")
    sigma_bem_sorted = sigma_bem[sort_idx]

    _fig_density_convergence(
        cases,
        os.path.join(figures_dir, "phase0_density_convergence.png"),
    )
    _fig_comparison(
        cases, sigma_bem_sorted, arc_s, vertex_arcs, sing_idx_arr,
        os.path.join(figures_dir, "phase0_comparison.png"),
    )
    _fig_loss(
        cases,
        os.path.join(figures_dir, "phase0_loss.png"),
    )

    print(f"\n  Total wall time: {time.perf_counter()-t_global:.1f}s")
    print("  Done.")


if __name__ == "__main__":
    main()
