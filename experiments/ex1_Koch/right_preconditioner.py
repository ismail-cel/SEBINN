"""
Experiment: Right-preconditioning via change of variable σ = −W̃ρ.

Mathematical background
-----------------------
All previous Calderón experiments tried to LEFT-precondition the standard
loss ||Vσ − g||².  Left-preconditioning requires an SPD preconditioner,
but T̃_sym has 23 negative eigenvalues — the Calderón advantage is lost as
soon as a sufficient SPD shift is applied.

RIGHT-PRECONDITIONING avoids this entirely by changing the VARIABLE,
not the loss.

Change of variable
------------------
Instead of learning σ directly, we learn ρ (smooth!) and define

    σ = −W̃ ρ

Substituting into the single-layer BIE  V σ = g:

    V (−W̃ ρ) = g
    (−V W̃) ρ = g
    B ρ = g

where B ≡ −V W̃.

Conditioning
------------
By the Calderón identity V W̃ = ¼(K̃² − I):

    cond_eig(B) = cond_eig(V W̃) ≈ 15   (matches validated CHECK 7)

This is the SAME conditioning advantage as left-preconditioning, but:
  - No SPD requirement on any operator
  - The loss ||Bρ − g||² is always a valid L2 loss
  - ρ is smooth → network approximates it well

With singular enrichment (Case E)
-----------------------------------
The enriched ansatz is

    σ = −W̃ ρ + γ σ_s

Substituting into V σ = g:

    V (−W̃ ρ + γ σ_s) = g
    B ρ + (V σ_s) γ = g    where (V σ_s) = V @ σ_s_Yq ∈ ℝ^{Nq × n_γ}

Loss:

    L_E = ||B ρ − g + (V σ_s) γ||² / Nq
        = ||B ρ + V_σs γ − g||² / Nq

Cases
-----
A: BINN       + standard loss   (γ frozen at 0)     — baseline
B: SE-BINN    + standard loss   (γ trainable)        — baseline + enrichment
C: SE-BINN    + exact V⁻¹ preconditioner             — reference / best known
D: right-prec + no enrichment   σ = −W̃ρ             — right-prec control
E: right-prec + enrichment      σ = −W̃ρ + γσ_s      — right-prec + singular

Budget: Adam [1000,1000,1000] at lr [1e-3, 3e-4, 1e-4] + LBFGS 15000.
Geometry: Koch(1), manufactured singular density.
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
from matplotlib.lines import Line2D

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
from src.quadrature.hypersingular import (
    assemble_hypersingular_direct,
    regularise_hypersingular,
)
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
    seed          = 0,
    n_per_edge    = 12,
    p_gl          = 16,
    m_col_base    = 16,
    w_base        = 1.0,
    w_corner      = 1.0,
    w_ring        = 1.0,
    eq_scale_mode = "none",
    gamma_true    = np.array([+1.0, -0.5, +1.0, -0.5, +1.0, -0.5]),
    gamma_init    = 0.0,
    hidden_width  = 80,
    n_hidden      = 4,
    cutoff_radius = 0.15,
    adam_iters    = [1000, 1000, 1000],
    adam_lrs      = [1e-3,  3e-4,  1e-4],
    lbfgs_iters   = 15000,
    lbfgs_memory  = 30,
    lbfgs_grad_tol= 1e-10,
    lbfgs_step_tol= 1e-12,
    log_every     = 200,
    lbfgs_log_every = 300,
    lbfgs_alpha0  = 1e-1,
    lbfgs_alpha_fb= [1e-2, 1e-3],
    lbfgs_armijo_c1 = 1e-4,
    lbfgs_beta    = 0.5,
    lbfgs_max_bt  = 20,
    n_grid_coarse = 51,
    n_grid_final  = 101,
)


# ===========================================================================
# Loss functions
# ===========================================================================

def standard_loss(model, op):
    from src.training.loss import sebinn_loss
    return sebinn_loss(model, op)


def exact_precond_loss(model, op):
    """L = ||V_h^{-1}(Vσ − g)||²  (exact Calderón left-preconditioner)."""
    res  = op.A @ model(op.Yq, op.sigma_s_q) + op.corr * model(op.Xc, op.sigma_s_c) - op.f
    prec = op.V_inv @ res
    loss = (op.wCol * prec**2).sum() / op.wCol_sum
    with torch.no_grad():
        dbg = {"loss": float(loss), "mean_abs_res": float(res.abs().mean()),
               "gamma": model.gamma_value()}
    return loss, dbg


def right_prec_loss_D(model, op):
    """
    Case D: right-prec, no enrichment.

    σ = −W̃ ρ,   Loss = ||B ρ − g||² / Nq

    where ρ = sigma_w(Yq),  B = −V W̃  (precomputed in op.B_prec),
    g = g_mfg at quadrature nodes (op.f_Yq).

    cond_eig(B) ≈ 15 (Calderón identity), no SPD required.
    """
    rho = model.sigma_w(op.Yq)               # (Nq,)   smooth, no enrichment
    res = op.B_prec @ rho - op.f_Yq          # (Nq,)
    loss = (res**2).mean() * op.prec_scale
    with torch.no_grad():
        dbg = {"loss": float(loss), "mean_abs_res": float(res.abs().mean()),
               "gamma": model.gamma_value()}
    return loss, dbg


def right_prec_loss_E(model, op):
    """
    Case E: right-prec + singular enrichment.

    σ = −W̃ ρ + γ σ_s
    Loss = ||B ρ + (V σ_s) γ − g||² / Nq

    where ρ = sigma_w(Yq),  B = −V W̃,  V_σs = V @ σ_s_Yq.
    γ is trainable (from model.gamma_module).
    """
    rho   = model.sigma_w(op.Yq)             # (Nq,)
    gamma = model.gamma_module()             # (n_gamma,)
    # V_sigma_s_q: (Nq, n_gamma)
    enrichment = op.V_sigma_s_q @ gamma      # (Nq,)
    res = op.B_prec @ rho + enrichment - op.f_Yq   # (Nq,)
    loss = (res**2).mean() * op.prec_scale
    with torch.no_grad():
        dbg = {"loss": float(loss), "mean_abs_res": float(res.abs().mean()),
               "gamma": model.gamma_value()}
    return loss, dbg


# ===========================================================================
# Helpers
# ===========================================================================

def make_u_exact_fn(Yq_T, wq, sigma_mfg):
    swq = sigma_mfg * wq
    def _u(xy):
        return _log_kernel_matrix(xy, Yq_T) @ swq
    return _u


def _lbfgs_cfg():
    return LBFGSConfig(
        max_iters      = CFG["lbfgs_iters"],
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


def _build_model(n_sing, freeze_gamma=False):
    m = SEBINNModel(
        hidden_width=CFG["hidden_width"], n_hidden=CFG["n_hidden"],
        n_gamma=n_sing, gamma_init=CFG["gamma_init"], dtype=torch.float64,
    )
    if freeze_gamma:
        for p in m.gamma_module.parameters():
            p.requires_grad_(False)
    return m


# ===========================================================================
# Training run
# ===========================================================================

def _train_one(label, loss_fn, init_state, shared,
               freeze_gamma=False,
               recover_sigma_fn=None,
               adam_iters=None, adam_lrs=None,
               verbose=True):
    """
    Train one case.

    Parameters
    ----------
    recover_sigma_fn : callable(model, Yq_t, sigma_s_t) -> np.ndarray
        How to compute the density σ from the model for diagnostics.
        Default (None): use model(Yq_t, sigma_s_t).numpy()  (standard).
        For right-prec cases: σ = −W̃ ρ [+ γ σ_s].
    """
    adam_iters = adam_iters or CFG["adam_iters"]
    adam_lrs   = adam_lrs   or CFG["adam_lrs"]

    if verbose:
        print(f"\n{'='*66}")
        print(f"  Case {label}")
        print(f"{'='*66}")

    t0         = time.perf_counter()
    op         = shared["op"]
    Yq_T       = shared["Yq_T"]
    wq         = shared["wq"]
    P          = shared["P"]
    sigma_bem  = shared["sigma_bem"]
    sigma_s_Yq = shared["sigma_s_Yq"]
    sort_idx   = shared["sort_idx"]
    u_exact    = shared["u_exact"]
    arc        = shared["arc"]
    n_gamma    = sigma_s_Yq.shape[1]

    # Default recover_sigma: standard enriched density
    if recover_sigma_fn is None:
        def recover_sigma_fn(m, Yq_t, ss_t):
            with torch.no_grad():
                return m(Yq_t, ss_t).numpy()

    model = _build_model(n_gamma, freeze_gamma=freeze_gamma)
    model.load_state_dict(copy.deepcopy(init_state), strict=False)
    if freeze_gamma:
        with torch.no_grad():
            for p in model.gamma_module.parameters():
                p.zero_()

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  trainable params: {n_tr}  (gamma {'frozen' if freeze_gamma else 'trainable'})")

    Yq_t      = torch.tensor(Yq_T, dtype=torch.float64)
    sigma_s_t = torch.tensor(sigma_s_Yq, dtype=torch.float64)
    traj      = []
    gamma_traj= []

    def _reldiff():
        s = recover_sigma_fn(model, Yq_t, sigma_s_t)
        return float(np.linalg.norm(s - sigma_bem)
                     / max(np.linalg.norm(sigma_bem), 1e-14)), s

    def _record(stage, n_iter, loss_val):
        d_err, s = _reldiff()
        i_out = reconstruct_interior(
            P=P, Yq=Yq_T, wq=wq, sigma=s,
            n_grid=CFG["n_grid_coarse"], u_exact=u_exact,
        )
        g     = model.gamma_value()
        g_arr = np.array(g if isinstance(g, list) else [float(g)])
        traj.append((n_iter, d_err, loss_val, g_arr.copy(), float(i_out.rel_L2)))
        gamma_traj.append((n_iter, g_arr.copy()))
        if verbose:
            gfmt = "[" + ",".join(f"{v:+.3f}" for v in g_arr) + "]"
            print(f"  [{label}] {stage}: loss={loss_val:.3e} | "
                  f"d_err={d_err:.4f} | iL2={i_out.rel_L2:.3e} | γ={gfmt}")

    with torch.no_grad():
        loss0, _ = loss_fn(model, op)
    _record("init", 0, float(loss0))

    adam_loss_all = []
    global_it = 0
    for ph_idx, (n_it, lr) in enumerate(zip(adam_iters, adam_lrs)):
        ph_cfg = AdamConfig(phase_iters=[n_it], phase_lrs=[lr],
                            log_every=CFG["log_every"])
        ph_res = run_adam_phases(model, op, ph_cfg, verbose=verbose,
                                 loss_fn=loss_fn)
        adam_loss_all.extend(ph_res.loss_hist)
        global_it += ph_res.n_iters
        _record(f"Adam-ph{ph_idx+1}", global_it, ph_res.final_loss)

    lbfgs_res = run_lbfgs(model, op, _lbfgs_cfg(), verbose=verbose,
                          loss_fn=loss_fn)
    _record("LBFGS",
            global_it + lbfgs_res.n_iters,
            lbfgs_res.loss_hist[-1] if lbfgs_res.loss_hist else float("nan"))

    # Final sigma from recover_sigma_fn
    sigma_final = recover_sigma_fn(model, Yq_t, sigma_s_t)

    final_out = reconstruct_interior(
        P=P, Yq=Yq_T, wq=wq, sigma=sigma_final,
        n_grid=CFG["n_grid_final"], u_exact=u_exact,
    )
    d_err_final = float(np.linalg.norm(sigma_final - sigma_bem)
                        / max(np.linalg.norm(sigma_bem), 1e-14))
    gamma_final = np.array(
        model.gamma_value() if isinstance(model.gamma_value(), list)
        else [float(model.gamma_value())]
    )
    gamma_err = float(np.linalg.norm(gamma_final - CFG["gamma_true"])
                      / max(np.linalg.norm(CFG["gamma_true"]), 1e-14))

    if verbose:
        print(f"\n  {label} final:")
        print(f"    Density rel-diff : {d_err_final:.4f}")
        print(f"    Interior L2      : {final_out.rel_L2:.3e}")
        print(f"    γ error          : {gamma_err:.4f}")
        print(f"    γ final          : [{', '.join(f'{v:+.4f}' for v in gamma_final)}]")
        print(f"    LBFGS reason     : {lbfgs_res.reason}")
        print(f"    Wall time        : {time.perf_counter()-t0:.1f}s")

    # Also record the smooth ρ (only meaningful for D, E)
    with torch.no_grad():
        rho_final = model.sigma_w(Yq_t).numpy()

    with torch.no_grad():
        bie_loss_final, _ = standard_loss(model, op)

    return dict(
        label=label,
        density_rel_diff=d_err_final,
        final_rel_L2=final_out.rel_L2,
        bie_loss_final=float(bie_loss_final),
        gamma_final=gamma_final,
        gamma_err=gamma_err,
        lbfgs_reason=lbfgs_res.reason,
        wall_time=time.perf_counter()-t0,
        loss_init=float(loss0),
        adam_lr_used=adam_lrs,
        sigma_final=sigma_final[sort_idx],
        rho_final=rho_final[sort_idx],
        traj=traj,
        gamma_traj=gamma_traj,
        final_out=final_out,
    )


# ===========================================================================
# Figures
# ===========================================================================

COLORS = {
    "A (BINN, std)":          "#9467bd",
    "B (SE-BINN, std)":       "#1f77b4",
    "C (SE-BINN, V⁻¹)":      "#2ca02c",
    "D (right-prec, ρ only)": "#d62728",
    "E (right-prec, +γσ_s)":  "#ff7f0e",
}
MARKERS = {
    "A (BINN, std)":          "s",
    "B (SE-BINN, std)":       "o",
    "C (SE-BINN, V⁻¹)":      "^",
    "D (right-prec, ρ only)": "D",
    "E (right-prec, +γσ_s)":  "P",
}


def _fig_convergence(cases, adam_cutoff, outpath):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for res in cases:
        lbl   = res["label"]
        traj  = res["traj"]
        iters  = [t[0] for t in traj]
        d_errs = [t[1] for t in traj]
        c  = COLORS.get(lbl, "gray")
        mk = MARKERS.get(lbl, "o")
        ax.semilogy(iters, d_errs, mk + "-", color=c, lw=1.8, ms=6,
                    markevery=max(1, len(iters)//8), label=lbl)
        ax.annotate(f"  {d_errs[-1]:.3f}",
                    xy=(iters[-1], d_errs[-1]), fontsize=9, color=c, va="center")

    ax.axvline(x=adam_cutoff, color="gray", ls=":", lw=1.0, alpha=0.6,
               label="Adam → L-BFGS")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(r"$\|\sigma - \sigma_{\mathrm{mfg}}\| / \|\sigma_{\mathrm{mfg}}\|$",
                  fontsize=12)
    ax.set_title(
        "Right-preconditioned BINN — density rel-diff, Koch(1)\n"
        r"D: $\sigma = -\tilde{W}\rho$"
        r"  |  E: $\sigma = -\tilde{W}\rho + \gamma\sigma_s$"
        "  |  cond(B) ≈ 15",
        fontsize=11,
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def _fig_density(cases, sigma_bem, arc, sort_idx, outpath):
    arc_s = arc[sort_idx]
    bem_s = sigma_bem[sort_idx]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.subplots_adjust(hspace=0.32)

    ax = axes[0]
    ax.plot(arc_s, bem_s, "k-", lw=2.2, label=r"$\sigma_{\mathrm{mfg}}$", zorder=6)
    for res in cases:
        lbl = res["label"]
        c   = COLORS.get(lbl, "gray")
        ax.plot(arc_s, res["sigma_final"], "-", color=c, lw=1.2, alpha=0.85,
                label=f"{lbl}  (d={res['density_rel_diff']:.3f})")
    ax.set_ylabel(r"$\sigma(s)$", fontsize=11)
    ax.set_title("Final recovered density σ", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, lw=0.3, alpha=0.4)

    ax = axes[1]
    for res in cases:
        lbl = res["label"]
        c   = COLORS.get(lbl, "gray")
        err = np.abs(res["sigma_final"] - bem_s)
        ax.semilogy(arc_s, err + 1e-20, "-", color=c, lw=1.2, alpha=0.85, label=lbl)
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$|\sigma(s) - \sigma_{\mathrm{mfg}}(s)|$", fontsize=11)
    ax.set_title("Pointwise density error", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)

    fig.suptitle("Right-preconditioned BINN — Koch(1)  (full budget)", fontsize=12)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def _fig_rho(cases_DE, rho_target_D, rho_target_E, arc, sort_idx, outpath):
    """
    Plot the smooth preimage ρ for cases D and E.

    ρ_target_D = arg min ||σ_bem + W̃ρ||  (smooth, no singularity)
    ρ_target_E = arg min ||σ_smooth + W̃ρ||  (residual after removing γ*σ_s)
    """
    arc_s = arc[sort_idx]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.subplots_adjust(hspace=0.32)

    ax = axes[0]
    if rho_target_D is not None:
        ax.plot(arc_s, rho_target_D[sort_idx], "k--", lw=2.0, alpha=0.7,
                label=r"$\rho^\star_D = -\tilde{W}^{-1}\sigma_\mathrm{mfg}$  (reference)")
    if rho_target_E is not None:
        ax.plot(arc_s, rho_target_E[sort_idx], "k:", lw=2.0, alpha=0.7,
                label=r"$\rho^\star_E = -\tilde{W}^{-1}\sigma_\mathrm{smooth}$  (reference)")

    for res in cases_DE:
        lbl = res["label"]
        c   = COLORS.get(lbl, "gray")
        ax.plot(arc_s, res["rho_final"], "-", color=c, lw=1.4, alpha=0.9, label=lbl)
    ax.set_ylabel(r"$\rho(s)$  (learned)", fontsize=11)
    ax.set_title(r"Smooth preimage $\rho$ (network output, before $-\tilde{W}\rho$)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, lw=0.3, alpha=0.4)

    ax = axes[1]
    for res, rho_tgt in zip(cases_DE, [rho_target_D, rho_target_E]):
        if rho_tgt is None:
            continue
        lbl = res["label"]
        c   = COLORS.get(lbl, "gray")
        err = np.abs(res["rho_final"] - rho_tgt[sort_idx])
        ax.semilogy(arc_s, err + 1e-20, "-", color=c, lw=1.4, alpha=0.9,
                    label=f"{lbl}  ρ-err")
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$|\rho(s) - \rho^\star(s)|$", fontsize=11)
    ax.set_title("Pointwise ρ error (vs reference target)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)

    fig.suptitle(r"Learned preimage $\rho$: smoothness of D vs E — Koch(1)", fontsize=12)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def _fig_gamma(cases, outpath):
    cases_with_gamma = [r for r in cases if r["label"] != "A (BINN, std)"]
    if not cases_with_gamma:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    gamma_true = CFG["gamma_true"]
    n_gamma    = len(gamma_true)
    linestyles = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,2))]

    for res in cases_with_gamma:
        lbl   = res["label"]
        c     = COLORS.get(lbl, "gray")
        traj  = res["gamma_traj"]
        iters = [t[0] for t in traj]
        for k in range(n_gamma):
            vals = [t[1][k] for t in traj]
            ls   = linestyles[k % len(linestyles)]
            ax.plot(iters, vals, ls=ls, color=c, lw=1.3, alpha=0.85,
                    label=f"{lbl} γ_{k+1}" if k == 0 else f"γ_{k+1}")

    for k, gk in enumerate(gamma_true):
        ax.axhline(y=gk, color="gray", ls=":", lw=0.8, alpha=0.6)
        ax.text(0, gk, f"  γ_{k+1}*={gk:+.1f}", fontsize=7.5,
                va="center", color="gray")

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(r"$\hat{\gamma}_c$", fontsize=11)
    ax.set_title("γ_c trajectories — B, C, E  (dashed = γ_true)", fontsize=11)
    ax.grid(True, lw=0.3, alpha=0.4)
    handles = [Line2D([0],[0], color=COLORS.get(r["label"],"gray"), lw=2,
                      label=r["label"]) for r in cases_with_gamma]
    ax.legend(handles=handles, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def _fig_operator_spectra(B_prec, V_h, W_tilde, outpath):
    """Compare cond_eig of B = −VW̃ vs V vs W̃."""
    eig_V  = np.sort(np.abs(np.linalg.eigvals(V_h)))[::-1]
    eig_WT = np.sort(np.abs(np.linalg.eigvals(W_tilde)))[::-1]
    eig_B  = np.sort(np.abs(np.linalg.eigvals(B_prec)))[::-1]

    cond_V  = float(eig_V[0]  / (eig_V[-1]  + 1e-14))
    cond_WT = float(eig_WT[0] / (eig_WT[-1] + 1e-14))
    cond_B  = float(eig_B[0]  / (eig_B[-1]  + 1e-14))

    fig, ax = plt.subplots(figsize=(9, 5))
    idx = np.arange(1, len(eig_V)+1)
    ax.semilogy(idx, eig_V,  "k-",  lw=1.8, label=rf"$|λ(V_h)|$      cond={cond_V:.2e}")
    ax.semilogy(idx, eig_WT, "b--", lw=1.8, label=rf"$|λ(\tilde{{W}})|$  cond={cond_WT:.2e}")
    ax.semilogy(idx, eig_B,  "r-",  lw=1.8, label=rf"$|λ(B)|$         cond={cond_B:.2e}")
    ax.set_xlabel("Index (sorted by magnitude)", fontsize=11)
    ax.set_ylabel(r"$|\lambda_k|$", fontsize=11)
    ax.set_title(
        r"Operator spectra: $V_h$,  $\tilde{W}$,  $B=-V\tilde{W}$"
        "\nKoch(1) — Calderón: cond(B) ≈ cond(VW̃) ≈ 15",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.text(0.5, 0.04,
            f"cond(V) = {cond_V:.2e}  |  cond(W̃) = {cond_WT:.2e}  |  cond(B) = {cond_B:.2e}",
            transform=ax.transAxes, ha="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    fig_dir = os.path.join(_HERE, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "="*70)
    print("RIGHT-PRECONDITIONED BINN: σ = −W̃ρ, loss = ||Bρ − g||²")
    print("B = −VW̃,   cond_eig(B) ≈ 15  (Calderón identity)")
    print("="*70)

    # ------------------------------------------------------------------
    # 1. Geometry + operators
    # ------------------------------------------------------------------
    geom   = make_koch_geometry(n=1)
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=CFG["n_per_edge"])
    label_corner_ring_panels(panels, P)
    qdata  = build_panel_quadrature(panels, p=CFG["p_gl"])
    nmat   = assemble_nystrom_matrix(qdata)
    V_h    = nmat.V                  # (Nq, Nq) full Nyström matrix
    Yq_T   = qdata.Yq.T             # (Nq, 2)
    wq     = qdata.wq
    Nq     = qdata.n_quad
    print(f"  Koch(1): N_panels={qdata.n_panels}, N_quad={Nq}")

    # ------------------------------------------------------------------
    # 2. W̃ assembly + right-preconditioning operator B
    # ------------------------------------------------------------------
    print("\n  === Assembling W̃ and right-preconditioned operator B ===")
    W_h, _  = assemble_hypersingular_direct(qdata)
    W_tilde = regularise_hypersingular(W_h, wq)    # (Nq, Nq)

    # B = −V W̃  (note: V_h is the full operator including self-panel correction)
    # Do NOT symmetrise: sym(VW̃) ≠ VW̃ in eigenstructure, and symmetrisation
    # destroys cond_eig (452 → 15). The loss ||Bρ−g||² doesn't need B symmetric.
    B_prec_np = V_h @ (-W_tilde)                   # (Nq, Nq)

    # cond_eig(B)
    eig_B = np.abs(np.linalg.eigvals(B_prec_np))
    cond_B = float(eig_B.max() / (eig_B.min() + 1e-14))

    # cond_eig(V) for reference
    eig_V = np.abs(np.linalg.eigvals(V_h))
    cond_V = float(eig_V.max() / (eig_V.min() + 1e-14))

    # cond_eig(VW̃) before symmetrisation
    VW_raw = np.abs(np.linalg.eigvals(V_h @ W_tilde))
    cond_VW = float(VW_raw.max() / (VW_raw.min() + 1e-14))

    print(f"  cond_eig(V_h)     = {cond_V:.2e}  (reference)")
    print(f"  cond_eig(VW̃)      = {cond_VW:.2f}  (raw, before symmetrise)")
    print(f"  cond_eig(B=−VW̃)   = {cond_B:.2f}  (after symmetrise)")
    print(f"  Expected: cond(B) ≈ 15 from Calderón identity")

    # Operator spectra figure
    _fig_operator_spectra(
        B_prec_np, V_h, W_tilde,
        os.path.join(fig_dir, "right_prec_spectra.png"),
    )

    # ------------------------------------------------------------------
    # 3. Manufactured density
    # ------------------------------------------------------------------
    print("\n  === Manufactured density ===")
    enrichment = SingularEnrichment(geom=geom, use_cutoff=True,
                                    cutoff_radius=CFG["cutoff_radius"],
                                    per_corner_gamma=True)
    n_sing     = enrichment.n_singular
    sigma_s_Yq = enrichment.precompute(Yq_T)    # (Nq, n_gamma)
    gamma_true = CFG["gamma_true"].copy()

    f_smooth     = Yq_T[:, 0]**2 - Yq_T[:, 1]**2
    sigma_smooth = solve_bem(nmat, f_smooth, tol=1e-12).sigma
    sigma_mfg    = sigma_smooth + sigma_s_Yq @ gamma_true
    g_mfg        = V_h @ sigma_mfg
    sigma_bem    = solve_bem(nmat, g_mfg, tol=1e-12).sigma

    energy = float(np.linalg.norm(sigma_s_Yq @ gamma_true)**2
                   / max(np.linalg.norm(sigma_mfg)**2, 1e-14))
    print(f"    Enrichment energy = {energy*100:.2f}%")

    # ------------------------------------------------------------------
    # 4. Reference ρ targets (for diagnostic figures)
    # ------------------------------------------------------------------
    # ρ_D^* : σ_bem = −W̃ ρ_D^*  →  ρ_D^* = −W̃^{-1} σ_bem  (lstsq)
    # ρ_E^* : σ_smooth = −W̃ ρ_E^*  →  ρ_E^* = −W̃^{-1} σ_smooth
    print("  Computing reference ρ targets via lstsq ...")
    rho_target_D, *_ = np.linalg.lstsq(-W_tilde, sigma_bem,   rcond=None)
    rho_target_E, *_ = np.linalg.lstsq(-W_tilde, sigma_smooth, rcond=None)
    print(f"    ||ρ_D^*||_2 = {np.linalg.norm(rho_target_D):.4f}")
    print(f"    ||ρ_E^*||_2 = {np.linalg.norm(rho_target_E):.4f}")
    rho_residual_D = np.linalg.norm(-W_tilde @ rho_target_D - sigma_bem)
    rho_residual_E = np.linalg.norm(-W_tilde @ rho_target_E - sigma_smooth)
    print(f"    lstsq residuals: ρ_D={rho_residual_D:.3e},  ρ_E={rho_residual_E:.3e}")

    # ------------------------------------------------------------------
    # 5. Precompute V σ_s (needed for Case E enrichment term)
    # ------------------------------------------------------------------
    # V_sigma_s_q[k, c] = sum_j V_h[k,j] * sigma_s_Yq[j,c]   (Nq, n_gamma)
    V_sigma_s_q_np = V_h @ sigma_s_Yq                         # (Nq, n_gamma)

    # ------------------------------------------------------------------
    # 6. Loss scale check (ρ=0 → residual ≈ −g_mfg)
    # ------------------------------------------------------------------
    r0 = -g_mfg                        # residual at ρ=0 (standard loss)
    loss_std_0  = float(np.mean(r0**2))
    loss_B_0    = float(np.mean((B_prec_np @ np.zeros(Nq) - g_mfg)**2))
    # note: B @ 0 - g = -g_mfg, same magnitude as r0
    # use f_Yq = g_mfg (evaluated at quadrature nodes)
    f_Yq_np     = g_mfg.copy()         # g at quadrature nodes

    loss_D_0    = float(np.mean((B_prec_np @ np.zeros(Nq) - f_Yq_np)**2))
    loss_E_0    = float(np.mean((B_prec_np @ np.zeros(Nq)
                                 + V_sigma_s_q_np @ np.zeros(n_sing)
                                 - f_Yq_np)**2))

    # Normalise D and E to match std loss at init
    prec_scale_D = loss_std_0 / max(loss_D_0, 1e-14)
    prec_scale_E = loss_std_0 / max(loss_E_0, 1e-14)

    print(f"\n  === Loss scale check (ρ=0, γ=0) ===")
    print(f"    std   loss ||r||²/N     = {loss_std_0:.3e}")
    print(f"    D     loss ||Bρ−g||²/N  = {loss_D_0:.3e}  "
          f"(ratio {loss_D_0/loss_std_0:.3f})")
    print(f"    prec_scale applied      = {prec_scale_D:.3e}")

    u_exact = make_u_exact_fn(Yq_T, wq, sigma_mfg)

    # ------------------------------------------------------------------
    # 7. Operator state (standard collocation-based: for A, B, C)
    # ------------------------------------------------------------------
    col_pts   = build_collocation_points(panels, m_col_panel=CFG["m_col_base"])
    panel_wts = panel_loss_weights(panels, w_base=CFG["w_base"],
                                   w_corner=CFG["w_corner"], w_ring=CFG["w_ring"])
    g_fn      = lambda xy: xy[:, 0]**2 - xy[:, 1]**2
    op, diag  = build_operator_state(
        colloc=col_pts, qdata=qdata, enrichment=enrichment, g=g_fn,
        panel_weights=panel_wts, eq_scale_mode="none", eq_scale_fixed=1.0,
        dtype=torch.float64, device="cpu",
    )
    assert abs(diag["eq_scale"] - 1.0) < 1e-12
    op.f = torch.tensor(g_mfg, dtype=torch.float64)

    V_inv_np    = np.linalg.inv(V_h)
    op.V_inv    = torch.tensor(V_inv_np,      dtype=torch.float64)

    # Right-preconditioning additions to op
    op.B_prec        = torch.tensor(B_prec_np,      dtype=torch.float64)
    op.f_Yq          = torch.tensor(f_Yq_np,        dtype=torch.float64)
    op.V_sigma_s_q   = torch.tensor(V_sigma_s_q_np, dtype=torch.float64)
    op.prec_scale    = prec_scale_D    # same for D and E (g_mfg is same scale)

    print(f"    ||V_inv V − I||_F/√N = "
          f"{np.linalg.norm(V_inv_np @ V_h - np.eye(Nq))/np.sqrt(Nq):.3e}")

    # ------------------------------------------------------------------
    # 8. Shared initial state + arc-length sort
    # ------------------------------------------------------------------
    torch.manual_seed(CFG["seed"])
    init_model = _build_model(n_sing, freeze_gamma=False)
    init_state = copy.deepcopy(init_model.state_dict())

    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc         = panel_start[qdata.pan_id] + qdata.s_on_panel
    sort_idx    = np.argsort(arc)

    shared = dict(
        op=op, Yq_T=Yq_T, wq=wq, P=P,
        sigma_bem=sigma_bem, sigma_s_Yq=sigma_s_Yq,
        sort_idx=sort_idx, u_exact=u_exact, arc=arc,
    )

    adam_cutoff = sum(CFG["adam_iters"])

    # ------------------------------------------------------------------
    # 9. recover_sigma functions for right-prec cases
    # ------------------------------------------------------------------
    # Pre-capture numpy arrays for closures (torch not needed in recover)
    _W_tilde_np     = W_tilde
    _sigma_s_Yq_np  = sigma_s_Yq

    def recover_std(m, Yq_t, ss_t):
        """Standard: σ = σ_w + γσ_s."""
        with torch.no_grad():
            return m(Yq_t, ss_t).numpy()

    def recover_D(m, Yq_t, ss_t):
        """Case D: σ = −W̃ ρ  (no enrichment)."""
        with torch.no_grad():
            rho = m.sigma_w(Yq_t).numpy()
        return -_W_tilde_np @ rho

    def recover_E(m, Yq_t, ss_t):
        """Case E: σ = −W̃ ρ + γ σ_s."""
        with torch.no_grad():
            rho   = m.sigma_w(Yq_t).numpy()
            g_val = m.gamma_value()
            gamma = np.array(g_val if isinstance(g_val, list) else [float(g_val)])
        return -_W_tilde_np @ rho + _sigma_s_Yq_np @ gamma

    # ------------------------------------------------------------------
    # 10. Training: 5 cases
    # ------------------------------------------------------------------
    print("\n  === Training (5 cases, full budget) ===")
    print(f"  Adam: {CFG['adam_iters']} iters at lr {CFG['adam_lrs']}")
    print(f"  LBFGS: {CFG['lbfgs_iters']} iters, memory {CFG['lbfgs_memory']}")

    cases = []

    # A: BINN baseline
    cases.append(_train_one(
        "A (BINN, std)", standard_loss, init_state, shared,
        freeze_gamma=True, recover_sigma_fn=recover_std,
    ))

    # B: SE-BINN standard
    cases.append(_train_one(
        "B (SE-BINN, std)", standard_loss, init_state, shared,
        freeze_gamma=False, recover_sigma_fn=recover_std,
    ))

    # C: SE-BINN exact V⁻¹ preconditioner (reference)
    cases.append(_train_one(
        "C (SE-BINN, V⁻¹)", exact_precond_loss, init_state, shared,
        freeze_gamma=False, recover_sigma_fn=recover_std,
    ))

    # D: right-prec, no enrichment
    cases.append(_train_one(
        "D (right-prec, ρ only)", right_prec_loss_D, init_state, shared,
        freeze_gamma=True, recover_sigma_fn=recover_D,
    ))

    # E: right-prec + singular enrichment
    cases.append(_train_one(
        "E (right-prec, +γσ_s)", right_prec_loss_E, init_state, shared,
        freeze_gamma=False, recover_sigma_fn=recover_E,
    ))

    # ------------------------------------------------------------------
    # 11. Final summary table
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    labels_short = ["A (BINN)", "B (SE-BINN)", "C (V⁻¹)", "D (rp, ρ)", "E (rp, +γ)"]
    hdr = f"  {'Metric':<22}" + "".join(f"  {l:>14}" for l in labels_short)
    print(hdr)
    print("  " + "-"*92)

    for name, key in [
        ("Density rel-diff",  "density_rel_diff"),
        ("Interior rel L2",   "final_rel_L2"),
        ("BIE loss (std)",    "bie_loss_final"),
        ("γ error",          "gamma_err"),
    ]:
        row = f"  {name:<22}"
        for r in cases:
            row += f"  {r[key]:>14.4e}"
        print(row)

    row = f"  {'init loss':<22}"
    for r in cases:
        row += f"  {r['loss_init']:>14.3e}"
    print(row)

    row = f"  {'LBFGS reason':<22}"
    for r in cases:
        row += f"  {r['lbfgs_reason']:>14}"
    print(row)

    row = f"  {'Wall time (s)':<22}"
    for r in cases:
        row += f"  {r['wall_time']:>14.1f}"
    print(row)

    print()
    d_A = cases[0]["density_rel_diff"]
    d_B = cases[1]["density_rel_diff"]
    d_C = cases[2]["density_rel_diff"]
    d_D = cases[3]["density_rel_diff"]
    d_E = cases[4]["density_rel_diff"]
    print(f"  A baseline (BINN): d_err = {d_A:.4f}")
    print(f"  B baseline (SEBINN): d_err = {d_B:.4f}")
    print(f"  C (V⁻¹ prec):      d_err = {d_C:.4f}  ({d_B/max(d_C,1e-6):.1f}× over B)")
    print(f"  D (right-prec ρ):  d_err = {d_D:.4f}  ({d_B/max(d_D,1e-6):.1f}× over B)")
    print(f"  E (right-prec +γ): d_err = {d_E:.4f}  ({d_B/max(d_E,1e-6):.1f}× over B)")
    print()
    print(f"  cond_eig(B) = {cond_B:.2f}  (expected ≈ 15 from Calderón identity)")
    print(f"  cond(V_h)   = {cond_V:.2e}  (reference)")

    # Expected outcome annotation
    if d_D < d_B * 0.5 or d_E < d_B * 0.5:
        print("\n  SUCCESS: right-preconditioning improved over B baseline.")
    elif d_E < d_B * 0.8:
        print("\n  PARTIAL: right-prec shows marginal improvement.")
    else:
        print("\n  NOTE: right-prec result comparable to standard baseline.")

    # ------------------------------------------------------------------
    # 12. Figures
    # ------------------------------------------------------------------
    _fig_convergence(
        cases, adam_cutoff,
        os.path.join(fig_dir, "right_prec_convergence.png"),
    )
    _fig_density(
        cases, sigma_bem, arc, sort_idx,
        os.path.join(fig_dir, "right_prec_density.png"),
    )
    _fig_rho(
        [cases[3], cases[4]],   # D and E
        rho_target_D,
        rho_target_E,
        arc, sort_idx,
        os.path.join(fig_dir, "right_prec_rho.png"),
    )
    _fig_gamma(
        cases,
        os.path.join(fig_dir, "right_prec_gamma.png"),
    )

    print(f"\n  All figures saved to {fig_dir}/")
    return cases


if __name__ == "__main__":
    main()
