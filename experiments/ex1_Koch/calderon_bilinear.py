"""
Experiment: Calderón bilinear-form preconditioned loss.

Mathematical background
-----------------------
All previous Calderón experiments used the SQUARED NORM

    L_sq = ||T̃ r||²  =  rᵀ T̃ᵀ T̃ r

This amplifies any residual r by ||T̃||² ~ (1.1e4)², and no scalar
normalisation can simultaneously fix the scale and preserve the
conditioning advantage.

The CORRECT Calderón-preconditioned loss is the BILINEAR FORM

    L_bil = rᵀ T̃ r  =  rᵀ T̃_sym r     (since rᵀ A_antisym r = 0)

where T̃_sym = (T̃ + T̃ᵀ)/2.

The gradient w.r.t. σ through r = Vσ − g is:
    ∂L/∂σ = 2 Vᵀ T̃_sym r

The condition number governing convergence is cond_eig(T̃_sym V), which
for a perfect Calderón preconditioner equals cond_eig(T̃V) = 13.6.

SPD requirement & diagnostic
-----------------------------
For L_bil to be a valid (non-negative) loss, T̃_sym must be PSD.

Diagnostic (precomputed):
    T̃_sym = (T̃ + T̃ᵀ)/2
    eigvalsh(T̃_sym): min = −3159.60,  max = 11200.60
    N_negative = 23

    → T̃_sym is NOT PSD. Required diagonal shift: α ≈ 3170.

    CRITICAL CONSEQUENCE: shift α = 3169.6 destroys conditioning.
        cond_eig(T̃_spd V)  =  5.6e+05   (was 13.6 before shift!)
        The Calderón advantage is COMPLETELY LOST.

Weighted bilinear alternative
------------------------------
The Galerkin-symmetric form is W T̃ where W = diag(wq):
    diag(wq) T_h is machine-symmetric (4.6e-17 asymmetry)
    → W T̃ is symmetric

    eigvalsh(W T̃): min = −0.943,  max = +...
    N_negative = 6

    Required shift: ε ≈ 0.943 + 1e-10  (much smaller)

    Weighted bilinear loss:
        L_wbil = rᵀ W T̃_spd_w r

    Gradient: ∂L/∂σ = 2 Vᵀ W T̃_spd_w r

Cases
-----
A: BINN       + standard loss   (γ frozen at 0)
B: SE-BINN    + standard loss   (γ trainable)
C: SE-BINN    + exact V⁻¹ preconditioner  (reference / best known)
D: SE-BINN    + bilinear T̃_spd r   (diagonal shift α ≈ 3170, destroys cond)
E: SE-BINN    + weighted bilinear W T̃_spd_w r  (shift ε ≈ 0.94, keeps structure)

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
    # Full budget
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
    """L = ||V_h^{-1}(Vσ − g)||²"""
    res  = op.A @ model(op.Yq, op.sigma_s_q) + op.corr * model(op.Xc, op.sigma_s_c) - op.f
    prec = op.V_inv @ res
    loss = (op.wCol * prec**2).sum() / op.wCol_sum
    with torch.no_grad():
        dbg = {"loss": float(loss), "mean_abs_res": float(res.abs().mean()),
               "gamma": model.gamma_value()}
    return loss, dbg


def bilinear_loss(model, op):
    """L = rᵀ T̃_spd r  (bilinear form, T̃ sym + diagonal shift α ≈ 3170).

    WARNING: the required shift α ≈ 3170 destroys cond_eig:
    13.6 (Calderón) → 5.6e+05 (shifted). No conditioning benefit expected.
    Included to verify this empirically.
    """
    res  = op.A @ model(op.Yq, op.sigma_s_q) + op.corr * model(op.Xc, op.sigma_s_c) - op.f
    bil  = torch.dot(res, op.T_tilde_spd @ res)
    loss = bil * op.bilinear_scale
    with torch.no_grad():
        dbg = {"loss": float(loss), "mean_abs_res": float(res.abs().mean()),
               "gamma": model.gamma_value()}
    return loss, dbg


def weighted_bilinear_loss(model, op):
    """L = rᵀ W T̃_spd_w r  where W = diag(wq), shift ε ≈ 0.94.

    W T̃ is machine-precision symmetric (Galerkin sense).
    Required shift is negligible compared to α ≈ 3170 for the unweighted form.
    """
    res  = op.A @ model(op.Yq, op.sigma_s_q) + op.corr * model(op.Xc, op.sigma_s_c) - op.f
    bil  = torch.dot(res, op.W_T_spd @ res)
    loss = bil * op.w_bilinear_scale
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
               adam_iters=None, adam_lrs=None,
               verbose=True):
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
        with torch.no_grad():
            s = model(Yq_t, sigma_s_t).numpy()
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

    with torch.no_grad():
        sigma_final = model(Yq_t, sigma_s_t).numpy()
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
        traj=traj,
        gamma_traj=gamma_traj,
        final_out=final_out,
    )


# ===========================================================================
# Figures
# ===========================================================================

COLORS = {
    "A (BINN, std)":            "#9467bd",
    "B (SE-BINN, std)":         "#1f77b4",
    "C (SE-BINN, V⁻¹)":        "#2ca02c",
    "D (SE-BINN, bil T̃)":      "#d62728",
    "E (SE-BINN, wbil WT̃)":    "#ff7f0e",
}
MARKERS = {
    "A (BINN, std)":            "s",
    "B (SE-BINN, std)":         "o",
    "C (SE-BINN, V⁻¹)":        "^",
    "D (SE-BINN, bil T̃)":      "D",
    "E (SE-BINN, wbil WT̃)":    "P",
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
        "Calderón bilinear form — density rel-diff, Koch(1)\n"
        "A: BINN  |  B: SE-BINN  |  C: V⁻¹  |  D: bil rᵀT̃r  |  E: wbil rᵀWT̃r",
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
    ax.set_title("Final density", fontsize=11)
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

    fig.suptitle("Calderón bilinear form — Koch(1)  (full budget)", fontsize=12)
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
    ax.set_title("γ_c trajectories — B, C, D, E  (dashed = γ_true)", fontsize=11)
    ax.grid(True, lw=0.3, alpha=0.4)
    handles = [Line2D([0],[0], color=COLORS.get(r["label"],"gray"), lw=2,
                      label=r["label"]) for r in cases_with_gamma]
    ax.legend(handles=handles, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def _fig_spd_spectra(T_sym, W_T, alpha_shift, eps_shift, V_h, outpath):
    """Eigenvalue spectra of T̃_sym, W T̃, and their shifted versions."""
    eig_T = np.linalg.eigvalsh(T_sym)
    eig_WT = np.linalg.eigvalsh(W_T)

    T_spd    = T_sym + alpha_shift * np.eye(len(T_sym))
    WT_spd   = W_T   + eps_shift   * np.eye(len(W_T))
    eig_T_spd  = np.linalg.eigvalsh(T_spd)
    eig_WT_spd = np.linalg.eigvalsh(WT_spd)

    eig_TV     = np.abs(np.linalg.eigvals(T_spd  @ V_h))
    eig_WTV    = np.abs(np.linalg.eigvals(WT_spd @ V_h))

    cond_TV  = eig_TV.max()  / (eig_TV.min()  + 1e-14)
    cond_WTV = eig_WTV.max() / (eig_WTV.min() + 1e-14)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    idx = np.arange(1, len(eig_T)+1)
    ax.plot(idx, np.sort(eig_T),    "b-",  lw=1.5,
            label=r"$\tilde{T}_\mathrm{sym}$ (before shift)")
    ax.plot(idx, np.sort(eig_T_spd),"b--", lw=1.5,
            label=rf"$\tilde{{T}}_\mathrm{{spd}}$ (α={alpha_shift:.0f})")
    ax.plot(idx, np.sort(eig_WT),   "r-",  lw=1.5,
            label=r"$W\tilde{T}$ (before shift)")
    ax.plot(idx, np.sort(eig_WT_spd),"r--",lw=1.5,
            label=rf"$W\tilde{{T}}_\mathrm{{spd}}$ (ε={eps_shift:.3f})")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Eigenvalue index (sorted)", fontsize=11)
    ax.set_ylabel(r"$\lambda_k$", fontsize=11)
    ax.set_title("Eigenvalue spectra (sorted)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.text(0.03, 0.96,
            rf"$\tilde{{T}}_{{sym}}$: {np.sum(eig_T<0)} neg eigs, λ_min={eig_T.min():.1f}"
            "\n"
            rf"$W\tilde{{T}}$: {np.sum(eig_WT<0)} neg eigs, λ_min={eig_WT.min():.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax = axes[1]
    idx2 = np.arange(1, len(eig_TV)+1)
    ax.semilogy(np.arange(1, len(V_h)+1),
                np.sort(np.abs(np.linalg.eigvals(V_h)))[::-1],
                "k-", lw=1.5, label=r"$|$λ$(V_h)|$ (ref)")
    ax.semilogy(np.sort(eig_TV)[::-1],   "b--", lw=1.5,
                label=rf"$|\lambda(\tilde{{T}}_{{spd}}V)|$  cond={cond_TV:.2e}")
    ax.semilogy(np.sort(eig_WTV)[::-1],  "r--", lw=1.5,
                label=rf"$|\lambda(W\tilde{{T}}_{{spd}}V)|$  cond={cond_WTV:.2e}")
    ax.set_xlabel("Index", fontsize=11)
    ax.set_ylabel(r"$|\lambda_k|$", fontsize=11)
    ax.set_title("cond_eig after shift (governs bilinear gradient)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.text(0.03, 0.04,
            f"cond_eig(T̃_spd V)  = {cond_TV:.2e}  (was 13.6 before shift)\n"
            f"cond_eig(WT̃_spd V) = {cond_WTV:.2e}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    fig.suptitle(
        "SPD diagnostic — T̃ symmetrization + diagonal shifts\n"
        "Koch(1), N_quad=2304",
        fontsize=11,
    )
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
    print("CALDERÓN BILINEAR FORM: rᵀ T̃ r")
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
    V_h    = nmat.V
    Yq_T   = qdata.Yq.T
    wq     = qdata.wq
    Nq     = qdata.n_quad
    print(f"  Koch(1): N_panels={qdata.n_panels}, N_quad={Nq}")

    # ------------------------------------------------------------------
    # 2. T̃ assembly + SPD diagnostic
    # ------------------------------------------------------------------
    print("\n  === Step 1: SPD diagnostic ===")
    T_h, _  = assemble_hypersingular_direct(qdata)
    T_tilde = regularise_hypersingular(T_h, wq)

    # Symmetrize
    T_sym  = 0.5 * (T_tilde + T_tilde.T)

    # Eigenvalues of T̃_sym
    eig_Tsym = np.linalg.eigvalsh(T_sym)
    lmin_T   = float(eig_Tsym.min())
    lmax_T   = float(eig_Tsym.max())
    n_neg_T  = int(np.sum(eig_Tsym < 0))

    # Weighted form W T̃
    W_T    = np.diag(wq) @ T_tilde
    eig_WT = np.linalg.eigvalsh(W_T)
    lmin_WT   = float(eig_WT.min())
    lmax_WT   = float(eig_WT.max())
    n_neg_WT  = int(np.sum(eig_WT < 0))

    # Shifts
    alpha_shift = abs(lmin_T)  + 1e-10   # for T̃_sym
    eps_shift   = abs(lmin_WT) + 1e-10   # for W T̃

    print(f"\n  T̃_sym = (T̃ + T̃ᵀ)/2:")
    print(f"    eigvalsh: min={lmin_T:.4f},  max={lmax_T:.4f}")
    print(f"    N_negative = {n_neg_T}")
    print(f"    Required shift α = {alpha_shift:.4f}")

    print(f"\n  W T̃  (W = diag(wq), Galerkin-symmetric):")
    print(f"    eigvalsh: min={lmin_WT:.4f},  max={lmax_WT:.4f}")
    print(f"    N_negative = {n_neg_WT}")
    print(f"    Required shift ε = {eps_shift:.6f}")

    # Shifted matrices
    T_spd  = T_sym  + alpha_shift * np.eye(Nq)    # T̃_spd
    WT_spd = W_T    + eps_shift   * np.eye(Nq)    # W T̃_spd_w

    # cond_eig after shift
    eig_TV   = np.abs(np.linalg.eigvals(T_spd  @ V_h))
    eig_WTV  = np.abs(np.linalg.eigvals(WT_spd @ V_h))
    cond_TV  = float(eig_TV.max()  / (eig_TV.min()  + 1e-14))
    cond_WTV = float(eig_WTV.max() / (eig_WTV.min() + 1e-14))

    cond_V   = float(np.linalg.cond(V_h))
    TV_before = np.abs(np.linalg.eigvals(T_tilde @ V_h))
    cond_TV_before = float(TV_before.max() / (TV_before.min() + 1e-14))

    print(f"\n  cond_eig(T̃ V)          = {cond_TV_before:.2f}  (Calderón identity)")
    print(f"  cond_eig(T̃_spd V)      = {cond_TV:.2e}  ← shift destroys conditioning!")
    print(f"  cond_eig(W T̃_spd V)    = {cond_WTV:.2e}")
    print(f"  cond(V_h)              = {cond_V:.2e}  (reference)")

    if cond_TV > 1e4:
        print(f"\n  WARNING: diagonal shift α={alpha_shift:.2f} raises cond_eig from "
              f"{cond_TV_before:.1f} to {cond_TV:.2e}.")
        print(f"  The Calderón advantage (cond_eig=13.6) is LOST for Case D.")
        print(f"  Case D is included only to verify this failure empirically.")
    if cond_WTV < cond_V:
        print(f"\n  INFO: cond_eig(WT̃_spd V) = {cond_WTV:.2e} < cond(V) = {cond_V:.2e}.")
        print(f"  Case E (weighted bilinear) may provide conditioning benefit.")

    # Save SPD figure
    _fig_spd_spectra(
        T_sym, W_T, alpha_shift, eps_shift, V_h,
        os.path.join(fig_dir, "calderon_bilinear_spd_spectra.png"),
    )

    # ------------------------------------------------------------------
    # 3. Manufactured density
    # ------------------------------------------------------------------
    print("\n  === Manufactured density ===")
    enrichment = SingularEnrichment(geom=geom, use_cutoff=True,
                                    cutoff_radius=CFG["cutoff_radius"],
                                    per_corner_gamma=True)
    n_sing     = enrichment.n_singular
    sigma_s_Yq = enrichment.precompute(Yq_T)
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
    # 4. Loss scale check (σ=0 → r ≈ −g_mfg)
    # ------------------------------------------------------------------
    r0 = -g_mfg
    loss_std_0    = float(np.mean(r0**2))
    loss_bil_0    = float(r0 @ T_spd  @ r0)
    loss_wbil_0   = float(r0 @ WT_spd @ r0)
    loss_vinv_0   = float(np.mean((np.linalg.inv(V_h) @ r0)**2))

    # Scalar normalisations to match standard loss at init
    bilinear_scale   = loss_std_0  / max(loss_bil_0,  1e-14)
    w_bilinear_scale = loss_std_0  / max(loss_wbil_0, 1e-14)

    print(f"\n  === Step 3: Initial loss scale (σ=0, r = −g_mfg) ===")
    print(f"    std   loss ||r||²/N       = {loss_std_0:.3e}")
    print(f"    V⁻¹   loss ||V⁻¹r||²/N   = {loss_vinv_0:.3e}  "
          f"(ratio {loss_vinv_0/loss_std_0:.3f})")
    print(f"    bil   loss rᵀT̃_spd r     = {loss_bil_0:.3e}  "
          f"(ratio {loss_bil_0/loss_std_0:.3e})")
    print(f"    wbil  loss rᵀWT̃_spd r    = {loss_wbil_0:.3e}  "
          f"(ratio {loss_wbil_0/loss_std_0:.3e})")
    print(f"\n    bilinear_scale   = {bilinear_scale:.3e}  (applied to D)")
    print(f"    w_bilinear_scale = {w_bilinear_scale:.3e}  (applied to E)")
    print(f"    After scaling, all losses start at ≈ {loss_std_0:.3e}")

    u_exact = make_u_exact_fn(Yq_T, wq, sigma_mfg)

    # ------------------------------------------------------------------
    # 5. Operator state
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

    V_inv_np = np.linalg.inv(V_h)
    op.V_inv        = torch.tensor(V_inv_np, dtype=torch.float64)
    op.T_tilde_spd  = torch.tensor(T_spd,   dtype=torch.float64)
    op.W_T_spd      = torch.tensor(WT_spd,  dtype=torch.float64)
    op.bilinear_scale   = bilinear_scale
    op.w_bilinear_scale = w_bilinear_scale

    print(f"    ||V_inv V − I||_F/√N = "
          f"{np.linalg.norm(V_inv_np @ V_h - np.eye(Nq))/np.sqrt(Nq):.3e}")

    # ------------------------------------------------------------------
    # 6. Shared initial state
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
    # 7. Training: 5 cases
    # ------------------------------------------------------------------
    print("\n  === Step 4: Training (5 cases, full budget) ===")
    print(f"  Adam: {CFG['adam_iters']} iters at lr {CFG['adam_lrs']}")
    print(f"  LBFGS: {CFG['lbfgs_iters']} iters, memory {CFG['lbfgs_memory']}")

    cases = []

    cases.append(_train_one(
        "A (BINN, std)", standard_loss, init_state, shared,
        freeze_gamma=True,
    ))
    cases.append(_train_one(
        "B (SE-BINN, std)", standard_loss, init_state, shared,
    ))
    cases.append(_train_one(
        "C (SE-BINN, V⁻¹)", exact_precond_loss, init_state, shared,
    ))
    cases.append(_train_one(
        "D (SE-BINN, bil T̃)", bilinear_loss, init_state, shared,
    ))
    cases.append(_train_one(
        "E (SE-BINN, wbil WT̃)", weighted_bilinear_loss, init_state, shared,
    ))

    # ------------------------------------------------------------------
    # 8. Final summary table
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    labels_short = ["A (BINN)", "B (SE-BINN)", "C (V⁻¹)", "D (bil)", "E (wbil)"]
    hdr = f"  {'Metric':<22}" + "".join(f"  {l:>13}" for l in labels_short)
    print(hdr)
    print("  " + "-"*88)

    for name, key in [
        ("Density rel-diff",  "density_rel_diff"),
        ("Interior rel L2",   "final_rel_L2"),
        ("BIE loss (std)",    "bie_loss_final"),
        ("γ error",          "gamma_err"),
    ]:
        row = f"  {name:<22}"
        for r in cases:
            row += f"  {r[key]:>13.4e}"
        print(row)

    row = f"  {'init loss':<22}"
    for r in cases:
        row += f"  {r['loss_init']:>13.3e}"
    print(row)

    row = f"  {'LBFGS reason':<22}"
    for r in cases:
        row += f"  {r['lbfgs_reason']:>13}"
    print(row)

    row = f"  {'Wall time (s)':<22}"
    for r in cases:
        row += f"  {r['wall_time']:>13.1f}"
    print(row)

    print()
    d_B = cases[1]["density_rel_diff"]
    d_C = cases[2]["density_rel_diff"]
    d_D = cases[3]["density_rel_diff"]
    d_E = cases[4]["density_rel_diff"]
    print(f"  B baseline:   d_err = {d_B:.4f}")
    print(f"  C (V⁻¹):     d_err = {d_C:.4f}  ({d_B/max(d_C,1e-6):.1f}× over B)")
    print(f"  D (bil T̃):   d_err = {d_D:.4f}  ({d_B/max(d_D,1e-6):.1f}× over B)  "
          f"[expected: no improvement — cond_eig={cond_TV:.0e}]")
    print(f"  E (wbil WT̃): d_err = {d_E:.4f}  ({d_B/max(d_E,1e-6):.1f}× over B)  "
          f"[cond_eig={cond_WTV:.0e}]")

    # ------------------------------------------------------------------
    # 9. Figures
    # ------------------------------------------------------------------
    _fig_convergence(
        cases, adam_cutoff,
        os.path.join(fig_dir, "calderon_bilinear_convergence.png"),
    )
    _fig_density(
        cases, sigma_bem, arc, sort_idx,
        os.path.join(fig_dir, "calderon_bilinear_density.png"),
    )
    _fig_gamma(
        cases,
        os.path.join(fig_dir, "calderon_bilinear_gamma.png"),
    )

    print(f"\n  All figures saved to {fig_dir}/")
    return cases


if __name__ == "__main__":
    main()
