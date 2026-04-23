"""
Experiment: Calderón preconditioner — final decisive run.

Normalisation rationale
-----------------------
Phase 2 showed cond_eig(T̃V) = 13.6 (Calderón identity works at the
discrete level) but training failed because T̃ has entries O(1/h²)
making the loss 10⁸× too large for Adam.

Phase 3 divided by ρ(T̃) = 1.1e4, which brought the eigenvalues of
T̃_s V down to ~3e-4 (1100× too small), causing L-BFGS to stall.

Correct normalisation: divide T̃ by median|λ(T̃V)|.

    T̃_norm = T̃ / median|λ(T̃V)|

Effect on eigenvalues of T̃_norm V:
    median|λ| = 1       (by construction)
    cond_eig  = 13.6    (unchanged)
    min|λ|    ≈ 0.27    max|λ| ≈ 3.7

Effect on loss scale:
    ||T̃_norm r||² ≈ ||r||²  (because typical |λ|≈1 means T̃_norm≈V⁻¹ on average)

This is deterministic (depends only on the operators, not initialisation).

Cases
-----
A: BINN       + standard loss  (γ frozen at 0)
B: SE-BINN    + standard loss  (γ trainable)
C: SE-BINN    + exact V_h^{-1} preconditioner
D: SE-BINN    + normalised Calderón T̃_norm

Budget: Adam [1000,1000,1000] at lr [1e-3, 3e-4, 1e-4] + LBFGS 15000.
Geometry: Koch(1), manufactured singular density.

Figures:
  calderon_final_convergence.png    — density rel-diff vs iteration (4 cases)
  calderon_final_density.png        — σ vs arc-length + |σ-σ_mfg| (4 cases)
  calderon_final_gamma.png          — γ_c trajectories (Cases B, C, D)
  calderon_eigenvalue_comparison.png — eigenvalue spectra V, T̃_norm V, V⁻¹V
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
    res = op.A @ model(op.Yq, op.sigma_s_q) + op.corr * model(op.Xc, op.sigma_s_c) - op.f
    prec = op.V_inv @ res
    loss = (op.wCol * prec**2).sum() / op.wCol_sum
    with torch.no_grad():
        dbg = {"loss": float(loss), "mean_abs_res": float(res.abs().mean()),
               "gamma": model.gamma_value()}
    return loss, dbg


def calderon_norm_loss(model, op):
    """L = ||T̃_norm (Vσ − g)||²  (median-eigenvalue normalised Calderón)."""
    res = op.A @ model(op.Yq, op.sigma_s_q) + op.corr * model(op.Xc, op.sigma_s_c) - op.f
    prec = op.T_tilde_norm @ res
    loss = (op.wCol * prec**2).sum() / op.wCol_sum
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

    t0        = time.perf_counter()
    op        = shared["op"]
    Yq_T      = shared["Yq_T"]
    wq        = shared["wq"]
    P         = shared["P"]
    sigma_bem = shared["sigma_bem"]
    sigma_s_Yq= shared["sigma_s_Yq"]
    sort_idx  = shared["sort_idx"]
    u_exact   = shared["u_exact"]
    arc       = shared["arc"]
    n_gamma   = sigma_s_Yq.shape[1]

    model = _build_model(n_gamma, freeze_gamma=freeze_gamma)
    model.load_state_dict(copy.deepcopy(init_state), strict=False)
    if freeze_gamma:
        # re-zero gamma (load_state_dict may have set it from init_state)
        with torch.no_grad():
            for p in model.gamma_module.parameters():
                p.zero_()

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  trainable params: {n_tr}  (gamma {'frozen' if freeze_gamma else 'trainable'})")

    Yq_t      = torch.tensor(Yq_T, dtype=torch.float64)
    sigma_s_t = torch.tensor(sigma_s_Yq, dtype=torch.float64)
    traj      = []
    gamma_traj= []   # (iter, gamma_array)

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
        g = model.gamma_value()
        g_arr = np.array(g if isinstance(g, list) else [float(g)])
        traj.append((n_iter, d_err, loss_val, g_arr.copy(), float(i_out.rel_L2)))
        gamma_traj.append((n_iter, g_arr.copy()))
        if verbose:
            gfmt = "[" + ",".join(f"{v:+.3f}" for v in g_arr) + "]"
            print(f"  [{label}] {stage}: loss={loss_val:.3e} | "
                  f"d_err={d_err:.4f} | iL2={i_out.rel_L2:.3e} | γ={gfmt}")

    # Initial state
    with torch.no_grad():
        loss0, _ = loss_fn(model, op)
    _record("init", 0, float(loss0))

    # Adam phases
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

    # L-BFGS
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

    # Also compute BIE loss at final state (standardised for comparison)
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
    "A (BINN, std)":          "#9467bd",
    "B (SE-BINN, std)":       "#1f77b4",
    "C (SE-BINN, V⁻¹)":      "#2ca02c",
    "D (SE-BINN, Calderón)":  "#d62728",
}
MARKERS = {"A (BINN, std)": "s", "B (SE-BINN, std)": "o",
           "C (SE-BINN, V⁻¹)": "^", "D (SE-BINN, Calderón)": "D"}


def _fig_convergence(cases, adam_cutoff, outpath):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for res in cases:
        lbl  = res["label"]
        traj = res["traj"]
        iters  = [t[0] for t in traj]
        d_errs = [t[1] for t in traj]
        c = COLORS.get(lbl, "gray")
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
    ax.set_title("Calderón preconditioner — density rel-diff, Koch(1)\n"
                 "A: BINN  |  B: SE-BINN  |  C: V⁻¹ precond  |  D: Calderón T̃_norm",
                 fontsize=11)
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

    fig.suptitle("Calderón preconditioner — Koch(1)  (full budget)", fontsize=12)
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
        lbl  = res["label"]
        c    = COLORS.get(lbl, "gray")
        traj = res["gamma_traj"]
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
    ax.set_title("γ_c trajectories — B, C, D  (dashed = γ_true)", fontsize=11)
    ax.grid(True, lw=0.3, alpha=0.4)
    # Compact legend: one entry per case
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color=COLORS.get(r["label"],"gray"), lw=2,
                      label=r["label"]) for r in cases_with_gamma]
    ax.legend(handles=handles, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def _fig_eigenvalues(V_h, T_tilde_norm, outpath):
    eig_V    = np.sort(np.abs(np.linalg.eigvals(V_h)))[::-1]
    eig_TV   = np.sort(np.abs(np.linalg.eigvals(T_tilde_norm @ V_h)))[::-1]
    eig_VinvV= np.ones(len(eig_V))    # exactly 1 by construction

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.35)

    ax = axes[0]
    idx = np.arange(1, len(eig_V)+1)
    ax.semilogy(idx, eig_V,     color="#1f77b4", lw=1.5, label=r"$V_h$")
    ax.semilogy(idx, eig_TV,    color="#d62728", lw=1.5, label=r"$\tilde{T}_\mathrm{norm} V_h$")
    ax.semilogy(idx, eig_VinvV, color="#2ca02c", lw=1.5, ls="--",
                label=r"$V_h^{-1} V_h$ (ideal)")
    ax.set_xlabel("Index (sorted)", fontsize=11)
    ax.set_ylabel(r"$|\lambda_k|$", fontsize=11)
    ax.set_title("Sorted eigenvalue magnitudes", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    cond_V  = np.linalg.cond(V_h)
    cond_TV = np.linalg.cond(T_tilde_norm @ V_h)
    ax.text(0.03, 0.04,
            f"cond$(V_h)$ = {cond_V:.2e}\n"
            f"cond$(\\tilde{{T}}_n V_h)$ = {cond_TV:.2e}\n"
            f"cond$(V^{{-1}}V)$ = 1.00e+00",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax = axes[1]
    eig_V_c  = np.linalg.eigvals(V_h)
    eig_TV_c = np.linalg.eigvals(T_tilde_norm @ V_h)
    ax.scatter(eig_V_c.real,  eig_V_c.imag,  s=3, alpha=0.4, color="#1f77b4",
               label=r"$V_h$")
    ax.scatter(eig_TV_c.real, eig_TV_c.imag, s=3, alpha=0.4, color="#d62728",
               label=r"$\tilde{T}_\mathrm{norm} V_h$")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.axvline(1, color="#2ca02c", lw=1.0, ls="--", alpha=0.6, label="ideal (1+0i)")
    ax.set_xlabel(r"Re$(\lambda)$", fontsize=11)
    ax.set_ylabel(r"Im$(\lambda)$", fontsize=11)
    ax.set_title("Eigenvalues in complex plane", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)

    fig.suptitle(
        r"Eigenvalue spectra: $V_h$ vs $\tilde{T}_{\mathrm{norm}} V_h$ vs ideal preconditioner"
        "\nKoch(1),  cond_eig(T̃_norm V) = 13.6  (Calderón identity at discrete level)",
        fontsize=11,
    )
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
    print("CALDERÓN FINAL: Median-eigenvalue normalised T̃_norm")
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
    # 2. T̃ assembly + median-eigenvalue normalisation
    # ------------------------------------------------------------------
    print("\n  Assembling T_h + computing normalisation...")
    T_h, _  = assemble_hypersingular_direct(qdata)
    T_tilde = regularise_hypersingular(T_h, wq)

    TV          = T_tilde @ V_h
    eigvals_TV  = np.linalg.eigvals(TV)
    mag_TV      = np.abs(eigvals_TV)
    median_eig  = float(np.median(mag_TV))
    T_tilde_norm = T_tilde / median_eig

    # Verify
    eigvals_TVn = np.linalg.eigvals(T_tilde_norm @ V_h)
    mag_TVn     = np.abs(eigvals_TVn)
    cond_eig_before = mag_TV.max() / (mag_TV.min() + 1e-14)
    cond_eig_after  = mag_TVn.max() / (mag_TVn.min() + 1e-14)

    print(f"\n  === Normalisation diagnostics ===")
    print(f"    median|λ(T̃V)|                = {median_eig:.6f}")
    print(f"    Before: eig(T̃V)  min={mag_TV.min():.4f}  "
          f"median={np.median(mag_TV):.4f}  max={mag_TV.max():.4f}  "
          f"cond={cond_eig_before:.1f}")
    print(f"    After:  eig(T̃_n V) min={mag_TVn.min():.4f}  "
          f"median={np.median(mag_TVn):.4f}  max={mag_TVn.max():.4f}  "
          f"cond={cond_eig_after:.1f}")
    print(f"    cond unchanged: {abs(cond_eig_before - cond_eig_after) < 0.01:.0f}  "
          f"({cond_eig_before:.2f} → {cond_eig_after:.2f})")

    # Loss scale check at t=0 (σ≈0 → r≈-g)
    r0 = -nmat.V @ np.zeros(Nq)   # will be set below; placeholder
    cond_V = np.linalg.cond(V_h)
    print(f"\n    cond(V_h)  = {cond_V:.3e}  (reference)")
    print(f"    cond(T̃_n V) = {np.linalg.cond(T_tilde_norm @ V_h):.3e}")

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

    # Loss scale verification with r0 = -g_mfg
    r0          = -g_mfg
    loss_std_0  = float(np.mean(r0**2))
    loss_prec_0 = float(np.mean((T_tilde_norm @ r0)**2))
    print(f"\n  === Initial loss scale (σ=0) ===")
    print(f"    std  loss (||r||²/N):       {loss_std_0:.3e}")
    print(f"    prec loss (||T̃_n r||²/N):  {loss_prec_0:.3e}")
    print(f"    ratio prec/std:             {loss_prec_0/max(loss_std_0,1e-14):.3f}  "
          f"(target: 1.0)")

    u_exact = make_u_exact_fn(Yq_T, wq, sigma_mfg)

    # ------------------------------------------------------------------
    # 4. Operator state
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
    op.V_inv        = torch.tensor(V_inv_np,    dtype=torch.float64)
    op.T_tilde_norm = torch.tensor(T_tilde_norm, dtype=torch.float64)

    print(f"    ||V_inv V − I||_F/√N = "
          f"{np.linalg.norm(V_inv_np @ V_h - np.eye(Nq))/np.sqrt(Nq):.3e}")

    # ------------------------------------------------------------------
    # 5. Shared initial state
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
    # 6. Training: 4 cases
    # ------------------------------------------------------------------
    print("\n  === Training: 4 cases (full budget) ===")
    print(f"  Adam: {CFG['adam_iters']} iters at lr {CFG['adam_lrs']}")
    print(f"  LBFGS: {CFG['lbfgs_iters']} iters, memory {CFG['lbfgs_memory']}")

    cases = []

    # Case A: BINN (γ frozen at 0)
    cases.append(_train_one(
        "A (BINN, std)", standard_loss, init_state, shared,
        freeze_gamma=True,
    ))

    # Case B: SE-BINN standard
    cases.append(_train_one(
        "B (SE-BINN, std)", standard_loss, init_state, shared,
    ))

    # Case C: SE-BINN + exact V^{-1}
    cases.append(_train_one(
        "C (SE-BINN, V⁻¹)", exact_precond_loss, init_state, shared,
    ))

    # Case D: SE-BINN + normalised Calderón
    cases.append(_train_one(
        "D (SE-BINN, Calderón)", calderon_norm_loss, init_state, shared,
    ))

    # ------------------------------------------------------------------
    # 7. Final summary table
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    hdr = f"  {'Metric':<22}  {'A (BINN)':>12}  {'B (SE-BINN)':>12}  {'C (V⁻¹)':>12}  {'D (Calderon)':>12}"
    print(hdr)
    print("  " + "-"*70)

    def _row(name, key, fmt=".4f"):
        vals = [r[key] for r in cases]
        row = f"  {name:<22}"
        for v in vals:
            row += f"  {v:{'>12' if isinstance(v,float) else '>12'}}"
        print(row)

    for name, key in [
        ("Density rel-diff",  "density_rel_diff"),
        ("Interior rel L2",   "final_rel_L2"),
        ("BIE loss (std)",     "bie_loss_final"),
        ("γ error",           "gamma_err"),
    ]:
        row  = f"  {name:<22}"
        for r in cases:
            v = r[key]
            row += f"  {v:>12.4e}"
        print(row)

    print(f"  {'init loss':<22}", end="")
    for r in cases:
        print(f"  {r['loss_init']:>12.3e}", end="")
    print()

    print(f"  {'Adam lr used':<22}", end="")
    for r in cases:
        lr0 = r['adam_lr_used'][0]
        print(f"  {lr0:>12.0e}", end="")
    print()

    print(f"  {'LBFGS reason':<22}", end="")
    for r in cases:
        print(f"  {r['lbfgs_reason']:>12}", end="")
    print()

    print(f"  {'Wall time (s)':<22}", end="")
    for r in cases:
        print(f"  {r['wall_time']:>12.1f}", end="")
    print()

    print()
    d_B = cases[1]["density_rel_diff"]
    d_C = cases[2]["density_rel_diff"]
    d_D = cases[3]["density_rel_diff"]
    print(f"  B baseline:  d_err = {d_B:.4f}")
    print(f"  C (V⁻¹):    d_err = {d_C:.4f}  ({d_B/max(d_C,1e-6):.1f}× improvement over B)")
    print(f"  D (Calderón): d_err = {d_D:.4f}  ({d_B/max(d_D,1e-6):.1f}× improvement over B)")
    gap = (d_D - d_C) / max(d_B - d_C, 1e-4) * 100
    print(f"  D closes {100-gap:.0f}% of the gap from B to C" if d_D < d_B else "  D did not improve over B")

    # ------------------------------------------------------------------
    # 8. Figures
    # ------------------------------------------------------------------
    _fig_convergence(
        cases, adam_cutoff,
        os.path.join(fig_dir, "calderon_final_convergence.png"),
    )
    _fig_density(
        cases, sigma_bem, arc, sort_idx,
        os.path.join(fig_dir, "calderon_final_density.png"),
    )
    _fig_gamma(
        cases,
        os.path.join(fig_dir, "calderon_final_gamma.png"),
    )
    _fig_eigenvalues(
        V_h, T_tilde_norm,
        os.path.join(fig_dir, "calderon_eigenvalue_comparison.png"),
    )

    print(f"\n  All figures saved to {fig_dir}/")
    return cases


if __name__ == "__main__":
    main()
