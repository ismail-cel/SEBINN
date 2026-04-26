"""
Experiment: Right-preconditioning using B_direct = K_h² − (1/4)I.

Mathematical background
-----------------------
The Calderón identity at operator level reads:
    V W = (1/4) I − K²

This motivates two candidate preconditioned operators:

    B_nystrom   ≡ −V W̃          (Nyström matrix product)
                                  cond_svd ≈ 304, non-normality ≈ 0.65
    B_direct    ≡ (1/4)I − K_h²  (assembled directly from double-layer K_h)
                                  cond_svd ≈ 198, non-normality ≈ 6×10⁻⁴

    B_direct_neg ≡ K_h² − (1/4)I  (same sign as B_nystrom = −VW ≈ K²−(1/4)I)

The near-normality of B_direct_neg means cond_svd ≈ cond_eig, so the
gradient Hessian condition number is B_direct_neg^T B_direct_neg ≈ 198²
rather than the much larger cond_svd(B_nystrom)² ≈ 304².

Change of variable
------------------
    σ = −W̃ ρ  (Case D, Case E)

Case D trains ρ with B_nystrom (baseline right-prec, carried over for comparison).
Case E trains ρ with B_direct_neg (new, nearly normal).

For Case E with singular enrichment:
    σ = −W̃ ρ + γ σ_s
    Vσ = g  →  B_direct_neg ρ + (V σ_s) γ = g

Preprocessing (Steps 1–3)
--------------------------
1. Assemble B_nystrom, B_direct, B_direct_neg; compare conditioning.
2. Sign test: solve both (B_direct ρ = g) and (B_direct_neg ρ = g),
   recover σ = −W̃ρ, compare to σ_mfg.
3. Smoothness check: plot ρ* = B^{-1} g vs σ_mfg.

Cases
-----
A: BINN       + standard loss   (γ frozen at 0)         — baseline
B: SE-BINN    + standard loss   (γ trainable)            — baseline + enrichment
C: SE-BINN    + exact V⁻¹ preconditioner                 — reference / best known
D: right-prec NYSTROM, no enrichment  σ = −W̃ρ           — B_nystrom
E: right-prec DIRECT, + enrichment    σ = −W̃ρ + γσ_s   — B_direct_neg (NEW)

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
    compute_panel_normals,
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
# Double-layer operator assembly
# ===========================================================================

def assemble_double_layer(qdata, normals):
    """
    Double-layer operator K_h for 2D Laplace.

    K(x,y) = (1/2π) · (y−x)·n_y / |x−y|²

    n_y = unit outward normal at SOURCE point y.
    Matrix entry: K_h[i,j] = K(x_i, y_j) · w_j.

    Self-panel: (y−x) is tangential on a straight panel → (y−x)·n_y = 0.
    All self-panel entries are exactly zero (no correction needed).
    """
    Yq = qdata.Yq       # (2, Nq)
    wq = qdata.wq       # (Nq,)
    Nq = qdata.n_quad

    # Broadcast per-panel normals to per-node shape (Nq, 2)
    node_normals = np.empty((Nq, 2))
    for pid in range(qdata.n_panels):
        js = qdata.idx_std[pid]
        node_normals[js, :] = normals[pid, :]

    # d[i,j] = y_j − x_i  (SOURCE minus TARGET)
    d_x = Yq[0, :][None, :] - Yq[0, :][:, None]   # (Nq, Nq)
    d_y = Yq[1, :][None, :] - Yq[1, :][:, None]

    rho2      = d_x**2 + d_y**2
    rho2_safe = np.where(rho2 > 1e-30, rho2, 1.0)

    ny_x = node_normals[:, 0]
    ny_y = node_normals[:, 1]
    d_dot_ny = d_x * ny_x[None, :] + d_y * ny_y[None, :]   # (Nq, Nq)

    kernel = (1.0 / (2.0 * np.pi)) * d_dot_ny / rho2_safe

    np.fill_diagonal(kernel, 0.0)
    kernel[rho2 < 1e-30] = 0.0

    K_h = kernel * wq[None, :]
    return K_h


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


def right_prec_nystrom_loss(model, op):
    """
    Case D: right-prec nystrom.

    σ = −W̃ ρ,   Loss = ||B_nystrom ρ − g||² / Nq

    B_nystrom = −V W̃,  no singular enrichment.
    """
    rho = model.sigma_w(op.Yq)
    res = op.B_nystrom @ rho - op.f_Yq
    loss = (res**2).mean() * op.prec_scale_nystrom
    with torch.no_grad():
        dbg = {"loss": float(loss), "mean_abs_res": float(res.abs().mean()),
               "gamma": model.gamma_value()}
    return loss, dbg


def right_prec_direct_loss(model, op):
    """
    Case E: right-prec direct + singular enrichment.

    σ = −W̃ ρ + γ σ_s
    Loss = ||B_direct_neg ρ + (V σ_s) γ − g||² / Nq

    B_direct_neg = K_h² − (1/4)I  (nearly normal, cond_svd ≈ 198).
    """
    rho   = model.sigma_w(op.Yq)
    gamma = model.gamma_module()
    enrichment = op.V_sigma_s_q @ gamma
    res = op.B_direct_neg @ rho + enrichment - op.f_Yq
    loss = (res**2).mean() * op.prec_scale_direct
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

    recover_sigma_fn : callable(model, Yq_t, sigma_s_t) -> np.ndarray
        How to compute σ from the model.  Default: model(Yq_t, sigma_s_t).
        Right-prec cases: σ = −W̃ρ [+ γσ_s].
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
    V_h_np     = shared["V_h_np"]
    g_mfg_np   = shared["g_mfg_np"]
    n_gamma    = sigma_s_Yq.shape[1]

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

    g_norm = float(np.linalg.norm(g_mfg_np))

    def _reldiff_and_bie():
        s = recover_sigma_fn(model, Yq_t, sigma_s_t)
        d_err   = float(np.linalg.norm(s - sigma_bem)
                        / max(np.linalg.norm(sigma_bem), 1e-14))
        bie_res = float(np.linalg.norm(V_h_np @ s - g_mfg_np) / max(g_norm, 1e-14))
        return d_err, bie_res, s

    def _record(stage, n_iter, loss_val):
        d_err, bie_res, s = _reldiff_and_bie()
        i_out = reconstruct_interior(
            P=P, Yq=Yq_T, wq=wq, sigma=s,
            n_grid=CFG["n_grid_coarse"], u_exact=u_exact,
        )
        g     = model.gamma_value()
        g_arr = np.array(g if isinstance(g, list) else [float(g)])
        traj.append((n_iter, d_err, loss_val, g_arr.copy(),
                     float(i_out.rel_L2), bie_res))
        gamma_traj.append((n_iter, g_arr.copy()))
        if verbose:
            gfmt = "[" + ",".join(f"{v:+.3f}" for v in g_arr) + "]"
            print(f"  [{label}] {stage}: loss={loss_val:.3e} | "
                  f"d_err={d_err:.4f} | bie={bie_res:.3e} | γ={gfmt}")

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

    sigma_final = recover_sigma_fn(model, Yq_t, sigma_s_t)

    final_out = reconstruct_interior(
        P=P, Yq=Yq_T, wq=wq, sigma=sigma_final,
        n_grid=CFG["n_grid_final"], u_exact=u_exact,
    )
    d_err_final = float(np.linalg.norm(sigma_final - sigma_bem)
                        / max(np.linalg.norm(sigma_bem), 1e-14))
    bie_final   = float(np.linalg.norm(V_h_np @ sigma_final - g_mfg_np)
                        / max(g_norm, 1e-14))
    gamma_final = np.array(
        model.gamma_value() if isinstance(model.gamma_value(), list)
        else [float(model.gamma_value())]
    )
    gamma_err = float(np.linalg.norm(gamma_final - CFG["gamma_true"])
                      / max(np.linalg.norm(CFG["gamma_true"]), 1e-14))

    if verbose:
        print(f"\n  {label} final:")
        print(f"    Density rel-diff : {d_err_final:.4f}")
        print(f"    BIE residual     : {bie_final:.3e}")
        print(f"    Interior L2      : {final_out.rel_L2:.3e}")
        print(f"    γ error          : {gamma_err:.4f}")
        print(f"    γ final          : [{', '.join(f'{v:+.4f}' for v in gamma_final)}]")
        print(f"    LBFGS reason     : {lbfgs_res.reason}")
        print(f"    Wall time        : {time.perf_counter()-t0:.1f}s")

    with torch.no_grad():
        rho_final = model.sigma_w(Yq_t).numpy()

    with torch.no_grad():
        bie_loss_final, _ = standard_loss(model, op)

    return dict(
        label=label,
        density_rel_diff=d_err_final,
        bie_residual=bie_final,
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
    "A (BINN, std)":              "#9467bd",
    "B (SE-BINN, std)":           "#1f77b4",
    "C (SE-BINN, V⁻¹)":          "#2ca02c",
    "D (right-prec, B_nystrom)":  "#d62728",
    "E (right-prec, B_direct)":   "#ff7f0e",
}
MARKERS = {
    "A (BINN, std)":              "s",
    "B (SE-BINN, std)":           "o",
    "C (SE-BINN, V⁻¹)":          "^",
    "D (right-prec, B_nystrom)":  "D",
    "E (right-prec, B_direct)":   "P",
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
        "Density error convergence — Koch(1)\n"
        r"D: $B_\mathrm{nystrom}=-V\tilde{W}$  (cond$_\mathrm{svd}$≈304)"
        r"  |  E: $B_\mathrm{direct}=K^2-\frac{1}{4}I$  (cond$_\mathrm{svd}$≈198)",
        fontsize=11,
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    try:
        fig.tight_layout()
    except Exception:
        pass
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

    fig.suptitle(
        r"Right-prec direct ($B_\mathrm{direct}=K^2-\frac{1}{4}I$) — Koch(1)",
        fontsize=12,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def _fig_rho(cases_DE, rho_targets, arc, sort_idx, outpath):
    """Plot learned ρ for cases D and E, with reference ρ* from lstsq."""
    arc_s = arc[sort_idx]
    labels = [r["label"] for r in cases_DE]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.subplots_adjust(hspace=0.32)

    ax = axes[0]
    for lbl, rho_tgt in rho_targets.items():
        ax.plot(arc_s, rho_tgt[sort_idx], "k--", lw=1.6, alpha=0.55,
                label=rf"$\rho^\star$ ({lbl})")
    for res in cases_DE:
        lbl = res["label"]
        c   = COLORS.get(lbl, "gray")
        ax.plot(arc_s, res["rho_final"], "-", color=c, lw=1.4, alpha=0.9, label=lbl)
    ax.set_ylabel(r"$\rho(s)$  (network output)", fontsize=11)
    ax.set_title(r"Smooth preimage $\rho$ vs reference $\rho^\star = B^{-1}g$", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, lw=0.3, alpha=0.4)

    ax = axes[1]
    for res in cases_DE:
        lbl = res["label"]
        c   = COLORS.get(lbl, "gray")
        key = lbl
        if key in rho_targets:
            err = np.abs(res["rho_final"] - rho_targets[key][sort_idx])
            ax.semilogy(arc_s, err + 1e-20, "-", color=c, lw=1.4, alpha=0.9,
                        label=f"{lbl}  ρ-err")
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$|\rho(s) - \rho^\star(s)|$", fontsize=11)
    ax.set_title("Pointwise ρ error vs lstsq reference", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)

    fig.suptitle(r"Learned $\rho$ for right-prec cases — Koch(1)", fontsize=12)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def _fig_bie(cases, adam_cutoff, outpath):
    """BIE residual ||Vσ − g|| / ||g|| vs iteration for all cases."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for res in cases:
        lbl   = res["label"]
        traj  = res["traj"]
        iters = [t[0] for t in traj]
        bies  = [t[5] for t in traj]        # index 5 = bie_res
        c  = COLORS.get(lbl, "gray")
        mk = MARKERS.get(lbl, "o")
        ax.semilogy(iters, bies, mk + "-", color=c, lw=1.8, ms=6,
                    markevery=max(1, len(iters)//8), label=lbl)
        ax.annotate(f"  {bies[-1]:.2e}",
                    xy=(iters[-1], bies[-1]), fontsize=9, color=c, va="center")

    ax.axvline(x=adam_cutoff, color="gray", ls=":", lw=1.0, alpha=0.6,
               label="Adam → L-BFGS")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(r"$\|V_h \sigma - g\| / \|g\|$", fontsize=12)
    ax.set_title(
        r"BIE residual $\|V_h\sigma - g\|/\|g\|$ — Koch(1)"
        "\n(for right-prec cases, σ recovered from −W̃ρ)",
        fontsize=11,
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
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


# ===========================================================================
# Main
# ===========================================================================

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    fig_dir = os.path.join(_HERE, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "="*70)
    print("RIGHT-PREC DIRECT: B_direct = K_h² − (1/4)I")
    print("Hypothesis: near-normality (non-norm ≈ 6×10⁻⁴) → better training")
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
    # 2. W̃ and K_h
    # ------------------------------------------------------------------
    W_h, _  = assemble_hypersingular_direct(qdata)
    W_tilde = regularise_hypersingular(W_h, wq)

    normals, _ = compute_panel_normals(qdata)
    K_h        = assemble_double_layer(qdata, normals)
    print(f"  K_h max|self-panel| = {max(np.abs(K_h[np.ix_(qdata.idx_std[p], qdata.idx_std[p])]).max() for p in range(qdata.n_panels)):.2e}")

    # ------------------------------------------------------------------
    # STEP 1: Operator comparison
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: Operator comparison")
    print("="*60)

    B_nystrom_np   = V_h @ (-W_tilde)            # −VW̃
    K2             = K_h @ K_h
    I              = np.eye(Nq)
    B_direct_np    = 0.25 * I - K2               # (1/4)I − K²
    B_direct_neg_np = K2 - 0.25 * I              # K² − (1/4)I  ≈ −VW̃

    def _cond_stats(M, name):
        svs  = np.linalg.svd(M, compute_uv=False)
        eigs = np.abs(np.linalg.eigvals(M))
        cond_svd = float(svs[0] / svs[-1])
        cond_eig = float(eigs.max() / eigs.min())
        non_norm = (np.linalg.norm(M.T @ M - M @ M.T)
                    / np.linalg.norm(M)**2)
        print(f"  {name:<30}  cond_svd={cond_svd:8.1f}  "
              f"cond_eig={cond_eig:6.1f}  non-norm={non_norm:.3e}")
        return cond_svd, cond_eig, non_norm

    cond_svd_V,   _, _  = _cond_stats(V_h,            "V_h")
    cond_svd_Bn,  _, _  = _cond_stats(B_nystrom_np,   "B_nystrom (−VW̃)")
    cond_svd_Bd,  _, _  = _cond_stats(B_direct_np,    "B_direct  ((1/4)I−K²)")
    cond_svd_Bdn, _, _  = _cond_stats(B_direct_neg_np,"B_direct_neg (K²−(1/4)I)")

    rel_err_calderon = (np.linalg.norm(B_nystrom_np - B_direct_neg_np, "fro")
                        / np.linalg.norm(B_direct_neg_np, "fro"))
    print(f"\n  ||B_nystrom − B_direct_neg||_F / ||B_direct_neg||_F = {rel_err_calderon:.4f}")
    print(f"  (0 = Calderón identity holds exactly at matrix level)")

    # ------------------------------------------------------------------
    # STEP 2: Sign test
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: Sign test — which operator/sign recovers σ_mfg best?")
    print("="*60)

    # Manufactured density (needed here for sign test)
    enrichment_setup = SingularEnrichment(geom=geom, use_cutoff=True,
                                          cutoff_radius=CFG["cutoff_radius"],
                                          per_corner_gamma=True)
    n_sing     = enrichment_setup.n_singular
    sigma_s_Yq = enrichment_setup.precompute(Yq_T)
    gamma_true = CFG["gamma_true"].copy()

    f_smooth     = Yq_T[:, 0]**2 - Yq_T[:, 1]**2
    sigma_smooth = solve_bem(nmat, f_smooth, tol=1e-12).sigma
    sigma_mfg    = sigma_smooth + sigma_s_Yq @ gamma_true
    g_mfg        = V_h @ sigma_mfg
    sigma_bem    = solve_bem(nmat, g_mfg, tol=1e-12).sigma

    energy = float(np.linalg.norm(sigma_s_Yq @ gamma_true)**2
                   / max(np.linalg.norm(sigma_mfg)**2, 1e-14))
    print(f"  Enrichment energy = {energy*100:.2f}%")

    def _sign_test(B, g_rhs, B_label, g_label):
        rho, *_ = np.linalg.lstsq(B, g_rhs, rcond=None)
        sigma_rec = -W_tilde @ rho
        d_err = float(np.linalg.norm(sigma_rec - sigma_bem)
                      / max(np.linalg.norm(sigma_bem), 1e-14))
        print(f"  {B_label:<30}  rhs={g_label:<8}  d_err={d_err:.4f}")
        return rho, d_err

    rho_Bn_pos, err_Bn_pos = _sign_test(B_nystrom_np,    g_mfg,  "B_nystrom",    "+g")
    rho_Bn_neg, err_Bn_neg = _sign_test(B_nystrom_np,   -g_mfg,  "B_nystrom",    "−g")
    rho_Bd_pos, err_Bd_pos = _sign_test(B_direct_np,     g_mfg,  "B_direct",     "+g")
    rho_Bd_neg, err_Bd_neg = _sign_test(B_direct_np,    -g_mfg,  "B_direct",     "−g")
    rho_Bdn_pos, err_Bdn_pos = _sign_test(B_direct_neg_np,  g_mfg, "B_direct_neg", "+g")
    rho_Bdn_neg, err_Bdn_neg = _sign_test(B_direct_neg_np, -g_mfg, "B_direct_neg", "−g")

    # Determine best sign for B_direct_neg (training operator for Case E)
    if err_Bdn_pos < err_Bdn_neg:
        rhs_sign_direct = +1.0
        rho_E_ref = rho_Bdn_pos
        print(f"\n  → B_direct_neg ρ = +g  is the correct sign (d_err={err_Bdn_pos:.4f})")
    else:
        rhs_sign_direct = -1.0
        rho_E_ref = rho_Bdn_neg
        print(f"\n  → B_direct_neg ρ = −g  is the correct sign (d_err={err_Bdn_neg:.4f})")

    rho_D_ref = rho_Bn_pos if err_Bn_pos < err_Bn_neg else rho_Bn_neg

    # ------------------------------------------------------------------
    # STEP 3: Smoothness check
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3: Smoothness of reference ρ*")
    print("="*60)
    print(f"  ||σ_mfg||_2       = {np.linalg.norm(sigma_mfg):.4f}  (has corner singularities)")
    print(f"  ||ρ*_D (nystrom)||_2 = {np.linalg.norm(rho_D_ref):.4f}")
    print(f"  ||ρ*_E (direct)||_2  = {np.linalg.norm(rho_E_ref):.4f}")
    print(f"  max|σ_mfg| = {np.abs(sigma_mfg).max():.4f}")
    print(f"  max|ρ*_D|  = {np.abs(rho_D_ref).max():.4f}")
    print(f"  max|ρ*_E|  = {np.abs(rho_E_ref).max():.4f}")

    # Smoothness proxy: ratio of max to mean absolute value
    def _smoothness(x, name):
        r = float(np.abs(x).max() / (np.abs(x).mean() + 1e-14))
        print(f"  max/mean  {name:<20} = {r:.2f}  "
              f"({'smooth' if r < 5 else 'singular/spiky'})")
    _smoothness(sigma_mfg, "σ_mfg")
    _smoothness(rho_D_ref, "ρ*_D (nystrom)")
    _smoothness(rho_E_ref, "ρ*_E (direct)")

    # ------------------------------------------------------------------
    # 4. Loss scales
    # ------------------------------------------------------------------
    f_Yq_np     = g_mfg.copy()
    V_sigma_s_q_np = V_h @ sigma_s_Yq

    r0_std = float(np.mean(g_mfg**2))  # initial std loss scale

    loss_D_0 = float(np.mean((B_nystrom_np @ np.zeros(Nq) - f_Yq_np)**2))
    loss_E_0 = float(np.mean((rhs_sign_direct * B_direct_neg_np @ np.zeros(Nq)
                               + V_sigma_s_q_np @ np.zeros(n_sing)
                               - f_Yq_np)**2))

    prec_scale_nystrom = r0_std / max(loss_D_0, 1e-14)
    prec_scale_direct  = r0_std / max(loss_E_0, 1e-14)

    print(f"\n  Loss scale check (ρ=0, γ=0):")
    print(f"    std   init loss = {r0_std:.3e}")
    print(f"    D     init loss = {loss_D_0:.3e}  prec_scale={prec_scale_nystrom:.3e}")
    print(f"    E     init loss = {loss_E_0:.3e}  prec_scale={prec_scale_direct:.3e}")

    # ------------------------------------------------------------------
    # 5. Operator state (standard collocation; used for A, B, C)
    # ------------------------------------------------------------------
    col_pts   = build_collocation_points(panels, m_col_panel=CFG["m_col_base"])
    panel_wts = panel_loss_weights(panels, w_base=CFG["w_base"],
                                   w_corner=CFG["w_corner"], w_ring=CFG["w_ring"])
    g_fn      = lambda xy: xy[:, 0]**2 - xy[:, 1]**2
    op, diag  = build_operator_state(
        colloc=col_pts, qdata=qdata, enrichment=enrichment_setup, g=g_fn,
        panel_weights=panel_wts, eq_scale_mode="none", eq_scale_fixed=1.0,
        dtype=torch.float64, device="cpu",
    )
    assert abs(diag["eq_scale"] - 1.0) < 1e-12
    op.f = torch.tensor(g_mfg, dtype=torch.float64)

    V_inv_np = np.linalg.inv(V_h)
    op.V_inv = torch.tensor(V_inv_np, dtype=torch.float64)

    # Right-prec additions to op
    op.B_nystrom        = torch.tensor(B_nystrom_np,    dtype=torch.float64)
    op.B_direct_neg     = torch.tensor(rhs_sign_direct * B_direct_neg_np,
                                       dtype=torch.float64)
    op.f_Yq             = torch.tensor(f_Yq_np,         dtype=torch.float64)
    op.V_sigma_s_q      = torch.tensor(V_sigma_s_q_np,  dtype=torch.float64)
    op.prec_scale_nystrom = prec_scale_nystrom
    op.prec_scale_direct  = prec_scale_direct

    print(f"    ||V_inv V − I||_F/√N = "
          f"{np.linalg.norm(V_inv_np @ V_h - np.eye(Nq))/np.sqrt(Nq):.3e}")

    # ------------------------------------------------------------------
    # 6. Shared initial state + arc-length sort
    # ------------------------------------------------------------------
    u_exact = make_u_exact_fn(Yq_T, wq, sigma_mfg)

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
        V_h_np=V_h, g_mfg_np=g_mfg,
    )

    adam_cutoff = sum(CFG["adam_iters"])

    # ------------------------------------------------------------------
    # 7. recover_sigma closures
    # ------------------------------------------------------------------
    _W_tilde_np    = W_tilde
    _sigma_s_Yq_np = sigma_s_Yq

    def recover_std(m, Yq_t, ss_t):
        with torch.no_grad():
            return m(Yq_t, ss_t).numpy()

    def recover_D(m, Yq_t, ss_t):
        """Case D: σ = −W̃ρ (no enrichment)."""
        with torch.no_grad():
            rho = m.sigma_w(Yq_t).numpy()
        return -_W_tilde_np @ rho

    def recover_E(m, Yq_t, ss_t):
        """Case E: σ = −W̃ρ + γσ_s."""
        with torch.no_grad():
            rho   = m.sigma_w(Yq_t).numpy()
            g_val = m.gamma_value()
            gamma = np.array(g_val if isinstance(g_val, list) else [float(g_val)])
        return -_W_tilde_np @ rho + _sigma_s_Yq_np @ gamma

    # ------------------------------------------------------------------
    # 8. Training: 5 cases
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("TRAINING (5 cases)")
    print(f"  Adam: {CFG['adam_iters']} iters at lr {CFG['adam_lrs']}")
    print(f"  LBFGS: {CFG['lbfgs_iters']} iters, memory {CFG['lbfgs_memory']}")
    print("="*70)

    cases = []

    cases.append(_train_one(
        "A (BINN, std)", standard_loss, init_state, shared,
        freeze_gamma=True, recover_sigma_fn=recover_std,
    ))

    cases.append(_train_one(
        "B (SE-BINN, std)", standard_loss, init_state, shared,
        freeze_gamma=False, recover_sigma_fn=recover_std,
    ))

    cases.append(_train_one(
        "C (SE-BINN, V⁻¹)", exact_precond_loss, init_state, shared,
        freeze_gamma=False, recover_sigma_fn=recover_std,
    ))

    cases.append(_train_one(
        "D (right-prec, B_nystrom)", right_prec_nystrom_loss, init_state, shared,
        freeze_gamma=True, recover_sigma_fn=recover_D,
    ))

    cases.append(_train_one(
        "E (right-prec, B_direct)", right_prec_direct_loss, init_state, shared,
        freeze_gamma=False, recover_sigma_fn=recover_E,
    ))

    # ------------------------------------------------------------------
    # 9. Summary table
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    labels_short = ["A (BINN)", "B (SE-BINN)", "C (V⁻¹)", "D (Bny)", "E (Bdir)"]
    hdr = f"  {'Metric':<22}" + "".join(f"  {l:>13}" for l in labels_short)
    print(hdr)
    print("  " + "-"*92)

    for name, key in [
        ("Density rel-diff",  "density_rel_diff"),
        ("BIE residual",      "bie_residual"),
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

    row = f"  {'Wall time (s)':<22}"
    for r in cases:
        row += f"  {r['wall_time']:>13.1f}"
    print(row)

    print()
    d_A = cases[0]["density_rel_diff"]
    d_B = cases[1]["density_rel_diff"]
    d_C = cases[2]["density_rel_diff"]
    d_D = cases[3]["density_rel_diff"]
    d_E = cases[4]["density_rel_diff"]
    print(f"  A (BINN):          d_err = {d_A:.4f}")
    print(f"  B (SE-BINN):       d_err = {d_B:.4f}")
    print(f"  C (V⁻¹ prec):      d_err = {d_C:.4f}  ({d_B/max(d_C,1e-6):.1f}× over B)")
    print(f"  D (B_nystrom):     d_err = {d_D:.4f}  ({d_B/max(d_D,1e-6):.1f}× over B)")
    print(f"  E (B_direct):      d_err = {d_E:.4f}  ({d_B/max(d_E,1e-6):.1f}× over B)")
    print()
    print(f"  H1 test: cond_svd(B_nystrom)={cond_svd_Bn:.0f}, "
          f"cond_svd(B_direct_neg)={cond_svd_Bdn:.0f}")
    if d_E < d_D * 0.8:
        print(f"\n  H1 SUPPORTED: B_direct improves over B_nystrom "
              f"({d_D/max(d_E,1e-6):.1f}× reduction in d_err).")
    elif d_E < d_D:
        print(f"\n  H1 MARGINAL: B_direct slightly better than B_nystrom.")
    else:
        print(f"\n  H1 REJECTED: B_direct does NOT improve over B_nystrom.")
        print(f"  → Near-normality alone insufficient; the recovery σ=−W̃ρ "
              f"still routes through V⁻¹.")

    # ------------------------------------------------------------------
    # 10. Figures
    # ------------------------------------------------------------------
    rho_targets = {
        "D (right-prec, B_nystrom)": rho_D_ref,
        "E (right-prec, B_direct)":  rho_E_ref,
    }

    _fig_convergence(
        cases, adam_cutoff,
        os.path.join(fig_dir, "right_prec_direct_convergence.png"),
    )
    _fig_density(
        cases, sigma_bem, arc, sort_idx,
        os.path.join(fig_dir, "right_prec_direct_density.png"),
    )
    _fig_rho(
        [cases[3], cases[4]], rho_targets, arc, sort_idx,
        os.path.join(fig_dir, "right_prec_direct_rho.png"),
    )
    _fig_bie(
        cases, adam_cutoff,
        os.path.join(fig_dir, "right_prec_direct_bie.png"),
    )
    _fig_gamma(
        cases,
        os.path.join(fig_dir, "right_prec_direct_gamma.png"),
    )

    print(f"\n  All figures saved to {fig_dir}/")
    return cases


if __name__ == "__main__":
    main()
