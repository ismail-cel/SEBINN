"""
Experiment: Calderón bilinear form with spectral-flipped W̃_+.

Background
----------
All previous Calderón attempts failed because:
  - ||T̃r||² amplifies by ||T̃||_op² (Phase 2/3/final)
  - rᵀ T̃_sym r requires T̃_sym PSD, but T̃_sym has 23 neg eigs (bilinear)
    → diagonal shift α≈3170 destroys cond_eig (13.6 → 8.5e16)

This experiment uses SPECTRAL FLIPPING instead of diagonal shifting:
  W̃ = Q Λ Qᵀ  →  W̃_+ = Q |Λ| Qᵀ

Only the 23 negative eigenvalues are flipped (1% of 2304).
The other 2281 eigenvalues are unchanged.
cond_eig(W̃_+ V) should remain ≈ 15 (vs 15.2 for W̃V).

The bilinear form loss is then:
    L = rᵀ W̃_+ r  ≥ 0  (valid because W̃_+ is SPD)

Gradient: ∂L/∂σ = 2 Vᵀ W̃_+ r  — effective cond ≈ 15 (Calderón).

Cases
-----
A: BINN       + standard loss   (γ frozen)
B: SE-BINN    + standard loss   (γ trainable)
C: SE-BINN    + exact V⁻¹       (gold standard: d_err=0.025)
D: SE-BINN    + rᵀ W̃_+ r       (spectral-flip bilinear)

Budget: Adam [1000,1000,1000] + LBFGS 15000.  Koch(1), manufactured density.
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
from src.training.loss import sebinn_loss
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
# Spectral flip
# ===========================================================================

def make_spd_spectral_flip(W_tilde: np.ndarray):
    """
    Make W̃ SPD by flipping negative eigenvalues.

    W̃ = Q Λ Qᵀ  →  W̃_+ = Q |Λ| Qᵀ

    Only the negative eigenvalues are changed; all positive ones are
    preserved exactly.  The correction is rank-n_neg (≪ Nq).

    Returns
    -------
    W_plus    : ndarray (Nq, Nq)  — SPD matrix
    n_flipped : int               — number of eigenvalues flipped
    """
    eigvals, Q = np.linalg.eigh(W_tilde)
    n_neg = int((eigvals < 0).sum())
    W_plus = Q @ np.diag(np.abs(eigvals)) @ Q.T
    W_plus = 0.5 * (W_plus + W_plus.T)   # remove numerical noise
    return W_plus, n_neg


# ===========================================================================
# Loss functions
# ===========================================================================

def standard_loss(model, op):
    return sebinn_loss(model, op)


def exact_precond_loss(model, op):
    """L = ||V⁻¹(Vσ − g)||²"""
    res  = op.A @ model(op.Yq, op.sigma_s_q) + op.corr * model(op.Xc, op.sigma_s_c) - op.f
    prec = op.V_inv @ res
    loss = (op.wCol * prec**2).sum() / op.wCol_sum
    with torch.no_grad():
        dbg = {"loss": float(loss), "mean_abs_res": float(res.abs().mean()),
               "gamma": model.gamma_value()}
    return loss, dbg


def calderon_spd_loss(model, op):
    """L = scale × rᵀ W̃_+ r   (bilinear form, W̃_+ SPD via spectral flip)."""
    res  = op.A @ model(op.Yq, op.sigma_s_q) + op.corr * model(op.Xc, op.sigma_s_c) - op.f
    bil  = torch.dot(res, op.W_plus @ res)
    loss = op.spd_scale * bil
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
               freeze_gamma=False, adam_lrs=None, verbose=True):
    adam_lrs = adam_lrs or CFG["adam_lrs"]

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
        print(f"  trainable params: {n_tr}  (γ {'frozen' if freeze_gamma else 'trainable'})")

    Yq_t      = torch.tensor(Yq_T, dtype=torch.float64)
    sigma_s_t = torch.tensor(sigma_s_Yq, dtype=torch.float64)
    traj       = []
    gamma_traj = []

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

    global_it = 0
    for ph_idx, (n_it, lr) in enumerate(zip(CFG["adam_iters"], adam_lrs)):
        ph_cfg = AdamConfig(phase_iters=[n_it], phase_lrs=[lr],
                            log_every=CFG["log_every"])
        ph_res = run_adam_phases(model, op, ph_cfg, verbose=verbose,
                                 loss_fn=loss_fn)
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
        adam_lrs_used=adam_lrs,
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
    "D (SE-BINN, W̃_+ bil)":  "#d62728",
}
MARKERS = {
    "A (BINN, std)": "s", "B (SE-BINN, std)": "o",
    "C (SE-BINN, V⁻¹)": "^", "D (SE-BINN, W̃_+ bil)": "D",
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
        "Calderón spectral-flip SPD — density rel-diff, Koch(1)\n"
        r"A: BINN  |  B: SE-BINN  |  C: V⁻¹  |  D: $r^\top\tilde{W}_+r$",
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
        ax.plot(arc_s, res["sigma_final"], "-", color=COLORS.get(lbl,"gray"),
                lw=1.2, alpha=0.85,
                label=f"{lbl}  (d={res['density_rel_diff']:.3f})")
    ax.set_ylabel(r"$\sigma(s)$", fontsize=11)
    ax.set_title("Final density", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, lw=0.3, alpha=0.4)
    ax = axes[1]
    for res in cases:
        lbl = res["label"]
        err = np.abs(res["sigma_final"] - bem_s)
        ax.semilogy(arc_s, err + 1e-20, "-", color=COLORS.get(lbl,"gray"),
                    lw=1.2, alpha=0.85, label=lbl)
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$|\sigma - \sigma_{\mathrm{mfg}}|$", fontsize=11)
    ax.set_title("Pointwise density error", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    fig.suptitle("Calderón spectral-flip — Koch(1)  (full budget)", fontsize=12)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def _fig_eigenvalues(V_h, W_tilde, W_plus, outpath):
    Nq = len(V_h)
    eig_V    = np.sort(np.abs(np.linalg.eigvals(V_h)))[::-1]
    eig_WV   = np.sort(np.abs(np.linalg.eigvals(W_tilde @ V_h)))[::-1]
    eig_WpV  = np.sort(np.abs(np.linalg.eigvals(W_plus  @ V_h)))[::-1]

    cond_WV  = eig_WV.max()  / (eig_WV.min()  + 1e-14)
    cond_WpV = eig_WpV.max() / (eig_WpV.min() + 1e-14)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    idx = np.arange(1, Nq+1)
    ax.semilogy(idx, eig_V,   color="#1f77b4", lw=1.5, label=r"$V_h$")
    ax.semilogy(idx, eig_WV,  color="#ff7f0e", lw=1.5,
                label=rf"$\tilde{{W}}V_h$  cond={cond_WV:.1f}")
    ax.semilogy(idx, eig_WpV, color="#d62728", lw=1.5, ls="--",
                label=rf"$\tilde{{W}}_+V_h$  cond={cond_WpV:.1f}")
    ax.axhline(0.25, color="gray", ls=":", lw=1.0, label="1/4 (Calderón target)")
    ax.set_xlabel("Index (sorted)", fontsize=11)
    ax.set_ylabel(r"$|\lambda_k|$", fontsize=11)
    ax.set_title("Eigenvalue magnitudes — spectral flip preserves cond_eig", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.text(0.03, 0.04,
            f"cond(V_h) = {np.linalg.cond(V_h):.2e}\n"
            f"cond_eig(W̃V) = {cond_WV:.1f}\n"
            f"cond_eig(W̃_+V) = {cond_WpV:.1f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax = axes[1]
    eig_Wsym  = np.linalg.eigvalsh(0.5*(W_tilde + W_tilde.T))
    eig_Wpsym = np.linalg.eigvalsh(0.5*(W_plus  + W_plus.T))
    idx2 = np.arange(1, Nq+1)
    ax.plot(idx2, np.sort(eig_Wsym),  color="#ff7f0e", lw=1.5,
            label=rf"$\tilde{{W}}_{{sym}}$  n_neg={int((eig_Wsym<0).sum())}")
    ax.plot(idx2, np.sort(eig_Wpsym), color="#d62728", lw=1.5, ls="--",
            label=r"$\tilde{W}_+$ (all positive)")
    ax.axhline(0, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Eigenvalue index (sorted)", fontsize=11)
    ax.set_ylabel(r"$\lambda_k$", fontsize=11)
    ax.set_title("Sorted eigenvalues of W̃ vs W̃_+", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, lw=0.3, alpha=0.5)

    fig.suptitle(
        r"Spectral flip: $\tilde{W}_+ = Q|\Lambda|Q^T$ — conditioning preserved"
        "\nKoch(1), N_quad=2304",
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
    print("CALDERÓN SPD: Spectral-flip bilinear form  rᵀ W̃_+ r")
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
    # 2. Step 1 & 2: Spectral flip + validation
    # ------------------------------------------------------------------
    print("\n  === Step 1–2: Spectral flip of W̃ ===")
    W_h, _  = assemble_hypersingular_direct(qdata)
    W_tilde = regularise_hypersingular(W_h, wq)
    W_plus, n_flipped = make_spd_spectral_flip(0.5*(W_tilde + W_tilde.T))

    # Verify SPD
    eigvals_plus = np.linalg.eigvalsh(W_plus)
    print(f"  n_flipped   = {n_flipped} / {Nq}  ({100*n_flipped/Nq:.2f}%)")
    print(f"  λ_min(W̃_+) = {eigvals_plus.min():.6e}  (should be > 0)")
    print(f"  λ_max(W̃_+) = {eigvals_plus.max():.6e}")
    print(f"  W̃_+ is SPD : {eigvals_plus.min() > 0}")

    # Conditioning comparison
    eig_WV   = np.abs(np.linalg.eigvals(W_tilde @ V_h))
    eig_WpV  = np.abs(np.linalg.eigvals(W_plus  @ V_h))
    cond_WV  = float(eig_WV.max()  / (eig_WV.min()  + 1e-14))
    cond_WpV = float(eig_WpV.max() / (eig_WpV.min() + 1e-14))
    cond_V   = float(np.linalg.cond(V_h))

    print(f"\n  Conditioning comparison:")
    print(f"    cond_eig(W̃ V)   = {cond_WV:.1f}  (from validation: 15.2)")
    print(f"    cond_eig(W̃_+ V) = {cond_WpV:.1f}  (after spectral flip)")
    print(f"    cond(V_h)        = {cond_V:.3e}")
    print(f"\n  Eigenvalues of W̃_+ V:")
    print(f"    median|λ| = {np.median(eig_WpV):.4f}  (target: ~0.25)")
    print(f"    min|λ|    = {eig_WpV.min():.4f}")
    print(f"    max|λ|    = {eig_WpV.max():.4f}")

    if cond_WpV > 200:
        print(f"\n  WARNING: cond_eig(W̃_+ V) = {cond_WpV:.1f} >> 15. "
              f"Spectral flip degraded conditioning.")
    else:
        print(f"\n  OK: cond_eig(W̃_+ V) = {cond_WpV:.1f} ≈ cond_eig(W̃V) = {cond_WV:.1f}")

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
    # 4. Step 4: Loss scale calibration
    # ------------------------------------------------------------------
    r0 = -g_mfg   # residual at σ=0
    loss_std_0 = float(np.mean(r0**2))
    loss_bil_0 = float(r0 @ W_plus @ r0)
    spd_scale  = loss_std_0 / max(loss_bil_0, 1e-30)

    print(f"\n  === Step 4: Loss scale calibration (σ=0, r = −g_mfg) ===")
    print(f"    std  loss ||r||²/N        = {loss_std_0:.3e}")
    print(f"    bil  loss rᵀW̃_+ r        = {loss_bil_0:.3e}")
    print(f"    ratio bil/std             = {loss_bil_0/loss_std_0:.3e}")
    print(f"    spd_scale (= std/bil)     = {spd_scale:.3e}")
    print(f"    After scaling: bil_scaled = {loss_bil_0*spd_scale:.3e} ≈ std")

    u_exact = make_u_exact_fn(Yq_T, wq, sigma_mfg)

    # ------------------------------------------------------------------
    # 5. Operator state
    # ------------------------------------------------------------------
    col_pts   = build_collocation_points(panels, m_col_panel=CFG["m_col_base"])
    panel_wts = panel_loss_weights(panels, w_base=CFG["w_base"],
                                   w_corner=CFG["w_corner"], w_ring=CFG["w_ring"])
    g_fn = lambda xy: xy[:, 0]**2 - xy[:, 1]**2
    op, diag = build_operator_state(
        colloc=col_pts, qdata=qdata, enrichment=enrichment, g=g_fn,
        panel_weights=panel_wts, eq_scale_mode="none", eq_scale_fixed=1.0,
        dtype=torch.float64, device="cpu",
    )
    op.f = torch.tensor(g_mfg, dtype=torch.float64)

    V_inv_np = np.linalg.inv(V_h)
    op.V_inv   = torch.tensor(V_inv_np, dtype=torch.float64)
    op.W_plus  = torch.tensor(W_plus,   dtype=torch.float64)
    op.spd_scale = float(spd_scale)

    print(f"    ||V⁻¹V − I||_F/√N = "
          f"{np.linalg.norm(V_inv_np @ V_h - np.eye(Nq))/np.sqrt(Nq):.3e}")

    # ------------------------------------------------------------------
    # 6. Shared initial model state
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
    # 7. Training: 4 cases
    # ------------------------------------------------------------------
    print("\n  === Step 5: Training (4 cases, full budget) ===")
    print(f"  Adam: {CFG['adam_iters']} iters at lr {CFG['adam_lrs']}")
    print(f"  LBFGS: {CFG['lbfgs_iters']} iters, memory {CFG['lbfgs_memory']}")

    cases = []
    cases.append(_train_one("A (BINN, std)",       standard_loss,     init_state, shared, freeze_gamma=True))
    cases.append(_train_one("B (SE-BINN, std)",    standard_loss,     init_state, shared))
    cases.append(_train_one("C (SE-BINN, V⁻¹)",   exact_precond_loss,init_state, shared))
    cases.append(_train_one("D (SE-BINN, W̃_+ bil)",calderon_spd_loss, init_state, shared))

    # ------------------------------------------------------------------
    # 8. Final summary table
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    labels_short = ["A (BINN)", "B (SE-BINN)", "C (V⁻¹)", "D (W̃_+ bil)"]
    hdr = f"  {'Metric':<22}" + "".join(f"  {l:>14}" for l in labels_short)
    print(hdr)
    print("  " + "-"*80)

    for name, key in [
        ("Density rel-diff",  "density_rel_diff"),
        ("Interior rel L2",   "final_rel_L2"),
        ("BIE loss (std)",    "bie_loss_final"),
        ("γ error",          "gamma_err"),
    ]:
        row = f"  {name:<22}" + "".join(f"  {r[key]:>14.4e}" for r in cases)
        print(row)

    row = f"  {'init loss':<22}" + "".join(f"  {r['loss_init']:>14.3e}" for r in cases)
    print(row)
    row = f"  {'LBFGS reason':<22}" + "".join(f"  {r['lbfgs_reason']:>14}" for r in cases)
    print(row)
    row = f"  {'Wall time (s)':<22}" + "".join(f"  {r['wall_time']:>14.1f}" for r in cases)
    print(row)

    print()
    d_B = cases[1]["density_rel_diff"]
    d_C = cases[2]["density_rel_diff"]
    d_D = cases[3]["density_rel_diff"]
    print(f"  B baseline :  d_err = {d_B:.4f}")
    print(f"  C (V⁻¹)   :  d_err = {d_C:.4f}  ({d_B/max(d_C,1e-6):.1f}× over B)")
    print(f"  D (W̃_+)   :  d_err = {d_D:.4f}  ({d_B/max(d_D,1e-6):.1f}× over B)")
    gap_closed = (d_B - d_D) / max(d_B - d_C, 1e-6) * 100
    if d_D < d_B:
        print(f"  D closes {gap_closed:.0f}% of the B→C gap")
    else:
        print(f"  D did NOT improve over B")

    print(f"\n  cond_eig(W̃V)   = {cond_WV:.1f}")
    print(f"  cond_eig(W̃_+V) = {cond_WpV:.1f}")
    print(f"  n_flipped      = {n_flipped}/{Nq}  ({100*n_flipped/Nq:.2f}%)")
    print(f"  spd_scale      = {spd_scale:.3e}")

    # ------------------------------------------------------------------
    # 9. Figures
    # ------------------------------------------------------------------
    _fig_convergence(
        cases, adam_cutoff,
        os.path.join(fig_dir, "calderon_spd_convergence.png"),
    )
    _fig_density(
        cases, sigma_bem, arc, sort_idx,
        os.path.join(fig_dir, "calderon_spd_density.png"),
    )
    _fig_eigenvalues(
        V_h, W_tilde, W_plus,
        os.path.join(fig_dir, "calderon_spd_eigenvalues.png"),
    )

    print(f"\n  All figures saved to {fig_dir}/")
    return cases


if __name__ == "__main__":
    main()
