"""
Experiment: Calderón preconditioner phase 2 — direct Nyström T_h kernel.

Mathematical background
-----------------------
Phase 1 used the Maue identity W_h = -D_h^T diag(w) V_h D_h.
This is a Galerkin discretisation living in a different discrete space than
the Nyström V_h, causing cond(W̃V) ≥ cond(V) (no improvement).

Phase 2 uses the direct collocational kernel:

    T_h[i,j] = -(1/π) Re[ τ_i · τ_j / (z_i - z_j)² ] · w_j   (i ≠ j)
    T_h[i,i] = (1/π)(1/(L_p - s₀) + 1/s₀) - Σ_{j∈panel(i),j≠i} T_h[i,j]

where τ = unit tangent (complex), z = x_1 + ix_2.  This is the same Nyström
framework as V_h, so T̃_h V_h lives in a single consistent discrete space.

Calderón identity: for a smooth closed curve,
    T V ≈ I/4 + compact  → cond(T̃V) ≈ O(1)

On Koch (non-compact K at reentrant corners ω=4π/3):
    cond(T̃V) < cond(V), hopefully with significant improvement.

Parts
-----
1. Operator assembly + diagnostics
2. Eigenvalue spectrum: cond(T̃V) vs cond(V)  (KEY TEST)
3. Quick training comparison:
   Case B:            SE-BINN + standard loss  (baseline)
   Case D_exact:      SE-BINN + exact precond  V_h^{-1}  (Phase 0 reference)
   Case D_calderon:   SE-BINN + Calderón precond  T̃_h    (Phase 2)

Geometry: Koch(1), n_per_edge=12, p_gl=16
Budget:   Adam [500,500] + L-BFGS 3000  (same as Phase 1 for fair comparison)
Figures:  experiments/ex1_Koch/figures/
  phase2_eigenvalues.png    — T̃V vs V eigenvalue comparison
  phase2_nullspace.png      — T_h @ ones and Galerkin symmetry
  phase2_training.png       — density trajectories B / D_exact / D_calderon
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
    seed            = 0,
    n_per_edge      = 12,
    p_gl            = 16,
    m_col_base      = 16,
    w_base          = 1.0,
    w_corner        = 1.0,
    w_ring          = 1.0,
    eq_scale_mode   = "none",
    gamma_true      = np.array([+1.0, -0.5, +1.0, -0.5, +1.0, -0.5]),
    gamma_init      = 0.0,
    hidden_width    = 80,
    n_hidden        = 4,
    adam_iters      = [500, 500],
    adam_lrs        = [1e-3, 1e-4],
    log_every       = 100,
    lbfgs_max_iters = 3000,
    lbfgs_grad_tol  = 1e-10,
    lbfgs_step_tol  = 1e-12,
    lbfgs_memory    = 20,
    lbfgs_log_every = 100,
    lbfgs_alpha0    = 1e-1,
    lbfgs_alpha_fb  = [1e-2, 1e-3],
    lbfgs_armijo_c1 = 1e-4,
    lbfgs_beta      = 0.5,
    lbfgs_max_bt    = 20,
    n_grid_coarse   = 51,
    n_grid_final    = 101,
)


# ===========================================================================
# Preconditioned losses
# ===========================================================================

def standard_loss(model, op):
    from src.training.loss import sebinn_loss
    return sebinn_loss(model, op)


def exact_precond_loss(model, op):
    """L = ||V_h^{-1}(Vσ − g)||²  (exact preconditioner, Phase 0 reference)."""
    sigma_std = model(op.Yq, op.sigma_s_q)
    sigma_c   = model(op.Xc, op.sigma_s_c)
    Vstd = op.A @ sigma_std
    Vsig = Vstd + op.corr * sigma_c
    res  = Vsig - op.f
    prec_res = op.V_inv @ res
    loss = (op.wCol * prec_res**2).sum() / op.wCol_sum
    with torch.no_grad():
        dbg = {
            "mean_abs_res": float(res.detach().abs().mean()),
            "mse_scaled": float((prec_res**2).detach().mean()),
            "loss": float(loss.detach()),
            "gamma": model.gamma_value(),
        }
    return loss, dbg


def calderon_loss(model, op):
    """L = ||T̃_h (Vσ − g)||²  (direct Nyström Calderón preconditioner, Phase 2)."""
    sigma_std = model(op.Yq, op.sigma_s_q)
    sigma_c   = model(op.Xc, op.sigma_s_c)
    Vstd = op.A @ sigma_std
    Vsig = Vstd + op.corr * sigma_c
    res  = Vsig - op.f
    prec_res = op.T_tilde @ res
    loss = (op.wCol * prec_res**2).sum() / op.wCol_sum
    with torch.no_grad():
        dbg = {
            "mean_abs_res": float(res.detach().abs().mean()),
            "mse_scaled": float((prec_res**2).detach().mean()),
            "loss": float(loss.detach()),
            "gamma": model.gamma_value(),
        }
    return loss, dbg


# ===========================================================================
# Helpers
# ===========================================================================

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

def _train_one(label, loss_fn, init_state, shared, verbose=True):
    if verbose:
        print(f"\n{'='*64}")
        print(f"  Case {label} | loss={loss_fn.__name__}")
        print(f"{'='*64}")

    t0 = time.perf_counter()
    op        = shared["op"]
    Yq_T      = shared["Yq_T"]
    wq        = shared["wq"]
    P         = shared["P"]
    sigma_bem = shared["sigma_bem"]
    sigma_s_Yq= shared["sigma_s_Yq"]
    sort_idx  = shared["sort_idx"]
    u_exact   = shared["u_exact"]
    n_gamma   = sigma_s_Yq.shape[1]

    model = SEBINNModel(
        hidden_width=CFG["hidden_width"], n_hidden=CFG["n_hidden"],
        n_gamma=n_gamma, gamma_init=CFG["gamma_init"], dtype=torch.float64,
    )
    model.load_state_dict(copy.deepcopy(init_state))

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  trainable params: {n_tr}")

    Yq_t      = torch.tensor(Yq_T, dtype=torch.float64)
    sigma_s_t = torch.tensor(sigma_s_Yq, dtype=torch.float64)

    traj = []

    def _density_reldiff():
        with torch.no_grad():
            s = model(Yq_t, sigma_s_t).numpy()
        return float(np.linalg.norm(s - sigma_bem)
                     / max(np.linalg.norm(sigma_bem), 1e-14)), s

    def _record(stage, n_iter, loss_val):
        d_err, s = _density_reldiff()
        i_out = reconstruct_interior(
            P=P, Yq=Yq_T, wq=wq, sigma=s,
            n_grid=CFG["n_grid_coarse"], u_exact=u_exact,
        )
        g = model.gamma_value()
        g_arr = np.array(g if isinstance(g, list) else [float(g)])
        traj.append((n_iter, d_err, loss_val, g_arr.copy()))
        if verbose:
            gfmt = "[" + ",".join(f"{v:+.3f}" for v in g_arr) + "]"
            print(f"  [{label}] {stage}: loss={loss_val:.3e} | "
                  f"d_err={d_err:.4f} | iL2={i_out.rel_L2:.3e} | γ={gfmt}")

    # Adam phases
    all_adam_loss = []
    global_it = 0
    for ph_idx, (n_it, lr) in enumerate(zip(CFG["adam_iters"], CFG["adam_lrs"])):
        ph_cfg = AdamConfig(phase_iters=[n_it], phase_lrs=[lr],
                            log_every=CFG["log_every"])
        ph_res = run_adam_phases(model, op, ph_cfg, verbose=verbose,
                                 loss_fn=loss_fn)
        all_adam_loss.extend(ph_res.loss_hist)
        global_it += ph_res.n_iters
        _record(f"Adam-ph{ph_idx+1}", global_it, ph_res.final_loss)

    # L-BFGS
    lbfgs_cfg = _make_lbfgs_cfg()
    lbfgs_res = run_lbfgs(model, op, lbfgs_cfg, verbose=verbose,
                          loss_fn=loss_fn)
    _record("LBFGS", global_it + lbfgs_res.n_iters,
            lbfgs_res.loss_hist[-1] if lbfgs_res.loss_hist else float("nan"))

    with torch.no_grad():
        sigma_final = model(Yq_t, sigma_s_t).numpy()
    final_out = reconstruct_interior(
        P=P, Yq=Yq_T, wq=wq, sigma=sigma_final,
        n_grid=CFG["n_grid_final"], u_exact=u_exact,
    )
    density_rel_diff = float(np.linalg.norm(sigma_final - sigma_bem)
                             / max(np.linalg.norm(sigma_bem), 1e-14))

    gamma_final = np.array(
        model.gamma_value() if isinstance(model.gamma_value(), list)
        else [float(model.gamma_value())]
    )
    gamma_err = float(np.linalg.norm(gamma_final - CFG["gamma_true"])
                      / max(np.linalg.norm(CFG["gamma_true"]), 1e-14))

    if verbose:
        print(f"\n  {label} final:")
        print(f"    Density rel-diff : {density_rel_diff:.4f}")
        print(f"    Interior L2      : {final_out.rel_L2:.3e}")
        print(f"    γ error          : {gamma_err:.4f}")
        print(f"    LBFGS reason     : {lbfgs_res.reason}")
        print(f"    Wall time        : {time.perf_counter()-t0:.1f}s")

    return dict(
        label=label,
        density_rel_diff=density_rel_diff,
        final_rel_L2=final_out.rel_L2,
        gamma_final=gamma_final,
        gamma_err=gamma_err,
        lbfgs_reason=lbfgs_res.reason,
        wall_time=time.perf_counter()-t0,
        loss_hist_adam=all_adam_loss,
        loss_hist_lbfgs=list(lbfgs_res.loss_hist),
        adam_n_iters=global_it,
        sigma_final=sigma_final[sort_idx],
        traj=traj,
        final_out=final_out,
    )


# ===========================================================================
# Figures
# ===========================================================================

def _fig_eigenvalues(V_h, T_tilde, outpath):
    """Eigenvalue magnitudes: T̃V vs V_h alone."""
    eigvals_V  = np.linalg.eigvals(V_h)
    eigvals_TV = np.linalg.eigvals(T_tilde @ V_h)

    mag_V  = np.sort(np.abs(eigvals_V))[::-1]
    mag_TV = np.sort(np.abs(eigvals_TV))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.35)

    ax = axes[0]
    ax.semilogy(mag_V,  color="#1f77b4", lw=1.4, label=r"$V_h$ alone")
    ax.semilogy(mag_TV, color="#d62728", lw=1.4, label=r"$\tilde{T}_h V_h$")
    ax.set_xlabel("Index (sorted by magnitude)", fontsize=11)
    ax.set_ylabel(r"$|\lambda|$", fontsize=11)
    ax.set_title("Sorted eigenvalue magnitudes", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    cond_V  = np.linalg.cond(V_h)
    cond_TV = np.linalg.cond(T_tilde @ V_h)
    ax.text(0.05, 0.07,
            f"cond$(V_h)$ = {cond_V:.2e}\ncond$(\tilde T_h V_h)$ = {cond_TV:.2e}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax = axes[1]
    ax.scatter(eigvals_V.real,  eigvals_V.imag,  s=2, alpha=0.5,
               color="#1f77b4", label=r"$V_h$")
    ax.scatter(eigvals_TV.real, eigvals_TV.imag, s=2, alpha=0.5,
               color="#d62728", label=r"$\tilde{T}_h V_h$")
    ax.set_xlabel(r"Re$(\lambda)$", fontsize=11)
    ax.set_ylabel(r"Im$(\lambda)$", fontsize=11)
    ax.set_title("Eigenvalues in complex plane", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)

    fig.suptitle(
        r"Phase 2: direct $T_h$ kernel — eigenvalue spectrum" + "\n"
        r"$\tilde{T}_h V_h$ vs $V_h$ on Koch(1)",
        fontsize=11,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved phase2_eigenvalues → {outpath}")


def _fig_nullspace(qdata, T_h, wq, outpath):
    """T_h @ ones and Galerkin symmetry check."""
    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc = panel_start[qdata.pan_id] + qdata.s_on_panel
    sort_idx = np.argsort(arc)
    arc_s = arc[sort_idx]

    ones  = np.ones(qdata.n_quad)
    Tones = T_h @ ones

    W_T = wq[:, None] * T_h
    asym = np.linalg.norm(W_T - W_T.T) / (np.linalg.norm(W_T) + 1e-14)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.38)

    ax = axes[0]
    ax.semilogy(arc_s, np.abs(Tones[sort_idx]) + 1e-20,
                color="#1f77b4", lw=1.2)
    rel_null = np.linalg.norm(Tones) / (np.linalg.norm(T_h) * np.linalg.norm(ones) + 1e-14)
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$|(T_h \cdot \mathbf{1})_i|$", fontsize=11)
    ax.set_title(rf"(a) $T_h \cdot \mathbf{{1}} \approx 0$: rel = {rel_null:.2e}", fontsize=12)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.text(0.05, 0.92, "Near-panel quadrature\nerror at Koch corners",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax = axes[1]
    W_T_dense = W_T.copy()
    W_T_asym  = np.abs(W_T_dense - W_T_dense.T)
    # Show a 50×50 subblock for visualisation
    nb = min(50, qdata.n_quad)
    im = ax.imshow(W_T_asym[:nb, :nb], aspect="auto",
                   norm=matplotlib.colors.LogNorm(vmin=1e-20, vmax=W_T_asym.max()+1e-20),
                   cmap="hot_r")
    plt.colorbar(im, ax=ax)
    ax.set_title(rf"(b) $|\mathrm{{diag}}(w)T_h - (\mathrm{{diag}}(w)T_h)^T|$ (block 0:50)"
                 f"\nasym = {asym:.2e}", fontsize=11)
    ax.set_xlabel("node j", fontsize=10)
    ax.set_ylabel("node i", fontsize=10)

    fig.suptitle(r"Phase 2: $T_h$ nullspace and Galerkin symmetry — Koch(1)", fontsize=12)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved phase2_nullspace → {outpath}")


def _fig_training(cases, outpath):
    """Density rel-diff trajectories."""
    COLORS = {
        "B (SE-BINN, std)":        "#1f77b4",
        "D_exact (SE-BINN, V⁻¹)":  "#2ca02c",
        "D_calderon (SE-BINN, T̃)": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    for res in cases:
        lbl = res["label"]
        traj = res["traj"]
        iters = [t[0] for t in traj]
        d_errs = [t[1] for t in traj]
        c = COLORS.get(lbl, "gray")
        ax.semilogy(iters, d_errs, "o-", color=c, lw=1.5, ms=5, label=lbl)

    ax.axvline(x=1000, color="gray", ls=":", lw=1.0, alpha=0.6, label="Adam→LBFGS")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(r"$\|\sigma - \sigma_{\mathrm{BEM}}\| / \|\sigma_{\mathrm{BEM}}\|$",
                  fontsize=11)
    ax.set_title("Phase 2: Training comparison — direct $T_h$ Calderón preconditioner",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    # Print final values
    for res in cases:
        ax.annotate(
            f"{res['density_rel_diff']:.3f}",
            xy=(res["traj"][-1][0], res["traj"][-1][1]),
            fontsize=8, color=COLORS.get(res["label"], "gray"),
        )

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved phase2_training → {outpath}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    fig_dir = os.path.join(_HERE, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Geometry + quadrature
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 2: Direct Nyström T_h Calderón Preconditioner")
    print("="*70)

    geom    = make_koch_geometry(n=1)
    P       = geom.vertices
    panels  = build_uniform_panels(P, n_per_edge=CFG["n_per_edge"])
    label_corner_ring_panels(panels, P)
    qdata   = build_panel_quadrature(panels, p=CFG["p_gl"])
    nmat    = assemble_nystrom_matrix(qdata)

    Yq_T = qdata.Yq.T   # (Nq, 2)
    wq   = qdata.wq      # (Nq,)
    Nq   = qdata.n_quad

    print(f"\n  Geometry: Koch(1), N_panels={qdata.n_panels}, N_quad={Nq}")

    # ------------------------------------------------------------------
    # 2. Assemble T_h (direct kernel)
    # ------------------------------------------------------------------
    print("\n  Assembling T_h (direct Nyström kernel)...")
    t0 = time.perf_counter()
    T_h, _  = assemble_hypersingular_direct(qdata)
    T_tilde = regularise_hypersingular(T_h, wq)
    print(f"    assembly time: {time.perf_counter()-t0:.2f}s")

    # ------------------------------------------------------------------
    # 3. Key diagnostics
    # ------------------------------------------------------------------
    print("\n  === Operator diagnostics ===")

    # Galerkin symmetry
    W_T  = wq[:, None] * T_h
    asym = np.linalg.norm(W_T - W_T.T) / (np.linalg.norm(W_T) + 1e-14)
    print(f"    Galerkin symmetry ||diag(w)T - (diag(w)T)^T|| / ||...||  = {asym:.3e}")

    # Nullspace
    ones  = np.ones(Nq)
    Tones = T_h @ ones
    rel_null = np.linalg.norm(Tones) / (np.linalg.norm(T_h) * np.sqrt(Nq) + 1e-14)
    print(f"    Nullspace ||T_h @ 1|| / (||T_h|| sqrt(Nq))               = {rel_null:.3e}")

    # Condition numbers
    V_h     = nmat.V
    cond_V  = np.linalg.cond(V_h)
    cond_T  = np.linalg.cond(T_tilde)
    TV      = T_tilde @ V_h
    cond_TV = np.linalg.cond(TV)
    ratio   = cond_V / max(cond_TV, 1e-14)

    print(f"\n  === Calderón condition numbers ===")
    print(f"    cond(V_h)   = {cond_V:.3e}")
    print(f"    cond(T̃_h)  = {cond_T:.3e}")
    print(f"    cond(T̃V)   = {cond_TV:.3e}")
    if ratio > 100:
        print(f"    Improvement: {ratio:.0f}×  ← Calderón EXCELLENT")
    elif ratio > 10:
        print(f"    Improvement: {ratio:.1f}×  ← Calderón WORKS")
    elif ratio > 1:
        print(f"    Improvement: {ratio:.2f}×  ← Calderón marginal")
    else:
        print(f"    Ratio: {ratio:.3f}×  ← Calderón NOT effective")

    # Eigenvalue clustering
    eigvals_TV = np.linalg.eigvals(TV)
    mag_TV     = np.abs(eigvals_TV)
    lam_med    = np.median(mag_TV)
    w_avg      = float(wq.mean())
    n_clust    = int(np.sum((mag_TV > 0.5*lam_med) & (mag_TV < 2.0*lam_med)))
    print(f"\n  === T̃V eigenvalue spectrum ===")
    print(f"    w_avg/4     = {w_avg/4:.6f}  (theory: smooth boundary cluster)")
    print(f"    min|λ|      = {mag_TV.min():.4f}")
    print(f"    max|λ|      = {mag_TV.max():.4f}")
    print(f"    median|λ|   = {lam_med:.4f}")
    print(f"    %  in [0.5,2]×median = {n_clust/len(mag_TV):.1%}")

    # ------------------------------------------------------------------
    # 4. Figures: eigenvalues + nullspace
    # ------------------------------------------------------------------
    _fig_eigenvalues(
        V_h, T_tilde,
        os.path.join(fig_dir, "phase2_eigenvalues.png"),
    )
    _fig_nullspace(
        qdata, T_h, wq,
        os.path.join(fig_dir, "phase2_nullspace.png"),
    )

    # ------------------------------------------------------------------
    # 5. Manufactured density + BEM reference solve
    # ------------------------------------------------------------------
    print("\n  === Manufactured density setup ===")
    R_cut = 0.15
    enrichment = SingularEnrichment(geom=geom, use_cutoff=True,
                                    cutoff_radius=R_cut, per_corner_gamma=True)
    n_sing     = enrichment.n_singular
    sigma_s_Yq = enrichment.precompute(Yq_T)   # (Nq, n_singular)

    gamma_true  = CFG["gamma_true"].copy()
    f_smooth    = Yq_T[:, 0]**2 - Yq_T[:, 1]**2
    bem_smooth  = solve_bem(nmat, f_smooth, tol=1e-12, max_iter=300)
    sigma_smooth = bem_smooth.sigma
    sigma_mfg   = sigma_smooth + sigma_s_Yq @ gamma_true
    g_mfg       = nmat.V @ sigma_mfg
    bem_mfg     = solve_bem(nmat, g_mfg, tol=1e-12, max_iter=300)
    sigma_bem   = bem_mfg.sigma

    energy = float(np.linalg.norm(sigma_s_Yq @ gamma_true)**2
                   / max(np.linalg.norm(sigma_mfg)**2, 1e-14))
    print(f"    γ_true          = {list(np.round(gamma_true, 2))}")
    print(f"    Enrichment energy = {energy*100:.2f}%")
    print(f"    BEM rel-res     = {bem_mfg.rel_res:.2e}")
    print(f"    ||σ_BEM - σ_mfg|| = {np.linalg.norm(sigma_bem - sigma_mfg):.4e}")

    u_exact = make_u_exact_fn(Yq_T, wq, sigma_mfg)

    # ------------------------------------------------------------------
    # 6. Operator state for training
    # ------------------------------------------------------------------
    print("\n  === Operator setup ===")
    col_pts = build_collocation_points(panels, m_col_panel=CFG["m_col_base"])
    panel_wts = panel_loss_weights(panels, w_base=CFG["w_base"],
                                   w_corner=CFG["w_corner"], w_ring=CFG["w_ring"])
    g_fn = lambda xy: xy[:, 0]**2 - xy[:, 1]**2
    op, op_diag = build_operator_state(
        colloc=col_pts, qdata=qdata, enrichment=enrichment, g=g_fn,
        panel_weights=panel_wts, eq_scale_mode="none", eq_scale_fixed=1.0,
        dtype=torch.float64, device="cpu",
    )
    assert abs(op_diag["eq_scale"] - 1.0) < 1e-12
    op.f = torch.tensor(g_mfg, dtype=torch.float64)

    # Attach V^{-1} for exact preconditioner reference
    V_inv_np = np.linalg.inv(V_h.astype(np.float64))
    op.V_inv  = torch.tensor(V_inv_np, dtype=torch.float64)
    print(f"    ||V_inv·V − I||_F/√N = {np.linalg.norm(V_inv_np @ V_h - np.eye(Nq))/np.sqrt(Nq):.3e}")

    # Attach T̃_h for Calderón preconditioner
    op.T_tilde = torch.tensor(T_tilde, dtype=torch.float64)
    print(f"    op.V_inv and op.T_tilde attached")

    # Initial model state (shared across cases)
    torch.manual_seed(CFG["seed"])
    init_model = SEBINNModel(
        hidden_width=CFG["hidden_width"], n_hidden=CFG["n_hidden"],
        n_gamma=n_sing, gamma_init=CFG["gamma_init"], dtype=torch.float64,
    )
    init_state = copy.deepcopy(init_model.state_dict())

    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc         = panel_start[qdata.pan_id] + qdata.s_on_panel
    sort_idx    = np.argsort(arc)

    shared = dict(
        op=op, Yq_T=Yq_T, wq=wq, P=P,
        sigma_bem=sigma_bem, sigma_s_Yq=sigma_s_Yq,
        sort_idx=sort_idx, u_exact=u_exact,
    )

    # ------------------------------------------------------------------
    # 7. Training comparison
    # ------------------------------------------------------------------
    print("\n  === Training comparison (B / D_exact / D_calderon) ===")
    cases = []

    # Case B: SE-BINN + standard loss
    cases.append(_train_one(
        "B (SE-BINN, std)", standard_loss, init_state, shared,
    ))

    # Case D_exact: SE-BINN + exact V^{-1} preconditioner
    cases.append(_train_one(
        "D_exact (SE-BINN, V⁻¹)", exact_precond_loss, init_state, shared,
    ))

    # Case D_calderon: SE-BINN + direct T̃_h preconditioner
    cases.append(_train_one(
        "D_calderon (SE-BINN, T̃)", calderon_loss, init_state, shared,
    ))

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 2 SUMMARY")
    print("="*70)
    print(f"  {'Case':<30}  {'d_err':>8}  {'iL2':>10}  {'γ_err':>8}  {'time':>7}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*7}")
    for r in cases:
        print(f"  {r['label']:<30}  {r['density_rel_diff']:>8.4f}  "
              f"{r['final_rel_L2']:>10.3e}  {r['gamma_err']:>8.4f}  "
              f"{r['wall_time']:>6.1f}s")

    d_B  = cases[0]["density_rel_diff"]
    d_TC = cases[2]["density_rel_diff"]
    if d_TC < d_B:
        impr = d_B / max(d_TC, 1e-6)
        print(f"\n  T̃ preconditioner improvement: {impr:.2f}×  (d_err: {d_B:.4f} → {d_TC:.4f})")
    else:
        print(f"\n  T̃ preconditioner: NO improvement (d_err: {d_B:.4f} → {d_TC:.4f})")

    # ------------------------------------------------------------------
    # 9. Training figure
    # ------------------------------------------------------------------
    _fig_training(cases, os.path.join(fig_dir, "phase2_training.png"))

    print(f"\n  All figures saved to {fig_dir}/")
    return cases


if __name__ == "__main__":
    main()
