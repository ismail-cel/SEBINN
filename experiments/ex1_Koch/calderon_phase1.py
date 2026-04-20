"""
Experiment: Calderón preconditioner phase 1 — validation and training test.

Mathematical background
-----------------------
The hypersingular operator W for the 2D Laplace equation admits the Maue
regularisation  W = -D_s V D_s, giving the discrete formula

    W_h = -D_h^T  diag(wq)  V_h  D_h      (symmetric Galerkin form)

where D_h is the block-diagonal tangential derivative matrix and V_h is
the Nyström matrix.  The Calderón identity on closed curves states

    -W V = I/4 - K^2 ≈ I/4 + compact,

predicting that eigenvalues of W̃ V_h cluster near a constant.

Parts
-----
1. Assembly validation: D_h accuracy, W_h symmetry/nullspace
2. Eigenvalue spectrum: cond(W̃V) vs cond(V) — the KEY diagnostic
3. Quick training comparison:
   Case B:            SE-BINN + standard loss  (baseline)
   Case D_exact:      SE-BINN + exact precond  (V_h^{-1}, from Phase 0)
   Case D_calderon:   SE-BINN + Calderón precond  (W̃_h)

Key finding (predicted from tests)
------------------------------------
cond(W̃V) ≈ cond(V) ≈ 1.3e4 on Koch snowflake.
The Calderón identity fails because the double-layer operator K is NOT
compact at reentrant corners (ω = 4π/3).  Therefore D_calderon ≈ B.

This contrasts with the exact preconditioner (Phase 0), which achieved
density rel-diff = 3.4% vs 47% for the standard loss.

Geometry: Koch(1), n_per_edge=12, p_gl=16
Figures:  experiments/ex1_Koch/figures/
  calderon_eigenvalues.png      — W̃V vs V eigenvalue comparison
  calderon_validation.png       — D_h accuracy + W_h nullspace
  calderon_vs_exact.png         — training density trajectories
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
from src.quadrature.tangential_derivative import (
    build_tangential_derivative_matrix,
)
from src.quadrature.hypersingular import (
    assemble_hypersingular_matrix,
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
    # Short training budget for quick trend comparison
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
    """L = ||V_h^{-1}(Vσ − g)||²  (Phase 0 exact preconditioner)."""
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
            "mse_unscaled": float((prec_res**2).detach().mean()),
            "loss": float(loss.detach()),
            "gamma": model.gamma_value(),
        }
    return loss, dbg


def calderon_loss(model, op):
    """L = ||W̃_h (Vσ − g)||²  (Calderón preconditioner via Maue)."""
    sigma_std = model(op.Yq, op.sigma_s_q)
    sigma_c   = model(op.Xc, op.sigma_s_c)
    Vstd = op.A @ sigma_std
    Vsig = Vstd + op.corr * sigma_c
    res  = Vsig - op.f
    prec_res = op.W_tilde @ res
    loss = (op.wCol * prec_res**2).sum() / op.wCol_sum
    with torch.no_grad():
        dbg = {
            "mean_abs_res": float(res.detach().abs().mean()),
            "mse_scaled": float((prec_res**2).detach().mean()),
            "mse_unscaled": float((prec_res**2).detach().mean()),
            "loss": float(loss.detach()),
            "gamma": model.gamma_value(),
        }
    return loss, dbg


# ===========================================================================
# Helpers
# ===========================================================================

def _boundary_arclength(qdata, n_per_edge):
    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc = panel_start[qdata.pan_id] + qdata.s_on_panel
    total = float(qdata.L_panel.sum())
    Nv = qdata.n_panels // n_per_edge
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

def _fig_eigenvalues(V_h, W_tilde, outpath):
    """Eigenvalue magnitudes of W̃V vs V_h alone."""
    eigvals_V  = np.linalg.eigvals(V_h)
    eigvals_WV = np.linalg.eigvals(W_tilde @ V_h)

    mag_V  = np.sort(np.abs(eigvals_V))[::-1]
    mag_WV = np.sort(np.abs(eigvals_WV))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.35)

    ax = axes[0]
    ax.semilogy(mag_V,  color="#1f77b4", lw=1.4, label=r"$V_h$ alone")
    ax.semilogy(mag_WV, color="#d62728", lw=1.4, label=r"$\tilde{W}_h V_h$")
    ax.set_xlabel("Index (sorted by magnitude)", fontsize=11)
    ax.set_ylabel(r"$|\lambda|$", fontsize=11)
    ax.set_title("Sorted eigenvalue magnitudes", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.text(0.05, 0.07, f"cond$(V_h)$ = {np.linalg.cond(V_h):.2e}\n"
            f"cond$(\tilde W V_h)$ = {np.linalg.cond(W_tilde @ V_h):.2e}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax = axes[1]
    ax.scatter(eigvals_V.real,  eigvals_V.imag,  s=2, alpha=0.5,
               color="#1f77b4", label=r"$V_h$")
    ax.scatter(eigvals_WV.real, eigvals_WV.imag, s=2, alpha=0.5,
               color="#d62728", label=r"$\tilde{W}_h V_h$")
    ax.set_xlabel(r"Re$(\lambda)$", fontsize=11)
    ax.set_ylabel(r"Im$(\lambda)$", fontsize=11)
    ax.set_title("Eigenvalues in complex plane", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)

    fig.suptitle(
        r"Eigenvalue spectrum — $\tilde{W}_h V_h$ vs $V_h$  (Koch(1))" + "\n"
        r"Calderón identity $-WV \approx I/4$ fails on polygonal domains",
        fontsize=11,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved calderon_eigenvalues → {outpath}")


def _fig_validation(qdata, D_h, W_h, arc, outpath):
    """D_h derivative accuracy + W_h nullspace."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.38)

    # Subplot (a): D_h accuracy on sin(2πs/L)
    L_tot = float(qdata.L_panel.sum())
    freq  = 2.0 * np.pi / L_tot
    phi       = np.sin(freq * arc)
    phi_prime = freq * np.cos(freq * arc)
    phi_Dh    = D_h @ phi
    sort_idx  = np.argsort(arc)
    arc_s = arc[sort_idx]

    ax = axes[0]
    ax.plot(arc_s, phi_prime[sort_idx], color="black", lw=1.6, label=r"$\varphi'$ exact")
    ax.plot(arc_s, phi_Dh[sort_idx],   color="#d62728", lw=1.2, ls="--",
            label=r"$D_h\varphi$ (Lagrange)")
    rel_err = np.linalg.norm(phi_Dh - phi_prime) / max(np.linalg.norm(phi_prime), 1e-14)
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$\varphi'(s)$", fontsize=11)
    ax.set_title(rf"(a) $D_h$ accuracy: rel.err = {rel_err:.2e}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.5)

    # Subplot (b): W_h @ ones ≈ 0
    Nq   = qdata.n_quad
    ones = np.ones(Nq)
    Wones = W_h @ ones
    rel_null = np.linalg.norm(Wones) / max(np.linalg.norm(W_h) * np.sqrt(Nq), 1e-14)

    ax = axes[1]
    ax.semilogy(arc_s, np.abs(Wones[sort_idx]) + 1e-20,
                color="#1f77b4", lw=1.2)
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$|(W_h \cdot \mathbf{1})_i|$", fontsize=11)
    ax.set_title(rf"(b) $W_h \cdot \mathbf{{1}} \approx 0$: rel = {rel_null:.2e}", fontsize=12)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(r"Hypersingular $W_h$ validation — Koch(1)", fontsize=12)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved calderon_validation → {outpath}")


def _fig_training(cases, outpath):
    """Density rel-diff trajectories for B / D_exact / D_calderon."""
    COLORS = {
        "B (SE-BINN, std)":       "#1f77b4",
        "D_exact (SE-BINN, V⁻¹)": "#2ca02c",
        "D_calderon (SE-BINN, W̃)": "#d62728",
    }
    LS = {
        "B (SE-BINN, std)":       "-",
        "D_exact (SE-BINN, V⁻¹)": "--",
        "D_calderon (SE-BINN, W̃)": "-.",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for c in cases:
        col = COLORS.get(c["label"], "gray")
        ls  = LS.get(c["label"], "-")
        traj = c["traj"]
        if not traj:
            continue
        iters  = np.array([t[0] for t in traj])
        d_errs = np.array([t[1] for t in traj])
        ax.semilogy(iters, d_errs, color=col, ls=ls, lw=2.0,
                    marker="o", markersize=5,
                    label=f"{c['label']}  (d_err={c['density_rel_diff']:.4f})")

    n_adam = cases[0]["adam_n_iters"]
    ax.axvline(n_adam, color="gray", lw=0.9, ls=":", alpha=0.6,
               label=f"Adam→L-BFGS ({n_adam})")
    ax.set_xlabel("Iteration (end of stage)", fontsize=12)
    ax.set_ylabel(r"$\|\sigma_\theta - \sigma_\mathrm{BEM}\|/\|\sigma_\mathrm{BEM}\|$",
                  fontsize=11)
    ax.set_title(
        r"Calderón vs exact preconditioner — Koch(1) manufactured density" + "\n"
        r"$D_\mathrm{exact}$: $V_h^{-1}$ precond (Phase 0).  "
        r"$D_\mathrm{calderon}$: $\tilde{W}_h$ precond (Maue).",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved calderon_vs_exact → {outpath}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("=" * 72)
    print("  Koch(1) — Calderón preconditioner phase 1")
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
    panels = build_uniform_panels(P, n_per_edge=CFG["n_per_edge"])
    label_corner_ring_panels(panels, P)
    qdata  = build_panel_quadrature(panels, p=CFG["p_gl"])
    Yq_T   = qdata.Yq.T
    wq     = qdata.wq
    Nq     = qdata.n_quad

    arc, vertex_arcs, total_arc = _boundary_arclength(qdata, CFG["n_per_edge"])
    sort_idx = np.argsort(arc)

    sing_verts  = P[geom.singular_corner_indices]
    d_min = float(min(np.linalg.norm(sing_verts[i]-sing_verts[j])
                      for i in range(len(sing_verts))
                      for j in range(i+1, len(sing_verts))))
    R_cut = 0.5 * d_min
    print(f"  Nq={Nq} | arc={total_arc:.4f} | d_min={d_min:.6f} | R={R_cut:.6f}")

    # ------------------------------------------------------------------
    # 2. Nyström matrix and hypersingular assembly
    # ------------------------------------------------------------------
    print("\n--- Hypersingular assembly ---")
    nmat = assemble_nystrom_matrix(qdata)

    t0 = time.perf_counter()
    D_h     = build_tangential_derivative_matrix(qdata)
    W_h     = assemble_hypersingular_matrix(qdata, nmat)
    W_tilde = regularise_hypersingular(W_h, qdata)

    cond_V  = np.linalg.cond(nmat.V)
    cond_WT = np.linalg.cond(W_tilde)
    cond_WV = np.linalg.cond(W_tilde @ nmat.V)

    asym_Wh = np.linalg.norm(W_h - W_h.T) / max(np.linalg.norm(W_h), 1e-14)
    Wones   = W_h @ np.ones(Nq)
    null_rel = np.linalg.norm(Wones) / max(np.linalg.norm(W_h)*np.sqrt(Nq), 1e-14)

    print(f"  Assembly time: {time.perf_counter()-t0:.2f}s")
    print(f"  W_h asymmetry ||W-W^T||/||W|| = {asym_Wh:.3e}  (should ≈ 0)")
    print(f"  W_h nullspace ||W·1||/||W|| = {null_rel:.3e}  (should ≈ 0)")
    print(f"  cond(V_h)       = {cond_V:.3e}")
    print(f"  cond(W_tilde)   = {cond_WT:.3e}")
    print(f"  cond(W̃ V_h)    = {cond_WV:.3e}")

    if cond_WV < cond_V / 10:
        print(f"  → Calderón WORKS: {cond_V/cond_WV:.0f}× improvement")
    elif cond_WV < cond_V:
        print(f"  → Calderón MARGINAL: {cond_V/cond_WV:.2f}× improvement")
    else:
        print(f"  → Calderón NOT EFFECTIVE on Koch polygon")
        print(f"     Root cause: -WV = I/4 - K^2; K not compact at reentrant corners (ω=4π/3)")

    # D_h accuracy check
    L_tot     = float(qdata.L_panel.sum())
    freq      = 2.0 * np.pi / L_tot
    phi       = np.sin(freq * arc)
    phi_prime = freq * np.cos(freq * arc)
    phi_Dh    = D_h @ phi
    dh_err    = np.linalg.norm(phi_Dh - phi_prime) / max(np.linalg.norm(phi_prime), 1e-14)
    print(f"  D_h derivative rel.err on sin(2πs/L): {dh_err:.3e}  (should < 1e-10)")

    # ------------------------------------------------------------------
    # 3. Exact preconditioner V_inv
    # ------------------------------------------------------------------
    print("\n--- Exact preconditioner ---")
    t0 = time.perf_counter()
    V_inv_np = la.inv(nmat.V)
    print(f"  V_inv computed in {time.perf_counter()-t0:.2f}s")
    print(f"  ‖V_inv·V − I‖_F/‖I‖_F = "
          f"{np.linalg.norm(V_inv_np @ nmat.V - np.eye(Nq))/np.sqrt(Nq):.3e}")

    # ------------------------------------------------------------------
    # 4. Manufactured density
    # ------------------------------------------------------------------
    print("\n--- Manufactured density ---")
    enrichment = SingularEnrichment(geom=geom, use_cutoff=True,
                                    cutoff_radius=R_cut, per_corner_gamma=True)
    n_sing     = enrichment.n_singular
    sigma_s_Yq = enrichment.precompute(Yq_T)

    f_smooth    = Yq_T[:,0]**2 - Yq_T[:,1]**2
    bem_smooth  = solve_bem(nmat, f_smooth, tol=1e-12, max_iter=300)
    sigma_smooth= bem_smooth.sigma

    gamma_true  = CFG["gamma_true"].copy()
    sigma_mfg   = sigma_smooth + sigma_s_Yq @ gamma_true
    g_mfg       = nmat.V @ sigma_mfg

    bem_mfg     = solve_bem(nmat, g_mfg, tol=1e-12, max_iter=300)
    sigma_bem   = bem_mfg.sigma

    energy = float(np.linalg.norm(sigma_s_Yq @ gamma_true)**2
                   / max(np.linalg.norm(sigma_mfg)**2, 1e-14))
    print(f"  γ_true = {list(np.round(gamma_true, 2))}")
    print(f"  Enrichment energy = {energy*100:.2f}%")
    print(f"  GMRES flag = {bem_mfg.flag} | rel_res = {bem_mfg.rel_res:.3e}")

    u_exact = make_u_exact_fn(Yq_T, wq, sigma_mfg)

    # ------------------------------------------------------------------
    # 5. Operator state
    # ------------------------------------------------------------------
    print("\n--- Operator setup ---")
    colloc  = build_collocation_points(panels, m_col_panel=CFG["m_col_base"])
    w_panel = panel_loss_weights(panels, w_base=CFG["w_base"],
                                 w_corner=CFG["w_corner"], w_ring=CFG["w_ring"])
    g_dummy = lambda xy: xy[:,0]**2 - xy[:,1]**2
    op, op_diag = build_operator_state(
        colloc=colloc, qdata=qdata, enrichment=enrichment, g=g_dummy,
        panel_weights=w_panel, eq_scale_mode="none", eq_scale_fixed=1.0,
        dtype=torch.float64, device="cpu",
    )
    assert abs(op_diag["eq_scale"] - 1.0) < 1e-12
    op.f = torch.tensor(g_mfg, dtype=torch.float64)

    # Attach preconditioners
    op.V_inv  = torch.tensor(V_inv_np,               dtype=torch.float64)
    op.W_tilde= torch.tensor(W_tilde,                dtype=torch.float64)

    print(f"  Nb={colloc.n_colloc} | eq_scale=1.0")
    print(f"  op.V_inv and op.W_tilde attached")

    # ------------------------------------------------------------------
    # 6. Shared initial model
    # ------------------------------------------------------------------
    torch.manual_seed(CFG["seed"])
    init_model = SEBINNModel(
        hidden_width=CFG["hidden_width"], n_hidden=CFG["n_hidden"],
        n_gamma=n_sing, gamma_init=CFG["gamma_init"], dtype=torch.float64,
    )
    init_state = copy.deepcopy(init_model.state_dict())
    shared = dict(op=op, Yq_T=Yq_T, wq=wq, P=P,
                  sigma_bem=sigma_bem, sigma_s_Yq=sigma_s_Yq,
                  sort_idx=sort_idx, u_exact=u_exact)

    # ------------------------------------------------------------------
    # 7. Training: three cases
    # ------------------------------------------------------------------
    print("\n--- Training (short budget) ---")
    res_b  = _train_one("B (SE-BINN, std)",       standard_loss,  init_state, shared)
    res_de = _train_one("D_exact (SE-BINN, V⁻¹)", exact_precond_loss, init_state, shared)
    res_dc = _train_one("D_calderon (SE-BINN, W̃)", calderon_loss, init_state, shared)

    cases = [res_b, res_de, res_dc]

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("  RESULTS — Calderón phase 1")
    print("=" * 80)
    print(f"  cond(V)    = {cond_V:.3e}")
    print(f"  cond(W̃V)  = {cond_WV:.3e}  ({'improves' if cond_WV < cond_V else 'does NOT improve'} conditioning)")
    print()

    col_labels = [c["label"] for c in cases]
    W = 28
    hdr = f"  {'Metric':<24} | " + " | ".join(f"{l:>{W}}" for l in col_labels)
    sep = "  " + "-" * (len(hdr)-2)
    print(hdr); print(sep)
    for name, key in [("Density rel-diff", "density_rel_diff"),
                      ("Interior L2", "final_rel_L2"),
                      ("γ error", "gamma_err")]:
        vals = [f"{c[key]:.3e}" for c in cases]
        print(f"  {name:<24} | " + " | ".join(f"{v:>{W}}" for v in vals))
    print(sep)
    wts = [f"{c['wall_time']:.1f}s" for c in cases]
    print(f"  {'Wall time':<24} | " + " | ".join(f"{v:>{W}}" for v in wts))
    print(sep)

    # Interpretation
    de_b  = cases[0]["density_rel_diff"]
    de_de = cases[1]["density_rel_diff"]
    de_dc = cases[2]["density_rel_diff"]
    print()
    print(f"  B (std)      : d_err = {de_b:.4f}")
    print(f"  D_exact (V⁻¹): d_err = {de_de:.4f}  ({de_b/de_de:.1f}× better than B)")
    print(f"  D_calderon   : d_err = {de_dc:.4f}  ({de_b/de_dc:.2f}× better than B)")
    if de_b / de_dc < 2.0:
        print()
        print("  FINDING: Calderón preconditioner (Maue W̃) does NOT improve training.")
        print("  Root cause: cond(W̃V) ≈ cond(V). Incompatible Galerkin/Nystrom discretizations")
        print("  + K^2 non-compact at Koch corners make -WV ≠ I/4.")
        print()
        print("  Recommendation: use V_h^{-1} (exact preconditioner) if affordable,")
        print("  or seek a sparse approximation to V_h^{-1} (e.g., H-matrix compression).")

    # ------------------------------------------------------------------
    # 9. Figures
    # ------------------------------------------------------------------
    print("\n--- Figures ---")
    _fig_eigenvalues(
        nmat.V, W_tilde,
        os.path.join(figures_dir, "calderon_eigenvalues.png"),
    )
    _fig_validation(
        qdata, D_h, W_h, arc,
        os.path.join(figures_dir, "calderon_validation.png"),
    )
    _fig_training(
        cases,
        os.path.join(figures_dir, "calderon_vs_exact.png"),
    )

    print(f"\n  Total wall time: {time.perf_counter()-t_global:.1f}s")
    print("  Done.")


if __name__ == "__main__":
    main()
