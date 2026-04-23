"""
Experiment: Calderón preconditioner phase 3 — spectral scaling of T̃_h.

Root cause of Phase 2 training failure
---------------------------------------
Phase 2 showed cond_eig(T̃V) = 13.6 — the Calderón identity works at the
discrete level.  But the preconditioned loss ||T̃ r||² failed to improve
training because T̃ has entries O(1/h²) from the hypersingular kernel:
    panel length h ≈ 3.12/144 ≈ 0.022  →  T entries ~ 1/(πh) ~ 14
    diagonal self-correction ~ 2/(πh) ~ 29
    ||T̃||_op ~ O(300)

So the initial Calderón loss is ~10⁵× larger than the standard loss,
overwhelming Adam's learning rate.  The gradient DIRECTIONS are correct
(eigenvalue ratio 13.6) but the MAGNITUDE is wrong.

Fix: spectral normalisation
---------------------------
Divide T̃ by its spectral radius ρ(T̃):

    T̃_s = T̃ / ρ(T̃)

Properties:
  - ρ(T̃_s) = 1   (unit spectral radius)
  - cond_eig(T̃_s V) = cond_eig(T̃ V)   (unchanged, ~13.6)
  - Initial loss scale ≈ initial standard loss

Parts
-----
Short budget (Adam [500,500] + LBFGS 3000):
  Case B:                  SE-BINN + standard loss
  Case D_exact:            SE-BINN + exact V_h^{-1}
  Case D_calderon_raw:     SE-BINN + T̃ unscaled   (Phase 2 reference)
  Case D_calderon_scaled:  SE-BINN + T̃_s scaled

If D_calderon_scaled d_err < B d_err × 0.5 → run full budget:
  Adam [1000,1000,1000] at lr [1e-3, 3e-4, 1e-4] + LBFGS 15000

Figures:
  calderon_scaled_convergence.png  — density trajectories all cases
  calderon_scaled_comparison.png   — density vs arc-length at final iter
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
    # Short budget
    short_adam_iters = [500, 500],
    short_adam_lrs   = [1e-3, 1e-4],
    short_lbfgs_iters = 3000,
    # Full budget
    full_adam_iters  = [1000, 1000, 1000],
    full_adam_lrs    = [1e-3, 3e-4, 1e-4],
    full_lbfgs_iters = 15000,
    log_every        = 100,
    lbfgs_grad_tol   = 1e-10,
    lbfgs_step_tol   = 1e-12,
    lbfgs_memory     = 20,
    lbfgs_full_memory= 30,
    lbfgs_log_every  = 200,
    lbfgs_alpha0     = 1e-1,
    lbfgs_alpha_fb   = [1e-2, 1e-3],
    lbfgs_armijo_c1  = 1e-4,
    lbfgs_beta       = 0.5,
    lbfgs_max_bt     = 20,
    n_grid_coarse    = 51,
    n_grid_final     = 101,
)

# Phase 2 unscaled result (for comparison table)
_PHASE2_UNSCALED = dict(
    label="D_calderon_raw (T̃, unscaled)",
    density_rel_diff=0.5354,
    final_rel_L2=2.491e-02,
    gamma_err=0.3086,
    wall_time=75.1,
    traj=[(4000, 0.5354, 6.959e-01, None)],
    note="Phase 2 result (from log)",
)


# ===========================================================================
# Loss functions
# ===========================================================================

def standard_loss(model, op):
    from src.training.loss import sebinn_loss
    return sebinn_loss(model, op)


def exact_precond_loss(model, op):
    """L = ||V_h^{-1}(Vσ − g)||²"""
    sigma_std = model(op.Yq, op.sigma_s_q)
    sigma_c   = model(op.Xc, op.sigma_s_c)
    res  = op.A @ sigma_std + op.corr * sigma_c - op.f
    prec_res = op.V_inv @ res
    loss = (op.wCol * prec_res**2).sum() / op.wCol_sum
    with torch.no_grad():
        dbg = {"loss": float(loss), "mean_abs_res": float(res.abs().mean()),
               "gamma": model.gamma_value()}
    return loss, dbg


def calderon_scaled_loss(model, op):
    """L = ||T̃_s (Vσ − g)||²  (spectrally normalised Calderón)."""
    sigma_std = model(op.Yq, op.sigma_s_q)
    sigma_c   = model(op.Xc, op.sigma_s_c)
    res  = op.A @ sigma_std + op.corr * sigma_c - op.f
    prec_res = op.T_tilde_scaled @ res
    loss = (op.wCol * prec_res**2).sum() / op.wCol_sum
    with torch.no_grad():
        dbg = {"loss": float(loss), "mean_abs_res": float(res.abs().mean()),
               "gamma": model.gamma_value()}
    return loss, dbg


# ===========================================================================
# Helpers
# ===========================================================================

def make_u_exact_fn(Yq_T, wq, sigma_mfg):
    sigma_wq = sigma_mfg * wq
    def _u(xy):
        return _log_kernel_matrix(xy, Yq_T) @ sigma_wq
    return _u


def _make_lbfgs_cfg(max_iters, memory):
    return LBFGSConfig(
        max_iters      = max_iters,
        grad_tol       = CFG["lbfgs_grad_tol"],
        step_tol       = CFG["lbfgs_step_tol"],
        memory         = memory,
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

def _train_one(label, loss_fn, init_state, shared,
               adam_iters, adam_lrs, lbfgs_max_iters, lbfgs_memory,
               verbose=True):
    if verbose:
        print(f"\n{'='*64}")
        print(f"  Case {label}")
        print(f"{'='*64}")

    t0        = time.perf_counter()
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

    Yq_t      = torch.tensor(Yq_T, dtype=torch.float64)
    sigma_s_t = torch.tensor(sigma_s_Yq, dtype=torch.float64)
    traj      = []

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
        traj.append((n_iter, d_err, loss_val, g_arr.copy()))
        if verbose:
            gfmt = "[" + ",".join(f"{v:+.3f}" for v in g_arr) + "]"
            print(f"  [{label}] {stage}: loss={loss_val:.3e} | "
                  f"d_err={d_err:.4f} | iL2={i_out.rel_L2:.3e} | γ={gfmt}")

    # Record initial state
    with torch.no_grad():
        loss_init, _ = loss_fn(model, op)
    loss_init = float(loss_init)
    _record("init", 0, loss_init)

    # Adam phases
    all_adam_loss = []
    global_it = 0
    for ph_idx, (n_it, lr) in enumerate(zip(adam_iters, adam_lrs)):
        ph_cfg = AdamConfig(phase_iters=[n_it], phase_lrs=[lr],
                            log_every=CFG["log_every"])
        ph_res = run_adam_phases(model, op, ph_cfg, verbose=verbose,
                                 loss_fn=loss_fn)
        all_adam_loss.extend(ph_res.loss_hist)
        global_it += ph_res.n_iters
        _record(f"Adam-ph{ph_idx+1}", global_it, ph_res.final_loss)

    # L-BFGS
    lbfgs_cfg = _make_lbfgs_cfg(lbfgs_max_iters, lbfgs_memory)
    lbfgs_res = run_lbfgs(model, op, lbfgs_cfg, verbose=verbose,
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
        print(f"    LBFGS reason     : {lbfgs_res.reason}")
        print(f"    Wall time        : {time.perf_counter()-t0:.1f}s")

    return dict(
        label=label, density_rel_diff=d_err_final,
        final_rel_L2=final_out.rel_L2,
        gamma_final=gamma_final, gamma_err=gamma_err,
        lbfgs_reason=lbfgs_res.reason,
        wall_time=time.perf_counter()-t0,
        loss_init=loss_init,
        loss_hist_adam=all_adam_loss,
        loss_hist_lbfgs=list(lbfgs_res.loss_hist),
        adam_n_iters=global_it,
        sigma_final=sigma_final[sort_idx],
        traj=traj, final_out=final_out,
    )


def _run_short(label, loss_fn, init_state, shared, verbose=True):
    return _train_one(
        label, loss_fn, init_state, shared,
        adam_iters=CFG["short_adam_iters"],
        adam_lrs=CFG["short_adam_lrs"],
        lbfgs_max_iters=CFG["short_lbfgs_iters"],
        lbfgs_memory=CFG["lbfgs_memory"],
        verbose=verbose,
    )


def _run_full(label, loss_fn, init_state, shared, verbose=True):
    return _train_one(
        label, loss_fn, init_state, shared,
        adam_iters=CFG["full_adam_iters"],
        adam_lrs=CFG["full_adam_lrs"],
        lbfgs_max_iters=CFG["full_lbfgs_iters"],
        lbfgs_memory=CFG["lbfgs_full_memory"],
        verbose=verbose,
    )


# ===========================================================================
# Figures
# ===========================================================================

COLORS = {
    "B (SE-BINN, std)":                "#1f77b4",
    "D_exact (SE-BINN, V⁻¹)":          "#2ca02c",
    "D_calderon_raw (T̃, unscaled)":    "#ff7f0e",
    "D_calderon_scaled (T̃_s)":         "#d62728",
    # Full-budget variants
    "B full":                           "#aec7e8",
    "D_exact full":                     "#98df8a",
    "D_calderon_scaled full":           "#ff9896",
}


def _fig_convergence(cases, title, outpath, adam_cutoff=1000):
    fig, ax = plt.subplots(figsize=(10, 5))
    for res in cases:
        lbl  = res["label"]
        traj = res.get("traj", [])
        if not traj:
            continue
        iters  = [t[0] for t in traj]
        d_errs = [t[1] for t in traj]
        c = COLORS.get(lbl, "gray")
        ls = "--" if "full" in lbl else "-"
        ax.semilogy(iters, d_errs, "o" + ls, color=c, lw=1.5, ms=5, label=lbl)
        ax.annotate(f"{d_errs[-1]:.3f}",
                    xy=(iters[-1], d_errs[-1]), xytext=(5, 0),
                    textcoords="offset points", fontsize=8, color=c)

    ax.axvline(x=adam_cutoff, color="gray", ls=":", lw=0.8, alpha=0.5,
               label="Adam→LBFGS")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(r"$\|\sigma - \sigma_{\mathrm{BEM}}\| / \|\sigma_{\mathrm{BEM}}\|$",
                  fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def _fig_density_comparison(cases, sigma_bem, arc, sigma_s_Yq, gamma_true, outpath):
    sort_idx = np.argsort(arc)
    arc_s = arc[sort_idx]
    sigma_mfg_s = (sigma_bem + sigma_s_Yq @ gamma_true)[sort_idx]  # approximate

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(arc_s, sigma_mfg_s, "k-", lw=2.0, label=r"$\sigma_{\mathrm{mfg}}$", zorder=5)

    for res in cases:
        lbl  = res["label"]
        sig  = res.get("sigma_final")
        if sig is None:
            continue
        c = COLORS.get(lbl, "gray")
        ls = "--" if "full" in lbl else "-"
        ax.plot(arc_s, sig, ls, color=c, lw=1.2, alpha=0.8,
                label=f"{lbl}  (d={res['density_rel_diff']:.3f})")

    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$\sigma(s)$", fontsize=11)
    ax.set_title("Phase 3: Final density comparison — Koch(1)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, lw=0.3, alpha=0.4)
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

    # ------------------------------------------------------------------
    # 1. Geometry + quadrature
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 3: Spectrally Scaled Calderón Preconditioner")
    print("="*70)

    geom   = make_koch_geometry(n=1)
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=CFG["n_per_edge"])
    label_corner_ring_panels(panels, P)
    qdata  = build_panel_quadrature(panels, p=CFG["p_gl"])
    nmat   = assemble_nystrom_matrix(qdata)
    V_h    = nmat.V

    Yq_T = qdata.Yq.T
    wq   = qdata.wq
    Nq   = qdata.n_quad
    print(f"  Koch(1): N_panels={qdata.n_panels}, N_quad={Nq}")

    # ------------------------------------------------------------------
    # 2. Assemble T_h and compute spectral scaling
    # ------------------------------------------------------------------
    print("\n  Assembling T_h...")
    T_h, _  = assemble_hypersingular_direct(qdata)
    T_tilde = regularise_hypersingular(T_h, wq)

    print("  Computing spectral radius of T̃...")
    eigvals_T = np.linalg.eigvals(T_tilde)
    spectral_radius = float(np.max(np.abs(eigvals_T)))
    T_tilde_scaled  = T_tilde / spectral_radius

    # Verify: cond_eig unchanged, spectral radius = 1
    TV_scaled    = T_tilde_scaled @ V_h
    eigvals_TVs  = np.linalg.eigvals(TV_scaled)
    mag_TVs      = np.abs(eigvals_TVs)
    cond_eig_TVs = mag_TVs.max() / (mag_TVs.min() + 1e-14)

    TV_unscaled  = T_tilde @ V_h
    eigvals_TV   = np.linalg.eigvals(TV_unscaled)
    mag_TV       = np.abs(eigvals_TV)
    cond_eig_TV  = mag_TV.max() / (mag_TV.min() + 1e-14)
    cond_V       = np.linalg.cond(V_h)

    print(f"\n  === Spectral scaling diagnostics ===")
    print(f"    spectral_radius(T̃)          = {spectral_radius:.3e}")
    print(f"    spectral_radius(T̃_s)        = {np.max(np.abs(np.linalg.eigvals(T_tilde_scaled))):.3e}  (should be 1.0)")
    print(f"    cond_eig(T̃V)  before scale  = {cond_eig_TV:.3e}")
    print(f"    cond_eig(T̃_sV) after scale  = {cond_eig_TVs:.3e}  (should match)")
    print(f"    max|eig(T̃_s V)|             = {mag_TVs.max():.6f}  (should be O(1))")
    print(f"    cond(V_h)                    = {cond_V:.3e}  (for reference)")

    w_avg = float(wq.mean())
    print(f"    w_avg/4 = {w_avg/4:.6f}  (theory: eigenvalue cluster for smooth ∂Ω)")

    # Expected loss scale ratio after scaling
    loss_ratio_est = spectral_radius**2
    print(f"\n    Unscaled Calderón loss ≈ T̃-scaled × {loss_ratio_est:.2e}")
    print(f"    After scaling, Calderón loss ≈ standard loss  (both O(||r||²))")

    # ------------------------------------------------------------------
    # 3. Manufactured density
    # ------------------------------------------------------------------
    print("\n  === Manufactured density ===")
    R_cut      = 0.15
    enrichment = SingularEnrichment(geom=geom, use_cutoff=True,
                                    cutoff_radius=R_cut, per_corner_gamma=True)
    n_sing     = enrichment.n_singular
    sigma_s_Yq = enrichment.precompute(Yq_T)

    gamma_true   = CFG["gamma_true"].copy()
    f_smooth     = Yq_T[:, 0]**2 - Yq_T[:, 1]**2
    sigma_smooth = solve_bem(nmat, f_smooth, tol=1e-12).sigma
    sigma_mfg    = sigma_smooth + sigma_s_Yq @ gamma_true
    g_mfg        = V_h @ sigma_mfg
    sigma_bem    = solve_bem(nmat, g_mfg, tol=1e-12).sigma

    energy = float(np.linalg.norm(sigma_s_Yq @ gamma_true)**2
                   / max(np.linalg.norm(sigma_mfg)**2, 1e-14))
    print(f"    Enrichment energy = {energy*100:.2f}%")
    u_exact = make_u_exact_fn(Yq_T, wq, sigma_mfg)

    # ------------------------------------------------------------------
    # 4. Operator state
    # ------------------------------------------------------------------
    col_pts   = build_collocation_points(panels, m_col_panel=CFG["m_col_base"])
    panel_wts = panel_loss_weights(panels, w_base=CFG["w_base"],
                                   w_corner=CFG["w_corner"], w_ring=CFG["w_ring"])
    g_fn      = lambda xy: xy[:, 0]**2 - xy[:, 1]**2
    op, op_diag = build_operator_state(
        colloc=col_pts, qdata=qdata, enrichment=enrichment, g=g_fn,
        panel_weights=panel_wts, eq_scale_mode="none", eq_scale_fixed=1.0,
        dtype=torch.float64, device="cpu",
    )
    assert abs(op_diag["eq_scale"] - 1.0) < 1e-12
    op.f = torch.tensor(g_mfg, dtype=torch.float64)

    V_inv_np = np.linalg.inv(V_h)
    op.V_inv          = torch.tensor(V_inv_np, dtype=torch.float64)
    op.T_tilde_scaled = torch.tensor(T_tilde_scaled, dtype=torch.float64)

    # ------------------------------------------------------------------
    # 5. Initial model state
    # ------------------------------------------------------------------
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
    # 6. Short-budget training comparison (4 cases)
    # ------------------------------------------------------------------
    print("\n  === Short-budget training (Adam [500,500] + LBFGS 3000) ===")
    cases_short = []

    res_B  = _run_short("B (SE-BINN, std)", standard_loss,      init_state, shared)
    cases_short.append(res_B)

    res_DE = _run_short("D_exact (SE-BINN, V⁻¹)", exact_precond_loss, init_state, shared)
    cases_short.append(res_DE)

    res_CS = _run_short("D_calderon_scaled (T̃_s)", calderon_scaled_loss, init_state, shared)
    cases_short.append(res_CS)

    # Append Phase 2 unscaled reference (no re-run)
    cases_short.append(_PHASE2_UNSCALED)

    # ------------------------------------------------------------------
    # 7. Short-budget summary
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 3 — SHORT BUDGET SUMMARY")
    print("="*70)
    print(f"  {'Case':<38}  {'d_err':>8}  {'iL2':>10}  {'γ_err':>8}  {'init_loss':>12}")
    print(f"  {'-'*38}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*12}")
    for r in cases_short:
        il2  = r.get("final_rel_L2", float("nan"))
        gerr = r.get("gamma_err",    float("nan"))
        l0   = r.get("loss_init",    float("nan"))
        print(f"  {r['label']:<38}  {r['density_rel_diff']:>8.4f}  "
              f"{il2:>10.3e}  {gerr:>8.4f}  {l0:>12.3e}")

    d_B  = res_B["density_rel_diff"]
    d_CS = res_CS["density_rel_diff"]
    d_DE = res_DE["density_rel_diff"]
    print(f"\n  loss_init ratio (Calderón_scaled / std): "
          f"{res_CS['loss_init'] / max(res_B['loss_init'], 1e-14):.3f}  "
          f"(should be ≈ 1.0 after scaling)")
    print(f"\n  B   d_err = {d_B:.4f}")
    print(f"  D_exact   d_err = {d_DE:.4f}  ({d_B/max(d_DE,1e-6):.1f}× over B)")
    print(f"  D_scaled  d_err = {d_CS:.4f}  ({d_B/max(d_CS,1e-6):.1f}× over B)")

    # ------------------------------------------------------------------
    # 8. Figure: short-budget convergence
    # ------------------------------------------------------------------
    _fig_convergence(
        cases_short,
        title="Phase 3: Scaled Calderón — short budget (Koch(1))",
        outpath=os.path.join(fig_dir, "calderon_scaled_convergence_short.png"),
        adam_cutoff=sum(CFG["short_adam_iters"]),
    )

    # ------------------------------------------------------------------
    # 9. Full budget if scaled Calderón improved by >2×
    # ------------------------------------------------------------------
    run_full = (d_CS < d_B * 0.5)
    print(f"\n  Full-budget run: {'YES' if run_full else 'NO'} "
          f"(threshold: d_scaled < 0.5 × d_B = {0.5*d_B:.4f})")

    cases_all = list(cases_short)   # will extend with full-budget

    if run_full:
        print("\n  === Full-budget training (Adam [1000,1000,1000] + LBFGS 15000) ===")

        res_B_full  = _run_full("B full",                 standard_loss,
                                init_state, shared)
        res_DE_full = _run_full("D_exact full",           exact_precond_loss,
                                init_state, shared)
        res_CS_full = _run_full("D_calderon_scaled full", calderon_scaled_loss,
                                init_state, shared)

        cases_full = [res_B_full, res_DE_full, res_CS_full]
        cases_all.extend(cases_full)

        print("\n" + "="*70)
        print("PHASE 3 — FULL BUDGET SUMMARY")
        print("="*70)
        print(f"  {'Case':<38}  {'d_err':>8}  {'iL2':>10}  {'γ_err':>8}")
        print(f"  {'-'*38}  {'-'*8}  {'-'*10}  {'-'*8}")
        for r in cases_full:
            print(f"  {r['label']:<38}  {r['density_rel_diff']:>8.4f}  "
                  f"{r['final_rel_L2']:>10.3e}  {r['gamma_err']:>8.4f}")

        d_Bf  = res_B_full["density_rel_diff"]
        d_CSf = res_CS_full["density_rel_diff"]
        d_DEf = res_DE_full["density_rel_diff"]
        print(f"\n  D_exact full  vs B full: {d_Bf/max(d_DEf,1e-6):.1f}× improvement")
        print(f"  D_scaled full vs B full: {d_Bf/max(d_CSf,1e-6):.1f}× improvement")

        _fig_convergence(
            cases_full,
            title="Phase 3: Scaled Calderón — full budget (Koch(1))",
            outpath=os.path.join(fig_dir, "calderon_scaled_convergence_full.png"),
            adam_cutoff=sum(CFG["full_adam_iters"]),
        )
        _fig_density_comparison(
            cases_full, sigma_bem, arc, sigma_s_Yq, gamma_true,
            os.path.join(fig_dir, "calderon_scaled_comparison.png"),
        )
    else:
        # Still produce comparison figure from short budget
        _fig_density_comparison(
            [r for r in cases_short if r.get("sigma_final") is not None],
            sigma_bem, arc, sigma_s_Yq, gamma_true,
            os.path.join(fig_dir, "calderon_scaled_comparison.png"),
        )

    print(f"\n  All figures saved to {fig_dir}/")
    return cases_all


if __name__ == "__main__":
    main()
