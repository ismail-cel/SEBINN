"""
h-convergence study: SE-BINN vs BINN vs BEM on L-shaped domain.

Scientific question
-------------------
For u_exact = r^(2/3) sin(2θ/3), the boundary density σ has a r^{−1/3}
singularity at the reentrant corner (0,0).  Standard uniform-panel Nyström
converges at algebraic rate limited by this singularity.

We compare three methods across N_per_edge ∈ {4, 6, 8, 12, 16, 24}:

  BEM     : direct Nyström solve (no neural network) — establishes baseline
            convergence rate O(N^{2/3}) limited by the corner singularity.

  BINN    : neural σ_w trained to fit collocation residual, γ=0 frozen.
            Network must implicitly represent the singularity.

  SE-BINN : σ = σ_w + γ σ_s, γ trainable.  Singularity pre-handled by
            the analytic enrichment, σ_w should be smooth.

Expected behaviour
------------------
  BEM    : converges at O(N^{2/3}) in interior rel L2.
  BINN   : converges slower (network architecture limits, singularity not
           pre-removed), may plateau at each N.
  SE-BINN: converges faster per panel count, approximating O(N^p) with
           p > 2/3 because σ_w is smooth after enrichment.

If enrichment helps:  SE-BINN curve lies strictly below BINN curve,
and the log-log slope of SE-BINN is steeper.

Training budget
---------------
Each (N, method) pair uses:
  Stage 1: Adam [500, 500, 500] at lr [1e-3, 3e-4, 1e-4]  (1500 iters)
  Stage 2: L-BFGS max_iters=5000

This is intentionally modest so the study runs in < 30 min.
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
from matplotlib.ticker import LogLocator

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
from src.training.lbfgs import LBFGSConfig, _loss_and_grad, _two_loop, _armijo_line_search
from src.reconstruction.interior import reconstruct_interior


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Panel counts to sweep (panels per edge; L-shape has 6 edges)
N_LIST = [4, 6, 8, 12, 16, 24]

CFG = dict(
    seed           = 0,
    p_gl           = 16,
    m_col_base     = 4,
    eq_scale_mode  = "fixed",
    eq_scale_fixed = 10.0,
    gmres_tol      = 1e-12,
    gmres_maxiter  = 300,
    hidden_width   = 64,
    n_hidden       = 3,
    gamma_init     = 0.0,
    # Adam
    adam_iters     = [500, 500, 500],
    adam_lrs       = [1e-3, 3e-4, 1e-4],
    adam_log_every = 200,
    # L-BFGS
    lbfgs_max_iters  = 5000,
    lbfgs_memory     = 20,
    lbfgs_grad_tol   = 1e-9,
    lbfgs_log_every  = 500,
    lbfgs_step_tol   = 1e-12,
    lbfgs_alpha0     = 1e-1,
    lbfgs_alpha_fb   = [1e-2, 1e-3],
    lbfgs_armijo_c1  = 1e-4,
    lbfgs_beta       = 0.5,
    lbfgs_max_bt     = 20,
    # Evaluation
    n_grid_final   = 101,   # interior grid for all accuracy evaluations
)


# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------

def u_exact(xy: np.ndarray) -> np.ndarray:
    """u = r^(2/3) sin(2θ/3), θ clockwise from +x at (0,0)."""
    r     = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    theta = (-np.arctan2(xy[:, 1], xy[:, 0])) % (2.0 * np.pi)
    return np.where(r < 1e-15, 0.0, r ** (2.0 / 3.0) * np.sin(2.0 * theta / 3.0))


# ---------------------------------------------------------------------------
# Adam loop (minimal, no tracking)
# ---------------------------------------------------------------------------

def _run_adam(model, op, phase_iters, phase_lrs, log_every, label, verbose):
    from src.training.loss import sebinn_loss
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=phase_lrs[0], betas=(0.9, 0.999), eps=1e-8,
    )
    global_it = 0
    for ph, (n_it, lr) in enumerate(zip(phase_iters, phase_lrs)):
        for pg in opt.param_groups:
            pg["lr"] = lr
        for j in range(n_it):
            opt.zero_grad()
            loss, _ = sebinn_loss(model, op)
            loss.backward()
            opt.step()
            global_it += 1
            if verbose and (j == 0 or (global_it % log_every == 0)):
                print(f"  [{label}] Adam {global_it:5d} | loss={float(loss):.3e} | "
                      f"γ={model.gamma_value():.4f}")
    return global_it


# ---------------------------------------------------------------------------
# L-BFGS loop (inline, matching run.py implementation)
# ---------------------------------------------------------------------------

def _run_lbfgs(model, op, cfg, label, log_every, verbose):
    max_iters    = cfg.max_iters
    grad_tol     = cfg.grad_tol
    step_tol     = cfg.step_tol
    memory       = cfg.memory
    alpha_starts = [cfg.alpha0] + list(cfg.alpha_fallback)

    theta = model.to_vector()
    f, g  = _loss_and_grad(theta, model, op)

    S, Y, rho_list = [], [], []
    n_ls_fail = 0
    reason    = "maxIters"

    for k in range(1, max_iters + 1):
        gnorm = float(g.norm())
        if gnorm < grad_tol:
            reason = "gradTol"
            break

        p_dir = -_two_loop(g, S, Y, rho_list) if S else -g

        alpha, theta_new, f_new, g_new, accepted = _armijo_line_search(
            theta, f, g, p_dir, model, op, cfg, alpha_starts)

        if not accepted:
            n_ls_fail += 1
            S.clear(); Y.clear(); rho_list.clear()
            alpha, theta_new, f_new, g_new, accepted = _armijo_line_search(
                theta, f, g, -g, model, op, cfg, alpha_starts)
            if not accepted:
                reason = "lsFailure"
                break

        s_vec = theta_new - theta
        y_vec = g_new - g
        snorm = float(s_vec.norm())
        if snorm < step_tol:
            reason = "stepTol"
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

        if verbose and (k == 1 or k % log_every == 0 or k == max_iters):
            print(f"  [{label}] LBFGS {k:5d} | loss={f:.3e} | "
                  f"gNorm={gnorm:.3e} | γ={model.gamma_value():.4f}")

    model.from_vector(theta)
    if verbose:
        print(f"  [{label}] LBFGS done | k={k} | reason={reason}")
    return k, reason


# ---------------------------------------------------------------------------
# Run one (N, method) pair
# ---------------------------------------------------------------------------

def _run_one(n_per_edge, freeze_gamma, label, verbose=False):
    """
    Train one model and return interior rel_L2, linf, and final gamma.

    Parameters
    ----------
    n_per_edge   : int — panels per edge
    freeze_gamma : bool — True → BINN (σ = σ_w only)
    label        : str — for logging
    """
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    geom   = make_lshape_geometry()
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=n_per_edge)
    label_corner_ring_panels(panels, P)
    qdata  = build_panel_quadrature(panels, p=CFG["p_gl"])
    Yq_np  = qdata.Yq.T      # (Nq, 2)
    wq     = qdata.wq

    nmat      = assemble_nystrom_matrix(qdata)
    bem_sol   = solve_bem(nmat, u_exact(Yq_np),
                          tol=CFG["gmres_tol"], max_iter=CFG["gmres_maxiter"])
    sigma_bem = bem_sol.sigma

    enrichment  = SingularEnrichment(geom=geom, per_corner_gamma=False)
    sigma_s_np  = enrichment.evaluate(Yq_np)
    w_panel     = panel_loss_weights(panels, w_base=1.0, w_corner=1.0, w_ring=1.0)
    colloc      = build_collocation_points(panels, m_col_panel=CFG["m_col_base"])
    op, _       = build_operator_state(
        colloc=colloc, qdata=qdata, enrichment=enrichment, g=u_exact,
        panel_weights=w_panel,
        eq_scale_mode=CFG["eq_scale_mode"],
        eq_scale_fixed=CFG["eq_scale_fixed"],
        dtype=torch.float64, device="cpu",
    )

    model = SEBINNModel(
        hidden_width=CFG["hidden_width"], n_hidden=CFG["n_hidden"],
        n_gamma=enrichment.n_gamma, gamma_init=CFG["gamma_init"],
        dtype=torch.float64,
    )
    if freeze_gamma:
        model.gamma_module.gamma.requires_grad_(False)

    # Adam
    _run_adam(model, op, CFG["adam_iters"], CFG["adam_lrs"],
              CFG["adam_log_every"], label, verbose)

    # L-BFGS
    cfg_lbfgs = LBFGSConfig(
        max_iters=CFG["lbfgs_max_iters"],
        memory=CFG["lbfgs_memory"],
        grad_tol=CFG["lbfgs_grad_tol"],
        step_tol=CFG["lbfgs_step_tol"],
        alpha0=CFG["lbfgs_alpha0"],
        alpha_fallback=CFG["lbfgs_alpha_fb"],
        armijo_c1=CFG["lbfgs_armijo_c1"],
        backtrack_beta=CFG["lbfgs_beta"],
        max_backtrack=CFG["lbfgs_max_bt"],
    )
    _run_lbfgs(model, op, cfg_lbfgs, label, CFG["lbfgs_log_every"], verbose)

    # Reconstruct interior
    sigma_s_t = torch.tensor(sigma_s_np, dtype=torch.float64)
    Yq_t      = torch.tensor(Yq_np,      dtype=torch.float64)
    with torch.no_grad():
        sigma_nn = model(Yq_t, sigma_s_t).numpy()

    out = reconstruct_interior(
        P=P, Yq=Yq_np, wq=wq, sigma=sigma_nn,
        n_grid=CFG["n_grid_final"], u_exact=u_exact,
        x_range=(-1.0, 1.0), y_range=(-1.0, 1.0),
    )

    density_err = float(
        np.linalg.norm(sigma_nn - sigma_bem) / max(np.linalg.norm(sigma_bem), 1e-14)
    )

    return dict(
        rel_L2       = out.rel_L2,
        linf         = out.linf,
        density_err  = density_err,
        gamma_final  = model.gamma_value(),
        n_per_edge   = n_per_edge,
        Nq           = len(wq),
    )


def _bem_accuracy(n_per_edge):
    """BEM reference interior accuracy."""
    geom   = make_lshape_geometry()
    panels = build_uniform_panels(geom.vertices, n_per_edge=n_per_edge)
    label_corner_ring_panels(panels, geom.vertices)
    qdata  = build_panel_quadrature(panels, p=CFG["p_gl"])
    Yq     = qdata.Yq.T
    wq     = qdata.wq
    nmat   = assemble_nystrom_matrix(qdata)
    sol    = solve_bem(nmat, u_exact(Yq), tol=1e-12, max_iter=300)
    out    = reconstruct_interior(
        P=geom.vertices, Yq=Yq, wq=wq, sigma=sol.sigma,
        n_grid=CFG["n_grid_final"], u_exact=u_exact,
        x_range=(-1.0, 1.0), y_range=(-1.0, 1.0),
    )
    return dict(rel_L2=out.rel_L2, linf=out.linf, n_per_edge=n_per_edge, Nq=len(wq))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _fig_convergence(bem_rows, binn_rows, sebinn_rows, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.35)

    N_vals      = np.array([r["n_per_edge"] for r in bem_rows])
    h_vals      = 1.0 / N_vals          # panel size proxy

    bem_L2      = np.array([r["rel_L2"]      for r in bem_rows])
    binn_L2     = np.array([r["rel_L2"]      for r in binn_rows])
    sebinn_L2   = np.array([r["rel_L2"]      for r in sebinn_rows])
    binn_derr   = np.array([r["density_err"] for r in binn_rows])
    sebinn_derr = np.array([r["density_err"] for r in sebinn_rows])

    # --- Left: interior rel L2 vs N ---
    ax = axes[0]
    ax.loglog(N_vals, bem_L2,    "k-o",  lw=1.5, ms=5, label="BEM")
    ax.loglog(N_vals, binn_L2,   "b-s",  lw=1.5, ms=5, label="BINN ($\\gamma=0$)")
    ax.loglog(N_vals, sebinn_L2, "r-^",  lw=1.5, ms=5, label="SE-BINN ($\\gamma$ free)")

    # Reference slopes
    N_ref = N_vals[[0, -1]].astype(float)
    for slope, style, txt in [
        (-2/3, "k:",   r"$N^{-2/3}$"),
        (-1.0, "k--",  r"$N^{-1}$"),
    ]:
        mid = 0.5 * (np.log(N_ref[0]) + np.log(N_ref[-1]))
        N_mid = np.exp(mid)
        ref   = bem_L2[0] * (N_vals / N_vals[0]) ** slope
        ax.loglog(N_vals, ref, style, lw=0.8, alpha=0.6, label=txt)

    ax.set_xlabel("Panels per edge $N$", fontsize=12)
    ax.set_ylabel("Interior rel $L^2$ error", fontsize=12)
    ax.set_title("(a) Interior accuracy vs $N$", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    # --- Right: density rel-diff vs N ---
    ax2 = axes[1]
    ax2.semilogy(N_vals, binn_derr,   "b-s",  lw=1.5, ms=5, label="BINN")
    ax2.semilogy(N_vals, sebinn_derr, "r-^",  lw=1.5, ms=5, label="SE-BINN")
    ax2.set_xlabel("Panels per edge $N$", fontsize=12)
    ax2.set_ylabel(r"$\|\sigma_\theta - \sigma_\mathrm{BEM}\| / \|\sigma_\mathrm{BEM}\|$",
                   fontsize=12)
    ax2.set_title("(b) Boundary density rel-diff vs $N$", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(
        r"h-Convergence: L-shape, $u = r^{2/3}\sin(2\theta/3)$" + "\n"
        r"($\alpha = \pi/\omega - 1 = -1/3$,  expected BEM rate $N^{-2/3}$)",
        fontsize=12, y=1.02,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def _print_table(bem_rows, binn_rows, sebinn_rows):
    print()
    print("=" * 80)
    print("  h-CONVERGENCE TABLE (L-shape, u = r^(2/3) sin(2θ/3))")
    print("=" * 80)
    hdr = f"  {'N':>4}  {'Nq':>6}  {'BEM rel_L2':>12}  " \
          f"{'BINN rel_L2':>12}  {'SEBINN rel_L2':>13}  " \
          f"{'BINN derr':>10}  {'SEBINN derr':>11}  {'γ_final':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for b, bn, se in zip(bem_rows, binn_rows, sebinn_rows):
        gf = se["gamma_final"]
        gstr = f"{gf:.4f}" if not isinstance(gf, list) else f"{gf[0]:.4f}"
        print(f"  {b['n_per_edge']:>4}  {b['Nq']:>6}  "
              f"{b['rel_L2']:>12.3e}  {bn['rel_L2']:>12.3e}  {se['rel_L2']:>13.3e}  "
              f"{bn['density_err']:>10.4f}  {se['density_err']:>11.4f}  {gstr:>8}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("=" * 65)
    print("  h-Convergence study: SE-BINN on L-shaped domain")
    print(f"  N_list = {N_LIST}")
    print("=" * 65)

    figures_dir = os.path.join(_HERE, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    bem_rows    = []
    binn_rows   = []
    sebinn_rows = []

    for n_per_edge in N_LIST:
        t0 = time.perf_counter()
        Ntot = 6 * n_per_edge
        print(f"\n--- N_per_edge={n_per_edge} | N_total={Ntot} ---")

        # BEM reference
        print("  BEM reference ...")
        r_bem = _bem_accuracy(n_per_edge)
        bem_rows.append(r_bem)
        print(f"  BEM: rel_L2={r_bem['rel_L2']:.3e}  linf={r_bem['linf']:.3e}")

        # BINN (gamma frozen)
        print("  BINN ...")
        r_binn = _run_one(n_per_edge, freeze_gamma=True,  label="BINN",    verbose=False)
        binn_rows.append(r_binn)
        print(f"  BINN: rel_L2={r_binn['rel_L2']:.3e}  derr={r_binn['density_err']:.4f}")

        # SE-BINN (gamma free)
        print("  SE-BINN ...")
        r_se = _run_one(n_per_edge,  freeze_gamma=False, label="SE-BINN", verbose=False)
        sebinn_rows.append(r_se)
        gf = r_se["gamma_final"]
        gstr = f"{gf:.4f}" if not isinstance(gf, list) else f"{gf[0]:.4f}"
        print(f"  SE-BINN: rel_L2={r_se['rel_L2']:.3e}  derr={r_se['density_err']:.4f}  γ={gstr}")

        print(f"  Wall: {time.perf_counter()-t0:.1f}s")

    _print_table(bem_rows, binn_rows, sebinn_rows)

    _fig_convergence(
        bem_rows, binn_rows, sebinn_rows,
        outpath=os.path.join(figures_dir, "convergence_study.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
