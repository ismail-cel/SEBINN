"""
A/B comparison: BINN baseline (γ=0 frozen) vs SE-BINN (γ free, per-corner).

Purpose
-------
Isolate the effect of singular enrichment by running two training runs that
are identical except for whether γ is trainable:

  Case A — BINN:    SEBINNModel with γ frozen at zero → σ = σ_w only
  Case B — SE-BINN: SEBINNModel with per_corner_gamma=True, γ trainable

Both runs start from the SAME initial network weights (shared state_dict).
All hyperparameters are identical.  The only difference is γ trainability.

Geometry / problem
------------------
Koch(1) snowflake, u_exact = x² − y², Dirichlet BVP.

Outputs
-------
- Console: side-by-side comparison table
- experiments/ex1_Koch/figures/ab_comparison.png  (3 subplots)
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
from src.singular.enrichment import SingularEnrichment
from src.models.sebinn import SEBINNModel
from src.training.collocation import build_collocation_points
from src.training.operator import build_operator_state
from src.training.adam_phase import AdamConfig, run_adam_phases
from src.training.lbfgs import LBFGSConfig, run_lbfgs
from src.reconstruction.interior import reconstruct_interior


# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

CFG = dict(
    seed          = 0,
    # geometry
    n_per_edge    = 12,
    p_gl          = 16,
    m_col_base    = 4,
    w_base        = 1.0,
    w_corner      = 1.0,
    w_ring        = 1.0,
    # equation scaling
    eq_scale_mode  = "fixed",
    eq_scale_fixed = 10.0,
    # BEM
    gmres_tol     = 1e-12,
    gmres_maxiter = 300,
    # model
    hidden_width  = 80,
    n_hidden      = 4,
    gamma_init    = 0.0,
    # Adam — two phases, 1000 iters each
    adam_iters    = [1000, 1000],
    adam_lrs      = [1e-3, 1e-4],
    log_every     = 200,
    # L-BFGS
    lbfgs_max_iters  = 5000,
    lbfgs_grad_tol   = 1e-9,
    lbfgs_step_tol   = 1e-12,
    lbfgs_memory     = 20,
    lbfgs_log_every  = 50,
    lbfgs_alpha0     = 1e-1,
    lbfgs_alpha_fb   = [1e-2, 1e-3],
    lbfgs_armijo_c1  = 1e-4,
    lbfgs_beta       = 0.5,
    lbfgs_max_bt     = 20,
    # grid
    n_grid        = 201,
)


# ---------------------------------------------------------------------------
# Boundary data
# ---------------------------------------------------------------------------

def u_exact(xy: np.ndarray) -> np.ndarray:
    return xy[:, 0] ** 2 - xy[:, 1] ** 2


# ---------------------------------------------------------------------------
# Arc-length helper  (copied from run.py)
# ---------------------------------------------------------------------------

def _boundary_arclength(qdata, n_per_edge: int):
    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc = panel_start[qdata.pan_id] + qdata.s_on_panel

    Npan         = qdata.n_panels
    total_length = float(qdata.L_panel.sum())
    Nv           = Npan // n_per_edge

    v_panel_idx       = np.arange(Nv) * n_per_edge
    panel_start_full  = np.concatenate([[0.0], np.cumsum(qdata.L_panel)])
    vertex_arcs       = np.append(panel_start_full[v_panel_idx], total_length)

    return arc, vertex_arcs, total_length


# ---------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------

def _train_one(
    label: str,
    freeze_gamma: bool,
    init_state: dict,
    shared: dict,
    verbose: bool = True,
) -> dict:
    """
    Run one case (A or B).

    Parameters
    ----------
    label        : "BINN" or "SE-BINN"
    freeze_gamma : if True, call requires_grad_(False) on gamma after loading
                   the shared initial state_dict
    init_state   : shared initial model state_dict (loaded from the zero-start
                   model before any training)
    shared       : precomputed shared objects (op, enrichment, Yq_T, wq, etc.)
    """
    if verbose:
        frozen_str = " [γ frozen=0]" if freeze_gamma else " [γ trainable]"
        print(f"\n{'='*60}")
        print(f"  Case {label}{frozen_str}")
        print(f"{'='*60}")

    t0 = time.perf_counter()

    op         = shared["op"]
    enrichment = shared["enrichment"]
    Yq_T       = shared["Yq_T"]
    wq         = shared["wq"]
    P          = shared["P"]
    sigma_bem  = shared["sigma_bem"]
    sort_idx   = shared["sort_idx"]

    # Build model and load shared initial weights
    model = SEBINNModel(
        hidden_width = CFG["hidden_width"],
        n_hidden     = CFG["n_hidden"],
        n_gamma      = enrichment.n_gamma,
        gamma_init   = CFG["gamma_init"],
        dtype        = torch.float64,
    )
    model.load_state_dict(copy.deepcopy(init_state))

    if freeze_gamma:
        model.gamma_module.gamma.requires_grad_(False)
        if verbose:
            print(f"  gamma frozen at {model.gamma_module.gamma.detach().tolist()}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  trainable params: {n_trainable}")

    # Adam
    adam_cfg = AdamConfig(
        phase_iters = CFG["adam_iters"],
        phase_lrs   = CFG["adam_lrs"],
        log_every   = CFG["log_every"],
    )
    adam_res = run_adam_phases(model, op, adam_cfg, verbose=verbose)
    adam_n_iters = adam_res.n_iters

    # L-BFGS
    lbfgs_cfg = LBFGSConfig(
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
    lbfgs_res = run_lbfgs(model, op, lbfgs_cfg, verbose=verbose)

    # Final density
    sigma_s_Yq = enrichment.precompute(Yq_T)
    with torch.no_grad():
        sigma_final = model(
            torch.tensor(Yq_T,       dtype=torch.float64),
            torch.tensor(sigma_s_Yq, dtype=torch.float64),
        ).numpy()

    final_out = reconstruct_interior(
        P=P, Yq=Yq_T, wq=wq, sigma=sigma_final,
        n_grid=CFG["n_grid"], u_exact=u_exact,
    )

    density_rel_diff = float(
        np.linalg.norm(sigma_final - sigma_bem)
        / max(np.linalg.norm(sigma_bem), 1e-14)
    )

    t_total = time.perf_counter() - t0

    # Combined loss history (Adam then L-BFGS, 1-indexed iteration axis)
    loss_hist_adam  = list(adam_res.loss_hist)
    loss_hist_lbfgs = list(lbfgs_res.loss_hist)

    gamma_vals = model.gamma_value()

    if verbose:
        print(f"\n  {label} results:")
        print(f"    Interior rel L2  : {final_out.rel_L2:.3e}")
        print(f"    Interior L∞      : {final_out.linf:.3e}")
        print(f"    Final loss       : {lbfgs_res.loss_hist[-1]:.3e}")
        print(f"    Density rel-diff : {density_rel_diff:.3e}")
        print(f"    Wall time        : {t_total:.1f}s")
        if not freeze_gamma:
            gv = gamma_vals if isinstance(gamma_vals, list) else [gamma_vals]
            print(f"    γ_c values       : {[f'{v:.4f}' for v in gv]}")

    return dict(
        label            = label,
        freeze_gamma     = freeze_gamma,
        final_rel_L2     = final_out.rel_L2,
        final_linf       = final_out.linf,
        final_loss       = lbfgs_res.loss_hist[-1],
        density_rel_diff = density_rel_diff,
        wall_time        = t_total,
        gamma_vals       = gamma_vals,
        loss_hist_adam   = loss_hist_adam,
        loss_hist_lbfgs  = loss_hist_lbfgs,
        adam_n_iters     = adam_n_iters,
        sigma_final      = sigma_final[sort_idx],
        final_out        = final_out,
    )


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _print_table(res_a: dict, res_b: dict, bem_rel_L2: float, bem_linf: float):
    col_w = 20
    print()
    print("=" * 62)
    print("  A/B COMPARISON: BINN (γ=0) vs SE-BINN (γ free, per-corner)")
    print("=" * 62)

    header = f"{'Metric':<24} | {'BINN (γ=0)':>{col_w}} | {'SE-BINN (γ free)':>{col_w}}"
    sep    = "-" * len(header)
    print(header)
    print(sep)

    def row(name, va, vb):
        print(f"{name:<24} | {va:>{col_w}} | {vb:>{col_w}}")

    row("Interior rel L2",
        f"{res_a['final_rel_L2']:.3e}",
        f"{res_b['final_rel_L2']:.3e}")
    row("Interior L∞",
        f"{res_a['final_linf']:.3e}",
        f"{res_b['final_linf']:.3e}")
    row("Final loss",
        f"{res_a['final_loss']:.3e}",
        f"{res_b['final_loss']:.3e}")
    row("Density rel-diff",
        f"{res_a['density_rel_diff']:.3e}",
        f"{res_b['density_rel_diff']:.3e}")
    row("Wall time (s)",
        f"{res_a['wall_time']:.1f}",
        f"{res_b['wall_time']:.1f}")
    print(sep)
    row("BEM rel L2 (ref)", f"{bem_rel_L2:.3e}", f"{bem_rel_L2:.3e}")
    print(sep)

    # gamma
    gv = res_b["gamma_vals"]
    if isinstance(gv, list):
        gv_str = "[" + ", ".join(f"{v:.3f}" for v in gv) + "]"
    else:
        gv_str = f"{float(gv):.6f}"
    row("γ_c values", "(frozen at 0)", gv_str)
    print("=" * 62)

    # improvement factor
    if res_a["final_rel_L2"] > 0 and res_b["final_rel_L2"] > 0:
        factor = res_a["final_rel_L2"] / res_b["final_rel_L2"]
        print(f"\n  SE-BINN interior error improvement: {factor:.2f}×")
    density_factor = res_a["density_rel_diff"] / max(res_b["density_rel_diff"], 1e-14)
    print(f"  SE-BINN density error improvement : {density_factor:.2f}×")
    print()


# ---------------------------------------------------------------------------
# Comparison figure (3 subplots)
# ---------------------------------------------------------------------------

def _plot_comparison(
    res_a: dict,
    res_b: dict,
    sigma_bem_sorted: np.ndarray,
    arc: np.ndarray,
    vertex_arcs: np.ndarray,
    singular_corner_indices,
    outpath: str,
):
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.subplots_adjust(hspace=0.42)

    n_adam  = res_a["adam_n_iters"]      # same for both (identical config)
    total_a = len(res_a["loss_hist_adam"]) + len(res_a["loss_hist_lbfgs"])
    total_b = len(res_b["loss_hist_adam"]) + len(res_b["loss_hist_lbfgs"])

    # ---- (a) Loss history ----
    ax = axes[0]

    def _plot_loss(res, color, label):
        ha = res["loss_hist_adam"]
        hl = res["loss_hist_lbfgs"]
        n_a = len(ha)
        n_l = len(hl)
        iters_a = np.arange(1, n_a + 1)
        iters_l = np.arange(n_a + 1, n_a + n_l + 1)
        ax.semilogy(iters_a, ha, color=color, lw=1.5, alpha=0.85, label=label + " (Adam)")
        if n_l:
            ax.semilogy(iters_l, hl, color=color, lw=1.5, ls="--", alpha=0.85,
                        label=label + " (L-BFGS)")

    _plot_loss(res_a, "#1f77b4", "BINN")
    _plot_loss(res_b, "#d62728", "SE-BINN")

    # Vertical line at Adam→L-BFGS transition (same for both)
    ax.axvline(n_adam, color="gray", lw=1.0, ls=":", label=f"Adam→L-BFGS (iter {n_adam})")

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("(a) Loss history: BINN vs SE-BINN", fontsize=12)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    # ---- (b) Density comparison ----
    ax = axes[1]

    # Vertical lines at reentrant corner arclength positions
    # Reentrant corners are at odd indices for Koch(1)
    n_per_edge = CFG["n_per_edge"]
    total_length = vertex_arcs[-1]
    for ci in singular_corner_indices:
        ax.axvline(vertex_arcs[ci], color="#999999", lw=0.7, ls="--", alpha=0.7)

    ax.plot(arc, sigma_bem_sorted,         color="black",   lw=1.0, alpha=0.9,
            label=r"$\sigma_\mathrm{BEM}$ (reference)")
    ax.plot(arc, res_a["sigma_final"],     color="#1f77b4", lw=1.0, alpha=0.85, ls="-",
            label=r"$\sigma_\mathrm{BINN}$ ($\gamma=0$)")
    ax.plot(arc, res_b["sigma_final"],     color="#d62728", lw=1.0, alpha=0.85, ls="--",
            label=r"$\sigma_\mathrm{SE\text{-}BINN}$ ($\gamma$ free)")

    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$\sigma(s)$", fontsize=11)
    ax.set_title("(b) Boundary density comparison", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, lw=0.3, alpha=0.5)

    # Annotate reentrant corners
    ax.annotate("↕ reentrant corners", xy=(vertex_arcs[1], ax.get_ylim()[0]),
                fontsize=7, color="#666666",
                xytext=(vertex_arcs[1] + 0.05, ax.get_ylim()[0]),
                textcoords="data")

    # ---- (c) Density error ----
    ax = axes[2]

    err_binn   = np.abs(res_a["sigma_final"] - sigma_bem_sorted)
    err_sebinn = np.abs(res_b["sigma_final"] - sigma_bem_sorted)

    for ci in singular_corner_indices:
        ax.axvline(vertex_arcs[ci], color="#999999", lw=0.7, ls="--", alpha=0.7)

    ax.semilogy(arc, err_binn,   color="#1f77b4", lw=1.0, alpha=0.85,
                label=r"$|\sigma_\mathrm{BINN} - \sigma_\mathrm{BEM}|$")
    ax.semilogy(arc, err_sebinn, color="#d62728", lw=1.0, alpha=0.85, ls="--",
                label=r"$|\sigma_\mathrm{SE\text{-}BINN} - \sigma_\mathrm{BEM}|$")

    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$|\sigma - \sigma_\mathrm{BEM}|$", fontsize=11)
    ax.set_title("(c) Pointwise density error", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(
        "A/B comparison: BINN (γ=0) vs SE-BINN (per-corner γ free)\n"
        "Koch(1), $u = x^2 - y^2$,  Adam 2000 + L-BFGS 5000",
        fontsize=13, y=1.01,
    )

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved ab_comparison → {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("=" * 60)
    print("  SE-BINN A/B Comparison — Koch(1), u = x² − y²")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Geometry, quadrature, BEM (shared)
    # ------------------------------------------------------------------
    print("\n--- Shared setup ---")
    geom   = make_koch_geometry(n=1)
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=CFG["n_per_edge"])
    label_corner_ring_panels(panels, P)

    qdata = build_panel_quadrature(panels, p=CFG["p_gl"])
    Yq_T  = qdata.Yq.T          # (Nq, 2)
    wq    = qdata.wq
    Nq    = qdata.n_quad

    arc, vertex_arcs, total_arc = _boundary_arclength(qdata, CFG["n_per_edge"])
    sort_idx = np.argsort(arc)

    print(f"  Geometry: Koch(1) | vertices={geom.n_vertices} | panels={len(panels)}")
    print(f"  Quadrature: Nq={Nq} | total arc-length={total_arc:.4f}")

    # BEM reference
    nmat      = assemble_nystrom_matrix(qdata)
    f_bnd     = u_exact(Yq_T)
    bem_sol   = solve_bem(nmat, f_bnd,
                          tol=CFG["gmres_tol"], max_iter=CFG["gmres_maxiter"])
    sigma_bem = bem_sol.sigma

    bem_out = reconstruct_interior(
        P=P, Yq=Yq_T, wq=wq, sigma=sigma_bem,
        n_grid=CFG["n_grid"], u_exact=u_exact,
    )
    print(f"  BEM: rel_L2={bem_out.rel_L2:.3e} | linf={bem_out.linf:.3e}")

    # ------------------------------------------------------------------
    # 2. Operator state (shared: both cases see the same linear system)
    #    We use per_corner_gamma=True enrichment.  For Case A the γ
    #    parameters exist but are frozen, so σ_s terms are assembled but
    #    never updated — this is equivalent to σ = σ_w.
    # ------------------------------------------------------------------
    enrichment = SingularEnrichment(geom=geom, per_corner_gamma=True)

    w_panel = panel_loss_weights(panels, w_base=1.0, w_corner=1.0, w_ring=1.0)
    colloc  = build_collocation_points(panels, m_col_panel=CFG["m_col_base"])

    op, op_diag = build_operator_state(
        colloc         = colloc,
        qdata          = qdata,
        enrichment     = enrichment,
        g              = u_exact,
        panel_weights  = w_panel,
        eq_scale_mode  = CFG["eq_scale_mode"],
        eq_scale_fixed = CFG["eq_scale_fixed"],
        dtype          = torch.float64,
        device         = "cpu",
    )
    print(f"  Operator: Nb={colloc.n_colloc} | eq_scale={op_diag['eq_scale']:.2e}")

    # ------------------------------------------------------------------
    # 3. Initial model (one shared initialisation for both cases)
    # ------------------------------------------------------------------
    torch.manual_seed(CFG["seed"])   # re-seed to guarantee same init
    init_model = SEBINNModel(
        hidden_width = CFG["hidden_width"],
        n_hidden     = CFG["n_hidden"],
        n_gamma      = enrichment.n_gamma,
        gamma_init   = CFG["gamma_init"],
        dtype        = torch.float64,
    )
    init_state = copy.deepcopy(init_model.state_dict())
    print(f"  Model: n_params={init_model.n_params()} | n_gamma={enrichment.n_gamma}")
    print(f"  Initial state_dict saved — both cases will start from here.")

    shared = dict(
        op                     = op,
        enrichment             = enrichment,
        Yq_T                   = Yq_T,
        wq                     = wq,
        P                      = P,
        sigma_bem              = sigma_bem,
        sort_idx               = sort_idx,
    )

    # ------------------------------------------------------------------
    # 4. Case A — BINN baseline (γ frozen at 0)
    # ------------------------------------------------------------------
    res_a = _train_one(
        label        = "BINN",
        freeze_gamma = True,
        init_state   = init_state,
        shared       = shared,
        verbose      = True,
    )

    # ------------------------------------------------------------------
    # 5. Case B — SE-BINN (γ trainable, per-corner)
    # ------------------------------------------------------------------
    res_b = _train_one(
        label        = "SE-BINN",
        freeze_gamma = False,
        init_state   = init_state,
        shared       = shared,
        verbose      = True,
    )

    # ------------------------------------------------------------------
    # 6. Comparison table
    # ------------------------------------------------------------------
    _print_table(res_a, res_b, bem_out.rel_L2, bem_out.linf)

    # ------------------------------------------------------------------
    # 7. Figure
    # ------------------------------------------------------------------
    figures_dir = os.path.join(_HERE, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    outpath = os.path.join(figures_dir, "ab_comparison.png")

    _plot_comparison(
        res_a            = res_a,
        res_b            = res_b,
        sigma_bem_sorted = sigma_bem[sort_idx],
        arc              = arc[sort_idx],
        vertex_arcs      = vertex_arcs,
        singular_corner_indices = list(geom.singular_corner_indices),
        outpath          = outpath,
    )


if __name__ == "__main__":
    main()
