"""
Experiment 1: SE-BINN on Koch(1) snowflake, u_exact = x² − y².

MATLAB reference: bem_pinn_nystrom_comparison.m (static workflow, lines 1-320)

Geometry
--------
Koch(1) snowflake: 12 edges, 12 vertices.
Reentrant (singular) corners at odd indices {1,3,5,7,9,11}, interior angle
omega = 4π/3, singular exponent alpha = π/omega − 1 = −1/4.

Boundary condition
------------------
Dirichlet: g(x,y) = x² − y²  (harmonic, smooth — tests whether SEBINN
recovers a solution whose singularity is mild enough to be captured by the
enrichment even when u itself is smooth).

Training
--------
Phase 1 : Adam,   500 iters, lr = 1e-4
Phase 2 : L-BFGS, up to 3500 iters, Armijo backtracking

Evaluation
----------
BEM Nyström reference (GMRES) + SE-BINN reconstruction on a 201×201 grid.
Reports relative L2 and L∞ errors vs u_exact.
"""

import sys
import os
import time
import argparse

import numpy as np
import torch

# ---- project root on path ----
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


# ===========================================================================
# Configuration  (mirrors MATLAB cfg struct)
# ===========================================================================

def default_cfg() -> dict:
    return dict(
        seed          = 0,

        # --- geometry / discretisation ---
        n_per_edge    = 12,       # cfg.NpEdge = 12
        p_gl          = 16,       # cfg.pGL = 16
        m_col_base    = 4,        # 4 GL nodes per panel → Nb = 576
        w_base        = 1.0,
        w_corner      = 1.0,
        w_ring        = 1.0,

        # --- equation scaling ---
        # MATLAB default: useEquationScaling=false → none.
        # 'fixed' at 10.0 gives mild conditioning improvement without
        # inflating gradients to the degree that breaks L-BFGS line search.
        eq_scale_mode  = "fixed",
        eq_scale_fixed = 10.0,

        # --- BEM solve ---
        gmres_tol     = 1e-12,
        gmres_maxiter = 300,

        # --- SE-BINN model ---
        hidden_width  = 80,
        n_hidden      = 4,
        gamma_init    = 0.0,

        # --- Adam  (two-phase: warm-up at 1e-3, refine at 1e-4) ---
        adam_iters    = [500, 500],
        adam_lrs      = [1e-3, 1e-4],
        log_every     = 100,

        # --- L-BFGS ---
        use_lbfgs         = True,
        lbfgs_max_iters   = 3500,
        lbfgs_grad_tol    = 1e-8,
        lbfgs_step_tol    = 1e-12,
        lbfgs_memory      = 10,
        lbfgs_log_every   = 25,
        lbfgs_alpha0      = 1e-1,
        lbfgs_alpha_fb    = [1e-2, 1e-3],
        lbfgs_armijo_c1   = 1e-4,
        lbfgs_beta        = 0.5,
        lbfgs_max_bt      = 20,

        # --- reconstruction / plotting ---
        n_grid        = 201,
    )


# ===========================================================================
# Boundary data
# ===========================================================================

def u_exact(xy: np.ndarray) -> np.ndarray:
    """u(x,y) = x² − y².  MATLAB: cfg.u_exact = @(x,y) x.^2 - y.^2."""
    return xy[:, 0] ** 2 - xy[:, 1] ** 2


# ===========================================================================
# Arc-length helper
# ===========================================================================

def _boundary_arclength(qdata, n_per_edge: int):
    """
    Compute arc-length coordinate for every quadrature node and the
    arc-length positions of the polygon vertices.

    Returns
    -------
    arc          : ndarray (Nq,)   arc-length of each quad node (sorted by panel order)
    vertex_arcs  : ndarray (Nv+1,) arc-lengths at each vertex (0 … total_length)
    total_length : float
    """
    # cumulative panel starts: panel_start[k] = sum(L[0:k])
    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc = panel_start[qdata.pan_id] + qdata.s_on_panel   # (Nq,)

    Npan         = qdata.n_panels
    total_length = float(qdata.L_panel.sum())
    Nv           = Npan // n_per_edge                     # number of vertices

    # vertex i starts at panel i*n_per_edge
    v_panel_idx  = np.arange(Nv) * n_per_edge
    panel_start_full = np.concatenate([[0.0], np.cumsum(qdata.L_panel)])
    vertex_arcs  = np.append(panel_start_full[v_panel_idx], total_length)

    return arc, vertex_arcs, total_length


# ===========================================================================
# Main training + evaluation loop
# ===========================================================================

def run(cfg: dict, verbose: bool = True) -> dict:
    """
    Full SE-BINN pipeline on Koch(1) with u_exact = x²−y².

    Returns a rich dict containing scalar metrics AND the arrays needed for
    plotting (densities, InteriorResult objects, geometry, arc-length).
    """
    torch.manual_seed(cfg["seed"])

    t0 = time.perf_counter()

    # ------------------------------------------------------------------ #
    # 1. Geometry                                                          #
    # ------------------------------------------------------------------ #
    geom   = make_koch_geometry(n=1)
    P      = geom.vertices                     # (12, 2)
    panels = build_uniform_panels(P, n_per_edge=cfg["n_per_edge"])
    label_corner_ring_panels(panels, P)
    Npan   = len(panels)

    if verbose:
        n_corner = sum(p.is_corner for p in panels)
        n_ring   = sum(p.is_ring   for p in panels)
        print(f"Geometry: Koch(1) | vertices={geom.n_vertices} | panels={Npan}")
        print(f"  corner panels={n_corner} | ring panels={n_ring}")

    # ------------------------------------------------------------------ #
    # 2. Quadrature  (also serves as Nyström nodes for BEM)               #
    # ------------------------------------------------------------------ #
    qdata = build_panel_quadrature(panels, p=cfg["p_gl"])
    Yq_T  = qdata.Yq.T          # (Nq, 2)
    wq    = qdata.wq             # (Nq,)
    Nq    = qdata.n_quad

    arc, vertex_arcs, total_arc = _boundary_arclength(qdata, cfg["n_per_edge"])
    sort_idx = np.argsort(arc)   # indices that sort nodes by arc-length

    if verbose:
        print(f"Quadrature: Nq={Nq} | p={cfg['p_gl']}")
        print(f"  total arc-length={total_arc:.4f}")

    # ------------------------------------------------------------------ #
    # 3. BEM / Nyström reference                                           #
    # ------------------------------------------------------------------ #
    if verbose:
        print("\n=== Part A: BEM reference ===")

    nmat      = assemble_nystrom_matrix(qdata)
    f_bnd     = u_exact(Yq_T)
    bem_sol   = solve_bem(nmat, f_bnd,
                          tol=cfg["gmres_tol"],
                          max_iter=cfg["gmres_maxiter"])
    sigma_bem = bem_sol.sigma

    if verbose:
        print(f"BEM solve: flag={bem_sol.flag} | rel_res={bem_sol.rel_res:.3e} "
              f"| iters={bem_sol.n_iter} | direct={bem_sol.used_direct}")

    bem_out = reconstruct_interior(
        P=P, Yq=Yq_T, wq=wq, sigma=sigma_bem,
        n_grid=cfg["n_grid"], u_exact=u_exact,
    )
    if verbose:
        print(f"BEM interior: rel_L2={bem_out.rel_L2:.3e} | linf={bem_out.linf:.3e}")

    # ------------------------------------------------------------------ #
    # 4. SE-BINN setup                                                     #
    # ------------------------------------------------------------------ #
    if verbose:
        print("\n=== Part B: SE-BINN training ===")

    enrichment = SingularEnrichment(geom=geom, per_corner_gamma=False)

    w_panel = panel_loss_weights(
        panels,
        w_base=cfg["w_base"],
        w_corner=cfg["w_corner"],
        w_ring=cfg["w_ring"],
    )

    colloc  = build_collocation_points(panels, m_col_panel=cfg["m_col_base"])
    Nb      = colloc.n_colloc

    if verbose:
        print(f"Collocation: Nb={Nb}")

    op, op_diag = build_operator_state(
        colloc=colloc,
        qdata=qdata,
        enrichment=enrichment,
        g=u_exact,
        panel_weights=w_panel,
        eq_scale_mode=cfg["eq_scale_mode"],
        eq_scale_fixed=cfg.get("eq_scale_fixed", 1.0),
        dtype=torch.float64,
        device="cpu",
    )

    if verbose:
        print(f"Operator: mean|A|={op_diag['mean_abs_A_before']:.3e} "
              f"| eq_scale={op_diag['eq_scale']:.3e}")

    model = SEBINNModel(
        hidden_width=cfg["hidden_width"],
        n_hidden=cfg["n_hidden"],
        n_gamma=1,
        gamma_init=cfg["gamma_init"],
        dtype=torch.float64,
    )
    if verbose:
        print(f"Model: n_params={model.n_params()} | gamma_init={cfg['gamma_init']}")

    # ------------------------------------------------------------------ #
    # 5. Adam                                                              #
    # ------------------------------------------------------------------ #
    adam_cfg = AdamConfig(
        phase_iters=cfg["adam_iters"],
        phase_lrs=cfg["adam_lrs"],
        log_every=cfg["log_every"],
    )
    adam_res = run_adam_phases(model, op, adam_cfg, verbose=verbose)

    sigma_s_Yq = enrichment.precompute(Yq_T)
    with torch.no_grad():
        sigma_adam = model(
            torch.tensor(Yq_T,      dtype=torch.float64),
            torch.tensor(sigma_s_Yq, dtype=torch.float64),
        ).numpy()

    adam_out = reconstruct_interior(
        P=P, Yq=Yq_T, wq=wq, sigma=sigma_adam,
        n_grid=cfg["n_grid"], u_exact=u_exact,
    )
    if verbose:
        print(f"Adam-end: rel_L2={adam_out.rel_L2:.3e} | linf={adam_out.linf:.3e} "
              f"| gamma={model.gamma_value():.6f}")

    # ------------------------------------------------------------------ #
    # 6. L-BFGS                                                            #
    # ------------------------------------------------------------------ #
    lbfgs_res = None
    if cfg["use_lbfgs"]:
        lbfgs_cfg = LBFGSConfig(
            max_iters=cfg["lbfgs_max_iters"],
            grad_tol=cfg["lbfgs_grad_tol"],
            step_tol=cfg["lbfgs_step_tol"],
            memory=cfg["lbfgs_memory"],
            log_every=cfg["lbfgs_log_every"],
            alpha0=cfg["lbfgs_alpha0"],
            alpha_fallback=cfg["lbfgs_alpha_fb"],
            armijo_c1=cfg["lbfgs_armijo_c1"],
            backtrack_beta=cfg["lbfgs_beta"],
            max_backtrack=cfg["lbfgs_max_bt"],
        )
        lbfgs_res = run_lbfgs(model, op, lbfgs_cfg, verbose=verbose)

    # ------------------------------------------------------------------ #
    # 7. Final density and reconstruction                                  #
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        sigma_final = model(
            torch.tensor(Yq_T,      dtype=torch.float64),
            torch.tensor(sigma_s_Yq, dtype=torch.float64),
        ).numpy()

    final_out = reconstruct_interior(
        P=P, Yq=Yq_T, wq=wq, sigma=sigma_final,
        n_grid=cfg["n_grid"], u_exact=u_exact,
    )

    t_total = time.perf_counter() - t0

    if verbose:
        print("\n=== Interior error summary ===")
        print(f"BEM reference  : rel_L2={bem_out.rel_L2:.3e} | linf={bem_out.linf:.3e}")
        print(f"SE-BINN Adam   : rel_L2={adam_out.rel_L2:.3e} | linf={adam_out.linf:.3e}")
        print(f"SE-BINN final  : rel_L2={final_out.rel_L2:.3e} | linf={final_out.linf:.3e}")
        print(f"gamma (final)  : {model.gamma_value():.6f}")
        print(f"Total wall time: {t_total:.1f}s")

    density_rel_diff = (
        float(np.linalg.norm(sigma_final - sigma_bem))
        / max(float(np.linalg.norm(sigma_bem)), 1e-14)
    )

    return dict(
        # --- scalar metrics ---
        bem_rel_L2       = bem_out.rel_L2,
        bem_linf         = bem_out.linf,
        adam_rel_L2      = adam_out.rel_L2,
        adam_linf        = adam_out.linf,
        final_rel_L2     = final_out.rel_L2,
        final_linf       = final_out.linf,
        gamma_final      = model.gamma_value(),
        density_rel_diff = density_rel_diff,
        wall_time        = t_total,

        # --- loss histories ---
        adam_loss_hist   = adam_res.loss_hist,
        lbfgs_loss_hist  = lbfgs_res.loss_hist if lbfgs_res else [],

        # --- density arrays (sorted by arc-length for plotting) ---
        arc              = arc[sort_idx],
        sigma_bem        = sigma_bem[sort_idx],
        sigma_adam       = sigma_adam[sort_idx],
        sigma_final      = sigma_final[sort_idx],
        vertex_arcs      = vertex_arcs,
        singular_corner_indices = list(geom.singular_corner_indices),

        # --- InteriorResult objects for grid plots ---
        bem_out          = bem_out,
        adam_out         = adam_out,
        final_out        = final_out,

        # --- geometry ---
        P                = P,
        geom             = geom,
    )


# ===========================================================================
# Plotting
# ===========================================================================

def plot_results(results: dict, outdir: str) -> None:
    """
    Produce and save all diagnostic figures.

    Figures saved
    -------------
    fig1_density.pdf/png     — σ_BEM vs σ_SE-BINN (Adam + final) vs arc-length
    fig2_interior_bem.pdf/png   — BEM interior: u_num | u_exact | log10|error|
    fig3_interior_sebinn.pdf/png — SE-BINN final interior: same triplet
    fig4_loss.pdf/png         — Adam + L-BFGS loss history
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    os.makedirs(outdir, exist_ok=True)

    P            = results["P"]
    arc          = results["arc"]
    vertex_arcs  = results["vertex_arcs"]
    sing_idx     = results["singular_corner_indices"]
    total_arc    = vertex_arcs[-1]

    # ------------------------------------------------------------------
    # Helper: draw polygon boundary on an axis
    # ------------------------------------------------------------------
    def _draw_polygon(ax, P, color="k", lw=0.8):
        Pc = np.vstack([P, P[0]])
        ax.plot(Pc[:, 0], Pc[:, 1], color=color, lw=lw)

    # ------------------------------------------------------------------
    # Figure 1: Density comparison
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    sigma_bem   = results["sigma_bem"]
    sigma_adam  = results["sigma_adam"]
    sigma_final = results["sigma_final"]

    # Top: full density
    ax = axes[0]
    ax.plot(arc, sigma_bem,   color="k",        lw=1.4, label="BEM (reference)")
    ax.plot(arc, sigma_adam,  color="#1f77b4",  lw=1.0, ls="--", alpha=0.8,
            label=f"SE-BINN Adam  (rel-L2={results['adam_rel_L2']:.2e})")
    ax.plot(arc, sigma_final, color="#d62728",  lw=1.1, alpha=0.9,
            label=f"SE-BINN final (rel-L2={results['final_rel_L2']:.2e})")

    # Shade corner regions
    n_vert = len(P)
    for vi in range(n_vert):
        va  = vertex_arcs[vi]
        vb  = vertex_arcs[vi + 1] if vi + 1 < len(vertex_arcs) else total_arc
        is_sing = vi in sing_idx
        color = "#ffcccc" if is_sing else "#e8f4e8"
        ax.axvspan(va, vb, alpha=0.25, color=color, linewidth=0)

    # Vertical lines at vertices
    for vi, va in enumerate(vertex_arcs[:-1]):
        is_sing = vi in sing_idx
        ax.axvline(va, color="#aa0000" if is_sing else "#448844",
                   lw=0.8, ls=":", alpha=0.7)

    ax.set_ylabel(r"$\sigma(s)$", fontsize=12)
    ax.set_title(
        f"Boundary density — Koch(1),  $u = x^2 - y^2$\n"
        r"$\gamma_\mathrm{final}$" + f" = {results['gamma_final']:.4f}  |  "
        f"density rel-diff = {results['density_rel_diff']:.2e}",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, lw=0.3, alpha=0.5)

    # Bottom: pointwise error |σ_final - σ_BEM|
    ax2 = axes[1]
    ax2.semilogy(arc, np.abs(sigma_final - sigma_bem), color="#d62728",
                 lw=0.9, label=r"$|\sigma_\mathrm{final} - \sigma_\mathrm{BEM}|$")
    ax2.semilogy(arc, np.abs(sigma_adam - sigma_bem),  color="#1f77b4",
                 lw=0.9, ls="--", alpha=0.7,
                 label=r"$|\sigma_\mathrm{Adam} - \sigma_\mathrm{BEM}|$")

    for vi, va in enumerate(vertex_arcs[:-1]):
        is_sing = vi in sing_idx
        ax2.axvline(va, color="#aa0000" if is_sing else "#448844",
                    lw=0.8, ls=":", alpha=0.7)

    ax2.set_xlabel("Arc-length $s$", fontsize=12)
    ax2.set_ylabel(r"$|\sigma - \sigma_\mathrm{BEM}|$", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, lw=0.3, alpha=0.5, which="both")

    # Custom legend patches for background shading
    legend_patches = [
        Patch(facecolor="#ffcccc", alpha=0.5, label="reentrant corner edge (singular)"),
        Patch(facecolor="#e8f4e8", alpha=0.5, label="convex corner edge"),
    ]
    axes[0].legend(
        handles=axes[0].get_legend_handles_labels()[0] + legend_patches,
        labels=axes[0].get_legend_handles_labels()[1]
        + [p.get_label() for p in legend_patches],
        fontsize=8, loc="upper right",
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig1_density.{ext}"), dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  saved fig1_density  → {outdir}")

    # ------------------------------------------------------------------
    # Figures 2–3: Interior solution triplets
    # ------------------------------------------------------------------
    def _interior_triplet(res, title_prefix, fname):
        """3-column panel: u_num | u_exact | log10|error|."""
        xv, yv = res.xv, res.yv

        # Shared colour limits for u panels
        u_all  = res.Ugrid[~np.isnan(res.Ugrid)]
        ue_all = res.Uexgrid[~np.isnan(res.Uexgrid)]
        vmin   = min(u_all.min(), ue_all.min())
        vmax   = max(u_all.max(), ue_all.max())

        err_abs = np.abs(res.Egrid)
        # floor at 1e-16 before log
        log_err = np.full_like(res.Egrid, np.nan)
        mask    = ~np.isnan(res.Egrid)
        log_err[mask] = np.log10(np.maximum(err_abs[mask], 1e-16))

        fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))

        def _imshow(ax, data, cmap, label, **kw):
            im = ax.imshow(
                data, origin="lower",
                extent=[xv[0], xv[-1], yv[0], yv[-1]],
                cmap=cmap, **kw,
            )
            _draw_polygon(ax, P)
            ax.set_aspect("equal")
            ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
            ax.set_title(label, fontsize=11)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        _imshow(axes2[0], res.Ugrid,   "RdBu_r",    r"$u_\mathrm{num}$",
                vmin=vmin, vmax=vmax)
        _imshow(axes2[1], res.Uexgrid, "RdBu_r",    r"$u_\mathrm{exact}$",
                vmin=vmin, vmax=vmax)
        _imshow(axes2[2], log_err,     "hot_r",
                r"$\log_{10}|u_\mathrm{num} - u_\mathrm{exact}|$")

        rel_l2 = res.rel_L2
        linf   = res.linf
        fig2.suptitle(
            f"{title_prefix}   rel-L2 = {rel_l2:.2e}   L∞ = {linf:.2e}",
            fontsize=12,
        )
        fig2.tight_layout()
        for ext in ("pdf", "png"):
            fig2.savefig(os.path.join(outdir, f"{fname}.{ext}"), dpi=150,
                         bbox_inches="tight")
        plt.close(fig2)
        print(f"  saved {fname}  → {outdir}")

    _interior_triplet(
        results["bem_out"],
        "BEM Nyström (reference)",
        "fig2_interior_bem",
    )
    _interior_triplet(
        results["final_out"],
        f"SE-BINN final  (γ = {results['gamma_final']:.4f})",
        "fig3_interior_sebinn",
    )

    # ------------------------------------------------------------------
    # Figure 4: Loss history
    # ------------------------------------------------------------------
    adam_hist  = results["adam_loss_hist"]
    lbfgs_hist = results["lbfgs_loss_hist"]
    n_adam     = len(adam_hist)
    n_lbfgs    = len(lbfgs_hist)

    fig4, ax4 = plt.subplots(figsize=(9, 4))
    iters_a = np.arange(1, n_adam + 1)
    ax4.semilogy(iters_a, adam_hist, color="#1f77b4", lw=1.2, label="Adam")

    if n_lbfgs > 0:
        iters_l = n_adam + np.arange(1, n_lbfgs + 1)
        ax4.semilogy(iters_l, lbfgs_hist, color="#d62728", lw=1.2, label="L-BFGS")
        ax4.axvline(n_adam, color="k", ls="--", lw=0.9, alpha=0.6,
                    label="Adam → L-BFGS")

    ax4.set_xlabel("Iteration", fontsize=12)
    ax4.set_ylabel("Weighted BIE residual loss", fontsize=12)
    ax4.set_title("Training loss history — Koch(1)", fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, lw=0.3, alpha=0.5, which="both")
    fig4.tight_layout()
    for ext in ("pdf", "png"):
        fig4.savefig(os.path.join(outdir, f"fig4_loss.{ext}"), dpi=150,
                     bbox_inches="tight")
    plt.close(fig4)
    print(f"  saved fig4_loss     → {outdir}")


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SE-BINN experiment 1: Koch(1), u=x²−y²"
    )
    parser.add_argument("--no-lbfgs",    action="store_true",
                        help="Skip L-BFGS phase (Adam only)")
    parser.add_argument("--no-plot",     action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--adam-iters",  type=int, default=None,
                        help="Override total Adam iters (single phase)")
    parser.add_argument("--lbfgs-iters", type=int, default=3500)
    parser.add_argument("--n-per-edge",  type=int, default=12)
    parser.add_argument("--seed",        type=int, default=0)
    args = parser.parse_args()

    cfg = default_cfg()
    cfg["seed"]            = args.seed
    cfg["lbfgs_max_iters"] = args.lbfgs_iters
    cfg["use_lbfgs"]       = not args.no_lbfgs
    cfg["n_per_edge"]      = args.n_per_edge
    # Only override multi-phase schedule if explicitly requested
    if args.adam_iters is not None:
        cfg["adam_iters"] = [args.adam_iters]
        cfg["adam_lrs"]   = [cfg["adam_lrs"][-1]]

    results = run(cfg, verbose=True)

    if not args.no_plot:
        figures_dir = os.path.join(_HERE, "figures")
        print(f"\nGenerating figures → {figures_dir}")
        plot_results(results, figures_dir)
        print("Done.")
