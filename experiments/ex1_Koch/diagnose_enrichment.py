"""
Diagnostic: how much of σ_BEM can the enrichment basis S capture?

Mathematical background (theory/sigma_s_mellin_derivation.md)
-------------------------------------------------------------
By Mellin analysis (von Petersdorff–Stephan 2014), the BEM density has a
leading singularity at every reentrant corner of the form

    σ(y) = a_0^(c) · r_c^{π/ω − 1}  +  σ_reg(y),

where a_0^(c) is a corner-specific amplitude depending on g and ω.
For g = x² − y² on Koch(1) (ω = 4π/3, 6 reentrant corners):

  - Each a_0^(c) ≠ 0  (V⁻¹ creates singularities even from smooth data)
  - The 6 corners split into 3 pairs under 2-fold symmetry of g,
    so a single shared γ must compromise across corners.

This script quantifies:
  1. The individual γ_c* = argmin ||σ_BEM − S @ γ||²  (per-corner)
  2. The shared γ* (summed σ_s column)
  3. Energy fractions: how much of σ_BEM is captured by each enrichment

Geometry  : Koch(1), n_per_edge=12, p_gl=16
Benchmark : g(x,y) = x² − y²
Output    : experiments/ex1_Koch/outputs/
"""

import sys
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from src.boundary.polygon import make_koch_geometry
from src.boundary.panels import build_uniform_panels, label_corner_ring_panels
from src.quadrature.panel_quad import build_panel_quadrature
from src.quadrature.nystrom import assemble_nystrom_matrix, solve_bem
from src.singular.enrichment import SingularEnrichment


OUTDIR = os.path.join(_HERE, "outputs")
N_PER_EDGE = 12
P_GL       = 16


# ---------------------------------------------------------------------------
# Boundary data
# ---------------------------------------------------------------------------

def g(xy: np.ndarray) -> np.ndarray:
    return xy[:, 0] ** 2 - xy[:, 1] ** 2


# ---------------------------------------------------------------------------
# Arc-length helper (re-used from run.py)
# ---------------------------------------------------------------------------

def _arclength(qdata, n_per_edge):
    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc  = panel_start[qdata.pan_id] + qdata.s_on_panel
    Npan = qdata.n_panels
    Nv   = Npan // n_per_edge
    panel_start_full = np.concatenate([[0.0], np.cumsum(qdata.L_panel)])
    v_arcs = np.append(panel_start_full[np.arange(Nv) * n_per_edge],
                       float(qdata.L_panel.sum()))
    return arc, v_arcs


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def diagnose():
    os.makedirs(OUTDIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Geometry + quadrature                                             #
    # ------------------------------------------------------------------ #
    geom   = make_koch_geometry(n=1)
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=N_PER_EDGE)
    label_corner_ring_panels(panels, P)

    qdata = build_panel_quadrature(panels, p=P_GL)
    Yq_T  = qdata.Yq.T   # (Nq, 2)
    wq    = qdata.wq      # (Nq,)
    Nq    = qdata.n_quad

    arc, v_arcs = _arclength(qdata, N_PER_EDGE)
    sort_idx    = np.argsort(arc)
    arc_s       = arc[sort_idx]

    sing_idx = list(geom.singular_corner_indices)  # 0-indexed vertex IDs
    n_sing   = len(sing_idx)

    print(f"Koch(1): Nq={Nq}, reentrant corners={n_sing} at vertex indices {sing_idx}")
    print(f"  ω = 4π/3 = {4*np.pi/3:.6f}  |  α - 1 = π/ω - 1 = {np.pi/(4*np.pi/3) - 1:.6f}")

    # ------------------------------------------------------------------ #
    # 2. BEM reference solve                                               #
    # ------------------------------------------------------------------ #
    nmat      = assemble_nystrom_matrix(qdata)
    f_bnd     = g(Yq_T)
    bem_sol   = solve_bem(nmat, f_bnd, tol=1e-12, max_iter=300)
    sigma_bem = bem_sol.sigma

    norm_bem = float(np.linalg.norm(sigma_bem))
    print(f"\nBEM solve: flag={bem_sol.flag}  ||σ_BEM|| = {norm_bem:.6e}")

    # ------------------------------------------------------------------ #
    # 3. Build S matrix: per-corner enrichment basis                       #
    # ------------------------------------------------------------------ #
    # S[:, c] = σ_s^(c)(Yq)  (no cutoff — we want the pure r^α behaviour)
    enrichment_pc = SingularEnrichment(geom=geom, per_corner_gamma=True,
                                       use_cutoff=False)
    S = enrichment_pc.evaluate_per_corner(Yq_T)   # (Nq, n_sing)

    # Shared enrichment: sum all corner columns
    enrichment_sh = SingularEnrichment(geom=geom, per_corner_gamma=False,
                                       use_cutoff=False)
    s_shared = enrichment_sh.evaluate(Yq_T)        # (Nq,)  = S.sum(axis=1)

    print(f"\nEnrichment basis  S  shape: {S.shape}")
    print(f"  ||S[:,c]|| per corner: "
          + "  ".join(f"{np.linalg.norm(S[:,c]):.3e}" for c in range(n_sing)))

    # ------------------------------------------------------------------ #
    # 4. Least-squares fit: per-corner                                     #
    # ------------------------------------------------------------------ #
    # γ* = argmin ||σ_BEM - S γ||²
    gamma_pc, res_pc, rank_pc, sv_pc = np.linalg.lstsq(S, sigma_bem, rcond=None)
    sigma_sing_pc   = S @ gamma_pc                   # fitted singular part
    sigma_smooth_pc = sigma_bem - sigma_sing_pc      # remainder for σ_w

    res_norm_pc  = float(np.linalg.norm(sigma_smooth_pc))
    rel_res_pc   = res_norm_pc / norm_bem
    energy_pc    = 1.0 - (res_norm_pc / norm_bem) ** 2

    # ------------------------------------------------------------------ #
    # 5. Least-squares fit: shared γ                                       #
    # ------------------------------------------------------------------ #
    gamma_sh_arr, _, _, _ = np.linalg.lstsq(
        s_shared[:, None], sigma_bem, rcond=None
    )
    gamma_sh      = float(gamma_sh_arr[0])
    sigma_sing_sh  = gamma_sh * s_shared
    sigma_smooth_sh = sigma_bem - sigma_sing_sh

    res_norm_sh  = float(np.linalg.norm(sigma_smooth_sh))
    rel_res_sh   = res_norm_sh / norm_bem
    energy_sh    = 1.0 - (res_norm_sh / norm_bem) ** 2

    # ------------------------------------------------------------------ #
    # 6. Print report                                                       #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 62)
    print("  ENRICHMENT DIAGNOSTIC REPORT")
    print("  Koch(1),  g(x,y) = x² − y²")
    print("=" * 62)

    print(f"\n  ||σ_BEM||                   = {norm_bem:.6e}")
    print(f"  ||σ_BEM||_∞                 = {np.max(np.abs(sigma_bem)):.6e}")

    print(f"\n  --- Per-corner fit (6 independent γ_c) ---")
    for c in range(n_sing):
        v_idx = sing_idx[c]
        print(f"    γ_{c}*  (vertex {v_idx:2d}) = {gamma_pc[c]:+.6f}")
    print(f"  Rank of S                   = {rank_pc}")
    print(f"  Singular values of S:  "
          + "  ".join(f"{sv:.2e}" for sv in sv_pc))
    print(f"  ||σ_BEM - S γ*||            = {res_norm_pc:.6e}")
    print(f"  Relative residual           = {rel_res_pc:.4f}  ({rel_res_pc*100:.2f}%)")
    print(f"  Energy captured             = {energy_pc:.6f}  ({energy_pc*100:.4f}%)")

    print(f"\n  --- Shared γ fit (1 parameter) ---")
    print(f"  γ*  (shared)                = {gamma_sh:+.6f}")
    print(f"  ||σ_BEM - γ* s_shared||     = {res_norm_sh:.6e}")
    print(f"  Relative residual           = {rel_res_sh:.4f}  ({rel_res_sh*100:.2f}%)")
    print(f"  Energy captured             = {energy_sh:.6f}  ({energy_sh*100:.4f}%)")

    print(f"\n  --- Per-corner vs shared comparison ---")
    print(f"  Energy gain from per-corner = {(energy_pc - energy_sh)*100:.4f}% additional")
    print(f"  γ_c spread: min={gamma_pc.min():+.6f}  max={gamma_pc.max():+.6f}"
          f"  std={gamma_pc.std():.6f}")
    print(f"  Ratio max/min |γ_c|         = "
          f"{np.max(np.abs(gamma_pc)) / max(np.min(np.abs(gamma_pc)), 1e-14):.3f}")

    print("\n" + "=" * 62)

    # ------------------------------------------------------------------ #
    # 7. Figures                                                           #
    # ------------------------------------------------------------------ #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def _draw_vertex_lines(ax, v_arcs, sing_idx, n_vert):
        for vi in range(n_vert):
            is_sing = vi in sing_idx
            ax.axvline(v_arcs[vi],
                       color="#aa0000" if is_sing else "#448844",
                       lw=0.7, ls=":", alpha=0.7)

    def _shade_arcs(ax, v_arcs, sing_idx, n_vert):
        for vi in range(n_vert):
            va = v_arcs[vi]; vb = v_arcs[vi + 1]
            c  = "#ffcccc" if vi in sing_idx else "#e8f4e8"
            ax.axvspan(va, vb, alpha=0.22, color=c, linewidth=0)

    n_vert  = len(P)
    arc_s   = arc[sort_idx]
    bem_s   = sigma_bem[sort_idx]
    sing_s  = sigma_sing_pc[sort_idx]
    smth_s  = sigma_smooth_pc[sort_idx]

    # --- Figure 1: σ_BEM decomposition (per-corner fit) ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    _shade_arcs(ax, v_arcs, sing_idx, n_vert)
    ax.plot(arc_s, bem_s,   "k",          lw=1.4, label=r"$\sigma_\mathrm{BEM}$")
    ax.plot(arc_s, sing_s,  "#d62728",    lw=1.1, alpha=0.85,
            label=r"$S\,\gamma^*$ (fitted singular part, per-corner)")
    ax.plot(arc_s, smth_s,  "#1f77b4",    lw=1.1, alpha=0.85,
            label=r"$\sigma_\mathrm{BEM} - S\,\gamma^*$ (smooth remainder $\sigma_w$)")
    _draw_vertex_lines(ax, v_arcs, sing_idx, n_vert)
    ax.set_ylabel(r"$\sigma(s)$", fontsize=12)
    ax.set_title(
        r"$\sigma_\mathrm{BEM}$ decomposition — Koch(1),  $g = x^2 - y^2$"
        f"\nPer-corner: energy captured = {energy_pc*100:.3f}%   "
        f"rel. residual = {rel_res_pc*100:.2f}%",
        fontsize=11,
    )
    legend_patches = [
        Patch(facecolor="#ffcccc", alpha=0.5, label="reentrant corner edge (singular)"),
        Patch(facecolor="#e8f4e8", alpha=0.5, label="convex corner edge"),
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_patches,
              labels=ax.get_legend_handles_labels()[1]
              + [p.get_label() for p in legend_patches],
              fontsize=8, loc="upper right")
    ax.grid(True, lw=0.3, alpha=0.5)

    # Bottom: γ_c bar chart as inset + pointwise singular part
    ax2 = axes[1]
    _shade_arcs(ax2, v_arcs, sing_idx, n_vert)
    ax2.semilogy(arc_s, np.abs(sing_s),  "#d62728", lw=1.0,
                 label=r"$|S\,\gamma^*|$  (per-corner)")
    ax2.semilogy(arc_s, np.abs(gamma_sh * s_shared[sort_idx]),
                 "#ff7f0e", lw=1.0, ls="--", alpha=0.8,
                 label=r"$|\gamma^*\, s_\mathrm{shared}|$  (shared)")
    ax2.semilogy(arc_s, np.abs(smth_s),  "#1f77b4", lw=1.0, alpha=0.7,
                 label=r"$|\sigma_\mathrm{BEM} - S\,\gamma^*|$ (smooth residual)")
    _draw_vertex_lines(ax2, v_arcs, sing_idx, n_vert)
    ax2.set_xlabel("Arc-length $s$", fontsize=12)
    ax2.set_ylabel("Magnitude  (log scale)", fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, lw=0.3, alpha=0.5, which="both")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(OUTDIR, f"diag_decomposition.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved diag_decomposition  → {OUTDIR}")

    # --- Figure 2: per-corner γ_c* bar chart ---
    fig2, ax3 = plt.subplots(figsize=(7, 4))
    colors_bar = ["#d62728" if vi in sing_idx else "#1f77b4"
                  for vi in sing_idx]
    bars = ax3.bar(range(n_sing), gamma_pc, color=colors_bar, alpha=0.85,
                   edgecolor="k", linewidth=0.6)
    ax3.axhline(gamma_sh, color="#ff7f0e", ls="--", lw=1.5,
                label=f"shared γ* = {gamma_sh:.5f}")
    ax3.axhline(0, color="k", lw=0.6)
    ax3.set_xticks(range(n_sing))
    ax3.set_xticklabels([f"corner {c}\n(vertex {sing_idx[c]})"
                         for c in range(n_sing)], fontsize=9)
    ax3.set_ylabel(r"$\gamma_c^*$", fontsize=12)
    ax3.set_title(
        r"Optimal per-corner $\gamma_c^*$ — Koch(1),  $g = x^2 - y^2$"
        f"\nstd = {gamma_pc.std():.5f}   "
        f"max/min |γ| = {np.max(np.abs(gamma_pc))/max(np.min(np.abs(gamma_pc)),1e-14):.2f}",
        fontsize=11,
    )
    ax3.legend(fontsize=9)
    ax3.grid(True, axis="y", lw=0.3, alpha=0.5)
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(OUTDIR, f"diag_gamma_bars.{ext}")
        fig2.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  saved diag_gamma_bars      → {OUTDIR}")

    # --- Figure 3: close-up near one reentrant corner ---
    # Find the corner with the largest |γ_c*|
    c_max    = int(np.argmax(np.abs(gamma_pc)))
    v_max    = sing_idx[c_max]
    arc_ctr  = v_arcs[v_max]           # arc-length at that vertex
    half_win = 0.3                     # ±0.3 around corner

    mask_win = np.abs(arc_s - arc_ctr) < half_win

    fig3, ax4 = plt.subplots(figsize=(8, 4))
    ax4.plot(arc_s[mask_win] - arc_ctr, bem_s[mask_win],
             "k", lw=1.6, label=r"$\sigma_\mathrm{BEM}$")
    ax4.plot(arc_s[mask_win] - arc_ctr, sing_s[mask_win],
             "#d62728", lw=1.4, alpha=0.85,
             label=r"$S\,\gamma^*$ (per-corner singular fit)")
    ax4.plot(arc_s[mask_win] - arc_ctr, smth_s[mask_win],
             "#1f77b4", lw=1.2, alpha=0.85,
             label=r"smooth remainder $\sigma_w = \sigma_\mathrm{BEM} - S\,\gamma^*$")
    ax4.axvline(0, color="#aa0000", ls="--", lw=1.0, alpha=0.7,
                label=f"reentrant corner (vertex {v_max})")
    ax4.set_xlabel(r"Arc-length relative to corner,  $s - s_c$", fontsize=11)
    ax4.set_ylabel(r"$\sigma(s)$", fontsize=12)
    ax4.set_title(
        f"Close-up: corner {c_max} (vertex {v_max}),  γ_{c_max}* = {gamma_pc[c_max]:+.5f}\n"
        r"$r^{-1/4}$ blow-up vs smooth remainder",
        fontsize=11,
    )
    ax4.legend(fontsize=9)
    ax4.grid(True, lw=0.3, alpha=0.5)
    fig3.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(OUTDIR, f"diag_corner_closeup.{ext}")
        fig3.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  saved diag_corner_closeup  → {OUTDIR}")

    return dict(
        gamma_pc=gamma_pc,
        gamma_sh=gamma_sh,
        energy_pc=energy_pc,
        energy_sh=energy_sh,
        rel_res_pc=rel_res_pc,
        rel_res_sh=rel_res_sh,
        norm_bem=norm_bem,
        sigma_bem=sigma_bem,
        sigma_sing_pc=sigma_sing_pc,
        sigma_smooth_pc=sigma_smooth_pc,
    )


if __name__ == "__main__":
    diagnose()
