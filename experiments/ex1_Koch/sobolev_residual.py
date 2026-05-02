"""
Sobolev (H¹) residual loss for the single-layer BIE.

Mathematical motivation
-----------------------
The standard loss ||Vσ−g||²_L² weights Fourier mode n by |σ_n(V)|² ~ 1/n²,
because V is a smoothing operator of order -1 (symbol ~ 1/|n|).  High-frequency
corner modes receive exponentially weaker gradient signal, causing the plateau.

The tangential derivative D_s has symbol ~ i|n| (order +1), so the composition
D_s V has symbol ~ 1:

    (D_s V)(n) ~ i|n| · 1/|n| = i  (order 0 pseudodifferential operator).

Therefore cond_svd(D_s V_h) = O(1), and the H¹ seminorm of the residual

    L_H1 = ||D_s(Vσ − g)||²

has Hessian condition O(1) — no V⁻¹, no hypersingular W̃, no Calderón identity.

Combined Sobolev loss:
    L(σ; α) = ||Vσ − g||² + α ||D_s(Vσ − g)||²
            = ||[V; √α D_s V] σ − [g; √α D_s g]||²

with Hessian cond = cond_svd([V; √α D_s V])² — interpolating between
cond(V)² ≈ 1.8×10⁸ at α=0 and cond(D_s V)² at α→∞.

Cases compared (same 4×80 tanh network, Koch(1), g=x²−y², no enrichment):
  A: Standard          L = ||Vσ − g||²                       (baseline)
  B: Left V⁻¹          L = ||σ − σ_BEM||²                    (gold standard)
  C: H¹ combined       L = ||Vσ−g||² + α ||D_s(Vσ−g)||²     (best α from sweep)
  D: Right V⁻¹         σ = V⁻¹ρ, L = ||ρ − g||²             (reference)
  E: Pure H¹           L = ||D_s(Vσ − g)||²                  (α → ∞)

Geometry: Koch(1), g(x,y) = x²−y², n_per_edge=12, p=16.
"""

from __future__ import annotations

import sys
import os
import time
import copy

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from src.boundary.polygon import make_koch_geometry
from src.boundary.panels import build_uniform_panels, label_corner_ring_panels
from src.quadrature.panel_quad import build_panel_quadrature
from src.quadrature.nystrom import assemble_nystrom_matrix, solve_bem
from src.quadrature.tangential_derivative import build_tangential_derivative_matrix
from src.models.sigma_w_net import build_sigma_w_network
from src.reconstruction.interior import reconstruct_interior

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED         = 0
N_PER_EDGE   = 12
P_GL         = 16
HIDDEN_WIDTH = 80
N_HIDDEN     = 4
LR_SCHEDULE  = [(1000, 1e-3), (1000, 3e-4), (1000, 1e-4)]
N_LBFGS      = 15000
LBFGS_MEMORY = 30
LOG_EVERY    = 200
N_GRID_FINAL = 201

SWEEP_LR     = [(500, 1e-3), (500, 1e-4)]
SWEEP_LBFGS  = 3000

ALPHA_SWEEP  = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

COLORS = {
    "A": "#1f77b4", "B": "#d62728", "C": "#2ca02c",
    "D": "#ff7f0e", "E": "#9467bd", "BEM": "black",
}
LWIDTH = 2.0


# ---------------------------------------------------------------------------
# Boundary data
# ---------------------------------------------------------------------------

def g_fn(xy: np.ndarray) -> np.ndarray:
    """Dirichlet data g(x,y) = x² − y² (harmonic polynomial)."""
    return xy[:, 0] ** 2 - xy[:, 1] ** 2


def u_exact_fn(xy: np.ndarray) -> np.ndarray:
    return xy[:, 0] ** 2 - xy[:, 1] ** 2


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup():
    geom   = make_koch_geometry(n=1)
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=N_PER_EDGE)
    label_corner_ring_panels(panels, P)
    qdata  = build_panel_quadrature(panels, p=P_GL)
    nmat   = assemble_nystrom_matrix(qdata)

    Yq_T      = qdata.Yq.T          # (Nq, 2)
    wq        = qdata.wq            # (Nq,)
    g_values  = g_fn(Yq_T)         # (Nq,)
    sigma_BEM = solve_bem(nmat, g_values).sigma
    V_h       = nmat.V              # (Nq, Nq)
    V_inv     = np.linalg.inv(V_h)
    Nq        = len(sigma_BEM)

    svs    = np.linalg.svd(V_h, compute_uv=False)
    cond_V = float(svs[0] / svs[-1])

    print(f"  Koch(1): N_panels={qdata.n_panels}, N_quad={Nq}")
    print(f"  cond_svd(V_h) = {cond_V:.0f}  (Hessian cond ≈ {cond_V**2:.2e})")

    # Arc-length for plotting
    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc         = panel_start[qdata.pan_id] + qdata.s_on_panel
    sort_idx    = np.argsort(arc)

    # Reentrant corner positions (odd vertices of Koch(1))
    corner_arcs = []
    for vi in range(1, len(P), 2):
        c = P[vi]
        dists = np.linalg.norm(Yq_T - c[None, :], axis=1)
        corner_arcs.append(arc[np.argmin(dists)])
    corner_arcs = sorted(corner_arcs)

    return dict(
        P=P, Yq_T=Yq_T, wq=wq, qdata=qdata,
        g_values=g_values, sigma_BEM=sigma_BEM,
        V_h=V_h, V_inv=V_inv, svs=svs, cond_V=cond_V, Nq=Nq,
        arc=arc, sort_idx=sort_idx, corner_arcs=corner_arcs,
    )


# ---------------------------------------------------------------------------
# D_h validation
# ---------------------------------------------------------------------------

def validate_Dh(D_h: np.ndarray, qdata, arc: np.ndarray) -> None:
    """
    Two checks:
      1. D_h @ arc ≈ 1 everywhere (arc-length derivative is 1; exact for p≥2)
      2. D_h @ sin(2πs/L) ≈ (2π/L) cos(2πs/L) (smooth function, p=16 → < 1e-10)
    """
    Nq      = len(arc)
    L_total = arc.max() + qdata.L_panel[-1] / 2  # rough total perimeter

    # Test 1: d(arc)/ds = 1
    D_arc = D_h @ arc
    ones  = np.ones(Nq)
    err1  = np.linalg.norm(D_arc - ones) / np.linalg.norm(ones)
    print(f"  D_h @ arc ≈ 1:  rel_err = {err1:.3e}  (should be ≈ 0)")

    # Test 2: d/ds sin(2πs/L) = (2π/L) cos(2πs/L)
    L_total = np.sum(qdata.L_panel)
    phi     = np.sin(2 * np.pi * arc / L_total)
    Dphi    = D_h @ phi
    Dphi_ex = (2 * np.pi / L_total) * np.cos(2 * np.pi * arc / L_total)
    err2    = np.linalg.norm(Dphi - Dphi_ex) / np.linalg.norm(Dphi_ex)
    print(f"  D_h @ sin(2πs/L) ≈ (2π/L)cos:  rel_err = {err2:.3e}  (should be < 1e-10)")


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def spectral_analysis(V_h: np.ndarray, DV: np.ndarray) -> float:
    """Print condition numbers; return best α from theory (cond table)."""
    Nq = V_h.shape[0]
    sv_V  = np.linalg.svd(V_h,  compute_uv=False)
    sv_DV = np.linalg.svd(DV, compute_uv=False)
    cond_V  = sv_V[0]  / sv_V[-1]
    cond_DV = sv_DV[0] / sv_DV[-1]

    print(f"\n  Reference:")
    print(f"    cond_svd(V)      = {cond_V:.1f}  "
          f"→ Hessian cond(||Vσ−g||²) ≈ {cond_V**2:.2e}")
    print(f"    cond_svd(D_s V)  = {cond_DV:.1f}  "
          f"→ Hessian cond(||D_s Vσ−D_s g||²) ≈ {cond_DV**2:.2e}")
    print(f"    cond_svd(I)      = 1.0   (V⁻¹ preconditioner, ideal)")

    alphas_plot = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    print(f"\n  {'α':>8s} | {'cond_svd([V;√α·DV])':>22s} | {'Hessian cond':>14s}")
    print(f"  {'-'*8}-+-{'-'*22}-+-{'-'*14}")

    best_alpha = None
    best_cond  = cond_V
    for alpha in alphas_plot:
        stacked = np.vstack([V_h, np.sqrt(alpha) * DV])
        sv      = np.linalg.svd(stacked, compute_uv=False)
        cond    = sv[0] / sv[-1]
        h_cond  = cond ** 2
        print(f"  {alpha:>8.3f} | {cond:>22.1f} | {h_cond:>14.2e}")
        if cond < best_cond:
            best_cond  = cond
            best_alpha = alpha

    if best_alpha is None:
        best_alpha = 1.0
    print(f"\n  Best α (minimum cond_svd): α = {best_alpha}")
    return best_alpha


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _new_model():
    return build_sigma_w_network(HIDDEN_WIDTH, N_HIDDEN).double()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, loss_fn, sigma_BEM_np, Yq_t,
          lr_schedule, n_lbfgs,
          case_label="?", recover_fn=None, verbose=True):
    if recover_fn is None:
        def recover_fn(m):
            with torch.no_grad():
                return m(Yq_t).squeeze(-1).numpy()

    history = {"iter": [], "loss": [], "density_reldiff": []}

    def _record(it, loss_val=None):
        with torch.no_grad():
            if loss_val is None:
                loss_val = float(loss_fn(model).detach())
        sigma = recover_fn(model)
        d_err = float(np.linalg.norm(sigma - sigma_BEM_np)
                      / np.linalg.norm(sigma_BEM_np))
        history["iter"].append(it)
        history["loss"].append(loss_val)
        history["density_reldiff"].append(d_err)
        if verbose and it % LOG_EVERY == 0:
            print(f"  [{case_label}] iter={it:6d} | loss={loss_val:.3e} "
                  f"| d_err={d_err:.4f}")

    # Adam
    opt  = torch.optim.Adam(model.parameters(), lr=1e-3)
    itr  = 0
    _record(0)
    for n_iters, lr in lr_schedule:
        for pg in opt.param_groups:
            pg["lr"] = lr
        for _ in range(n_iters):
            opt.zero_grad()
            loss = loss_fn(model)
            loss.backward()
            opt.step()
            itr += 1
            if itr % LOG_EVERY == 0:
                _record(itr, float(loss.detach()))
    _record(itr)
    if verbose:
        print(f"  [{case_label}] Adam done:  d_err={history['density_reldiff'][-1]:.4f}")

    # L-BFGS
    opt_lb = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        history_size=LBFGS_MEMORY, line_search_fn="strong_wolfe",
    )
    n_outer = n_lbfgs // 20
    lb_its  = 0
    for _ in range(n_outer):
        def closure():
            opt_lb.zero_grad()
            loss = loss_fn(model)
            loss.backward()
            return loss
        opt_lb.step(closure)
        lb_its += 20
        if lb_its % LOG_EVERY == 0:
            _record(itr + lb_its)
    _record(itr + lb_its)
    if verbose:
        print(f"  [{case_label}] LBFGS done: d_err={history['density_reldiff'][-1]:.4f}")

    return history


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _corner_lines(ax, corner_arcs, alpha=0.3, color="lightgray"):
    for ca in corner_arcs:
        ax.axvline(x=ca, color=color, lw=1.2, ls="--", alpha=alpha, zorder=0)


def fig_convergence(histories, adam_cutoff, outpath):
    labels = {
        "A": r"A: Standard $\|V\sigma-g\|^2$",
        "B": r"B: Left $V^{-1}$ (gold)",
        "C": r"C: H¹ combined",
        "D": r"D: Right $V^{-1}$",
        "E": r"E: Pure H¹",
    }
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for key, hist in histories.items():
        ax.semilogy(hist["iter"], hist["density_reldiff"],
                    "-", color=COLORS[key], lw=LWIDTH, label=labels[key])
        ax.annotate(f"  {hist['density_reldiff'][-1]:.4f}",
                    xy=(hist["iter"][-1], hist["density_reldiff"][-1]),
                    fontsize=8, color=COLORS[key], va="center")
    ax.axvline(x=adam_cutoff, color="gray", ls=":", lw=1.0, alpha=0.7,
               label="Adam → L-BFGS")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(r"$\|\sigma_\theta - \sigma_\mathrm{BEM}\| / \|\sigma_\mathrm{BEM}\|$",
                  fontsize=12)
    ax.set_title(r"Density convergence: Koch(1), $g = x^2 - y^2$, no enrichment"
                 "\nSobolev residual loss study", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_spectral(V_h, DV, outpath):
    """Log-log plot of singular values of V and D_s V."""
    sv_V  = np.linalg.svd(V_h,  compute_uv=False)
    sv_DV = np.linalg.svd(DV, compute_uv=False)
    Nq    = len(sv_V)
    idx   = np.arange(1, Nq + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.semilogy(idx, sv_V,  "-", color=COLORS["A"], lw=LWIDTH, label=r"$\sigma_k(V_h)$")
    ax.semilogy(idx, sv_DV, "-", color=COLORS["C"], lw=LWIDTH, label=r"$\sigma_k(D_s V_h)$")
    ax.set_xlabel(r"Index $k$", fontsize=12)
    ax.set_ylabel("Singular value", fontsize=12)
    ax.set_title(r"Singular values of $V_h$ vs $D_s V_h$"
                 f"\n$\\mathrm{{cond}}(V)={sv_V[0]/sv_V[-1]:.0f}$,"
                 f"  $\\mathrm{{cond}}(D_s V)={sv_DV[0]/sv_DV[-1]:.1f}$",
                 fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    ax = axes[1]
    alphas = np.logspace(-3, 3, 60)
    conds  = []
    for a in alphas:
        stacked = np.vstack([V_h, np.sqrt(a) * DV])
        sv      = np.linalg.svd(stacked, compute_uv=False)
        conds.append(sv[0] / sv[-1])
    ax.loglog(alphas, conds, "-", color=COLORS["C"], lw=LWIDTH)
    ax.axhline(y=sv_V[0]/sv_V[-1],  color=COLORS["A"], ls="--", lw=1.5,
               label=r"$\mathrm{cond}(V)$")
    ax.axhline(y=sv_DV[0]/sv_DV[-1], color=COLORS["E"], ls="--", lw=1.5,
               label=r"$\mathrm{cond}(D_s V)$")
    ax.set_xlabel(r"$\alpha$", fontsize=12)
    ax.set_ylabel(r"$\mathrm{cond}([V;\,\sqrt{\alpha}D_s V])$", fontsize=12)
    ax.set_title(r"Condition of combined operator vs $\alpha$", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(
        r"Spectral analysis: $D_s V$ vs $V$ — Koch(1), $N_q=%d$" % Nq,
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_density(sigmas, sigma_BEM, arc, sort_idx, corner_arcs, outpath):
    arc_s = arc[sort_idx]
    bem_s = sigma_BEM[sort_idx]
    ymin  = min(bem_s.min(), min(s[sort_idx].min() for s in sigmas.values())) * 1.05
    ymax  = max(bem_s.max(), max(s[sort_idx].max() for s in sigmas.values())) * 1.05

    keys   = list(sigmas.keys())
    ncols  = len(keys)
    labels = {
        "A": "(a) Standard",
        "B": r"(b) Left $V^{-1}$",
        "C": r"(c) H¹ combined",
        "D": r"(d) Right $V^{-1}$",
        "E": "(e) Pure H¹",
    }
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        sig   = sigmas[key][sort_idx]
        d_err = np.linalg.norm(sigmas[key] - sigma_BEM) / np.linalg.norm(sigma_BEM)
        ax.plot(arc_s, bem_s, "k--", lw=1.8, alpha=0.6,
                label=r"$\sigma_\mathrm{BEM}$")
        ax.plot(arc_s, sig,   "-", lw=LWIDTH, color=COLORS[key], alpha=0.9,
                label=f"d={d_err:.4f}")
        _corner_lines(ax, corner_arcs)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Arc-length $s$", fontsize=10)
        ax.set_title(labels[key], fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.3, alpha=0.4)
    axes[0].set_ylabel(r"$\sigma(s)$", fontsize=11)
    fig.suptitle(
        r"Final density $\sigma$ — Koch(1), $g = x^2 - y^2$, no enrichment",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_sweep(sweep_results, outpath):
    """α sweep: d_err and cond vs α."""
    alphas  = [r["alpha"] for r in sweep_results]
    d_errs  = [r["d_err"] for r in sweep_results]
    conds   = [r["cond"]  for r in sweep_results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    x_labels = ["0\n(std)"] + [str(a) for a in ALPHA_SWEEP] + ["∞\n(H¹)"]
    x_pos    = range(len(sweep_results))
    colors   = [COLORS["A"]] + [COLORS["C"]] * len(ALPHA_SWEEP) + [COLORS["E"]]
    ax.bar(x_pos, d_errs, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel(r"$\alpha$", fontsize=12)
    ax.set_ylabel("Density rel-diff", fontsize=12)
    ax.set_title(r"$\alpha$ sweep: density error (short budget)", fontsize=11)
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    for xi, de in zip(x_pos, d_errs):
        ax.text(xi, de + 0.005, f"{de:.3f}", ha="center", fontsize=7)

    ax = axes[1]
    ax.bar(x_pos, conds, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel(r"$\alpha$", fontsize=12)
    ax.set_ylabel(r"$\mathrm{cond}([V;\,\sqrt{\alpha}D_s V])$", fontsize=12)
    ax.set_title(r"$\alpha$ sweep: combined operator condition", fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)

    fig.suptitle(
        r"$\alpha$ sweep — Koch(1), $g = x^2 - y^2$, short budget "
        f"(Adam {SWEEP_LR} + L-BFGS {SWEEP_LBFGS})",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    fig_dir = os.path.join(_HERE, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("SOBOLEV RESIDUAL LOSS — Koch(1), g=x²−y², no enrichment")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Geometry and operators
    # ------------------------------------------------------------------
    print("\n--- Step 1: Setup ---")
    data       = setup()
    Yq_T       = data["Yq_T"]
    wq         = data["wq"]
    P          = data["P"]
    qdata      = data["qdata"]
    g_values   = data["g_values"]
    sigma_BEM  = data["sigma_BEM"]
    V_h        = data["V_h"]
    V_inv      = data["V_inv"]
    cond_V     = data["cond_V"]
    Nq         = data["Nq"]
    arc        = data["arc"]
    sort_idx   = data["sort_idx"]
    corner_arcs= data["corner_arcs"]

    # ------------------------------------------------------------------
    # Step 2: Build D_h and validate
    # ------------------------------------------------------------------
    print("\n--- Step 2: Build and validate D_h ---")
    D_h = build_tangential_derivative_matrix(qdata)
    validate_Dh(D_h, qdata, arc)

    # Precompute D_s applied to V and g
    DV = D_h @ V_h        # (Nq, Nq) — D_s(V·σ) = (D_h V_h) σ
    Dg = D_h @ g_values   # (Nq,)   — D_s g

    # ------------------------------------------------------------------
    # Step 3: Spectral analysis
    # ------------------------------------------------------------------
    print("\n--- Step 3: Spectral analysis ---")
    best_alpha_theory = spectral_analysis(V_h, DV)

    # ------------------------------------------------------------------
    # Step 4: Build shared initial weights and tensors
    # ------------------------------------------------------------------
    print("\n--- Step 4: Building initial model ---")
    torch.manual_seed(SEED)
    base_model = _new_model()
    init_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    n_params   = sum(p.numel() for p in base_model.parameters())
    print(f"  Parameters: {n_params}")

    def _fresh_model():
        m = _new_model()
        m.load_state_dict({k: v.clone() for k, v in init_state.items()})
        return m

    # Torch tensors (double precision)
    Yq_t       = torch.tensor(Yq_T,      dtype=torch.float64)
    g_t        = torch.tensor(g_values,  dtype=torch.float64)
    V_h_t      = torch.tensor(V_h,       dtype=torch.float64)
    V_inv_t    = torch.tensor(V_inv,     dtype=torch.float64)
    DV_t       = torch.tensor(DV,        dtype=torch.float64)
    Dg_t       = torch.tensor(Dg,        dtype=torch.float64)
    sigma_BEM_t= torch.tensor(sigma_BEM, dtype=torch.float64)

    # ------------------------------------------------------------------
    # Loss callables
    # ------------------------------------------------------------------
    def loss_A(model):
        sigma = model(Yq_t).squeeze(-1)
        res   = V_h_t @ sigma - g_t
        return (res ** 2).mean()

    def loss_B(model):
        sigma = model(Yq_t).squeeze(-1)
        res   = V_h_t @ sigma - g_t
        return (V_inv_t @ res).pow(2).mean()

    def loss_H1(model, alpha):
        sigma = model(Yq_t).squeeze(-1)
        res   = V_h_t @ sigma - g_t
        Dres  = DV_t @ sigma - Dg_t
        return (res ** 2).mean() + alpha * (Dres ** 2).mean()

    def loss_D(model):
        rho = model(Yq_t).squeeze(-1)
        return (rho - g_t).pow(2).mean()

    def recover_D(model):
        with torch.no_grad():
            rho = model(Yq_t).squeeze(-1)
            return (V_inv_t @ rho).numpy()

    def loss_E(model):
        sigma = model(Yq_t).squeeze(-1)
        Dres  = DV_t @ sigma - Dg_t
        return (Dres ** 2).mean()

    # ------------------------------------------------------------------
    # Step 5: α sweep (short budget, all from same init)
    # ------------------------------------------------------------------
    print("\n--- Step 5: α sweep (short budget) ---")
    print(f"  Budget: Adam {SWEEP_LR} + L-BFGS {SWEEP_LBFGS}")
    print(f"  {'α':>8s} | {'cond([V;√α·DV])':>18s} | {'d_err':>8s} | {'BIE res':>10s}")
    print(f"  {'-'*8}-+-{'-'*18}-+-{'-'*8}-+-{'-'*10}")

    sweep_results = []

    def _sweep_case(loss_fn, alpha, cond_val):
        m = _fresh_model()
        h = train(m, loss_fn, sigma_BEM, Yq_t,
                  lr_schedule=SWEEP_LR, n_lbfgs=SWEEP_LBFGS,
                  case_label=f"α={alpha}", verbose=False)
        with torch.no_grad():
            sig = m(Yq_t).squeeze(-1).numpy()
        d_err = float(np.linalg.norm(sig - sigma_BEM) / np.linalg.norm(sigma_BEM))
        bie   = float(np.linalg.norm(V_h @ sig - g_values) / np.linalg.norm(g_values))
        label = f"{alpha}" if alpha != 0 and alpha != float("inf") else ("0\n(std)" if alpha == 0 else "∞")
        print(f"  {str(alpha):>8s} | {cond_val:>18.1f} | {d_err:>8.4f} | {bie:>10.3e}")
        return {"alpha": alpha, "cond": cond_val, "d_err": d_err, "bie": bie}

    # α = 0 (standard)
    sv_V0 = np.linalg.svd(V_h, compute_uv=False)
    sweep_results.append(_sweep_case(loss_A, 0, sv_V0[0] / sv_V0[-1]))

    # α > 0 combined
    for alpha in ALPHA_SWEEP:
        stacked  = np.vstack([V_h, np.sqrt(alpha) * DV])
        sv_comb  = np.linalg.svd(stacked, compute_uv=False)
        cond_val = sv_comb[0] / sv_comb[-1]
        sweep_results.append(_sweep_case(
            lambda model, a=alpha: loss_H1(model, a),
            alpha, cond_val,
        ))

    # α = ∞ (pure H¹)
    sv_DV = np.linalg.svd(DV, compute_uv=False)
    sweep_results.append(_sweep_case(loss_E, float("inf"), sv_DV[0] / sv_DV[-1]))

    # Best α from sweep
    finite_results = [r for r in sweep_results if r["alpha"] not in (0, float("inf"))]
    best_alpha = min(finite_results, key=lambda r: r["d_err"])["alpha"]
    print(f"\n  Best α from sweep: {best_alpha}"
          f"  (d_err = {min(r['d_err'] for r in finite_results):.4f})")

    # ------------------------------------------------------------------
    # Step 6: Full training with FULL budget
    # ------------------------------------------------------------------
    adam_cutoff = sum(n for n, _ in LR_SCHEDULE)

    print("\n" + "=" * 60)
    print("Case A: Standard ||Vσ−g||²")
    print("=" * 60)
    model_A = _fresh_model()
    t0 = time.perf_counter()
    hist_A = train(model_A, loss_A, sigma_BEM, Yq_t,
                   lr_schedule=LR_SCHEDULE, n_lbfgs=N_LBFGS, case_label="A")
    time_A = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print("Case B: Left V⁻¹  ||σ−σ_BEM||²")
    print("=" * 60)
    model_B = _fresh_model()
    t0 = time.perf_counter()
    hist_B = train(model_B, loss_B, sigma_BEM, Yq_t,
                   lr_schedule=LR_SCHEDULE, n_lbfgs=N_LBFGS, case_label="B")
    time_B = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print(f"Case C: H¹ combined  ||Vσ−g||² + {best_alpha}·||D_s(Vσ−g)||²")
    print("=" * 60)
    model_C = _fresh_model()
    loss_C  = lambda m, a=best_alpha: loss_H1(m, a)
    t0 = time.perf_counter()
    hist_C = train(model_C, loss_C, sigma_BEM, Yq_t,
                   lr_schedule=LR_SCHEDULE, n_lbfgs=N_LBFGS, case_label="C")
    time_C = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print("Case D: Right V⁻¹  σ = V⁻¹ρ,  ||ρ−g||²")
    print("=" * 60)
    model_D = _fresh_model()
    t0 = time.perf_counter()
    hist_D = train(model_D, loss_D, sigma_BEM, Yq_t,
                   lr_schedule=LR_SCHEDULE, n_lbfgs=N_LBFGS,
                   case_label="D", recover_fn=recover_D)
    time_D = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print("Case E: Pure H¹  ||D_s(Vσ−g)||²")
    print("=" * 60)
    model_E = _fresh_model()
    t0 = time.perf_counter()
    hist_E = train(model_E, loss_E, sigma_BEM, Yq_t,
                   lr_schedule=LR_SCHEDULE, n_lbfgs=N_LBFGS, case_label="E")
    time_E = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Step 7: Final evaluation
    # ------------------------------------------------------------------
    print("\n--- Step 7: Final evaluation ---")
    with torch.no_grad():
        sigma_A = model_A(Yq_t).squeeze(-1).numpy()
        sigma_B = model_B(Yq_t).squeeze(-1).numpy()
        sigma_C = model_C(Yq_t).squeeze(-1).numpy()
        sigma_D = recover_D(model_D)
        sigma_E = model_E(Yq_t).squeeze(-1).numpy()

    def _d_err(s):
        return float(np.linalg.norm(s - sigma_BEM) / np.linalg.norm(sigma_BEM))

    def _bie(s):
        return float(np.linalg.norm(V_h @ s - g_values) / np.linalg.norm(g_values))

    def _interior(s):
        res = reconstruct_interior(
            P=P, Yq=Yq_T, wq=wq, sigma=s,
            n_grid=N_GRID_FINAL, u_exact=u_exact_fn,
        )
        return float(res.rel_L2)

    results = {}
    for label, sig, hist, t in [
        ("A", sigma_A, hist_A, time_A),
        ("B", sigma_B, hist_B, time_B),
        ("C", sigma_C, hist_C, time_C),
        ("D", sigma_D, hist_D, time_D),
        ("E", sigma_E, hist_E, time_E),
    ]:
        results[label] = {
            "d_err":  _d_err(sig),
            "bie":    _bie(sig),
            "iL2":    _interior(sig),
            "loss":   hist["loss"][-1],
            "time":   t,
            "sigma":  sig,
            "hist":   hist,
        }

    # ------------------------------------------------------------------
    # Step 8: Summary table
    # ------------------------------------------------------------------
    sv_DV_full = np.linalg.svd(DV, compute_uv=False)
    cond_DV    = sv_DV_full[0] / sv_DV_full[-1]

    # Stacked operator condition for Case C
    stacked_C  = np.vstack([V_h, np.sqrt(best_alpha) * DV])
    sv_C       = np.linalg.svd(stacked_C, compute_uv=False)
    cond_C     = sv_C[0] / sv_C[-1]

    hess_labels = {
        "A": f"≈{cond_V**2:.1e}",
        "B": "1",
        "C": f"≈{cond_C**2:.0f}",
        "D": "1",
        "E": f"≈{cond_DV**2:.0f}",
    }
    loss_labels = {
        "A": "||Vσ−g||²",
        "B": "||σ−σ_BEM||²",
        "C": f"||Vσ−g||²+{best_alpha}·||D_sVσ−D_sg||²",
        "D": "||ρ−g||²  (σ=V⁻¹ρ)",
        "E": "||D_s(Vσ−g)||²",
    }

    print(f"\n{'='*80}")
    print(f"SOBOLEV RESIDUAL RESULTS — Koch(1), g=x²−y², no enrichment")
    print(f"{'='*80}")
    print(f"  Nq={Nq}, network={N_HIDDEN}×{HIDDEN_WIDTH} tanh, seed={SEED}")
    print(f"  cond_svd(V)      = {cond_V:.0f}   Hessian cond ≈ {cond_V**2:.2e}")
    print(f"  cond_svd(D_s V)  = {cond_DV:.1f}   Hessian cond ≈ {cond_DV**2:.2e}")
    print(f"  Best α (sweep)   = {best_alpha}")
    print(f"  cond_svd([V;√α·DV]) = {cond_C:.1f}  Hessian cond ≈ {cond_C**2:.2e}")

    w = 16
    sep = f"{'─'*30}─┬─{'─'*w}─┬─{'─'*w}─┬─{'─'*w}─┬─{'─'*w}─┬─{'─'*w}"
    hdr = (f"{'Metric':<30s} │ {'A: Standard':>{w}s} │ {'B: Left V⁻¹':>{w}s} │ "
           f"{'C: H¹ comb':>{w}s} │ {'D: Right V⁻¹':>{w}s} │ {'E: Pure H¹':>{w}s}")
    print(f"\n{sep}\n{hdr}\n{sep}")

    d_ref = results["A"]["d_err"]
    rows = [
        ("Density rel-diff",
         f"{results['A']['d_err']:.4f}", f"{results['B']['d_err']:.4f}",
         f"{results['C']['d_err']:.4f}", f"{results['D']['d_err']:.4f}",
         f"{results['E']['d_err']:.4f}"),
        ("BIE residual",
         f"{results['A']['bie']:.2e}", f"{results['B']['bie']:.2e}",
         f"{results['C']['bie']:.2e}", f"{results['D']['bie']:.2e}",
         f"{results['E']['bie']:.2e}"),
        ("Interior rel L2",
         f"{results['A']['iL2']:.2e}", f"{results['B']['iL2']:.2e}",
         f"{results['C']['iL2']:.2e}", f"{results['D']['iL2']:.2e}",
         f"{results['E']['iL2']:.2e}"),
        ("Final loss",
         f"{results['A']['loss']:.2e}", f"{results['B']['loss']:.2e}",
         f"{results['C']['loss']:.2e}", f"{results['D']['loss']:.2e}",
         f"{results['E']['loss']:.2e}"),
        ("Wall time (s)",
         f"{results['A']['time']:.1f}", f"{results['B']['time']:.1f}",
         f"{results['C']['time']:.1f}", f"{results['D']['time']:.1f}",
         f"{results['E']['time']:.1f}"),
        ("Hessian cond (theory)",
         hess_labels["A"], hess_labels["B"], hess_labels["C"],
         hess_labels["D"], hess_labels["E"]),
    ]
    for name, *vals in rows:
        print(f"{name:<30s} │ " + " │ ".join(f"{v:>{w}s}" for v in vals))
    print(sep)

    # Improvement vs A
    impr = {k: d_ref / max(results[k]["d_err"], 1e-9) for k in "BCDE"}
    print(f"{'Improvement vs A':<30s} │ {'—':>{w}s} │ "
          + " │ ".join(f"{impr[k]:>{w}.1f}×" for k in "BCDE"))
    print(sep)

    # Key verdict
    c_err = results["C"]["d_err"]
    if c_err < 0.10:
        verdict = f"H¹ combined WORKS: d_err={c_err:.2%} < 10% — no V⁻¹ needed!"
    elif c_err < results["A"]["d_err"]:
        verdict = f"H¹ combined PARTIAL: d_err={c_err:.2%} (vs A={results['A']['d_err']:.2%})"
    else:
        verdict = f"H¹ combined NO IMPROVEMENT: d_err={c_err:.2%} ≥ A={results['A']['d_err']:.2%}"
    print(f"\nVERDICT: {verdict}")
    print(f"{'='*80}")

    # ------------------------------------------------------------------
    # Step 9: Figures
    # ------------------------------------------------------------------
    print("\n--- Step 9: Figures ---")

    histories = {k: results[k]["hist"] for k in "ABCDE"}
    fig_convergence(
        histories, adam_cutoff,
        os.path.join(fig_dir, "sobolev_convergence.png"),
    )

    fig_spectral(
        V_h, DV,
        os.path.join(fig_dir, "sobolev_spectral.png"),
    )

    fig_density(
        {k: results[k]["sigma"] for k in "ABCDE"},
        sigma_BEM, arc, sort_idx, corner_arcs,
        os.path.join(fig_dir, "sobolev_density.png"),
    )

    fig_sweep(
        sweep_results,
        os.path.join(fig_dir, "sobolev_alpha_sweep.png"),
    )

    print(f"\n  All figures saved to {fig_dir}/")
    return results


if __name__ == "__main__":
    main()
