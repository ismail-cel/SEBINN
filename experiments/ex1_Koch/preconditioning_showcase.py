"""
Preconditioning showcase: ill-conditioning of the single-layer BIE.

Mathematical background
-----------------------
We solve the Dirichlet problem

    -Δu = 0  in Ω,    u = g  on ∂Ω

using the single-layer representation

    u(x) = ∫_{∂Ω} G(x,y) σ(y) ds(y),

where the density σ satisfies the boundary integral equation

    V σ = g,    V : H^{-1/2}(∂Ω) → H^{1/2}(∂Ω).

The operator V is a compact smoothing operator — its eigenvalues cluster
toward zero at algebraic rate.  For Koch(1), cond(V_h) ≈ 13,000.

The loss landscape of ||V_h σ - g||² is therefore highly anisotropic:
the gradient with respect to the k-th singular mode of V is proportional
to σ_k², so the singular-density modes (associated with corner
singularities) receive exponentially weaker gradient signal and stall.

Three preconditioned formulations are compared:

  A  Standard:          L = ||V_h σ_θ - g||²
  B  Left V⁻¹:         L = ||V_h⁻¹(V_h σ_θ - g)||² = ||σ_θ - σ_BEM||²
  C  Right V⁻¹:        σ = V_h⁻¹ ρ_θ,  L = ||ρ_θ - g||²

All three use the same plain 4×80 tanh network, no singular enrichment.

Geometry: Koch(1), g(x,y) = x²−y², n_per_edge=12, p_gl=16.
Network: 4 hidden layers × 80 neurons, tanh activations, shared init.
Training: Adam 3×1000 iters at [1e-3, 3e-4, 1e-4] + L-BFGS 15000 iters.
"""

import sys
import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from src.boundary.polygon import make_koch_geometry
from src.boundary.panels import build_uniform_panels, label_corner_ring_panels
from src.quadrature.panel_quad import build_panel_quadrature
from src.quadrature.nystrom import assemble_nystrom_matrix, solve_bem
from src.models.sigma_w_net import build_sigma_w_network
from src.reconstruction.interior import reconstruct_interior

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED          = 0
N_PER_EDGE    = 12
P_GL          = 16
HIDDEN_WIDTH  = 80
N_HIDDEN      = 4
LR_SCHEDULE   = [(1000, 1e-3), (1000, 3e-4), (1000, 1e-4)]
N_LBFGS       = 15000
LBFGS_MEMORY  = 30
LOG_EVERY     = 200
N_GRID_COARSE = 51
N_GRID_FINAL  = 201

COLORS = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c", "BEM": "black"}
LWIDTH = 2.0


# ---------------------------------------------------------------------------
# Boundary data and exact solution
# ---------------------------------------------------------------------------

def g_fn(xy: np.ndarray) -> np.ndarray:
    """Dirichlet data g(x,y) = x² − y² (harmonic polynomial)."""
    return xy[:, 0]**2 - xy[:, 1]**2


def u_exact_fn(xy: np.ndarray) -> np.ndarray:
    """Exact interior solution u(x,y) = x² − y²."""
    return xy[:, 0]**2 - xy[:, 1]**2


# ---------------------------------------------------------------------------
# Step 0: Geometry and operators
# ---------------------------------------------------------------------------

def setup():
    geom   = make_koch_geometry(n=1)
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=N_PER_EDGE)
    label_corner_ring_panels(panels, P)
    qdata  = build_panel_quadrature(panels, p=P_GL)
    nmat   = assemble_nystrom_matrix(qdata)

    Yq_T       = qdata.Yq.T                         # (Nq, 2)
    wq         = qdata.wq                           # (Nq,)
    g_values   = g_fn(Yq_T)                         # (Nq,)
    sigma_BEM  = solve_bem(nmat, g_values).sigma     # (Nq,)
    V_h        = nmat.V                             # (Nq, Nq)
    V_inv      = np.linalg.inv(V_h)                 # (Nq, Nq) — once

    svs   = np.linalg.svd(V_h, compute_uv=False)
    cond_V = float(svs[0] / svs[-1])
    Nq     = len(sigma_BEM)

    print(f"  Koch(1): N_panels={qdata.n_panels}, N_quad={Nq}")
    print(f"  cond(V_h)        = {cond_V:.0f}")
    print(f"  max|σ_BEM|       = {np.abs(sigma_BEM).max():.3f}  (corner spikes)")
    print(f"  median|σ_BEM|    = {np.median(np.abs(sigma_BEM)):.3f}  (smooth regions)")
    print(f"  ||σ_BEM||_2      = {np.linalg.norm(sigma_BEM):.4f}")
    print(f"  ||g||_2          = {np.linalg.norm(g_values):.4f}")

    # Corner arc-length positions for figure annotations
    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc         = panel_start[qdata.pan_id] + qdata.s_on_panel
    sort_idx    = np.argsort(arc)

    # Reentrant corner arc-lengths: corners occur at vertices with angle > π
    # For Koch(1) the 12 vertices alternate convex / reentrant; the reentrant
    # ones (interior angle 4π/3) are indices 1, 3, 5, 7, 9, 11.
    # Closest quadrature nodes to each vertex:
    corner_arcs = []
    for vi in range(1, len(P), 2):   # odd vertices = reentrant
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
# Step 1: Model builder
# ---------------------------------------------------------------------------

def _new_model():
    return build_sigma_w_network(HIDDEN_WIDTH, N_HIDDEN).double()


# ---------------------------------------------------------------------------
# Step 2: Loss functions
# ---------------------------------------------------------------------------

def make_losses(V_h_t, V_inv_t, g_t, sigma_BEM_t, Yq_t):
    """Return the three loss callables: loss(model) -> scalar."""

    def loss_A(model):
        """Standard: L = ||V_h σ_θ − g||²."""
        sigma = model(Yq_t).squeeze(-1)
        res   = V_h_t @ sigma - g_t
        return (res**2).mean()

    def loss_B(model):
        """Left V⁻¹: L = ||V_h⁻¹(V_h σ_θ − g)||² = ||σ_θ − σ_BEM||²."""
        sigma = model(Yq_t).squeeze(-1)
        res   = V_h_t @ sigma - g_t
        prec  = V_inv_t @ res          # = σ − σ_BEM
        return (prec**2).mean()

    def loss_C(model):
        """Right V⁻¹: network learns ρ ≈ g, σ = V⁻¹ρ in post-processing."""
        rho = model(Yq_t).squeeze(-1)
        res = rho - g_t
        return (res**2).mean()

    return loss_A, loss_B, loss_C


def recover_sigma_C(model, Yq_t, V_inv_t):
    """Case C: σ = V_h⁻¹ ρ  (post-processing step)."""
    with torch.no_grad():
        rho   = model(Yq_t).squeeze(-1)
        sigma = V_inv_t @ rho
    return sigma.numpy()


# ---------------------------------------------------------------------------
# Step 3: Training loop
# ---------------------------------------------------------------------------

def train(model, loss_fn, sigma_BEM_np, Yq_t, V_inv_t,
          lr_schedule=LR_SCHEDULE, n_lbfgs=N_LBFGS,
          case_label="?", recover_fn=None, verbose=True):
    """
    Train model with Adam followed by L-BFGS.

    recover_fn : callable(model) -> np.ndarray
        Extract density from model for diagnostics.  Default: model(Yq).
    """
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
        if verbose and (it % LOG_EVERY == 0 or it <= 1):
            print(f"  [{case_label}] iter={it:6d} | loss={loss_val:.3e} "
                  f"| d_err={d_err:.4f}")

    # --- Adam ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    global_iter = 0
    _record(0)
    for n_iters, lr in lr_schedule:
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        for _ in range(n_iters):
            optimizer.zero_grad()
            loss = loss_fn(model)
            loss.backward()
            optimizer.step()
            global_iter += 1
            if global_iter % LOG_EVERY == 0:
                _record(global_iter, float(loss.detach()))
    _record(global_iter)

    if verbose:
        print(f"  [{case_label}] Adam done: d_err={history['density_reldiff'][-1]:.4f}")

    # --- L-BFGS (PyTorch built-in with strong Wolfe line search) ---
    opt_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        history_size=LBFGS_MEMORY, line_search_fn="strong_wolfe",
    )
    n_outer   = N_LBFGS // 20        # outer steps (each does ≤20 sub-iters)
    lbfgs_its = 0

    for step in range(n_outer):
        def closure():
            opt_lbfgs.zero_grad()
            loss = loss_fn(model)
            loss.backward()
            return loss
        opt_lbfgs.step(closure)
        lbfgs_its += 20
        it_total   = global_iter + lbfgs_its
        if lbfgs_its % LOG_EVERY == 0:
            _record(it_total)

    _record(global_iter + lbfgs_its)
    if verbose:
        print(f"  [{case_label}] LBFGS done: d_err={history['density_reldiff'][-1]:.4f}")

    return history


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _corner_lines(ax, corner_arcs, ymin, ymax, alpha=0.25, color="lightgray"):
    for ca in corner_arcs:
        ax.axvline(x=ca, color=color, lw=1.2, ls="--", alpha=alpha, zorder=0)


def fig_convergence(hist_A, hist_B, hist_C, adam_cutoff, outpath):
    fig, ax = plt.subplots(figsize=(10, 4))
    for hist, label, key in [
        (hist_A, "A: Standard $||V\\sigma - g||^2$",           "A"),
        (hist_B, "B: Left $V^{-1}$ preconditioned",            "B"),
        (hist_C, "C: Right $V^{-1}$ preconditioned",           "C"),
    ]:
        iters = hist["iter"]
        derr  = hist["density_reldiff"]
        ax.semilogy(iters, derr, "-", color=COLORS[key], lw=LWIDTH, label=label)
        ax.annotate(f"  {derr[-1]:.4f}", xy=(iters[-1], derr[-1]),
                    fontsize=9, color=COLORS[key], va="center")

    ax.axvline(x=adam_cutoff, color="gray", ls=":", lw=1.0, alpha=0.7,
               label="Adam → L-BFGS")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(r"$\|\sigma_\theta - \sigma_\mathrm{BEM}\| / \|\sigma_\mathrm{BEM}\|$",
                  fontsize=12)
    ax.set_title(
        r"Density convergence: Koch(1), $g = x^2 - y^2$, no enrichment",
        fontsize=12,
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_density(sigma_A, sigma_B, sigma_C, sigma_BEM, arc, sort_idx,
                corner_arcs, outpath):
    arc_s   = arc[sort_idx]
    bem_s   = sigma_BEM[sort_idx]
    A_s     = sigma_A[sort_idx]
    B_s     = sigma_B[sort_idx]
    C_s     = sigma_C[sort_idx]

    ymin = min(bem_s.min(), A_s.min(), B_s.min(), C_s.min()) * 1.05
    ymax = max(bem_s.max(), A_s.max(), B_s.max(), C_s.max()) * 1.05

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    titles = ["(a) Standard BINN", r"(b) Left $V^{-1}$", r"(c) Right $V^{-1}$"]
    sigs   = [A_s, B_s, C_s]
    keys   = ["A", "B", "C"]
    d_errs = [
        np.linalg.norm(sigma_A - sigma_BEM) / np.linalg.norm(sigma_BEM),
        np.linalg.norm(sigma_B - sigma_BEM) / np.linalg.norm(sigma_BEM),
        np.linalg.norm(sigma_C - sigma_BEM) / np.linalg.norm(sigma_BEM),
    ]

    for ax, title, sig, key, de in zip(axes, titles, sigs, keys, d_errs):
        ax.plot(arc_s, bem_s, "k--", lw=1.8, alpha=0.7,
                label=r"$\sigma_\mathrm{BEM}$")
        ax.plot(arc_s, sig,   "-",   lw=LWIDTH, color=COLORS[key], alpha=0.9,
                label=f"d={de:.4f}")
        _corner_lines(ax, corner_arcs, ymin, ymax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Arc-length $s$", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, lw=0.3, alpha=0.4)

    axes[0].set_ylabel(r"$\sigma(s)$", fontsize=11)
    fig.suptitle(
        r"Final density $\sigma$ — Koch(1), $g = x^2 - y^2$",
        fontsize=12, y=1.01,
    )
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_density_error(sigma_A, sigma_B, sigma_C, sigma_BEM,
                      arc, sort_idx, corner_arcs, outpath):
    arc_s = arc[sort_idx]
    bem_s = sigma_BEM[sort_idx]
    sigs  = [sigma_A[sort_idx], sigma_B[sort_idx], sigma_C[sort_idx]]
    keys  = ["A", "B", "C"]
    titles = ["(a) Standard BINN", r"(b) Left $V^{-1}$", r"(c) Right $V^{-1}$"]

    errs = [np.abs(s - bem_s) for s in sigs]
    ymax = max(e.max() for e in errs) * 2.0
    ymin = 1e-5

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, title, err, key in zip(axes, titles, errs, keys):
        ax.semilogy(arc_s, err + 1e-15, "-", lw=LWIDTH, color=COLORS[key])
        _corner_lines(ax, corner_arcs, ymin, ymax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Arc-length $s$", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)

    axes[0].set_ylabel(r"$|\sigma_\theta(s) - \sigma_\mathrm{BEM}(s)|$", fontsize=11)
    fig.suptitle(
        r"Pointwise density error — Koch(1), $g = x^2 - y^2$",
        fontsize=12, y=1.01,
    )
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_rho(rho_C, sigma_C, sigma_BEM, g_values_sorted, arc, sort_idx,
            corner_arcs, outpath):
    arc_s  = arc[sort_idx]
    rho_s  = rho_C[sort_idx]
    sig_s  = sigma_C[sort_idx]
    bem_s  = sigma_BEM[sort_idx]
    g_s    = g_values_sorted

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(arc_s, rho_s, "-", lw=LWIDTH, color=COLORS["C"],
            label=r"$\rho_\theta$ (network output)")
    ax.plot(arc_s, g_s,   "k--", lw=1.5, alpha=0.7,
            label=r"$g = x^2 - y^2$ (target)")
    _corner_lines(ax, corner_arcs, rho_s.min(), rho_s.max())
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$\rho(s)$", fontsize=11)
    ax.set_title(r"(a) Learned $\rho_\theta \approx g$ (smooth)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, lw=0.3, alpha=0.4)

    ax = axes[1]
    ax.plot(arc_s, bem_s, "k--", lw=1.8, alpha=0.7,
            label=r"$\sigma_\mathrm{BEM}$ (reference)")
    ax.plot(arc_s, sig_s, "-", lw=LWIDTH, color=COLORS["C"],
            label=r"$\sigma = V_h^{-1}\rho_\theta$ (recovered)")
    _corner_lines(ax, corner_arcs, min(bem_s.min(), sig_s.min()),
                  max(bem_s.max(), sig_s.max()))
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$\sigma(s)$", fontsize=11)
    ax.set_title(
        r"(b) Recovered $\sigma = V_h^{-1}\rho_\theta$ (singular spikes via $V_h^{-1}$)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, lw=0.3, alpha=0.4)

    fig.suptitle(
        r"Right $V^{-1}$ preconditioning: learn smooth $\rho$, recover singular $\sigma$",
        fontsize=12, y=1.01,
    )
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_singular_values(svs, cond_V, Nq, outpath):
    fig, ax = plt.subplots(figsize=(9, 4))
    idx = np.arange(1, Nq + 1)
    ax.semilogy(idx, svs, "-", color=COLORS["A"], lw=LWIDTH)

    # Annotate extremes
    ax.annotate(
        rf"$\sigma_1 = {svs[0]:.3f}$",
        xy=(1, svs[0]), xytext=(Nq * 0.08, svs[0] * 0.5),
        fontsize=10, arrowprops=dict(arrowstyle="->", color="gray"),
    )
    ax.annotate(
        rf"$\sigma_{{N_q}} = {svs[-1]:.4f}$",
        xy=(Nq, svs[-1]), xytext=(Nq * 0.55, svs[-1] * 50),
        fontsize=10, arrowprops=dict(arrowstyle="->", color="gray"),
    )
    ax.text(0.5, 0.07,
            rf"$\mathrm{{cond}}(V_h) = \sigma_1/\sigma_{{N_q}} = {cond_V:.0f}$"
            "\n"
            r"Gradient $\propto \sigma_k^2$: last mode "
            rf"receives ${cond_V**2:.0e}$× weaker signal",
            transform=ax.transAxes, ha="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    ax.set_xlabel(r"Index $k$ (sorted descending)", fontsize=12)
    ax.set_ylabel(r"Singular value $\sigma_k(V_h)$", fontsize=12)
    ax.set_title(
        r"Singular values of $V_h$: ill-conditioning of the single-layer BIE"
        f"\nKoch(1), $N_q = {Nq}$",
        fontsize=12,
    )
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    try:
        fig.tight_layout()
    except Exception:
        pass
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

    print("\n" + "="*70)
    print("PRECONDITIONING SHOWCASE — Koch(1), g = x²−y², no enrichment")
    print("="*70)

    # ------------------------------------------------------------------
    # 0. Setup
    # ------------------------------------------------------------------
    print("\n--- Setup ---")
    data = setup()
    Yq_T       = data["Yq_T"]
    wq         = data["wq"]
    P          = data["P"]
    g_values   = data["g_values"]
    sigma_BEM  = data["sigma_BEM"]
    V_h        = data["V_h"]
    V_inv      = data["V_inv"]
    svs        = data["svs"]
    cond_V     = data["cond_V"]
    Nq         = data["Nq"]
    arc        = data["arc"]
    sort_idx   = data["sort_idx"]
    corner_arcs= data["corner_arcs"]

    Yq_t        = torch.tensor(Yq_T,      dtype=torch.float64)
    g_t         = torch.tensor(g_values,  dtype=torch.float64)
    sigma_BEM_t = torch.tensor(sigma_BEM, dtype=torch.float64)
    V_h_t       = torch.tensor(V_h,       dtype=torch.float64)
    V_inv_t     = torch.tensor(V_inv,     dtype=torch.float64)

    loss_A, loss_B, loss_C = make_losses(V_h_t, V_inv_t, g_t, sigma_BEM_t, Yq_t)

    # ------------------------------------------------------------------
    # 1. Build models from same initial weights
    # ------------------------------------------------------------------
    print("\n--- Building models from same initial weights ---")
    torch.manual_seed(SEED)
    base_model = _new_model()
    init_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    model_A = _new_model(); model_A.load_state_dict({k: v.clone() for k, v in init_state.items()})
    model_B = _new_model(); model_B.load_state_dict({k: v.clone() for k, v in init_state.items()})
    model_C = _new_model(); model_C.load_state_dict({k: v.clone() for k, v in init_state.items()})

    n_params = sum(p.numel() for p in model_A.parameters())
    print(f"  Parameters per model: {n_params}")

    # ------------------------------------------------------------------
    # 2. Initial loss check and scale normalisation
    # ------------------------------------------------------------------
    print("\n--- Initial loss check ---")
    with torch.no_grad():
        init_A = float(loss_A(model_A))
        init_B = float(loss_B(model_B))
        init_C = float(loss_C(model_C))
    print(f"  Initial losses: A={init_A:.3e}, B={init_B:.3e}, C={init_C:.3e}")

    # Normalise losses so all three start at the same value (= init_A).
    # This makes the lr schedule comparable across cases.
    scale_A = 1.0
    scale_B = init_A / max(init_B, 1e-14)
    scale_C = init_A / max(init_C, 1e-14)
    print(f"  Normalisation scales: A={scale_A:.3f}, B={scale_B:.3f}, C={scale_C:.3f}")

    def loss_A_scaled(m): return scale_A * loss_A(m)
    def loss_B_scaled(m): return scale_B * loss_B(m)
    def loss_C_scaled(m): return scale_C * loss_C(m)

    # ------------------------------------------------------------------
    # 3. Training
    # ------------------------------------------------------------------
    adam_cutoff = sum(n for n, _ in LR_SCHEDULE)

    print("\n" + "="*60)
    print("Case A: Standard BINN  (||Vσ − g||²)")
    print("="*60)
    t0     = time.perf_counter()
    hist_A = train(model_A, loss_A_scaled, sigma_BEM,
                   Yq_t, V_inv_t, case_label="A")
    time_A = time.perf_counter() - t0

    print("\n" + "="*60)
    print("Case B: Left V⁻¹ preconditioned  (||σ − σ_BEM||²)")
    print("="*60)
    t0     = time.perf_counter()
    hist_B = train(model_B, loss_B_scaled, sigma_BEM,
                   Yq_t, V_inv_t, case_label="B")
    time_B = time.perf_counter() - t0

    print("\n" + "="*60)
    print("Case C: Right V⁻¹ preconditioned  (||ρ − g||², σ = V⁻¹ρ)")
    print("="*60)
    recover_C = lambda m: recover_sigma_C(m, Yq_t, V_inv_t)
    t0        = time.perf_counter()
    hist_C    = train(model_C, loss_C_scaled, sigma_BEM,
                      Yq_t, V_inv_t, case_label="C", recover_fn=recover_C)
    time_C    = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # 4. Final densities
    # ------------------------------------------------------------------
    print("\n--- Final densities ---")
    with torch.no_grad():
        sigma_A = model_A(Yq_t).squeeze(-1).numpy()
        sigma_B = model_B(Yq_t).squeeze(-1).numpy()
        sigma_C = recover_sigma_C(model_C, Yq_t, V_inv_t)
        rho_C   = model_C(Yq_t).squeeze(-1).numpy()

    d_err_A = float(np.linalg.norm(sigma_A - sigma_BEM) / np.linalg.norm(sigma_BEM))
    d_err_B = float(np.linalg.norm(sigma_B - sigma_BEM) / np.linalg.norm(sigma_BEM))
    d_err_C = float(np.linalg.norm(sigma_C - sigma_BEM) / np.linalg.norm(sigma_BEM))

    bie_A = float(np.linalg.norm(V_h @ sigma_A - g_values) / np.linalg.norm(g_values))
    bie_B = float(np.linalg.norm(V_h @ sigma_B - g_values) / np.linalg.norm(g_values))
    bie_C = float(np.linalg.norm(V_h @ sigma_C - g_values) / np.linalg.norm(g_values))

    # Interior reconstruction
    def _interior(sigma):
        res = reconstruct_interior(
            P=P, Yq=Yq_T, wq=wq, sigma=sigma,
            n_grid=N_GRID_FINAL, u_exact=u_exact_fn,
        )
        return float(res.rel_L2)

    iL2_A = _interior(sigma_A)
    iL2_B = _interior(sigma_B)
    iL2_C = _interior(sigma_C)

    overhead_per_iter_ms = (time_B - time_A) / (adam_cutoff + N_LBFGS) * 1000.0

    # ------------------------------------------------------------------
    # 5. Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"PRECONDITIONING SHOWCASE — Koch(1), g = x²−y², no enrichment")
    print(f"{'='*70}")
    print(f"cond(V_h) = {cond_V:.0f}")
    print(f"N_q = {Nq}, network = {N_HIDDEN}×{HIDDEN_WIDTH} tanh")
    print(f"Training: Adam {LR_SCHEDULE} + L-BFGS {N_LBFGS}")
    print(f"{'='*70}")
    print()
    hdr = (f"{'Metric':<25s} | {'A: Standard':>14s} | "
           f"{'B: Left V⁻¹':>14s} | {'C: Right V⁻¹':>14s}")
    sep = f"{'-'*25}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}"
    print(hdr); print(sep)
    print(f"{'Density rel-diff':<25s} | {d_err_A:>14.4f} | {d_err_B:>14.4f} | {d_err_C:>14.4f}")
    print(f"{'BIE residual':<25s} | {bie_A:>14.2e} | {bie_B:>14.2e} | {bie_C:>14.2e}")
    print(f"{'Interior rel L2':<25s} | {iL2_A:>14.2e} | {iL2_B:>14.2e} | {iL2_C:>14.2e}")
    print(f"{'Final train loss':<25s} | {hist_A['loss'][-1]:>14.2e} | "
          f"{hist_B['loss'][-1]:>14.2e} | {hist_C['loss'][-1]:>14.2e}")
    print(f"{'Wall time (s)':<25s} | {time_A:>14.1f} | {time_B:>14.1f} | {time_C:>14.1f}")
    print(sep)
    print(f"{'Improvement vs A':<25s} | {'—':>14s} | "
          f"{d_err_A/max(d_err_B,1e-9):>13.1f}x | "
          f"{d_err_A/max(d_err_C,1e-9):>13.1f}x")

    # ------------------------------------------------------------------
    # 6. Figures
    # ------------------------------------------------------------------
    print("\n--- Generating figures ---")

    fig_convergence(
        hist_A, hist_B, hist_C, adam_cutoff,
        os.path.join(fig_dir, "preconditioning_convergence.png"),
    )
    fig_density(
        sigma_A, sigma_B, sigma_C, sigma_BEM, arc, sort_idx, corner_arcs,
        os.path.join(fig_dir, "preconditioning_density.png"),
    )
    fig_density_error(
        sigma_A, sigma_B, sigma_C, sigma_BEM, arc, sort_idx, corner_arcs,
        os.path.join(fig_dir, "preconditioning_density_error.png"),
    )

    g_sorted = g_values[sort_idx]
    fig_rho(
        rho_C, sigma_C, sigma_BEM, g_sorted, arc, sort_idx, corner_arcs,
        os.path.join(fig_dir, "preconditioning_rho.png"),
    )
    fig_singular_values(
        svs, cond_V, Nq,
        os.path.join(fig_dir, "preconditioning_singular_values.png"),
    )

    print(f"\n  All figures saved to {fig_dir}/")

    # ------------------------------------------------------------------
    # 7. Explanatory commentary
    # ------------------------------------------------------------------
    print(f"""
EXPLANATION OF RESULTS
{'='*70}

1. WHY THE STANDARD BINN PLATEAUS:

   The single-layer operator V is a smoothing integral operator of order -1.
   Its singular values span {cond_V:.0f}:1, meaning the loss ||Vσ−g||² weights
   the k-th singular mode by σ_k².  The modes carrying the corner singularity
   r^{{−1/4}} correspond to the smallest singular values of V and receive
   gradient signal {cond_V**2:.0e}× weaker than the dominant modes.  The
   optimizer converges the well-conditioned modes quickly but stalls on the
   ill-conditioned ones, producing a density that satisfies the BIE (residual
   {bie_A:.1e}) but differs from σ_BEM by {d_err_A*100:.0f}%.

2. WHY LEFT-PRECONDITIONING WORKS:

   The loss ||V_h⁻¹(Vσ−g)||² = ||σ − σ_BEM||² directly measures the density
   error.  All modes receive equal gradient weight (condition number = 1).
   The optimizer sees the corner singularity as clearly as the smooth modes,
   achieving {d_err_B*100:.2f}% density error.

3. WHY RIGHT-PRECONDITIONING WORKS:

   The substitution σ = V_h⁻¹ρ transforms the BIE Vσ = g into ρ = g.  The
   network learns ρ ≈ g = x²−y² (a smooth polynomial), which is trivial.
   The density σ = V_h⁻¹ρ is computed in post-processing, where V_h⁻¹
   introduces the corner singularities.  The network never encounters the
   singularity — it only learns a smooth function.

4. THE KEY INSIGHT:

   Both preconditioners achieve equivalent accuracy ({d_err_B*100:.2f}% and
   {d_err_C*100:.2f}% density error) via different mechanisms:
     Left V⁻¹:  changes the LOSS METRIC (all modes equally weighted)
     Right V⁻¹: changes the UNKNOWN (learn smooth ρ, recover singular σ)

   Both require V_h⁻¹, computed once at O(N_q³) cost.  Each loss evaluation
   adds one O(N_q²) matvec.  For Koch(1) (N_q={Nq}), this adds
   ~{overhead_per_iter_ms:.1f} ms per iteration — negligible.
""")

    return dict(
        sigma_A=sigma_A, sigma_B=sigma_B, sigma_C=sigma_C, rho_C=rho_C,
        sigma_BEM=sigma_BEM,
        d_err_A=d_err_A, d_err_B=d_err_B, d_err_C=d_err_C,
        bie_A=bie_A, bie_B=bie_B, bie_C=bie_C,
        iL2_A=iL2_A, iL2_B=iL2_B, iL2_C=iL2_C,
        cond_V=cond_V, Nq=Nq,
    )


if __name__ == "__main__":
    main()
