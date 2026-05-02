"""
Calderón-preconditioned training with the corrected hypersingular assembly.

Mathematical background
-----------------------
The standard BIE loss ||Vσ - g||² has Hessian condition number cond(V)² ≈ 1.8×10⁸.
The corrected hypersingular W̃ satisfies cond_svd(W̃V) = 8.6, so the left-
preconditioned loss

    L_C = ||W̃(Vσ - g)||²

has Hessian cond = cond_svd(W̃V)² = 74 — a 2.4×10⁶ improvement.

Previous attempts (Phase 2) used the old W̃ assembly where W̃V was highly
non-normal (non-normality = 0.82), giving cond_svd(W̃V) = 6185 despite
cond_eig(W̃V) = 13.6.  The new corrected assembly (full self-panel block
+ neighbour corrections, port of MATLAB hypsing_correction_matrix) gives
non-normality = 0.015 → cond_svd ≈ cond_eig ≈ 8.6.

Four cases (same init weights, no enrichment, plain 4×80 tanh):
  A: Standard          L = ||Vσ − g||²              Hessian cond ≈ 1.8×10⁸
  B: Left V⁻¹          L = ||σ − σ_BEM||²            Hessian cond = 1   (gold)
  C: Left Calderón     L = ||W̃_s(Vσ − g)||²         Hessian cond ≈ 74  (THE TEST)
  D: Right V⁻¹         σ = V⁻¹ρ, L = ||ρ − g||²     Hessian cond = 1   (reference)

Geometry: Koch(1), g(x,y) = x²−y², n_per_edge=12, p_gl=16.
Network: 4 hidden layers × 80 neurons, tanh activations, shared init.
Training: Adam 3×1000 at [1e-3,3e-4,1e-4] + L-BFGS 15000 iters.
"""

import sys
import os
import time

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
from src.quadrature.hypersingular import (
    assemble_hypersingular_corrected,
    regularise_hypersingular,
)
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

COLORS = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c", "D": "#ff7f0e", "BEM": "black"}
LWIDTH = 2.0


# ---------------------------------------------------------------------------
# Boundary data
# ---------------------------------------------------------------------------

def g_fn(xy: np.ndarray) -> np.ndarray:
    return xy[:, 0] ** 2 - xy[:, 1] ** 2

def u_exact_fn(xy: np.ndarray) -> np.ndarray:
    return xy[:, 0] ** 2 - xy[:, 1] ** 2


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup():
    print("  Building geometry and operators...")
    geom   = make_koch_geometry(n=1)
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=N_PER_EDGE)
    label_corner_ring_panels(panels, P)
    qdata  = build_panel_quadrature(panels, p=P_GL)
    nmat   = assemble_nystrom_matrix(qdata)

    Yq_T      = qdata.Yq.T                        # (Nq, 2)
    wq        = qdata.wq
    g_values  = g_fn(Yq_T)
    sigma_BEM = solve_bem(nmat, g_values).sigma
    V_h       = nmat.V
    V_inv     = np.linalg.inv(V_h)

    print(f"  Koch(1): N_panels={qdata.n_panels}, N_quad={qdata.n_quad}")
    cond_V = float(np.linalg.svd(V_h, compute_uv=False)[0]
                   / np.linalg.svd(V_h, compute_uv=False)[-1])
    print(f"  cond_svd(V_h) = {cond_V:.0f}")

    # Corrected hypersingular
    print("  Assembling corrected W̃ (full panel correction)...")
    W_h, _      = assemble_hypersingular_corrected(qdata)
    W_tilde     = regularise_hypersingular(W_h, wq)

    # Initial residual ≈ −g (network output ≈ 0 at init)
    r0            = -g_values
    loss_std_init = float(np.mean(r0 ** 2))
    Wr0           = W_tilde @ r0
    loss_prec_init = float(np.mean(Wr0 ** 2))

    # Scale W̃ so initial losses match case A
    scale         = float(np.sqrt(loss_std_init / max(loss_prec_init, 1e-30)))
    W_tilde_scaled = W_tilde * scale

    sv_WV         = np.linalg.svd(W_tilde_scaled @ V_h, compute_uv=False)
    cond_svd_WV   = float(sv_WV[0] / sv_WV[-1])

    print(f"  Initial std loss:       {loss_std_init:.3e}")
    print(f"  Initial W̃ loss (raw):   {loss_prec_init:.3e}")
    print(f"  Scale factor:           {scale:.3e}")
    Wr0s = W_tilde_scaled @ r0
    print(f"  Initial W̃ loss (scaled):{float(np.mean(Wr0s**2)):.3e}  (ratio ≈ 1)")
    print(f"  cond_svd(W̃_s V):        {cond_svd_WV:.2f}  (Hessian cond ≈ {cond_svd_WV**2:.0f})")

    # Arc-length ordering for figures
    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc         = panel_start[qdata.pan_id] + qdata.s_on_panel
    sort_idx    = np.argsort(arc)

    corner_arcs = []
    for vi in range(1, len(P), 2):
        c = P[vi]
        dists = np.linalg.norm(Yq_T - c[None, :], axis=1)
        corner_arcs.append(arc[np.argmin(dists)])
    corner_arcs = sorted(corner_arcs)

    return dict(
        P=P, Yq_T=Yq_T, wq=wq, qdata=qdata,
        g_values=g_values, sigma_BEM=sigma_BEM,
        V_h=V_h, V_inv=V_inv,
        W_tilde=W_tilde, W_tilde_scaled=W_tilde_scaled, scale=scale,
        cond_svd_WV=cond_svd_WV,
        arc=arc, sort_idx=sort_idx, corner_arcs=corner_arcs,
        Nq=qdata.n_quad,
    )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _new_model():
    return build_sigma_w_network(HIDDEN_WIDTH, N_HIDDEN).double()


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def make_losses(V_h_t, V_inv_t, W_s_t, g_t, Yq_t):
    """Return (loss_A, loss_B, loss_C, loss_D) callables: fn(model) -> scalar."""

    def loss_A(model):
        """Standard: L = ||Vσ − g||²."""
        sigma = model(Yq_t).squeeze(-1)
        res   = V_h_t @ sigma - g_t
        return (res ** 2).mean()

    def loss_B(model):
        """Left V⁻¹: L = ||V⁻¹(Vσ−g)||² = ||σ − σ_BEM||²."""
        sigma = model(Yq_t).squeeze(-1)
        res   = V_h_t @ sigma - g_t
        return (V_inv_t @ res).pow(2).mean()

    def loss_C(model):
        """Left Calderón: L = ||W̃_s(Vσ−g)||²."""
        sigma = model(Yq_t).squeeze(-1)
        res   = V_h_t @ sigma - g_t
        return (W_s_t @ res).pow(2).mean()

    def loss_D(model):
        """Right V⁻¹: network learns ρ ≈ g; σ = V⁻¹ρ in post-processing."""
        rho = model(Yq_t).squeeze(-1)
        return (rho - g_t).pow(2).mean()

    return loss_A, loss_B, loss_C, loss_D


def recover_sigma_D(model, Yq_t, V_inv_t):
    with torch.no_grad():
        rho   = model(Yq_t).squeeze(-1)
        sigma = V_inv_t @ rho
    return sigma.numpy()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, loss_fn, sigma_BEM_np, Yq_t, V_inv_t,
          lr_schedule=LR_SCHEDULE, n_lbfgs=N_LBFGS,
          case_label="?", recover_fn=None, verbose=True):
    if recover_fn is None:
        def recover_fn(m):
            with torch.no_grad():
                return m(Yq_t).squeeze(-1).numpy()

    history = {"iter": [], "loss": [], "density_reldiff": [], "grad_norm": []}

    def _record(it, loss_val=None, grad_norm=None):
        with torch.no_grad():
            if loss_val is None:
                loss_val = float(loss_fn(model).detach())
        sigma = recover_fn(model)
        d_err = float(np.linalg.norm(sigma - sigma_BEM_np)
                      / np.linalg.norm(sigma_BEM_np))
        history["iter"].append(it)
        history["loss"].append(loss_val)
        history["density_reldiff"].append(d_err)
        history["grad_norm"].append(grad_norm if grad_norm is not None else float("nan"))
        if verbose and (it % LOG_EVERY == 0 or it <= 1):
            gn_str = f" | gnorm={grad_norm:.2e}" if grad_norm is not None else ""
            print(f"  [{case_label}] iter={it:6d} | loss={loss_val:.3e}"
                  f" | d_err={d_err:.4f}{gn_str}")

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
            # Capture gradient norm every LOG_EVERY steps
            gn = None
            if (global_iter + 1) % LOG_EVERY == 0:
                gn = float(sum(
                    p.grad.norm() ** 2 for p in model.parameters()
                    if p.grad is not None
                ) ** 0.5)
            optimizer.step()
            global_iter += 1
            if global_iter % LOG_EVERY == 0:
                _record(global_iter, float(loss.detach()), gn)

    _record(global_iter)
    if verbose:
        print(f"  [{case_label}] Adam done:  d_err={history['density_reldiff'][-1]:.4f}")

    # --- L-BFGS ---
    opt_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        history_size=LBFGS_MEMORY, line_search_fn="strong_wolfe",
    )
    n_outer   = N_LBFGS // 20
    lbfgs_its = 0

    for _ in range(n_outer):
        def closure():
            opt_lbfgs.zero_grad()
            loss = loss_fn(model)
            loss.backward()
            return loss
        opt_lbfgs.step(closure)
        lbfgs_its += 20
        if lbfgs_its % LOG_EVERY == 0:
            _record(global_iter + lbfgs_its)

    _record(global_iter + lbfgs_its)
    if verbose:
        print(f"  [{case_label}] LBFGS done: d_err={history['density_reldiff'][-1]:.4f}")

    return history


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _corner_lines(ax, corner_arcs, alpha=0.25, color="lightgray"):
    ymin, ymax = ax.get_ylim()
    for ca in corner_arcs:
        ax.axvline(x=ca, color=color, lw=1.0, ls="--", alpha=alpha, zorder=0)


def fig_convergence(hists, adam_cutoff, corner_arcs, outpath):
    labels = {
        "A": r"A: Standard $\|V\sigma-g\|^2$",
        "B": r"B: Left $V^{-1}$  $\|\sigma-\sigma^*\|^2$",
        "C": r"C: Left Calderón  $\|\tilde{W}_s(V\sigma-g)\|^2$",
        "D": r"D: Right $V^{-1}$  $\|\rho-g\|^2$",
    }
    fig, ax = plt.subplots(figsize=(11, 4))
    for key, hist in hists.items():
        iters = hist["iter"]
        derr  = hist["density_reldiff"]
        ax.semilogy(iters, derr, "-", color=COLORS[key], lw=LWIDTH, label=labels[key])
        ax.annotate(f"  {derr[-1]:.4f}",
                    xy=(iters[-1], derr[-1]), fontsize=9,
                    color=COLORS[key], va="center")

    ax.axvline(x=adam_cutoff, color="gray", ls=":", lw=1.0, alpha=0.7,
               label="Adam → L-BFGS")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(
        r"$\|\sigma_\theta - \sigma_\mathrm{BEM}\| / \|\sigma_\mathrm{BEM}\|$",
        fontsize=12)
    ax.set_title(
        r"Density convergence — Koch(1), $g=x^2-y^2$, no enrichment"
        "\n"
        r"Cases A–D: standard / left $V^{-1}$ / left Calderón / right $V^{-1}$",
        fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_density(sigmas, sigma_BEM, arc, sort_idx, corner_arcs, d_errs, outpath):
    arc_s  = arc[sort_idx]
    bem_s  = sigma_BEM[sort_idx]
    keys   = ["A", "B", "C", "D"]
    titles = [
        "(a) Standard BINN",
        r"(b) Left $V^{-1}$",
        r"(c) Left Calderón $\tilde{W}$",
        r"(d) Right $V^{-1}$",
    ]
    all_sigs = [s[sort_idx] for s in [sigmas[k] for k in keys]]
    ymin = min(bem_s.min(), *(s.min() for s in all_sigs)) * 1.05
    ymax = max(bem_s.max(), *(s.max() for s in all_sigs)) * 1.05

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, title, sig, key, de in zip(axes, titles, all_sigs, keys, d_errs):
        ax.plot(arc_s, bem_s, "k--", lw=1.5, alpha=0.7,
                label=r"$\sigma_\mathrm{BEM}$")
        ax.plot(arc_s, sig, "-", lw=LWIDTH, color=COLORS[key], alpha=0.9,
                label=f"d={de:.4f}")
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Arc-length $s$", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.3, alpha=0.4)
        _corner_lines(ax, corner_arcs)

    axes[0].set_ylabel(r"$\sigma(s)$", fontsize=11)
    fig.suptitle(
        r"Final density — Koch(1), $g=x^2-y^2$, no enrichment",
        fontsize=11, y=1.01)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_density_error(sigmas, sigma_BEM, arc, sort_idx, corner_arcs, outpath):
    arc_s = arc[sort_idx]
    bem_s = sigma_BEM[sort_idx]
    keys  = ["A", "B", "C", "D"]
    labels = {
        "A": r"A: Standard",
        "B": r"B: Left $V^{-1}$",
        "C": r"C: Left Calderón",
        "D": r"D: Right $V^{-1}$",
    }
    fig, ax = plt.subplots(figsize=(11, 4))
    ymin = 1e-5
    for key in keys:
        err = np.abs(sigmas[key][sort_idx] - bem_s)
        ax.semilogy(arc_s, err + 1e-15, "-", lw=LWIDTH,
                    color=COLORS[key], label=labels[key])
    _corner_lines(ax, corner_arcs)
    ax.set_xlabel("Arc-length $s$", fontsize=12)
    ax.set_ylabel(r"$|\sigma_\theta(s) - \sigma_\mathrm{BEM}(s)|$", fontsize=12)
    ax.set_title(
        r"Pointwise density error — Koch(1), $g=x^2-y^2$",
        fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
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

    print("\n" + "=" * 72)
    print("CALDERÓN CORRECTED TRAINING — Koch(1), g=x²−y², no enrichment")
    print("=" * 72)

    # ------------------------------------------------------------------
    print("\n--- Step 1: Setup ---")
    data = setup()
    Yq_T       = data["Yq_T"]
    P          = data["P"]
    wq         = data["wq"]
    qdata      = data["qdata"]
    g_values   = data["g_values"]
    sigma_BEM  = data["sigma_BEM"]
    V_h        = data["V_h"]
    V_inv      = data["V_inv"]
    W_tilde_sc = data["W_tilde_scaled"]
    scale      = data["scale"]
    cond_svd_WV= data["cond_svd_WV"]
    arc        = data["arc"]
    sort_idx   = data["sort_idx"]
    corner_arcs= data["corner_arcs"]
    Nq         = data["Nq"]

    # Torch tensors
    Yq_t    = torch.tensor(Yq_T,        dtype=torch.float64)
    g_t     = torch.tensor(g_values,    dtype=torch.float64)
    V_h_t   = torch.tensor(V_h,         dtype=torch.float64)
    V_inv_t = torch.tensor(V_inv,       dtype=torch.float64)
    W_s_t   = torch.tensor(W_tilde_sc,  dtype=torch.float64)

    loss_A, loss_B, loss_C, loss_D = make_losses(V_h_t, V_inv_t, W_s_t, g_t, Yq_t)

    # ------------------------------------------------------------------
    print("\n--- Step 2: Build models from shared init ---")
    torch.manual_seed(SEED)
    init_state = {k: v.clone()
                  for k, v in _new_model().state_dict().items()}

    def _fresh():
        m = _new_model()
        m.load_state_dict({k: v.clone() for k, v in init_state.items()})
        return m

    model_A = _fresh()
    model_B = _fresh()
    model_C = _fresh()
    model_D = _fresh()

    n_params = sum(p.numel() for p in model_A.parameters())
    print(f"  Parameters per model: {n_params}")
    print(f"  Hessian cond: A≈{float(np.linalg.svd(V_h,compute_uv=False)[0]/np.linalg.svd(V_h,compute_uv=False)[-1])**2:.2e}"
          f"  |  B=1  |  C≈{cond_svd_WV**2:.0f}  |  D=1")

    # ------------------------------------------------------------------
    print("\n--- Step 3: Initial loss check ---")
    with torch.no_grad():
        init_vals = {
            "A": float(loss_A(model_A)),
            "B": float(loss_B(model_B)),
            "C": float(loss_C(model_C)),
            "D": float(loss_D(model_D)),
        }
    for k, v in init_vals.items():
        print(f"  init loss {k}: {v:.3e}")

    adam_cutoff = sum(n for n, _ in LR_SCHEDULE)

    # ------------------------------------------------------------------
    hists   = {}
    sigmas  = {}
    times   = {}
    recover = {}

    cases = [
        ("A", model_A, loss_A, "Standard ||Vσ−g||²",         None),
        ("B", model_B, loss_B, "Left V⁻¹ ||σ−σ*||²",         None),
        ("C", model_C, loss_C, "Left Calderón ||W̃_s(Vσ−g)||²", None),
        ("D", model_D, loss_D, "Right V⁻¹ ||ρ−g||², σ=V⁻¹ρ",
         lambda m: recover_sigma_D(m, Yq_t, V_inv_t)),
    ]

    for key, model, loss_fn, title, rec_fn in cases:
        print(f"\n{'='*60}")
        print(f"Case {key}: {title}")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        hists[key] = train(
            model, loss_fn, sigma_BEM, Yq_t, V_inv_t,
            case_label=key, recover_fn=rec_fn,
        )
        times[key] = time.perf_counter() - t0
        recover[key] = rec_fn

    # ------------------------------------------------------------------
    print("\n--- Step 4: Final densities ---")
    with torch.no_grad():
        sigmas["A"] = model_A(Yq_t).squeeze(-1).numpy()
        sigmas["B"] = model_B(Yq_t).squeeze(-1).numpy()
        sigmas["C"] = model_C(Yq_t).squeeze(-1).numpy()
        sigmas["D"] = recover_sigma_D(model_D, Yq_t, V_inv_t)

    def _derr(s):
        return float(np.linalg.norm(s - sigma_BEM) / np.linalg.norm(sigma_BEM))

    def _bie(s):
        return float(np.linalg.norm(V_h @ s - g_values) / np.linalg.norm(g_values))

    def _interior(s):
        res = reconstruct_interior(
            P=P, Yq=Yq_T, wq=wq, sigma=s,
            n_grid=N_GRID_FINAL, u_exact=u_exact_fn,
        )
        return float(res.rel_L2)

    d_errs  = {k: _derr(sigmas[k]) for k in "ABCD"}
    bie_res = {k: _bie(sigmas[k])  for k in "ABCD"}
    iL2     = {k: _interior(sigmas[k]) for k in "ABCD"}

    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"CALDERÓN TRAINING RESULTS — Koch(1), g=x²−y², no enrichment")
    print(f"{'='*72}")
    print(f"  Nq={Nq}, network={N_HIDDEN}×{HIDDEN_WIDTH} tanh")
    print(f"  Training: Adam {LR_SCHEDULE} + L-BFGS {N_LBFGS}")
    print(f"  cond_svd(W̃_s V) = {cond_svd_WV:.2f}  (Hessian cond ≈ {cond_svd_WV**2:.0f})")
    print(f"  W̃ scale factor = {scale:.3e}")
    print()

    hdr = (f"{'Metric':<28s} | {'A: Standard':>13s} | "
           f"{'B: Left V⁻¹':>13s} | {'C: Calderón':>13s} | {'D: Right V⁻¹':>13s}")
    sep = f"{'-'*28}-+-{'-'*13}-+-{'-'*13}-+-{'-'*13}-+-{'-'*13}"
    print(hdr); print(sep)

    def _row(name, vals, fmt):
        return (f"{name:<28s} | {fmt.format(vals['A']):>13s} | "
                f"{fmt.format(vals['B']):>13s} | {fmt.format(vals['C']):>13s} | "
                f"{fmt.format(vals['D']):>13s}")

    print(_row("Density rel-diff",   d_errs,  "{:.4f}"))
    print(_row("BIE residual",       bie_res, "{:.2e}"))
    print(_row("Interior rel L2",    iL2,     "{:.2e}"))
    print(_row("Final loss",
               {k: hists[k]["loss"][-1] for k in "ABCD"},  "{:.2e}"))
    print(_row("Wall time (s)",       times,   "{:.1f}"))
    print(sep)
    impr = {k: d_errs["A"] / max(d_errs[k], 1e-9) for k in "ABCD"}
    print(f"{'Improvement vs A':<28s} | {'—':>13s} | "
          f"{impr['B']:>12.1f}x | {impr['C']:>12.1f}x | {impr['D']:>12.1f}x")

    hess_cond = {"A": "≈1.8e8", "B": "1", "C": f"≈{cond_svd_WV**2:.0f}", "D": "1"}
    print(f"{'Hessian cond (theory)':<28s} | {hess_cond['A']:>13s} | "
          f"{hess_cond['B']:>13s} | {hess_cond['C']:>13s} | {hess_cond['D']:>13s}")

    # Verdict
    print(f"\n{'='*72}")
    if d_errs["C"] < 0.20:
        verdict = f"SUCCESS: Calderón preconditioner WORKS! d_err={d_errs['C']:.4f} < 20%."
    elif d_errs["C"] < 0.50:
        verdict = f"PARTIAL: Some improvement. d_err={d_errs['C']:.4f}  (20–50% range)."
    else:
        verdict = f"INCONCLUSIVE: d_err={d_errs['C']:.4f} ≥ 50%."
    print(f"VERDICT: {verdict}")

    # ------------------------------------------------------------------
    print("\n--- Step 5: Generating figures ---")

    fig_convergence(
        hists, adam_cutoff, corner_arcs,
        os.path.join(fig_dir, "calderon_corrected_convergence.png"),
    )
    fig_density(
        sigmas, sigma_BEM, arc, sort_idx, corner_arcs,
        [d_errs[k] for k in "ABCD"],
        os.path.join(fig_dir, "calderon_corrected_density.png"),
    )
    fig_density_error(
        sigmas, sigma_BEM, arc, sort_idx, corner_arcs,
        os.path.join(fig_dir, "calderon_corrected_density_error.png"),
    )

    print(f"\n  All figures saved to {fig_dir}/")
    return d_errs


if __name__ == "__main__":
    main()
