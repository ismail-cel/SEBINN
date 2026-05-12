"""
2×2 enrichment × preconditioning experiment — Koch(1), manufactured σ.

Decomposes the two independent improvements studied in SEBINN:
  (1) Enrichment: SE-BINN (σ = σ_w + Σ γ_c σ_s^c) vs BINN (σ = σ_w)
  (2) Preconditioning: combined Calderón loss vs standard L² loss

Together these fill the 2×2 table:

  ┌──────────────────┬─────────────────┬────────────────────────────────┐
  │                  │  Standard loss  │  Combined Calderón (β=10)     │
  ├──────────────────┼─────────────────┼────────────────────────────────┤
  │  BINN  (γ=0)     │    Case A       │    Case C                      │
  │  SE-BINN (γ≠0)   │    Case B       │    Case D                      │
  └──────────────────┴─────────────────┴────────────────────────────────┘

The standard loss is  L = ||Vσ−g||²  (Cases 1 and 2 from six_loss_comparison).
The preconditioned loss is  L = ||Vσ−g||² + β||W̃(Vσ−g)||²  (Case 4 there).

Manufactured density
--------------------
  σ_mfg = σ_smooth + S @ γ_true,   γ_true = [+1,−0.5,+1,−0.5,+1,−0.5]
  g_mfg = V · σ_mfg                (exact, self-consistent BIE forcing)
  σ_BEM = σ_mfg                    (exact by construction)
  enrichment energy ≈ 31.5%

Koch(1): 12 edges, n_per_edge=12, p=16, Nq=2304, 6 reentrant corners (ω=4π/3)
Network: 4×80 tanh, shared σ_w initialisation; γ init=0 for SE-BINN cases.
Training: Adam 3×1000 + L-BFGS 15000 iters.
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
from src.quadrature.hypersingular import (
    assemble_hypersingular_corrected,
    regularise_hypersingular,
)
from src.singular.enrichment import SingularEnrichment
from src.models.sigma_w_net import build_sigma_w_network
from src.models.sebinn import SEBINNModel
from src.reconstruction.interior import reconstruct_interior

# ---------------------------------------------------------------------------
# Config
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
GMRES_TOL    = 1e-12
GMRES_MAX    = 3000
BETA         = 10.0   # Combined Calderón weight

GAMMA_TRUE = np.array([+1.0, -0.5, +1.0, -0.5, +1.0, -0.5])

CASES = ["A", "B", "C", "D"]
CASE_LABELS = {
    "A": r"A: BINN, standard $\|Vσ−g\|^2$",
    "B": r"B: SE-BINN, standard $\|Vσ−g\|^2$",
    "C": r"C: BINN, comb. Calderón ($\beta=10$)",
    "D": r"D: SE-BINN, comb. Calderón ($\beta=10$)",
}
COLORS = {"A": "#1f77b4", "B": "#2ca02c", "C": "#d62728", "D": "#ff7f0e"}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup():
    print("  Building Koch(1) geometry …")
    geom   = make_koch_geometry(n=1)
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=N_PER_EDGE)
    label_corner_ring_panels(panels, P)
    qdata  = build_panel_quadrature(panels, p=P_GL)
    Yq_T   = qdata.Yq.T          # (Nq, 2)
    wq     = qdata.wq
    Nq     = qdata.n_quad

    # Cutoff radius = 0.5 × min corner–corner separation
    sing_idx  = geom.singular_corner_indices
    sing_v    = P[sing_idx]
    dists     = [float(np.linalg.norm(sing_v[i] - sing_v[j]))
                 for i in range(len(sing_v)) for j in range(i+1, len(sing_v))]
    R_cut = 0.5 * min(dists)

    # Singular enrichment — per-corner, with cutoff
    enrichment = SingularEnrichment(
        geom=geom, use_cutoff=True, cutoff_radius=R_cut, per_corner_gamma=True,
    )
    sigma_s_Yq = enrichment.precompute(Yq_T)   # (Nq, 6)
    n_sing     = enrichment.n_singular          # 6

    # Nyström matrix
    print("  Assembling Nyström matrix …")
    nmat = assemble_nystrom_matrix(qdata)
    V_h  = nmat.V                   # (Nq, Nq)

    # Smooth density from g_smooth = x²−y²
    f_smooth   = Yq_T[:, 0]**2 - Yq_T[:, 1]**2
    bem_smooth = solve_bem(nmat, f_smooth, tol=GMRES_TOL, max_iter=GMRES_MAX)
    sigma_smooth = bem_smooth.sigma

    # Manufactured density
    sigma_mfg = sigma_smooth + sigma_s_Yq @ GAMMA_TRUE   # (Nq,)
    g_mfg     = V_h @ sigma_mfg                          # g = V·σ_mfg

    # Verify: BEM on g_mfg should recover sigma_mfg
    bem_mfg   = solve_bem(nmat, g_mfg, tol=GMRES_TOL, max_iter=GMRES_MAX)
    sigma_BEM = bem_mfg.sigma      # should ≈ sigma_mfg
    rec_err   = float(np.linalg.norm(sigma_BEM - sigma_mfg)
                      / max(np.linalg.norm(sigma_mfg), 1e-14))
    print(f"  BEM recovery error = {rec_err:.3e}  (target < 1e-8)")

    energy = float(np.linalg.norm(sigma_s_Yq @ GAMMA_TRUE)**2
                   / max(np.linalg.norm(sigma_mfg)**2, 1e-14))
    print(f"  Nq={Nq}, n_sing={n_sing}, enrichment energy={energy*100:.2f}%")

    # Corrected hypersingular W̃
    print("  Assembling corrected W̃ …")
    W_h, _  = assemble_hypersingular_corrected(qdata)
    W_tilde = regularise_hypersingular(W_h, wq)    # (Nq, Nq)

    # Arc-length for plotting
    pan_start  = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc        = pan_start[qdata.pan_id] + qdata.s_on_panel
    sort_idx   = np.argsort(arc)
    corner_arcs = []
    for vi in sing_idx:
        c     = P[vi]
        dists_pt = np.linalg.norm(Yq_T - c[None, :], axis=1)
        corner_arcs.append(arc[np.argmin(dists_pt)])
    corner_arcs = sorted(corner_arcs)

    # Interior u_exact via single-layer quadrature
    def u_exact_fn(xy: np.ndarray) -> np.ndarray:
        Nq_ = len(sigma_mfg)
        _LOG = -1.0 / (2.0 * np.pi)
        N = len(xy)
        u = np.zeros(N)
        for i in range(N):
            diff = Yq_T - xy[i]
            r    = np.linalg.norm(diff, axis=1)
            r    = np.maximum(r, 1e-14)
            u[i] = (_LOG * np.log(r) * wq * sigma_mfg).sum()
        return u

    return dict(
        P=P, geom=geom, Yq_T=Yq_T, wq=wq, qdata=qdata,
        sigma_s_Yq=sigma_s_Yq, n_sing=n_sing,
        V_h=V_h, W_tilde=W_tilde,
        g_mfg=g_mfg, sigma_mfg=sigma_mfg, sigma_BEM=sigma_BEM,
        sigma_smooth=sigma_smooth,
        Nq=Nq, rec_err=rec_err, energy=energy,
        arc=arc, sort_idx=sort_idx, corner_arcs=corner_arcs,
        u_exact_fn=u_exact_fn,
    )


# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------

def _make_binn(init_sw_state: dict) -> torch.nn.Module:
    m = build_sigma_w_network(HIDDEN_WIDTH, N_HIDDEN).double()
    m.load_state_dict({k: v.clone() for k, v in init_sw_state.items()})
    return m


def _make_sebinn(init_sw_state: dict, n_sing: int) -> SEBINNModel:
    m = SEBINNModel(
        hidden_width=HIDDEN_WIDTH, n_hidden=N_HIDDEN,
        n_gamma=n_sing, gamma_init=0.0,
        dtype=torch.float64,
    )
    m.sigma_w_net.load_state_dict({k: v.clone() for k, v in init_sw_state.items()})
    return m


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _sigma_binn(model, Yq_t):
    return model(Yq_t).squeeze(-1)   # (Nq,)


def _sigma_sebinn(model, Yq_t, sigma_s_q_t):
    return model(Yq_t, sigma_s_q_t)  # (Nq,)


def loss_standard_binn(model, Yq_t, V_h_t, g_t):
    s   = _sigma_binn(model, Yq_t)
    res = V_h_t @ s - g_t
    return (res**2).mean()


def loss_standard_sebinn(model, Yq_t, sigma_s_q_t, V_h_t, g_t):
    s   = _sigma_sebinn(model, Yq_t, sigma_s_q_t)
    res = V_h_t @ s - g_t
    return (res**2).mean()


def loss_calderon_binn(model, Yq_t, V_h_t, g_t, W_tilde_t, beta):
    s    = _sigma_binn(model, Yq_t)
    res  = V_h_t @ s - g_t
    Wres = W_tilde_t @ res
    return (res**2).mean() + beta * (Wres**2).mean()


def loss_calderon_sebinn(model, Yq_t, sigma_s_q_t, V_h_t, g_t, W_tilde_t, beta):
    s    = _sigma_sebinn(model, Yq_t, sigma_s_q_t)
    res  = V_h_t @ s - g_t
    Wres = W_tilde_t @ res
    return (res**2).mean() + beta * (Wres**2).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model,
    loss_fn,           # callable(model) -> scalar tensor
    sigma_BEM_np: np.ndarray,
    recover_fn,        # callable(model) -> np.ndarray (Nq,)
    case_label: str,
    verbose: bool = True,
):
    history = {"iter": [], "loss": [], "density_reldiff": [], "gamma": []}

    def _record(it, loss_val=None):
        with torch.no_grad():
            if loss_val is None:
                loss_val = float(loss_fn(model).detach())
        sigma  = recover_fn(model)
        d_err  = float(np.linalg.norm(sigma - sigma_BEM_np)
                       / np.linalg.norm(sigma_BEM_np))
        gamma  = (model.gamma_value() if isinstance(model, SEBINNModel)
                  else None)
        history["iter"].append(it)
        history["loss"].append(loss_val)
        history["density_reldiff"].append(d_err)
        history["gamma"].append(gamma)
        if verbose and it % LOG_EVERY == 0:
            gstr = (f" | γ={np.round(gamma, 3)}" if gamma is not None else "")
            print(f"  [{case_label}] iter={it:6d} | loss={loss_val:.3e}"
                  f" | d_err={d_err:.4f}{gstr}")

    # Adam
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    itr = 0
    _record(0)
    for n_iters, lr in LR_SCHEDULE:
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
        print(f"  [{case_label}] Adam done: d_err={history['density_reldiff'][-1]:.4f}")

    # L-BFGS
    opt_lb = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20,
        history_size=LBFGS_MEMORY, line_search_fn="strong_wolfe",
    )
    n_outer = N_LBFGS // 20
    lb_its  = 0
    loss_at_start = history["loss"][-1]

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

    loss_at_end = history["loss"][-1]
    lbfgs_ratio = loss_at_start / max(loss_at_end, 1e-30)
    made_progress = (loss_at_end < 0.99 * loss_at_start)

    if verbose:
        if made_progress:
            print(f"  [{case_label}] LBFGS done: d_err={history['density_reldiff'][-1]:.4f}"
                  f"  (loss {lbfgs_ratio:.1f}×)")
        else:
            print(f"  [{case_label}] LBFGS STALLED at {loss_at_end:.3e}")

    history["adam_cutoff"]      = itr
    history["lbfgs_ratio"]      = lbfgs_ratio
    history["lbfgs_made_progress"] = made_progress
    return history


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _vlines(ax, corner_arcs, alpha=0.3):
    for ca in corner_arcs:
        ax.axvline(x=ca, color="lightgray", lw=1.2, ls="--", alpha=alpha, zorder=0)


def fig_convergence(results, adam_cutoff, corner_arcs, outpath):
    fig, ax = plt.subplots(figsize=(12, 5))
    for cid in CASES:
        hist = results[cid]["hist"]
        ax.semilogy(hist["iter"], hist["density_reldiff"],
                    "-", color=COLORS[cid], lw=2.0, label=CASE_LABELS[cid])
        ax.annotate(f" {hist['density_reldiff'][-1]:.4f}",
                    xy=(hist["iter"][-1], hist["density_reldiff"][-1]),
                    fontsize=8, color=COLORS[cid], va="center")
    ax.axvline(x=adam_cutoff, color="gray", ls=":", lw=1.0, alpha=0.7,
               label="Adam → L-BFGS")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(r"$\|\sigma_\theta - \sigma_\mathrm{mfg}\| / \|\sigma_\mathrm{mfg}\|$",
                  fontsize=12)
    ax.set_title(
        r"Enrichment $\times$ preconditioning — Koch(1), manufactured $\sigma$",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_density_error(results, sigma_BEM, arc, sort_idx, corner_arcs, outpath):
    arc_s = arc[sort_idx]
    bem_s = sigma_BEM[sort_idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    for cid in CASES:
        sig = results[cid]["sigma"][sort_idx]
        ax.semilogy(arc_s, np.abs(sig - bem_s) + 1e-15,
                    "-", color=COLORS[cid], lw=1.8, label=CASE_LABELS[cid])
    _vlines(ax, corner_arcs, alpha=0.5)
    ax.set_xlabel("Arc-length $s$", fontsize=12)
    ax.set_ylabel(r"$|\sigma_\theta(s) - \sigma_\mathrm{mfg}(s)|$", fontsize=12)
    ax.set_title(
        r"Pointwise density error — Koch(1), manufactured $\sigma$",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_gamma_trajectories(results, outpath):
    se_cases = [c for c in CASES if results[c]["is_sebinn"]]
    if not se_cases:
        return

    n_corners = len(GAMMA_TRUE)
    fig, axes = plt.subplots(1, len(se_cases), figsize=(7 * len(se_cases), 4),
                             sharey=True)
    if len(se_cases) == 1:
        axes = [axes]

    corner_colors = plt.cm.tab10(np.linspace(0, 0.9, n_corners))

    for ax, cid in zip(axes, se_cases):
        hist = results[cid]["hist"]
        gammas = np.array([g for g in hist["gamma"] if g is not None])
        iters  = [it for it, g in zip(hist["iter"], hist["gamma"]) if g is not None]
        for c in range(n_corners):
            ax.plot(iters, gammas[:, c], color=corner_colors[c],
                    lw=1.6, label=f"γ_{c+1} (true={GAMMA_TRUE[c]:+.1f})")
            ax.axhline(GAMMA_TRUE[c], color=corner_colors[c], lw=0.8, ls=":", alpha=0.7)
        ax.axvline(x=results[cid]["hist"]["adam_cutoff"], color="gray",
                   ls=":", lw=1.0, alpha=0.7)
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel(r"$\gamma_c$", fontsize=11)
        ax.set_title(CASE_LABELS[cid], fontsize=10)
        ax.legend(fontsize=7.5, ncol=2)
        ax.grid(True, lw=0.3, alpha=0.4)

    fig.suptitle(
        r"$\gamma_c$ trajectory — SE-BINN cases, Koch(1), manufactured $\sigma$",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_2x2_summary(results, outpath):
    d_errs = {c: results[c]["d_err"] for c in CASES}
    matrix = np.array([
        [d_errs["A"], d_errs["C"]],
        [d_errs["B"], d_errs["D"]],
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    x      = np.arange(2)
    width  = 0.35
    bars_binn   = ax.bar(x - width/2, matrix[0], width, label="BINN (no enrichment)",
                         color=["#1f77b4", "#d62728"], edgecolor="black", lw=1.2)
    bars_sebinn = ax.bar(x + width/2, matrix[1], width, label="SE-BINN (enriched)",
                         color=["#2ca02c", "#ff7f0e"], edgecolor="black", lw=1.2,
                         hatch="//")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["Standard loss", r"Combined Calderón ($\beta=10$)"], fontsize=11)
    ax.set_ylabel("Density rel-diff (log scale)", fontsize=11)
    ax.set_title(
        r"2×2: enrichment × preconditioning — Koch(1), manufactured $\sigma$",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    for bars in [bars_binn, bars_sebinn]:
        for bar, d in zip(bars, [matrix[0 if bars is bars_binn else 1, 0],
                                  matrix[0 if bars is bars_binn else 1, 1]]):
            ax.text(bar.get_x() + bar.get_width() / 2, d * 1.5,
                    f"{d:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.grid(True, axis="y", which="both", lw=0.3, alpha=0.5)
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
    print("ENRICHMENT × PRECONDITIONING — Koch(1), manufactured σ")
    print("=" * 70)

    # --- Setup ---
    print("\n--- Step 1: Setup ---")
    data       = setup()
    Yq_T       = data["Yq_T"]
    V_h        = data["V_h"]
    W_tilde    = data["W_tilde"]
    g_mfg      = data["g_mfg"]
    sigma_mfg  = data["sigma_mfg"]
    sigma_BEM  = data["sigma_BEM"]
    sigma_s_Yq = data["sigma_s_Yq"]
    n_sing     = data["n_sing"]
    Nq         = data["Nq"]
    arc        = data["arc"]
    sort_idx   = data["sort_idx"]
    corner_arcs = data["corner_arcs"]
    P          = data["P"]
    wq         = data["wq"]
    u_exact_fn = data["u_exact_fn"]

    # --- Tensors ---
    Yq_t        = torch.tensor(Yq_T,        dtype=torch.float64)
    V_h_t       = torch.tensor(V_h,         dtype=torch.float64)
    W_tilde_t   = torch.tensor(W_tilde,     dtype=torch.float64)
    g_t         = torch.tensor(g_mfg,       dtype=torch.float64)
    sigma_s_q_t = torch.tensor(sigma_s_Yq,  dtype=torch.float64)

    # --- Shared sigma_w init ---
    print("\n--- Step 2: Shared σ_w initialisation ---")
    torch.manual_seed(SEED)
    base_sw      = build_sigma_w_network(HIDDEN_WIDTH, N_HIDDEN).double()
    init_sw_state = {k: v.clone() for k, v in base_sw.state_dict().items()}
    n_params_sw  = sum(p.numel() for p in base_sw.parameters())
    print(f"  σ_w parameters: {n_params_sw}")
    print(f"  SE-BINN adds {n_sing} γ parameters → total = {n_params_sw + n_sing}")

    # Initial loss values
    with torch.no_grad():
        m0   = _make_binn(init_sw_state)
        s0   = _sigma_binn(m0, Yq_t)
        res0 = V_h_t @ s0 - g_t
        il_std = float((res0**2).mean())
        il_cal = float(il_std + BETA * ((W_tilde_t @ res0)**2).mean())
    print(f"  Init standard loss : {il_std:.3e}")
    print(f"  Init Calderón loss : {il_cal:.3e}  (ratio {il_cal/il_std:.2f}×)")

    # --- Case definitions ---
    adam_cutoff = sum(n for n, _ in LR_SCHEDULE)
    results     = {}

    case_configs = {
        "A": dict(is_sebinn=False, loss_type="standard"),
        "B": dict(is_sebinn=True,  loss_type="standard"),
        "C": dict(is_sebinn=False, loss_type="calderon"),
        "D": dict(is_sebinn=True,  loss_type="calderon"),
    }

    for cid, cfg in case_configs.items():
        is_sebinn = cfg["is_sebinn"]
        loss_type = cfg["loss_type"]
        label_str = f"Case {cid}: {'SE-BINN' if is_sebinn else 'BINN'} + {'combined Calderón' if loss_type == 'calderon' else 'standard'}"

        print("\n" + "=" * 60)
        print(label_str)
        print("=" * 60)

        # Build model
        if is_sebinn:
            model = _make_sebinn(init_sw_state, n_sing)
        else:
            model = _make_binn(init_sw_state)

        # Build loss
        if loss_type == "standard":
            if is_sebinn:
                def _lf(m, _Yq=Yq_t, _ss=sigma_s_q_t, _V=V_h_t, _g=g_t):
                    return loss_standard_sebinn(m, _Yq, _ss, _V, _g)
            else:
                def _lf(m, _Yq=Yq_t, _V=V_h_t, _g=g_t):
                    return loss_standard_binn(m, _Yq, _V, _g)
        else:  # calderon
            if is_sebinn:
                def _lf(m, _Yq=Yq_t, _ss=sigma_s_q_t, _V=V_h_t, _g=g_t, _W=W_tilde_t):
                    return loss_calderon_sebinn(m, _Yq, _ss, _V, _g, _W, BETA)
            else:
                def _lf(m, _Yq=Yq_t, _V=V_h_t, _g=g_t, _W=W_tilde_t):
                    return loss_calderon_binn(m, _Yq, _V, _g, _W, BETA)

        # Recover function
        if is_sebinn:
            def recover(m, _Yq=Yq_t, _ss=sigma_s_q_t):
                with torch.no_grad():
                    return m(_Yq, _ss).numpy()
        else:
            def recover(m, _Yq=Yq_t):
                with torch.no_grad():
                    return m(_Yq).squeeze(-1).numpy()

        t0   = time.perf_counter()
        hist = train(model, _lf, sigma_BEM, recover, case_label=cid, verbose=True)
        wall = time.perf_counter() - t0

        # Final metrics
        sigma   = recover(model)
        d_err   = float(np.linalg.norm(sigma - sigma_BEM)
                        / np.linalg.norm(sigma_BEM))
        bie_res = float(np.linalg.norm(V_h @ sigma - g_mfg)
                        / np.linalg.norm(g_mfg))
        iL2     = float(reconstruct_interior(
            P=P, Yq=Yq_T, wq=wq, sigma=sigma,
            n_grid=N_GRID_FINAL, u_exact=u_exact_fn,
        ).rel_L2)

        gamma_final = None
        gamma_err   = None
        if is_sebinn:
            g_val = model.gamma_value()
            gamma_final = np.array(g_val if isinstance(g_val, list) else [float(g_val)])
            gamma_err   = float(np.linalg.norm(gamma_final - GAMMA_TRUE)
                                / np.linalg.norm(GAMMA_TRUE))
            print(f"  γ_final = {np.round(gamma_final, 4)}")
            print(f"  γ_true  = {np.round(GAMMA_TRUE, 4)}")
            print(f"  γ_err   = {gamma_err:.4f}")

        results[cid] = {
            "hist": hist, "sigma": sigma,
            "d_err": d_err, "bie": bie_res, "iL2": iL2,
            "wall": wall, "is_sebinn": is_sebinn,
            "gamma_final": gamma_final, "gamma_err": gamma_err,
        }

    # --- Summary table ---
    print(f"\n{'='*90}")
    print(f"ENRICHMENT × PRECONDITIONING SUMMARY — Koch(1), manufactured σ")
    print(f"{'='*90}")
    print(f"  Nq={Nq} | network={N_HIDDEN}×{HIDDEN_WIDTH} tanh | seed={SEED}")
    print(f"  γ_true = {list(np.round(GAMMA_TRUE, 2))}")
    print(f"  Training: Adam {LR_SCHEDULE} + L-BFGS {N_LBFGS}")

    w   = 12
    sep = "─" * 30 + "─┬─" + ("─" * w + "─┬─") * 3 + "─" * w
    hdr = f"{'Metric':<30s} │ " + " │ ".join(f"{'Case '+c:>{w}s}" for c in CASES)
    print(f"\n{sep}\n{hdr}\n{sep}")

    rows = [
        ("Model",            [("SE-BINN" if results[c]["is_sebinn"] else "BINN") for c in CASES]),
        ("Loss",             [("standard" if c in ("A","B") else "comb.Calderón") for c in CASES]),
        ("Density rel-diff", [f"{results[c]['d_err']:.4f}" for c in CASES]),
        ("BIE residual",     [f"{results[c]['bie']:.2e}"   for c in CASES]),
        ("Interior rel L2",  [f"{results[c]['iL2']:.2e}"   for c in CASES]),
        ("γ_err (SE-BINN)",  [f"{results[c]['gamma_err']:.4f}" if results[c]["gamma_err"] is not None else "—"
                               for c in CASES]),
        ("L-BFGS",           [("converges" if results[c]["hist"]["lbfgs_made_progress"] else "stalls") for c in CASES]),
        ("Wall time (s)",    [f"{results[c]['wall']:.1f}"  for c in CASES]),
    ]
    for name, vals in rows:
        print(f"{name:<30s} │ " + " │ ".join(f"{v:>{w}s}" for v in vals))
    print(sep)

    d_A = results["A"]["d_err"]
    improvements = []
    for c in CASES:
        d = results[c]["d_err"]
        if c == "A":
            improvements.append("—")
        else:
            improvements.append(f"{d_A / max(d, 1e-9):.1f}×")
    print(f"{'Improvement vs Case A':<30s} │ " + " │ ".join(f"{v:>{w}s}" for v in improvements))
    print(sep)

    # 2×2 effect decomposition
    d_A = results["A"]["d_err"]
    d_B = results["B"]["d_err"]
    d_C = results["C"]["d_err"]
    d_D = results["D"]["d_err"]
    print(f"\n  Effect of enrichment (standard loss):      A→B  {d_A/max(d_B,1e-9):.1f}×")
    print(f"  Effect of preconditioning (BINN):          A→C  {d_A/max(d_C,1e-9):.1f}×")
    print(f"  Effect of preconditioning (SE-BINN):       B→D  {d_B/max(d_D,1e-9):.1f}×")
    print(f"  Effect of enrichment (preconditioned loss): C→D  {d_C/max(d_D,1e-9):.1f}×")
    print(f"  Combined (D vs A):                         A→D  {d_A/max(d_D,1e-9):.1f}×")

    # --- Figures ---
    print("\n--- Step 3: Figures ---")
    fig_convergence(
        results, adam_cutoff, corner_arcs,
        os.path.join(fig_dir, "enr_prec_convergence.png"),
    )
    fig_density_error(
        results, sigma_BEM, arc, sort_idx, corner_arcs,
        os.path.join(fig_dir, "enr_prec_density_error.png"),
    )
    fig_gamma_trajectories(
        results,
        os.path.join(fig_dir, "enr_prec_gamma.png"),
    )
    fig_2x2_summary(
        results,
        os.path.join(fig_dir, "enr_prec_2x2_summary.png"),
    )

    print(f"\n  All figures saved to {fig_dir}/")
    return results


if __name__ == "__main__":
    main()
