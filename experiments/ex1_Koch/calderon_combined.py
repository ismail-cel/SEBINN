"""
Calderon combined residual loss showcase on Koch(1).

Loss under test
---------------
    L_beta(sigma) = ||r||_2^2 + beta ||W_tilde r||_2^2,
    r = V_h sigma - g_h.

This script compares five PINN/BINN-style training losses on the SAME frozen
Nystrom nodes/operators (no enrichment, no adaptive collocation):

A) Standard:      ||V_h sigma - g_h||^2
B) Left V^{-1}:   ||V_h^{-1}(V_h sigma - g_h)||^2 = ||sigma - sigma_BEM||^2
C) H1 combined:   ||V_h sigma - g_h||^2 + ||D_h(V_h sigma - g_h)||^2
D) Calderon comb: ||V_h sigma - g_h||^2 + beta ||W_tilde(V_h sigma - g_h)||^2
E) Right V^{-1}:  rho network, sigma = V_h^{-1} rho, loss ||rho - g_h||^2

All cases use: 4x80 tanh, seed=0, Koch(1), g=x^2-y^2, no enrichment.
"""

from __future__ import annotations

import copy
import os
import sys
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from src.boundary.polygon import make_koch_geometry
from src.boundary.panels import build_uniform_panels, label_corner_ring_panels
from src.models.sigma_w_net import build_sigma_w_network
from src.quadrature.hypersingular import (
    assemble_hypersingular_corrected,
    regularise_hypersingular,
)
from src.quadrature.nystrom import assemble_nystrom_matrix, solve_bem
from src.quadrature.panel_quad import build_panel_quadrature
from src.quadrature.tangential_derivative import build_tangential_derivative_matrix
from src.reconstruction.interior import reconstruct_interior


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 0
N_PER_EDGE = 12
P_GL = 16
HIDDEN_WIDTH = 80
N_HIDDEN = 4

SWEEP_LR_SCHEDULE = [(500, 1e-3), (500, 1e-4)]
SWEEP_N_LBFGS = 3000

FULL_LR_SCHEDULE = [(1000, 1e-3), (1000, 3e-4), (1000, 1e-4)]
FULL_N_LBFGS = 15000

LBFGS_MEMORY = 30
LOG_EVERY = 200

N_GRID_FINAL = 201

BETA_SWEEP = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
ALPHA_H1 = 1.0

COLORS = {
    "A": "#1f77b4",
    "B": "#d62728",
    "C": "#2ca02c",
    "D": "#9467bd",
    "E": "#ff7f0e",
    "BEM": "black",
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SetupData:
    P: np.ndarray
    qdata: object
    Yq_T: np.ndarray
    wq: np.ndarray
    g_values: np.ndarray
    sigma_BEM: np.ndarray
    V_h: np.ndarray
    V_inv: np.ndarray
    W_tilde: np.ndarray
    D_h: np.ndarray
    DV: np.ndarray
    Dg: np.ndarray
    arc: np.ndarray
    sort_idx: np.ndarray


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------

def g_fn(xy: np.ndarray) -> np.ndarray:
    return xy[:, 0] ** 2 - xy[:, 1] ** 2


def u_exact_fn(xy: np.ndarray) -> np.ndarray:
    return xy[:, 0] ** 2 - xy[:, 1] ** 2


def setup() -> SetupData:
    geom = make_koch_geometry(n=1)
    P = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=N_PER_EDGE)
    label_corner_ring_panels(panels, P)
    qdata = build_panel_quadrature(panels, p=P_GL)

    nmat = assemble_nystrom_matrix(qdata)
    V_h = nmat.V

    Yq_T = qdata.Yq.T
    wq = qdata.wq
    g_values = g_fn(Yq_T)

    bem = solve_bem(nmat, g_values)
    sigma_BEM = bem.sigma
    V_inv = np.linalg.inv(V_h)

    # Corrected hypersingular assembly (complex-corrected panel compensation)
    W_h, _ = assemble_hypersingular_corrected(qdata)
    W_tilde = regularise_hypersingular(W_h, wq)

    # Tangential derivative for H1 baseline
    D_h = build_tangential_derivative_matrix(qdata)
    DV = D_h @ V_h
    Dg = D_h @ g_values

    panel_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc = panel_start[qdata.pan_id] + qdata.s_on_panel
    sort_idx = np.argsort(arc)

    return SetupData(
        P=P,
        qdata=qdata,
        Yq_T=Yq_T,
        wq=wq,
        g_values=g_values,
        sigma_BEM=sigma_BEM,
        V_h=V_h,
        V_inv=V_inv,
        W_tilde=W_tilde,
        D_h=D_h,
        DV=DV,
        Dg=Dg,
        arc=arc,
        sort_idx=sort_idx,
    )


# ---------------------------------------------------------------------------
# Model + losses
# ---------------------------------------------------------------------------

def new_model() -> torch.nn.Module:
    return build_sigma_w_network(HIDDEN_WIDTH, N_HIDDEN).double()


def make_losses(
    V_h_t: torch.Tensor,
    V_inv_t: torch.Tensor,
    W_tilde_t: torch.Tensor,
    DV_t: torch.Tensor,
    Dg_t: torch.Tensor,
    g_t: torch.Tensor,
    Yq_t: torch.Tensor,
):
    def loss_standard(model: torch.nn.Module) -> torch.Tensor:
        sigma = model(Yq_t).squeeze(-1)
        res = V_h_t @ sigma - g_t
        return (res ** 2).mean()

    def loss_left_vinv(model: torch.nn.Module) -> torch.Tensor:
        sigma = model(Yq_t).squeeze(-1)
        res = V_h_t @ sigma - g_t
        pres = V_inv_t @ res
        return (pres ** 2).mean()

    def loss_h1_combined(model: torch.nn.Module, alpha: float = 1.0) -> torch.Tensor:
        sigma = model(Yq_t).squeeze(-1)
        res = V_h_t @ sigma - g_t
        dres = DV_t @ sigma - Dg_t
        return (res ** 2).mean() + alpha * (dres ** 2).mean()

    def loss_calderon_combined(model: torch.nn.Module, beta: float) -> torch.Tensor:
        sigma = model(Yq_t).squeeze(-1)
        res = V_h_t @ sigma - g_t
        wres = W_tilde_t @ res
        return (res ** 2).mean() + beta * (wres ** 2).mean()

    def loss_right_vinv(model: torch.nn.Module) -> torch.Tensor:
        rho = model(Yq_t).squeeze(-1)
        return ((rho - g_t) ** 2).mean()

    return (
        loss_standard,
        loss_left_vinv,
        loss_h1_combined,
        loss_calderon_combined,
        loss_right_vinv,
    )


def recover_sigma_right(model: torch.nn.Module, Yq_t: torch.Tensor, V_inv_t: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        rho = model(Yq_t).squeeze(-1)
        sigma = V_inv_t @ rho
    return sigma.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Training utility
# ---------------------------------------------------------------------------

def train_case(
    model: torch.nn.Module,
    loss_fn,
    sigma_BEM_np: np.ndarray,
    Yq_t: torch.Tensor,
    lr_schedule,
    n_lbfgs: int,
    case_label: str,
    recover_sigma_fn=None,
):
    if recover_sigma_fn is None:
        def recover_sigma_fn(m):
            with torch.no_grad():
                return m(Yq_t).squeeze(-1).detach().cpu().numpy()

    history = {"iter": [], "loss": [], "density_reldiff": []}
    stop_reason = "completed"

    def _record(it: int, loss_val: float | None = None):
        with torch.no_grad():
            if loss_val is None:
                loss_val = float(loss_fn(model).detach().cpu())
            sigma = recover_sigma_fn(model)
            derr = float(np.linalg.norm(sigma - sigma_BEM_np) / max(np.linalg.norm(sigma_BEM_np), 1e-30))
        history["iter"].append(it)
        history["loss"].append(loss_val)
        history["density_reldiff"].append(derr)
        if it % LOG_EVERY == 0:
            print(f"  [{case_label}] iter={it:6d} | loss={loss_val:.3e} | d_err={derr:.4f}")

    # Adam phase
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    it_global = 0
    _record(0)

    for n_it, lr in lr_schedule:
        for pg in opt.param_groups:
            pg["lr"] = lr

        for _ in range(n_it):
            opt.zero_grad()
            loss = loss_fn(model)
            if not torch.isfinite(loss):
                stop_reason = "non-finite loss in Adam"
                print(f"  [{case_label}] stopping Adam: {stop_reason}")
                _record(it_global, float("nan"))
                return history, stop_reason
            loss.backward()
            opt.step()

            it_global += 1
            if it_global % LOG_EVERY == 0:
                _record(it_global, float(loss.detach().cpu()))

    _record(it_global)

    # L-BFGS phase
    opt_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=LBFGS_MEMORY,
        line_search_fn="strong_wolfe",
    )

    n_outer = max(n_lbfgs // 20, 1)
    lbfgs_its = 0
    failed = False

    for _ in range(n_outer):
        def closure():
            opt_lbfgs.zero_grad()
            loss_c = loss_fn(model)
            if not torch.isfinite(loss_c):
                return torch.tensor(1e30, dtype=torch.float64, requires_grad=True)
            loss_c.backward()
            return loss_c

        try:
            loss_out = opt_lbfgs.step(closure)
            if isinstance(loss_out, torch.Tensor) and not torch.isfinite(loss_out):
                failed = True
                stop_reason = "non-finite loss in LBFGS"
                break
        except Exception as exc:  # pragma: no cover
            failed = True
            stop_reason = f"LBFGS exception: {exc}"
            break

        lbfgs_its += 20
        if lbfgs_its % LOG_EVERY == 0:
            _record(it_global + lbfgs_its)

    _record(it_global + lbfgs_its)
    if failed:
        print(f"  [{case_label}] warning: {stop_reason}")

    return history, stop_reason


# ---------------------------------------------------------------------------
# Metrics + plotting
# ---------------------------------------------------------------------------

def evaluate_metrics(
    sigma: np.ndarray,
    sigma_BEM: np.ndarray,
    V_h: np.ndarray,
    g_values: np.ndarray,
    B_mat: np.ndarray,
    f_B: np.ndarray,
    P: np.ndarray,
    Yq_T: np.ndarray,
    wq: np.ndarray,
):
    density_reldiff = float(np.linalg.norm(sigma - sigma_BEM) / max(np.linalg.norm(sigma_BEM), 1e-30))
    bie_residual = float(np.linalg.norm(V_h @ sigma - g_values) / max(np.linalg.norm(g_values), 1e-30))
    raw_mse = float(np.mean((V_h @ sigma - g_values) ** 2))
    b_mse = float(np.mean((B_mat @ sigma - f_B) ** 2))

    interior = reconstruct_interior(
        P=P,
        Yq=Yq_T,
        wq=wq,
        sigma=sigma,
        n_grid=N_GRID_FINAL,
        u_exact=u_exact_fn,
    )
    rel_l2 = float(interior.rel_L2)

    return {
        "density_reldiff": density_reldiff,
        "bie_residual": bie_residual,
        "raw_mse": raw_mse,
        "b_mse": b_mse,
        "interior_rel_l2": rel_l2,
    }


def plot_sweep(sweep_rows, best_beta, outpath):
    # beta=0 shown as horizontal reference; positive betas on log-x
    betas = np.array([r["beta"] for r in sweep_rows], dtype=float)
    derrs = np.array([r["density_reldiff"] for r in sweep_rows], dtype=float)

    mask_pos = betas > 0
    b_pos = betas[mask_pos]
    d_pos = derrs[mask_pos]
    d_std = float(derrs[betas == 0][0])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(b_pos, d_pos, "o-", color="#9467bd", lw=2, ms=6, label="Calderon combined")
    ax.axhline(d_std, color="#1f77b4", ls="--", lw=1.5, label="beta=0 (standard)")

    if best_beta > 0:
        i_best = np.argmin(np.abs(b_pos - best_beta))
        ax.plot(b_pos[i_best], d_pos[i_best], "*", color="red", ms=14, label=f"best beta={best_beta:g}")

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"Density rel-diff")
    ax.set_title(r"Calderon combined sweep: $\|r\|^2 + \beta\|\widetilde{W}r\|^2$")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {outpath}")


def plot_convergence(histories, outpath):
    labels = {
        "A": "A: Standard",
        "B": "B: Left V^{-1}",
        "C": "C: H1 (alpha=1)",
        "D": "D: Calderon combined",
        "E": "E: Right V^{-1}",
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    for key in ["A", "B", "C", "D", "E"]:
        h = histories[key]
        ax.semilogy(h["iter"], h["density_reldiff"], "-", lw=2, color=COLORS[key], label=labels[key])
        ax.annotate(f"{h['density_reldiff'][-1]:.3f}", xy=(h["iter"][-1], h["density_reldiff"][-1]),
                    textcoords="offset points", xytext=(4, 0), fontsize=8, color=COLORS[key])

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Density rel-diff vs BEM")
    ax.set_title("Density convergence across losses")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {outpath}")


def plot_density(sigmas, sigma_bem, arc, sort_idx, outpath):
    s = arc[sort_idx]
    bem = sigma_bem[sort_idx]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(s, bem, "k--", lw=1.8, label="BEM")
    ax.plot(s, sigmas["A"][sort_idx], color=COLORS["A"], lw=1.8, label="A: Standard")
    ax.plot(s, sigmas["B"][sort_idx], color=COLORS["B"], lw=1.8, label="B: Left V^{-1}")
    ax.plot(s, sigmas["C"][sort_idx], color=COLORS["C"], lw=1.8, label="C: H1")
    ax.plot(s, sigmas["D"][sort_idx], color=COLORS["D"], lw=1.8, label="D: Calderon comb")
    ax.plot(s, sigmas["E"][sort_idx], color=COLORS["E"], lw=1.8, label="E: Right V^{-1}")

    ax.set_xlabel("Arc-length s")
    ax.set_ylabel(r"$\sigma(s)$")
    ax.set_title(r"Density comparison vs BEM (Koch(1), $g=x^2-y^2$)")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {outpath}")


def plot_comparison_bar(results, outpath):
    order = ["A", "B", "C", "D", "E"]
    names = ["Standard", "Left V^{-1}", "H1", "Calderon comb", "Right V^{-1}"]
    vals = [results[k]["density_reldiff"] for k in order]

    fig, ax = plt.subplots(figsize=(8.5, 4))
    bars = ax.bar(np.arange(len(order)), vals,
                  color=[COLORS[k] for k in order], alpha=0.9)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Density rel-diff")
    ax.set_title("Method comparison (lower is better)")
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {outpath}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    fig_dir = os.path.join(_HERE, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "=" * 78)
    print("CALDERON COMBINED SHOWCASE — Koch(1), g=x^2-y^2, no enrichment")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Step 1: Assemble operators
    # ------------------------------------------------------------------
    print("\n--- Step 1: Assembling operators ---")
    data = setup()
    Nq = data.qdata.n_quad

    # Alias names matching requested notation
    V_h = data.V_h
    V_inv = data.V_inv
    sigma_BEM = data.sigma_BEM
    W_tilde = data.W_tilde
    D_h = data.D_h

    # For H1/Calderon derived operators
    DV = data.DV
    Dg = data.Dg
    B_mat = W_tilde @ V_h
    f_B = W_tilde @ data.g_values

    # Torch tensors
    Yq_t = torch.tensor(data.Yq_T, dtype=torch.float64)
    g_t = torch.tensor(data.g_values, dtype=torch.float64)
    V_h_t = torch.tensor(V_h, dtype=torch.float64)
    V_inv_t = torch.tensor(V_inv, dtype=torch.float64)
    W_tilde_t = torch.tensor(W_tilde, dtype=torch.float64)
    DV_t = torch.tensor(DV, dtype=torch.float64)
    Dg_t = torch.tensor(Dg, dtype=torch.float64)

    losses = make_losses(V_h_t, V_inv_t, W_tilde_t, DV_t, Dg_t, g_t, Yq_t)
    loss_standard, loss_left_vinv, loss_h1_combined, loss_calderon_combined, loss_right_vinv = losses

    print(f"  Koch(1): N_panels={data.qdata.n_panels}, N_quad={Nq}")
    print(f"  cond(V_h) = {np.linalg.cond(V_h):.3e}")
    print(f"  cond(B=W_tilde*V_h) = {np.linalg.cond(B_mat):.3e}")

    # ------------------------------------------------------------------
    # Step 2: Spectral analysis for combined losses
    # ------------------------------------------------------------------
    print("\n--- Step 2: Spectral analysis ---")
    beta_to_cond = {}
    I = np.eye(Nq)

    for beta in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        M_eff = I + beta * (W_tilde.T @ W_tilde)
        H_factor = V_h.T @ M_eff @ V_h
        sv = np.linalg.svd(H_factor, compute_uv=False)
        cond_H = float(sv[0] / sv[-1])
        beta_to_cond[beta] = cond_H
        print(f"  beta={beta:8.3f}: cond(V^T(I+beta W^T W)V) = {cond_H:.2e}")

    for alpha in [0.1, 1.0, 10.0]:
        M_eff_h1 = V_h.T @ V_h + alpha * (DV.T @ DV)
        sv_h1 = np.linalg.svd(M_eff_h1, compute_uv=False)
        cond_h1 = float(sv_h1[0] / sv_h1[-1])
        print(f"  H1 alpha={alpha:.1f}: cond(V^TV + alpha(DV)^T(DV)) = {cond_h1:.2e}")

    cond_std = float(np.linalg.cond(V_h.T @ V_h))
    print(f"\n  Standard: cond(V^TV) = {cond_std:.2e}")
    print("  V^{-1}:   cond = 1.0")

    # ------------------------------------------------------------------
    # Shared initialization
    # ------------------------------------------------------------------
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    init_state = {k: v.detach().clone() for k, v in new_model().state_dict().items()}

    def fresh_model() -> torch.nn.Module:
        m = new_model()
        m.load_state_dict({k: v.clone() for k, v in init_state.items()})
        return m

    # ------------------------------------------------------------------
    # Step 4: beta sweep (short budget)
    # ------------------------------------------------------------------
    print("\n--- Step 4: beta sweep (short budget) ---")
    print(f"  Budget: Adam {SWEEP_LR_SCHEDULE} + L-BFGS {SWEEP_N_LBFGS}")

    sweep_rows = []
    for beta in BETA_SWEEP:
        label = f"sweep beta={beta:g}"
        m = fresh_model()

        if beta == 0.0:
            loss_fn = loss_standard
            cond_hess = cond_std
        else:
            loss_fn = lambda model, b=beta: loss_calderon_combined(model, b)
            cond_hess = beta_to_cond[beta]

        t0 = time.perf_counter()
        hist, stop_reason = train_case(
            m,
            loss_fn,
            data.sigma_BEM,
            Yq_t,
            SWEEP_LR_SCHEDULE,
            SWEEP_N_LBFGS,
            label,
        )
        wall = time.perf_counter() - t0

        with torch.no_grad():
            sigma = m(Yq_t).squeeze(-1).detach().cpu().numpy()

        met = evaluate_metrics(
            sigma=sigma,
            sigma_BEM=data.sigma_BEM,
            V_h=V_h,
            g_values=data.g_values,
            B_mat=B_mat,
            f_B=f_B,
            P=data.P,
            Yq_T=data.Yq_T,
            wq=data.wq,
        )

        row = {
            "beta": beta,
            "cond_hessian": cond_hess,
            "density_reldiff": met["density_reldiff"],
            "bie_residual": met["bie_residual"],
            "final_loss": hist["loss"][-1],
            "wall_s": wall,
            "stop_reason": stop_reason,
        }
        sweep_rows.append(row)
        print(
            f"  beta={beta:7g} | cond(H)={cond_hess:.2e} | "
            f"d_err={row['density_reldiff']:.4f} | BIE={row['bie_residual']:.2e} | "
            f"loss={row['final_loss']:.2e} | time={wall:.1f}s"
        )

    # Best beta by density error among beta > 0
    pos_rows = [r for r in sweep_rows if r["beta"] > 0]
    best_row = min(pos_rows, key=lambda r: r["density_reldiff"])
    best_beta = float(best_row["beta"])

    print("\n  Sweep table:")
    print(f"  {'beta':>8s} | {'cond(H)':>12s} | {'density rel-diff':>16s} | {'BIE residual':>12s}")
    print("  " + "-" * 60)
    for r in sweep_rows:
        print(
            f"  {r['beta']:8g} | {r['cond_hessian']:12.2e} | "
            f"{r['density_reldiff']:16.4f} | {r['bie_residual']:12.2e}"
        )

    print(f"\n  Selected best beta (lowest density error, beta>0): {best_beta:g}")

    # ------------------------------------------------------------------
    # Step 5: full five-case training
    # ------------------------------------------------------------------
    print("\n--- Step 5: Full training comparison ---")
    print(f"  Budget: Adam {FULL_LR_SCHEDULE} + L-BFGS {FULL_N_LBFGS}")

    cases = {
        "A": {
            "title": "Standard ||Vsigma-g||^2",
            "loss": loss_standard,
            "recover": None,
        },
        "B": {
            "title": "Left V^{-1} ||sigma-sigma*||^2",
            "loss": loss_left_vinv,
            "recover": None,
        },
        "C": {
            "title": f"H1 combined alpha={ALPHA_H1:g}",
            "loss": lambda model: loss_h1_combined(model, ALPHA_H1),
            "recover": None,
        },
        "D": {
            "title": f"Calderon combined beta={best_beta:g}",
            "loss": lambda model, b=best_beta: loss_calderon_combined(model, b),
            "recover": None,
        },
        "E": {
            "title": "Right V^{-1}: ||rho-g||^2, sigma=V^{-1}rho",
            "loss": loss_right_vinv,
            "recover": lambda m: recover_sigma_right(m, Yq_t, V_inv_t),
        },
    }

    histories = {}
    sigmas = {}
    results = {}
    wall_times = {}

    for key in ["A", "B", "C", "D", "E"]:
        print(f"\n{'=' * 70}\nCase {key}: {cases[key]['title']}\n{'=' * 70}")
        m = fresh_model()
        t0 = time.perf_counter()
        hist, stop_reason = train_case(
            m,
            cases[key]["loss"],
            data.sigma_BEM,
            Yq_t,
            FULL_LR_SCHEDULE,
            FULL_N_LBFGS,
            key,
            recover_sigma_fn=cases[key]["recover"],
        )
        wall = time.perf_counter() - t0
        wall_times[key] = wall
        histories[key] = hist

        if cases[key]["recover"] is None:
            with torch.no_grad():
                sigma = m(Yq_t).squeeze(-1).detach().cpu().numpy()
        else:
            sigma = cases[key]["recover"](m)

        sigmas[key] = sigma

        met = evaluate_metrics(
            sigma=sigma,
            sigma_BEM=data.sigma_BEM,
            V_h=V_h,
            g_values=data.g_values,
            B_mat=B_mat,
            f_B=f_B,
            P=data.P,
            Yq_T=data.Yq_T,
            wq=data.wq,
        )
        met["final_loss"] = float(hist["loss"][-1])
        met["stop_reason"] = stop_reason
        results[key] = met

        print(
            f"  [{key}] final: d_err={met['density_reldiff']:.4f} | BIE={met['bie_residual']:.2e} | "
            f"iL2={met['interior_rel_l2']:.2e} | loss={met['final_loss']:.2e} | time={wall:.1f}s"
        )

    # ------------------------------------------------------------------
    # Step 6: comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"FINAL COMPARISON (best beta={best_beta:g})")
    print("=" * 78)
    hdr = (
        f"{'Metric':<24s} | {'A: Standard':>12s} | {'B: Left V^-1':>12s} | "
        f"{'C: H1':>12s} | {'D: Cald':>12s} | {'E: Right V^-1':>12s}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    def row(name, fmt, key):
        return (
            f"{name:<24s} | {fmt.format(results['A'][key]):>12s} | {fmt.format(results['B'][key]):>12s} | "
            f"{fmt.format(results['C'][key]):>12s} | {fmt.format(results['D'][key]):>12s} | {fmt.format(results['E'][key]):>12s}"
        )

    print(row("Density rel-diff", "{:.4f}", "density_reldiff"))
    print(row("BIE residual", "{:.2e}", "bie_residual"))
    print(row("Interior rel L2", "{:.2e}", "interior_rel_l2"))
    print(
        f"{'Wall time (s)':<24s} | {wall_times['A']:12.1f} | {wall_times['B']:12.1f} | "
        f"{wall_times['C']:12.1f} | {wall_times['D']:12.1f} | {wall_times['E']:12.1f}"
    )

    imp = {
        k: results["A"]["density_reldiff"] / max(results[k]["density_reldiff"], 1e-30)
        for k in ["B", "C", "D", "E"]
    }
    print(sep)
    print(
        f"{'Improvement vs A':<24s} | {'-':>12s} | {imp['B']:12.2f}x | {imp['C']:12.2f}x | "
        f"{imp['D']:12.2f}x | {imp['E']:12.2f}x"
    )

    # ------------------------------------------------------------------
    # Step 7: figures
    # ------------------------------------------------------------------
    print("\n--- Step 7: writing figures ---")
    plot_sweep(sweep_rows, best_beta, os.path.join(fig_dir, "calderon_combined_sweep.png"))
    plot_convergence(histories, os.path.join(fig_dir, "calderon_combined_convergence.png"))
    plot_density(sigmas, data.sigma_BEM, data.arc, data.sort_idx,
                 os.path.join(fig_dir, "calderon_combined_density.png"))
    plot_comparison_bar(results, os.path.join(fig_dir, "calderon_combined_comparison.png"))

    print(f"\nAll done. Figures in: {fig_dir}")


if __name__ == "__main__":
    main()
