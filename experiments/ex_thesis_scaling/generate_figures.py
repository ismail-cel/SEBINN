"""
Generate all thesis Section 5 figures and tables from saved Koch(1,2) NPZ data.
Koch(3) deferred (system crash); section written for two levels.

Loads: data_final/koch{1,2}_results.npz
Runs:  conditioning sweep at Koch(2) (matrix assembly only, no training)
Saves: figures_final/  (9 figures)
       tables_final/   (2 tables × {.tex,.csv})
"""

from __future__ import annotations

import sys, os, gc, time, warnings
import numpy as np
import scipy.sparse.linalg as spla

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..")
sys.path.insert(0, _ROOT)

from src.boundary.polygon import make_koch_geometry
from src.boundary.panels import build_uniform_panels, label_corner_ring_panels
from src.quadrature.panel_quad import build_panel_quadrature
from src.quadrature.nystrom import assemble_nystrom_matrix
from src.quadrature.hypersingular import (
    assemble_hypersingular_corrected, regularise_hypersingular,
)
from src.quadrature.tangential_derivative import lagrange_derivative_matrix
from src.quadrature.gauss import gauss_legendre

# ---------------------------------------------------------------------------
# Configuration (must match run_final.py)
# ---------------------------------------------------------------------------

LR_SCHEDULE = [(1000, 1e-3), (1000, 3e-4), (1000, 1e-4)]
METHODS     = ["A", "B", "C", "D"]
COLORS      = {"A": "#888888", "B": "#1f77b4", "C": "#2ca02c", "D": "#d62728"}
LINES       = {"A": "-",       "B": "-",       "C": "-",       "D": "--"}
MARKERS     = {"A": "o",       "B": "s",       "C": "^",       "D": "D"}

FIG_DIR  = os.path.join(_HERE, "figures_final")
TAB_DIR  = os.path.join(_HERE, "tables_final")
DATA_DIR = os.path.join(_HERE, "data_final")

LEVELS = [1, 2]   # Koch(3) deferred

# ---------------------------------------------------------------------------
# D_h helper (on-the-fly, numpy only — for cond_H1 sweep)
# ---------------------------------------------------------------------------

def _make_dh_info(qdata):
    xi, _ = gauss_legendre(qdata.p)
    D_ref  = lagrange_derivative_matrix(xi)
    panels = [(qdata.idx_std[pid], qdata.L_panel[pid])
              for pid in range(qdata.n_panels)]
    return D_ref, panels

def _dh_apply_vec(v: np.ndarray, D_ref, panels) -> np.ndarray:
    out = np.empty_like(v)
    for js, L in panels:
        out[js] = (2.0 / L) * (D_ref @ v[js])
    return out

# ---------------------------------------------------------------------------
# Condition number helpers
# ---------------------------------------------------------------------------

def _cond_V_lanczos(V_h: np.ndarray, k=6) -> float:
    Nq = V_h.shape[0]
    op = spla.LinearOperator((Nq, Nq),
                             matvec=lambda x: V_h @ x,
                             rmatvec=lambda x: V_h.T @ x, dtype=np.float64)
    try:
        sv_max = spla.svds(op, k=k, which='LM', return_singular_vectors=False,
                           maxiter=2000)
        sv_min = spla.svds(op, k=k, which='SM', return_singular_vectors=False,
                           maxiter=5000, tol=1e-6)
        return float(sv_max.max() / max(sv_min.min(), 1e-300))
    except Exception:
        return np.nan

def _cond_H1_hessian(V_h: np.ndarray, D_ref, panels, alpha=1.0, k=12) -> float:
    Nq = V_h.shape[0]
    if Nq <= 5000:
        DV = np.zeros_like(V_h)
        for js, L in panels:
            DV[js, :] = (2.0 / L) * (D_ref @ V_h[js, :])
        H  = V_h.T @ V_h + alpha * (DV.T @ DV)
        del DV; gc.collect()
        ev = np.linalg.eigvalsh(H)
        return float(ev[-1] / max(ev[0], 1e-300))
    else:
        def mv(x):
            Vx  = V_h @ x
            DVx = _dh_apply_vec(Vx, D_ref, panels)
            return V_h.T @ (Vx + alpha * DVx)
        op = spla.LinearOperator((Nq, Nq), matvec=mv, dtype=np.float64)
        try:
            ev_max = spla.eigsh(op, k=k, which='LM', return_eigenvectors=False)
            ev_min = spla.eigsh(op, k=k, which='SM', return_eigenvectors=False,
                                maxiter=8000, tol=1e-5)
            val = float(ev_max.max() / max(ev_min.min(), 1e-300))
            return val if val < 1e200 else np.nan
        except Exception as e:
            print(f"    [cond_H1 Lanczos] warning: {e}")
            return np.nan

# ---------------------------------------------------------------------------
# Load NPZ data
# ---------------------------------------------------------------------------

def load_level(n: int):
    """Return (data_dict, results_dict, grids_or_None)."""
    path = os.path.join(DATA_DIR, f"koch{n}_results.npz")
    f    = np.load(path, allow_pickle=True)

    raw_cH1 = float(f["cond_H1"])
    raw_nnWV = float(f["non_norm_WV"])

    data = dict(
        n          = int(f["n"]),
        Nq         = int(f["Nq"]),
        n_per_edge = int(f["n_per_edge"]),
        p_gl       = int(f["p_gl"]),
        cond_V     = float(f["cond_V"]),
        cond_eig_WV= float(f["cond_eig_WV"]),   # nan for Koch(2)
        cond_svd_WV= float(f["cond_svd_WV"]),
        non_norm_WV= raw_nnWV if raw_nnWV >= 0 else np.nan,
        cond_H1    = raw_cH1 if raw_cH1 < 1e200 else np.nan,
        null_Dh    = int(f["null_Dh"]),
        sigma_BEM  = f["sigma_BEM"],
        g_values   = f["g_values"],
    )

    results = {}
    for mid in METHODS:
        results[mid] = dict(
            sigma       = f[f"m{mid}_sigma"],
            d_err       = float(f[f"m{mid}_d_err"]),
            bie         = float(f[f"m{mid}_bie"]),
            iL2         = float(f[f"m{mid}_iL2"]),
            wall        = float(f[f"m{mid}_wall"]),
            lbfgs_reason= str(f[f"m{mid}_lbfgs"]),
            hist        = dict(
                iter            = f[f"m{mid}_hist_iter"].tolist(),
                loss            = f[f"m{mid}_hist_loss"].tolist(),
                density_reldiff = f[f"m{mid}_hist_derr"].tolist(),
            ),
        )

    grids = None
    if "grid_xv" in f:
        grids = {"xv": f["grid_xv"], "yv": f["grid_yv"]}
        for mid in METHODS:
            grids[mid] = dict(
                Ugrid   = f[f"grid_{mid}_U"],
                Uexgrid = f[f"grid_{mid}_Uex"],
                Egrid   = f[f"grid_{mid}_E"],
                rel_L2  = float(f[f"m{mid}_iL2"]),
            )

    return data, results, grids

# ---------------------------------------------------------------------------
# Rebuild geometry info (arc-length, corners — fast, no training)
# ---------------------------------------------------------------------------

def rebuild_geometry_info(n: int, n_per_edge: int, p_gl: int) -> dict:
    geom   = make_koch_geometry(n=n)
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=n_per_edge)
    label_corner_ring_panels(panels, P)
    qdata  = build_panel_quadrature(panels, p=p_gl)
    Yq_T   = qdata.Yq.T

    pan_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc       = pan_start[qdata.pan_id] + qdata.s_on_panel
    sort_idx  = np.argsort(arc)

    signed_area = 0.5 * np.sum(
        P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    ccw = (signed_area > 0)
    nv  = len(P)
    corner_arcs = []
    for vi in range(nv):
        v_prev = P[(vi - 1) % nv]
        v_curr = P[vi]
        v_next = P[(vi + 1) % nv]
        e1     = v_curr - v_prev
        e2     = v_next - v_curr
        cross  = e1[0] * e2[1] - e1[1] * e2[0]
        if (cross < 0) if ccw else (cross > 0):
            dists = np.linalg.norm(Yq_T - v_curr[None, :], axis=1)
            corner_arcs.append(arc[np.argmin(dists)])

    x_range = (P[:, 0].min() - 0.05, P[:, 0].max() + 0.05)
    y_range = (P[:, 1].min() - 0.05, P[:, 1].max() + 0.05)

    return dict(P=P, arc=arc, sort_idx=sort_idx, corner_arcs=corner_arcs,
                x_range=x_range, y_range=y_range)

# ---------------------------------------------------------------------------
# FIG 1-2: Density profiles
# ---------------------------------------------------------------------------

def fig_density(results: dict, data: dict, geom_info: dict,
                n_level: int, outpath: str):
    arc         = geom_info["arc"]
    sort_idx    = geom_info["sort_idx"]
    corner_arcs = geom_info["corner_arcs"]
    sigma_B     = data["sigma_BEM"]

    arc_s = arc[sort_idx]
    bem_s = sigma_B[sort_idx]
    margin = 0.15 * (bem_s.max() - bem_s.min())
    ymin   = bem_s.min() - margin
    ymax   = bem_s.max() + margin

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True, sharex=True)
    for ax, mid in zip(axes.flatten(), METHODS):
        sig   = results[mid]["sigma"][sort_idx]
        d_err = results[mid]["d_err"]
        n_lines   = min(len(corner_arcs), 30)
        c_alpha   = max(0.2, 0.55 - 0.01 * n_lines)
        for ca in corner_arcs[:n_lines]:
            ax.axvline(ca, color="#cccccc", lw=0.6, zorder=0, alpha=c_alpha)
        ax.plot(arc_s, bem_s, "k--", lw=1.5, alpha=0.7,
                label=r"$\sigma_{\mathrm{BEM}}$")
        ax.plot(arc_s, sig, LINES[mid], color=COLORS[mid], lw=1.6,
                label=f"d={d_err:.4f}")
        ax.set_ylim(ymin, ymax)
        ax.set_title(mid, fontsize=13, fontweight="bold")
        ax.legend(fontsize=8.5, loc="upper right")
        ax.grid(True, lw=0.3, alpha=0.4)

    for ax in axes[1]:
        ax.set_xlabel("Arc-length $s$", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\sigma(s)$", fontsize=11)

    fig.suptitle(
        rf"Density $\sigma_\theta(s)$ — Koch($n={n_level}$), "
        rf"$N_q={data['Nq']}$, $g=x^2-y^2$",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")

# ---------------------------------------------------------------------------
# FIG 3-4: Convergence (density rel-error vs iteration)
# ---------------------------------------------------------------------------

def fig_convergence(results: dict, n_level: int, Nq: int, outpath: str):
    adam_cutoff = sum(ni for ni, _ in LR_SCHEDULE)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for mid in METHODS:
        hist  = results[mid]["hist"]
        d_err = results[mid]["d_err"]
        ax.semilogy(hist["iter"], hist["density_reldiff"],
                    LINES[mid], color=COLORS[mid], lw=2.0, label=mid)
        ax.annotate(
            f" {d_err:.4f}",
            xy=(hist["iter"][-1], hist["density_reldiff"][-1]),
            fontsize=8, color=COLORS[mid], va="center",
        )

    ax.axvline(x=adam_cutoff, color="gray", ls=":", lw=1.2, alpha=0.8,
               label="Adam → L-BFGS")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(
        r"$\|\sigma_\theta - \sigma_{\mathrm{BEM}}\| / \|\sigma_{\mathrm{BEM}}\|$",
        fontsize=12,
    )
    ax.set_title(
        rf"Density relative error during training — Koch($n={n_level}$), "
        rf"$N_q={Nq}$, $g=x^2-y^2$",
        fontsize=11,
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")

# ---------------------------------------------------------------------------
# FIG 5: Conditioning vs Koch level (n=1,2)
# ---------------------------------------------------------------------------

def fig_conditioning_vs_level(all_data: dict, outpath: str):
    levels   = LEVELS
    cond_V   = [all_data[n]["cond_V"]    for n in levels]
    cond_svd = [all_data[n]["cond_svd_WV"] for n in levels]
    cond_H1  = [all_data[n]["cond_H1"]   for n in levels]
    Nqs      = [all_data[n]["Nq"]        for n in levels]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(levels, cond_V,   "k-o", lw=2.0, ms=9,
                label=r"$\kappa(V)$ (single-layer)")
    ax.semilogy(levels, cond_svd, "b-s", lw=2.0, ms=9,
                label=r"$\kappa_{\rm svd}(\widetilde{W}V)$ (Calderón product)")

    valid_H1 = [(n, c) for n, c in zip(levels, cond_H1) if not np.isnan(c)]
    if valid_H1:
        ns_h, cs_h = zip(*valid_H1)
        ax.semilogy(ns_h, cs_h, "g-^", lw=2.0, ms=9,
                    label=r"$\kappa(V^T(I+D_h^TD_h)V)$ ($H^1$ Hessian)")

    for n, cv, cs, Nq in zip(levels, cond_V, cond_svd, Nqs):
        ax.annotate(f" {cv:.1e}", xy=(n, cv), fontsize=8.5, color="black")
        if not np.isnan(cs):
            ax.annotate(f" {cs:.2f}", xy=(n, cs), fontsize=8.5,
                        color="blue", va="bottom")
    for n, ch in zip(levels, cond_H1):
        if not np.isnan(ch):
            ax.annotate(f" {ch:.1e}", xy=(n, ch), fontsize=8.5,
                        color="green", va="top")

    ax.set_xticks(levels)
    ax.set_xticklabels([f"Koch($n={n}$)\n$N_q={Nq}$"
                        for n, Nq in zip(levels, Nqs)])
    ax.set_ylabel("Condition number", fontsize=11)
    ax.set_title("Operator conditioning vs Koch level "
                 r"($n_{\rm pe}=12$, $p=16$)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")

# ---------------------------------------------------------------------------
# FIG 6: Conditioning vs N_q at Koch(2)  — sweep, no training
# ---------------------------------------------------------------------------

def compute_conditioning_sweep(n_level: int = 2,
                                n_pe_list=None,
                                p_gl: int = 16,
                                extra_records: list = None) -> list:
    """
    Compute conditioning for n_pe values where dense SVD is tractable (Nq<=5000).
    For larger Nq, pass pre-computed values via extra_records.
    Koch(2) n_pe ∈ {4,6}: Nq ∈ {3072,4608} — dense SVD reliable (~30-90s each).
    Larger n_pe (8,10,12) use Lanczos for SM singular values which diverges;
    instead the n_pe=12 point is taken from stored training data.
    """
    if n_pe_list is None:
        n_pe_list = [4, 6]   # dense SVD only; n_pe=12 added from stored data

    geom = make_koch_geometry(n=n_level)
    P    = geom.vertices
    records = []

    for n_pe in n_pe_list:
        t0    = time.perf_counter()
        pans  = build_uniform_panels(P, n_per_edge=n_pe)
        label_corner_ring_panels(pans, P)
        qdata = build_panel_quadrature(pans, p=p_gl)
        Nq    = qdata.n_quad
        wq    = qdata.wq

        nmat  = assemble_nystrom_matrix(qdata)
        V_h   = nmat.V

        sv_V   = np.linalg.svd(V_h, compute_uv=False)
        cond_V = float(sv_V[0] / sv_V[-1])

        W_h, _  = assemble_hypersingular_corrected(qdata)
        W_tilde = regularise_hypersingular(W_h, wq)
        del W_h; gc.collect()

        WV          = W_tilde @ V_h
        sv_WV       = np.linalg.svd(WV, compute_uv=False)
        cond_svd_WV = float(sv_WV[0] / sv_WV[-1])
        del WV, W_tilde; gc.collect()

        D_ref, dh_pans = _make_dh_info(qdata)
        cond_H1 = _cond_H1_hessian(V_h, D_ref, dh_pans)
        del V_h; gc.collect()

        records.append(dict(n_pe=n_pe, Nq=Nq, cond_V=cond_V,
                            cond_svd_WV=cond_svd_WV, cond_H1=cond_H1))
        print(f"    n_pe={n_pe:2d}  Nq={Nq:6d}  cond_V={cond_V:.3e}  "
              f"cond_svd_WV={cond_svd_WV:.2f}  cond_H1={cond_H1:.3e}  "
              f"({time.perf_counter()-t0:.1f}s)")

    if extra_records:
        for r in extra_records:
            records.append(r)
            ch = "nan" if np.isnan(r["cond_H1"]) else f"{r['cond_H1']:.3e}"
            print(f"    n_pe={r['n_pe']:2d}  Nq={r['Nq']:6d}  "
                  f"cond_V={r['cond_V']:.3e}  "
                  f"cond_svd_WV={r['cond_svd_WV']:.2f}  "
                  f"cond_H1={ch}  [from stored data]")

    records.sort(key=lambda r: r["Nq"])
    return records


def fig_conditioning_vs_Nq(records: list, n_level: int, outpath: str):
    Nqs      = [r["Nq"]         for r in records]
    cond_V   = [r["cond_V"]     for r in records]
    cond_svd = [r["cond_svd_WV"] for r in records]
    cond_H1  = [r["cond_H1"]    for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(Nqs, cond_V,   "k-o", lw=2.0, ms=8, label=r"$\kappa(V)$")
    ax.semilogy(Nqs, cond_svd, "b-s", lw=2.0, ms=8,
                label=r"$\kappa_{\rm svd}(\widetilde{W}V)$")

    valid_H1 = [(Nq, c) for Nq, c in zip(Nqs, cond_H1) if not np.isnan(c)]
    if valid_H1:
        ns_h, cs_h = zip(*valid_H1)
        ax.semilogy(ns_h, cs_h, "g-^", lw=2.0, ms=8,
                    label=r"$\kappa(V^T(I+D_h^TD_h)V)$")

    for Nq, cv, cs in zip(Nqs, cond_V, cond_svd):
        ax.annotate(f" {cv:.1e}", xy=(Nq, cv), fontsize=8, color="black")
        if not np.isnan(cs):
            ax.annotate(f" {cs:.2f}", xy=(Nq, cs), fontsize=8,
                        color="blue", va="bottom")
    for Nq, ch in zip(Nqs, cond_H1):
        if not np.isnan(ch):
            ax.annotate(f" {ch:.1e}", xy=(Nq, ch), fontsize=8,
                        color="green", va="top")

    ax.set_xlabel(r"$N_q$ (quadrature points)", fontsize=11)
    ax.set_ylabel("Condition number", fontsize=11)
    ax.set_title(
        rf"Operator conditioning vs $N_q$ — Koch($n={n_level}$), $p=16$",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")

# ---------------------------------------------------------------------------
# FIG 7: Density error scaling (n=1,2)
# ---------------------------------------------------------------------------

def fig_density_error_scaling(all_results: dict, all_data: dict, outpath: str):
    Nqs = [all_data[n]["Nq"] for n in LEVELS]

    fig, ax = plt.subplots(figsize=(7, 5))
    for mid in METHODS:
        d_errs = [all_results[n][mid]["d_err"] for n in LEVELS]
        ax.semilogy(LEVELS, d_errs, LINES[mid], color=COLORS[mid],
                    lw=2.0, marker=MARKERS[mid], ms=9, label=mid)
        for n, de in zip(LEVELS, d_errs):
            ax.annotate(f" {de:.4f}", xy=(n, de), fontsize=8,
                        color=COLORS[mid], va="center")

    ax.set_xticks(LEVELS)
    ax.set_xticklabels([f"Koch($n={n}$)\n$N_q={Nq}$"
                        for n, Nq in zip(LEVELS, Nqs)])
    ax.set_ylabel(
        r"$\|\sigma_\theta - \sigma_{\mathrm{BEM}}\| / \|\sigma_{\mathrm{BEM}}\|$",
        fontsize=11,
    )
    ax.set_title(r"Final density relative error vs Koch level, $g = x^2 - y^2$",
                 fontsize=11)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")

# ---------------------------------------------------------------------------
# FIG 8-9: Interior solution and error (Koch(2))
# ---------------------------------------------------------------------------

def fig_interior_solutions(grids: dict, data: dict, geom_info: dict,
                           outpath: str):
    xv = grids["xv"]
    yv = grids["yv"]
    P  = geom_info["P"]

    Uex  = grids[METHODS[0]]["Uexgrid"]
    vmin = np.nanmin(Uex)
    vmax = np.nanmax(Uex)

    fig = plt.figure(figsize=(17, 3.5))
    gs  = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.06],
                           wspace=0.05, left=0.04, right=0.96,
                           top=0.88, bottom=0.08)

    panels_data = [("Exact", Uex, "black")] + \
                  [(mid, grids[mid]["Ugrid"], COLORS[mid]) for mid in METHODS]
    ims = []
    for col, (label, Ugrid, col_color) in enumerate(panels_data):
        ax = fig.add_subplot(gs[0, col])
        im = ax.contourf(xv, yv, Ugrid, levels=50, vmin=vmin, vmax=vmax,
                         cmap="RdBu_r", extend="neither")
        ims.append(im)
        Pc = np.vstack([P, P[0:1]])
        ax.plot(Pc[:, 0], Pc[:, 1], "k-", lw=0.8)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=12, fontweight="bold",
                     color=col_color if label != "Exact" else "black")
        if col > 0:
            mid = METHODS[col - 1]
            ax.text(0.02, 0.02, f"$L^2$={grids[mid]['rel_L2']:.2e}",
                    transform=ax.transAxes, fontsize=7.5, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    cax = fig.add_subplot(gs[0, 5])
    fig.colorbar(ims[0], cax=cax)
    cax.tick_params(labelsize=8)

    fig.suptitle(
        r"Interior solution $u_\theta$ — Koch($n=2$), $g=x^2-y^2$",
        fontsize=11, y=0.98,
    )
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_interior_errors(grids: dict, data: dict, geom_info: dict,
                        outpath: str):
    xv = grids["xv"]
    yv = grids["yv"]
    P  = geom_info["P"]

    abs_errs = [np.abs(grids[mid]["Egrid"]) for mid in METHODS]
    all_err  = np.concatenate([e[~np.isnan(e)] for e in abs_errs])
    vmax_e   = float(np.nanmax(all_err))
    vmin_e   = max(float(np.nanmin(all_err[all_err > 0])), 1e-6 * vmax_e)
    norm     = mcolors.LogNorm(vmin=vmin_e, vmax=vmax_e)

    fig = plt.figure(figsize=(14, 3.5))
    gs  = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.07],
                           wspace=0.05, left=0.04, right=0.96,
                           top=0.88, bottom=0.08)
    ims = []
    for col, mid in enumerate(METHODS):
        ax  = fig.add_subplot(gs[0, col])
        Eg  = np.abs(grids[mid]["Egrid"])
        im  = ax.contourf(xv, yv, Eg, levels=50, norm=norm,
                          cmap="hot_r", extend="both")
        ims.append(im)
        Pc  = np.vstack([P, P[0:1]])
        ax.plot(Pc[:, 0], Pc[:, 1], "k-", lw=0.8)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(mid, fontsize=12, fontweight="bold", color=COLORS[mid])
        ax.text(0.02, 0.02, f"$L^2$={grids[mid]['rel_L2']:.2e}",
                transform=ax.transAxes, fontsize=7.5, va="bottom", color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

    cax = fig.add_subplot(gs[0, 4])
    fig.colorbar(ims[0], cax=cax, format="%5.1e")
    cax.tick_params(labelsize=8)
    cax.set_ylabel(r"$|u_\theta - u_{\rm ex}|$", fontsize=9)

    fig.suptitle(
        r"Interior error $|u_\theta - u_{\rm ex}|$ — Koch($n=2$), $g=x^2-y^2$",
        fontsize=11, y=0.98,
    )
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _f4(x):  return f"{x:.4f}" if not (np.isnan(x) or x > 1e200) else "—"
def _fe(x):  return f"{x:.2e}" if not (np.isnan(x) or x > 1e200) else "—"
def _ft(x):  return f"{x:.0f}" if not np.isnan(x) else "—"


def make_table_main(all_results: dict, all_data: dict, tables_dir: str):
    level_labels = {1: r"Koch($n=1$)", 2: r"Koch($n=2$)"}

    csv_rows = ["Level,Method,DensityErr,BIERes,InteriorL2,WallSec,LBFGS"]
    tex_rows = [
        r"\begin{table}[ht]",
        r"\centering",
        (r"\caption{High-resolution scaling study (Koch $n=1,2$): density error, "
         r"BIE residual, interior $L^2$ error, and wall time. "
         r"Geometry: Koch snowflake, $g(x,y)=x^2-y^2$, network $4\times80$ tanh, "
         r"seed=0. Training: Adam $3\times1000$ + L-BFGS~15\,000 iters. "
         r"$n_{\rm pe}=12$, $p=16$, $N_q = 2304$ (Koch $n=1$) / $9216$ (Koch $n=2$). "
         r"Method C: combined Sobolev $H^1$ loss, $\alpha=1$. Koch($n=3$) deferred.}"),
        r"\label{tab:scaling_main_hires}",
        r"\begin{tabular}{@{}llccccl@{}}",
        r"\toprule",
        (r"Level & Method "
         r"& $\|\sigma_\theta-\sigma^*\|/\|\sigma^*\|$ "
         r"& $\|V\sigma_\theta-g\|/\|g\|$ "
         r"& $\|u_\theta-u_{\rm ex}\|/\|u_{\rm ex}\|$ "
         r"& Wall (s) & L-BFGS \\"),
        r"\midrule",
    ]

    for i, n in enumerate(LEVELS):
        if i > 0:
            tex_rows.append(r"\midrule")
        first = True
        for mid in METHODS:
            r   = all_results[n][mid]
            lbl = level_labels[n] if first else ""
            tex_rows.append(
                f"  {lbl} & {mid} & "
                f"{_f4(r['d_err'])} & {_fe(r['bie'])} & "
                f"{_fe(r['iL2'])} & {_ft(r['wall'])} & "
                f"{r['lbfgs_reason']} \\\\")
            csv_rows.append(
                f"Koch{n},{mid},{r['d_err']:.6f},{r['bie']:.4e},"
                f"{r['iL2']:.4e},{r['wall']:.1f},{r['lbfgs_reason']}")
            first = False

    tex_rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(os.path.join(tables_dir, "main_comparison.tex"), "w") as f:
        f.write("\n".join(tex_rows))
    with open(os.path.join(tables_dir, "main_comparison.csv"), "w") as f:
        f.write("\n".join(csv_rows))
    print(f"  saved → tables_final/main_comparison.{{tex,csv}}")


def make_table_operator(all_data: dict, tables_dir: str):
    tex_rows = [
        r"\begin{table}[ht]",
        r"\centering",
        (r"\caption{Operator scaling for Koch $n=1,2$ ($n_{\rm pe}=12$, $p=16$). "
         r"$\kappa_{\rm eig}(\widetilde{W}V)$: omitted for $N_q>4000$ (Lanczos). "
         r"$\kappa(H^1\text{ Hess.}) = \kappa(V^T(I+D_h^TD_h)V)$. "
         r"Koch($n=3$) deferred.}"),
        r"\label{tab:operator_scaling_hires}",
        r"\begin{tabular}{@{}lrccccr@{}}",
        r"\toprule",
        (r"Level & $N_q$ & $\kappa_{\rm svd}(V)$ "
         r"& $\kappa_{\rm eig}(\widetilde{W}V)$ "
         r"& $\kappa_{\rm svd}(\widetilde{W}V)$ "
         r"& $\kappa(H^1\text{ Hess.})$ "
         r"& $N_{\rm null}(D_h)$ \\"),
        r"\midrule",
    ]
    csv_rows = ["Level,Nq,condV,condEigWV,condSvdWV,condH1,nullDh"]

    for n in LEVELS:
        d  = all_data[n]
        cv = d["cond_V"]
        ce = d["cond_eig_WV"]
        cs = d["cond_svd_WV"]
        ch = d["cond_H1"]
        nd = d["null_Dh"]
        Nq = d["Nq"]

        ce_s = "—" if (np.isnan(ce) or ce > 1e200) else f"{ce:.2f}"
        cs_s = "—" if (np.isnan(cs) or cs > 1e200) else f"{cs:.2f}"
        ch_s = "—" if (np.isnan(ch) or ch > 1e200) else f"{ch:.2e}"

        tex_rows.append(
            f"  Koch($n={n}$) & {Nq} & {cv:.3e} & {ce_s} & {cs_s} & "
            f"{ch_s} & {nd} \\\\")
        csv_rows.append(
            f"Koch{n},{Nq},{cv:.4e},"
            f"{'nan' if (np.isnan(ce) or ce > 1e200) else f'{ce:.4f}'},"
            f"{'nan' if (np.isnan(cs) or cs > 1e200) else f'{cs:.4f}'},"
            f"{'nan' if (np.isnan(ch) or ch > 1e200) else f'{ch:.4e}'},{nd}")

    tex_rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(os.path.join(tables_dir, "operator_scaling.tex"), "w") as f:
        f.write("\n".join(tex_rows))
    with open(os.path.join(tables_dir, "operator_scaling.csv"), "w") as f:
        f.write("\n".join(csv_rows))
    print(f"  saved → tables_final/operator_scaling.{{tex,csv}}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 68)
    print("THESIS SECTION 5 — FIGURE GENERATION (Koch n=1,2 from saved data)")
    print("=" * 68)
    print("  Koch(3) deferred (system crash). Section written for two levels.")

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load saved NPZ data
    # ------------------------------------------------------------------
    print("\n--- Loading saved data ---")
    all_data    = {}
    all_results = {}
    all_grids   = {}

    for n in LEVELS:
        data, results, grids = load_level(n)
        all_data[n]    = data
        all_results[n] = results
        all_grids[n]   = grids
        ch_str = "—" if np.isnan(data["cond_H1"]) else f"{data['cond_H1']:.2e}"
        print(f"  Koch({n}): Nq={data['Nq']}  cond_V={data['cond_V']:.3e}  "
              f"cond_svd_WV={data['cond_svd_WV']:.2f}  cond_H1={ch_str}")
        for mid in METHODS:
            r = results[mid]
            print(f"    {mid}: d_err={r['d_err']:.4f}  bie={r['bie']:.2e}  "
                  f"iL2={r['iL2']:.2e}  wall={r['wall']:.0f}s  {r['lbfgs_reason']}")

    # Sanity check
    dC1 = all_results[1]["C"]["d_err"]
    print(f"\n  Sanity: Method C Koch(1) d_err={dC1:.4f}  "
          f"{'PASS (<20%)' if dC1 < 0.20 else 'WARN: >=20%'}")
    assert dC1 < 0.20, f"Method C d_err={dC1:.4f} >= 0.20; wrong data?"

    # ------------------------------------------------------------------
    # Step 2: Rebuild geometry info (arc-length, corners — fast)
    # ------------------------------------------------------------------
    print("\n--- Rebuilding geometry info ---")
    all_geom = {}
    for n in LEVELS:
        print(f"  Koch({n}) …", end=" ", flush=True)
        gi = rebuild_geometry_info(n, all_data[n]["n_per_edge"],
                                      all_data[n]["p_gl"])
        all_geom[n] = gi
        print(f"Nv={len(gi['P'])}  n_reentrant_corners={len(gi['corner_arcs'])}")
        all_data[n]["P"] = gi["P"]   # needed by interior fig functions

    # ------------------------------------------------------------------
    # Step 3: Conditioning sweep at Koch(2)
    # Sweep n_pe ∈ {4,6} with dense SVD (Nq ≤ 5000); append n_pe=12
    # from stored training data (Lanczos diverges for larger matrices).
    # ------------------------------------------------------------------
    print("\n--- Conditioning sweep at Koch(2), p=16 ---")
    sweep_path  = os.path.join(DATA_DIR, "sweep_cond_koch2.npz")
    d2          = all_data[2]
    extra_pt    = [dict(n_pe=12, Nq=d2["Nq"], cond_V=d2["cond_V"],
                        cond_svd_WV=d2["cond_svd_WV"], cond_H1=d2["cond_H1"])]
    sweep_records = compute_conditioning_sweep(n_level=2, p_gl=16,
                                               extra_records=extra_pt)
    np.savez_compressed(sweep_path,
                        n_pe   = np.array([r["n_pe"]        for r in sweep_records]),
                        Nq     = np.array([r["Nq"]          for r in sweep_records]),
                        cond_V = np.array([r["cond_V"]      for r in sweep_records]),
                        cond_svd_WV = np.array([r["cond_svd_WV"] for r in sweep_records]),
                        cond_H1= np.array([r["cond_H1"]     for r in sweep_records]))
    print(f"  saved → data_final/sweep_cond_koch2.npz")

    # ------------------------------------------------------------------
    # Step 4: Figures
    # ------------------------------------------------------------------
    print(f"\n--- Generating figures → {FIG_DIR} ---")

    # FIG 1-2: Density profiles
    for n in LEVELS:
        fig_density(all_results[n], all_data[n], all_geom[n], n,
                    os.path.join(FIG_DIR, f"density_koch{n}.png"))

    # FIG 3-4: Convergence
    for n in LEVELS:
        fig_convergence(all_results[n], n, all_data[n]["Nq"],
                        os.path.join(FIG_DIR, f"convergence_koch{n}.png"))

    # FIG 5: Conditioning vs level
    fig_conditioning_vs_level(all_data,
                              os.path.join(FIG_DIR, "conditioning_vs_level.png"))

    # FIG 6: Conditioning vs N_q
    fig_conditioning_vs_Nq(sweep_records, n_level=2,
                           outpath=os.path.join(FIG_DIR, "conditioning_vs_Nq.png"))

    # FIG 7: Density error scaling
    fig_density_error_scaling(all_results, all_data,
                              os.path.join(FIG_DIR, "density_error_scaling.png"))

    # FIG 8-9: Interior (Koch(2))
    grids2 = all_grids[2]
    fig_interior_solutions(grids2, all_data[2], all_geom[2],
                           os.path.join(FIG_DIR, "interior_koch2_solutions.png"))
    fig_interior_errors(grids2, all_data[2], all_geom[2],
                        os.path.join(FIG_DIR, "interior_koch2_error.png"))

    # ------------------------------------------------------------------
    # Step 5: Tables
    # ------------------------------------------------------------------
    print(f"\n--- Generating tables → {TAB_DIR} ---")
    make_table_main(all_results, all_data, TAB_DIR)
    make_table_operator(all_data, TAB_DIR)

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print(f"\n{'='*68}")
    print("FINAL SUMMARY")
    print(f"{'='*68}")
    print(f"\n  {'Level':<12} {'Nq':>6}  {'M':>2}  "
          f"{'d_err':>8}  {'BIE':>8}  {'iL2':>8}  {'wall':>7}  L-BFGS")
    for n in LEVELS:
        Nq = all_data[n]["Nq"]
        for mid in METHODS:
            r = all_results[n][mid]
            print(f"  Koch({n})       {Nq:>6}  {mid:>2}  "
                  f"{r['d_err']:>8.4f}  {r['bie']:>8.2e}  "
                  f"{r['iL2']:>8.2e}  {r['wall']:>7.1f}  {r['lbfgs_reason']}")

    print(f"\n  Operator conditioning:")
    print(f"  {'Level':<10}  {'Nq':>6}  {'cond(V)':>10}  "
          f"{'cond_svd(WV)':>12}  {'cond_H1':>12}")
    for n in LEVELS:
        d = all_data[n]
        ch_str = "—" if np.isnan(d["cond_H1"]) else f"{d['cond_H1']:.3e}"
        print(f"  Koch({n})      {d['Nq']:>6}  {d['cond_V']:>10.3e}  "
              f"{d['cond_svd_WV']:>12.2f}  {ch_str:>12}")

    print(f"\n  Checklist:")
    print(f"  ✓ Method C Koch(1) d_err={dC1:.4f} < 20%")
    print(f"  ✓ Colorbars: gridspec dedicated column (interior) + no touching")
    print(f"  ✓ Labels A/B/C/D only (no 'Method' prefix, no 'no enrichment')")
    print(f"  ✓ Convergence plots show density rel-error (not loss)")
    print(f"  ✓ Koch(3) deferred — noted in table captions")
    print(f"\n  9 figures → figures_final/")
    print(f"  2 tables  → tables_final/")
    print(f"  Sweep data → data_final/sweep_cond_koch2.npz")


if __name__ == "__main__":
    main()
