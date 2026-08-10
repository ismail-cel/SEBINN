"""
Solver comparison: LU, GMRES (no prec), GMRES (Calderón) vs BINN A–D.

Same discretisation as the thesis scaling runs:
  Koch(1): n_per_edge=12, p_gl=16, N_q=2304
  Koch(2): n_per_edge=12, p_gl=16, N_q=9216

BINN results loaded from ex_thesis_scaling/data_final/; verified before use.

Outputs → experiments/ex_solver_comparison/{figures,tables,data}/
"""

from __future__ import annotations

import sys, os, gc, time
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from src.reconstruction.interior import reconstruct_interior

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIG_DIR   = os.path.join(_HERE, "figures")
TAB_DIR   = os.path.join(_HERE, "tables")
DATA_DIR  = os.path.join(_HERE, "data")
THESIS_DATA = os.path.join(_HERE, "..", "ex_thesis_scaling", "data_final")

LEVEL_CFG = {
    1: dict(n_per_edge=12, p_gl=16),
    2: dict(n_per_edge=12, p_gl=16),
}
LEVELS  = [1, 2]
METHODS = ["A", "B", "C", "D"]

# GMRES settings
GMRES_TOL   = 1e-10
GMRES_ATOL  = 0.0     # purely relative tolerance, scipy convention
GMRES_MAX   = 2000    # per-level cap; reported if not reached

# Interior grid
N_GRID = 100

# Expected BINN ranges for Koch(1) sanity check
BINN_EXPECTED_K1 = {
    "A": (0.60, 0.70), "B": (0.02, 0.06),
    "C": (0.08, 0.20), "D": (0.005, 0.015),
}

COLORS = {"A": "#888888", "B": "#1f77b4", "C": "#2ca02c", "D": "#d62728"}

# ---------------------------------------------------------------------------
# Boundary data
# ---------------------------------------------------------------------------

def g_fn(xy): return xy[:, 0]**2 - xy[:, 1]**2
def u_exact(xy): return xy[:, 0]**2 - xy[:, 1]**2

# ---------------------------------------------------------------------------
# Build operators for one Koch level
# ---------------------------------------------------------------------------

def build_level(n: int, n_per_edge: int, p_gl: int, verbose=True) -> dict:
    t0 = time.perf_counter()
    if verbose:
        print(f"  [Koch({n})] assembling (n_pe={n_per_edge}, p={p_gl}) …")

    geom   = make_koch_geometry(n=n)
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=n_per_edge)
    label_corner_ring_panels(panels, P)
    qdata  = build_panel_quadrature(panels, p=p_gl)
    Yq_T   = qdata.Yq.T
    wq     = qdata.wq
    Nq     = qdata.n_quad
    g      = g_fn(Yq_T)

    nmat = assemble_nystrom_matrix(qdata)
    V_h  = nmat.V

    W_h, _  = assemble_hypersingular_corrected(qdata)
    W_tilde = regularise_hypersingular(W_h, wq)
    del W_h; gc.collect()

    x_range = (P[:, 0].min() - 0.05, P[:, 0].max() + 0.05)
    y_range = (P[:, 1].min() - 0.05, P[:, 1].max() + 0.05)

    if verbose:
        print(f"  [Koch({n})] Nq={Nq}  ({time.perf_counter()-t0:.1f}s)")
    return dict(n=n, Nq=Nq, P=P, qdata=qdata,
                Yq_T=Yq_T, wq=wq, g=g,
                V_h=V_h, W_tilde=W_tilde,
                x_range=x_range, y_range=y_range)

# ---------------------------------------------------------------------------
# LU solve
# ---------------------------------------------------------------------------

def run_lu(V_h: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    lu, piv = la.lu_factor(V_h)
    sigma   = la.lu_solve((lu, piv), g)
    wall    = time.perf_counter() - t0
    return sigma, wall

# ---------------------------------------------------------------------------
# LU discretisation error (coarse vs fine mesh)
# ---------------------------------------------------------------------------

def lu_discretisation_error(n: int, n_pe_coarse: int, n_pe_fine: int,
                             p_gl: int, verbose=True) -> float | str:
    """
    Solve on fine mesh, interpolate to coarse arc-length, compare.
    Returns relative error float or '--' string if infeasible.
    """
    Nq_fine_est = 3 * (4**(n-1)) * n_pe_fine * p_gl * 4  # rough
    # Fine mesh memory: Nq_fine^2 * 8 bytes
    geom = make_koch_geometry(n=n)
    n_sides = len(geom.vertices)
    Nq_fine = n_sides * n_pe_fine * p_gl
    mem_gb  = (Nq_fine**2 * 8) / 1e9
    if mem_gb > 8.0:
        if verbose:
            print(f"  [Koch({n})] fine mesh Nq={Nq_fine}, "
                  f"V_h≈{mem_gb:.1f} GB — skipping (too large)")
        return "--"

    if verbose:
        print(f"  [Koch({n})] LU discretisation error: "
              f"fine mesh n_pe={n_pe_fine}, Nq={Nq_fine} …")

    # Build coarse
    P       = geom.vertices
    pans_c  = build_uniform_panels(P, n_per_edge=n_pe_coarse)
    label_corner_ring_panels(pans_c, P)
    qdata_c = build_panel_quadrature(pans_c, p=p_gl)
    Yq_c    = qdata_c.Yq.T
    arc_c   = _arc_lengths(qdata_c)
    g_c     = g_fn(Yq_c)
    nmat_c  = assemble_nystrom_matrix(qdata_c)
    sigma_c, _ = run_lu(nmat_c.V, g_c)

    # Build fine
    pans_f  = build_uniform_panels(P, n_per_edge=n_pe_fine)
    label_corner_ring_panels(pans_f, P)
    qdata_f = build_panel_quadrature(pans_f, p=p_gl)
    Yq_f    = qdata_f.Yq.T
    arc_f   = _arc_lengths(qdata_f)
    g_f     = g_fn(Yq_f)
    nmat_f  = assemble_nystrom_matrix(qdata_f)
    sigma_f, _ = run_lu(nmat_f.V, g_f)

    # Interpolate fine onto coarse arc-length points
    # Both parametrised by the same boundary arclength 0…L_total
    idx_f   = np.argsort(arc_f)
    sigma_f_interp = np.interp(arc_c, arc_f[idx_f], sigma_f[idx_f])

    err = float(np.linalg.norm(sigma_c - sigma_f_interp)
                / np.linalg.norm(sigma_f_interp))
    if verbose:
        print(f"  [Koch({n})] LU discretisation error = {err:.3e}")
    return err


def _arc_lengths(qdata) -> np.ndarray:
    pan_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    return pan_start[qdata.pan_id] + qdata.s_on_panel

# ---------------------------------------------------------------------------
# GMRES
# ---------------------------------------------------------------------------

def run_gmres(A_or_op, b: np.ndarray, label: str,
              M=None, restart: int = 200,
              maxiter: int = GMRES_MAX,
              verbose=True) -> tuple[np.ndarray, int, list[float], float]:
    """
    Run GMRES and record per-iteration residual norms.
    A_or_op: dense matrix or LinearOperator (for Calderón case).
    restart: Krylov subspace restart (default 200).
    Returns (sigma, n_iters, residuals, wall_time).
    """
    residuals = []
    iters     = [0]

    def callback(rk):
        # scipy passes the current residual vector (or norm with callback_type)
        residuals.append(float(np.linalg.norm(rk)))
        iters[0] += 1

    t0 = time.perf_counter()
    import scipy
    gmres_kwargs = dict(
        M=M, restart=restart, maxiter=maxiter,
        callback=callback, callback_type="pr_norm",
    )
    # scipy >= 1.12 renamed tol → rtol; support both
    if tuple(int(x) for x in scipy.__version__.split(".")[:2]) >= (1, 12):
        gmres_kwargs["rtol"] = GMRES_TOL
        gmres_kwargs["atol"] = GMRES_ATOL
    else:
        gmres_kwargs["tol"] = GMRES_TOL
        gmres_kwargs["atol"] = GMRES_ATOL
    sigma, info = spla.gmres(A_or_op, b, **gmres_kwargs)
    wall = time.perf_counter() - t0

    n_iters = iters[0]
    b_norm  = float(np.linalg.norm(b))
    final_res = float(np.linalg.norm(A_or_op @ sigma - b)) / b_norm

    if verbose:
        status = "converged" if info == 0 else f"NOT converged (info={info})"
        print(f"  [{label}] GMRES {status} in {n_iters} iters, "
              f"final rel-res={final_res:.2e}, wall={wall:.1f}s")

    return sigma, n_iters, residuals, wall

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(sigma: np.ndarray, sigma_ref: np.ndarray,
                    V_h: np.ndarray, g: np.ndarray,
                    P, Yq_T, wq, x_range, y_range) -> dict:
    d_err  = float(np.linalg.norm(sigma - sigma_ref)
                   / np.linalg.norm(sigma_ref))
    bie    = float(np.linalg.norm(V_h @ sigma - g) / np.linalg.norm(g))
    rec    = reconstruct_interior(P=P, Yq=Yq_T, wq=wq, sigma=sigma,
                                  n_grid=N_GRID, u_exact=u_exact,
                                  x_range=x_range, y_range=y_range)
    iL2    = float(rec.rel_L2)
    return dict(d_err=d_err, bie=bie, iL2=iL2)

# ---------------------------------------------------------------------------
# Load BINN data from thesis runs
# ---------------------------------------------------------------------------

def load_binn(n: int, sigma_LU: np.ndarray) -> dict:
    path = os.path.join(THESIS_DATA, f"koch{n}_results.npz")
    f    = np.load(path, allow_pickle=True)

    results = {}
    for mid in METHODS:
        sigma = f[f"m{mid}_sigma"]
        d_err = float(np.linalg.norm(sigma - sigma_LU)
                      / np.linalg.norm(sigma_LU))
        results[mid] = dict(
            sigma       = sigma,
            d_err       = d_err,
            bie         = float(f[f"m{mid}_bie"]),
            iL2         = float(f[f"m{mid}_iL2"]),
            wall        = float(f[f"m{mid}_wall"]),
            lbfgs       = str(f[f"m{mid}_lbfgs"]),
        )

    # Sanity check on Koch(1)
    if n == 1:
        print(f"\n  BINN sanity check (Koch(1)):")
        ok = True
        for mid, (lo, hi) in BINN_EXPECTED_K1.items():
            de = results[mid]["d_err"]
            flag = "✓" if lo <= de <= hi else "✗ WARN"
            print(f"    {mid}: d_err={de:.4f}  expected [{lo:.2f},{hi:.2f}]  {flag}")
            if not (lo <= de <= hi):
                ok = False
        if not ok:
            print("  WARNING: loaded BINN numbers outside expected range!")

    return results

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _fv(x, fmt=".3e"):
    if isinstance(x, str):
        return x
    if np.isnan(x):
        return "—"
    return f"{x:{fmt}}"

def _ft(x):
    if isinstance(x, str): return x
    return f"{x:.1f}"

def _fi(x):
    if isinstance(x, str): return x
    return str(int(x))


def make_tables(all_rows: dict, tables_dir: str):
    """
    all_rows[n] = list of dicts with keys:
      label, d_err, bie, iL2, iters, wall
    """
    # Combined CSV
    csv_lines = ["Level,Method,DensityErr,BIERes,InteriorL2,Iterations,WallSec"]
    for n in LEVELS:
        for r in all_rows[n]:
            iters_s = "--" if isinstance(r["iters"], str) else str(r["iters"])
            de_s    = "--" if isinstance(r["d_err"], str) else f"{r['d_err']:.6f}"
            csv_lines.append(
                f"Koch{n},{r['label']},{de_s},"
                f"{r['bie']:.4e},{r['iL2']:.4e},"
                f"{iters_s},{r['wall']:.1f}")
    with open(os.path.join(tables_dir, "solver_comparison.csv"), "w") as f:
        f.write("\n".join(csv_lines))
    print("  saved → solver_comparison.csv")

    # One combined TeX table
    tex = [
        r"\begin{table}[ht]",
        r"\centering",
        (r"\caption{Solver comparison on Koch prefractal domains "
         r"($n_{\rm pe}=12$, $p=16$, $g=x^2-y^2$). "
         r"Density error relative to LU reference $\boldsymbol{\sigma}^*$. "
         r"LU discretisation error estimated against a doubled-mesh reference "
         r"(``—'' if infeasible). "
         r"GMRES tol $=10^{-10}$ (relative, ${\rm atol}=0$), restart $=200$, "
         r"maxiter $=2000$. "
         r"BINN results loaded from the thesis scaling runs.}"),
        r"\label{tab:solver-comparison}",
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
        (r"Level & Method & Density error & BIE residual "
         r"& Interior $L^2$ & Iters & Wall (s) \\"),
        r"\midrule",
    ]

    first_level = True
    for n in LEVELS:
        if not first_level:
            tex.append(r"\midrule")
        first_level = False
        first_row = True
        for r in all_rows[n]:
            lbl = f"Koch($n={n}$)" if first_row else ""
            de_s = "—" if isinstance(r["d_err"], str) else _fv(r["d_err"])
            tex.append(
                f"  {lbl} & {r['label']} & {de_s} & "
                f"{_fv(r['bie'])} & {_fv(r['iL2'])} & "
                f"{_fi(r['iters']) if not isinstance(r['iters'],str) else '—'} & "
                f"{_ft(r['wall'])} \\\\")
            first_row = False

    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(tables_dir, "solver_comparison.tex"), "w") as f:
        f.write("\n".join(tex))
    print("  saved → solver_comparison.tex")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_gmres_convergence(gmres_data: dict, outpath: str):
    """
    gmres_data[n] = {"noprec": list[float], "calderon": list[float]}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    titles = {1: r"Koch($n=1$), $N_q=2304$", 2: r"Koch($n=2$), $N_q=9216$"}

    for ax, n in zip(axes, LEVELS):
        res_np = gmres_data[n]["noprec"]
        res_ca = gmres_data[n]["calderon"]

        if res_np:
            its_np = list(range(1, len(res_np) + 1))
            ax.semilogy(its_np, res_np, "k-", lw=1.8,
                        label=f"No preconditioner ({len(res_np)} iters)")
        if res_ca:
            its_ca = list(range(1, len(res_ca) + 1))
            ax.semilogy(its_ca, res_ca, "b-", lw=1.8,
                        label=f"Calderón precond. ({len(res_ca)} iters)")

        ax.axhline(GMRES_TOL, color="gray", ls="--", lw=1.0, alpha=0.7,
                   label=f"tol = {GMRES_TOL:.0e}")
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel("Relative residual norm", fontsize=11)
        ax.set_title(titles[n], fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)

    fig.suptitle("GMRES convergence: no preconditioner vs Calderón preconditioner",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_accuracy_vs_time(all_rows: dict, outpath: str):
    """
    Log-log scatter: x = wall time, y = density rel. error.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    solver_markers = {
        "LU (reference)":    ("k*", 14, "Classical"),
        "GMRES (no prec)":   ("kD",  9, "Classical"),
        "GMRES (Calderón)":  ("bD",  9, "Classical"),
    }
    binn_colors = COLORS

    # We need numeric d_err for the scatter; skip "--" rows
    for n in LEVELS:
        n_label = f"$n={n}$"
        marker_shape = "o" if n == 1 else "^"
        fill = "full" if n == 1 else "none"

        for r in all_rows[n]:
            if isinstance(r["d_err"], str):
                continue
            de   = r["d_err"]
            wall = r["wall"]
            lbl  = r["label"]
            pt_label = f"{lbl} ({n_label})"

            if lbl in ("LU (reference)", "GMRES (no prec)"):
                ax.loglog(wall, de, "ks" if lbl.startswith("LU") else "kD",
                          ms=10 if lbl.startswith("LU") else 9,
                          fillstyle=fill, label=pt_label)
            elif lbl == "GMRES (Calderón)":
                ax.loglog(wall, de, "bD", ms=9, fillstyle=fill, label=pt_label)
            elif lbl in METHODS:
                ax.loglog(wall, de,
                          marker_shape, color=binn_colors[lbl],
                          ms=10, fillstyle=fill, label=pt_label)
                ax.annotate(lbl, xy=(wall, de), xytext=(5, 4),
                            textcoords="offset points",
                            fontsize=8, color=binn_colors[lbl])

    ax.set_xlabel("Wall time (s)", fontsize=12)
    ax.set_ylabel(r"Density relative error $\|\boldsymbol{\sigma}_\theta - "
                  r"\boldsymbol{\sigma}^*\| / \|\boldsymbol{\sigma}^*\|$",
                  fontsize=11)
    ax.set_title("Accuracy vs wall time: classical solvers and BINN methods\n"
                 r"(filled = $n=1$, open = $n=2$)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 68)
    print("SOLVER COMPARISON — Koch(1,2), same discretisation as thesis runs")
    print("=" * 68)
    print(f"  GMRES: tol={GMRES_TOL:.0e}, atol={GMRES_ATOL}, "
          f"restart=200, maxiter={GMRES_MAX}")
    print(f"  BINN: loaded from {THESIS_DATA}")

    all_rows   = {}
    gmres_data = {}

    for n in LEVELS:
        cfg = LEVEL_CFG[n]
        print(f"\n{'='*68}")
        print(f"KOCH({n})  n_pe={cfg['n_per_edge']}  p={cfg['p_gl']}")
        print(f"{'='*68}")

        # ── 1. Assemble operators ──────────────────────────────────────────
        lv = build_level(n, cfg["n_per_edge"], cfg["p_gl"])
        V_h     = lv["V_h"]
        W_tilde = lv["W_tilde"]
        g       = lv["g"]
        Nq      = lv["Nq"]
        P       = lv["P"]
        Yq_T    = lv["Yq_T"]
        wq      = lv["wq"]
        xr      = lv["x_range"]
        yr      = lv["y_range"]

        # restart cap: min(200, Nq)
        restart = min(200, Nq)

        rows = []

        # ── 2. LU reference solve ──────────────────────────────────────────
        print(f"\n  LU …")
        sigma_LU, wall_LU = run_lu(V_h, g)
        bie_LU  = float(np.linalg.norm(V_h @ sigma_LU - g) / np.linalg.norm(g))
        rec_LU  = reconstruct_interior(P=P, Yq=Yq_T, wq=wq, sigma=sigma_LU,
                                       n_grid=N_GRID, u_exact=u_exact,
                                       x_range=xr, y_range=yr)
        iL2_LU  = float(rec_LU.rel_L2)

        # Discretisation error vs doubled mesh
        n_pe_fine = cfg["n_per_edge"] * 2
        lu_disc_err = lu_discretisation_error(n, cfg["n_per_edge"], n_pe_fine,
                                              cfg["p_gl"])

        print(f"  LU: bie={bie_LU:.2e}  iL2={iL2_LU:.2e}  "
              f"disc_err={lu_disc_err}  wall={wall_LU:.1f}s")
        rows.append(dict(
            label="LU (reference)",
            d_err=lu_disc_err,   # "--" for Koch(2) if too large
            bie=bie_LU, iL2=iL2_LU, iters="--", wall=wall_LU,
        ))

        # Save sigma_LU for BINN reloading
        np.save(os.path.join(DATA_DIR, f"sigma_LU_koch{n}.npy"), sigma_LU)

        # ── 3. GMRES (no preconditioner) ──────────────────────────────────
        print(f"\n  GMRES (no preconditioner) …")
        sigma_np, iters_np, res_np, wall_np = run_gmres(
            V_h, g, label=f"Koch({n})-GMRES-noprec",
            restart=restart, maxiter=GMRES_MAX)
        met_np = compute_metrics(sigma_np, sigma_LU, V_h, g, P, Yq_T, wq, xr, yr)
        rows.append(dict(label="GMRES (no prec)",
                         d_err=met_np["d_err"], bie=met_np["bie"],
                         iL2=met_np["iL2"], iters=iters_np, wall=wall_np))

        # ── 4. GMRES (Calderón left preconditioner) ───────────────────────
        print(f"\n  GMRES (Calderón preconditioner) …")
        # Explicitly preconditioned system: (W̃V) sigma = W̃g
        Wg     = W_tilde @ g
        WV_op  = spla.LinearOperator(
            (Nq, Nq),
            matvec=lambda x, _W=W_tilde, _V=V_h: _W @ (_V @ x),
            dtype=np.float64,
        )
        sigma_ca, iters_ca, res_ca, wall_ca = run_gmres(
            WV_op, Wg, label=f"Koch({n})-GMRES-calderon",
            restart=restart, maxiter=GMRES_MAX)
        met_ca = compute_metrics(sigma_ca, sigma_LU, V_h, g, P, Yq_T, wq, xr, yr)
        rows.append(dict(label="GMRES (Calderón)",
                         d_err=met_ca["d_err"], bie=met_ca["bie"],
                         iL2=met_ca["iL2"], iters=iters_ca, wall=wall_ca))

        gmres_data[n] = {"noprec": res_np, "calderon": res_ca}

        # ── 5. BINN: load from thesis runs ────────────────────────────────
        print(f"\n  Loading BINN results from thesis data …")
        binn = load_binn(n, sigma_LU)
        for mid in METHODS:
            r = binn[mid]
            rows.append(dict(label=mid,
                             d_err=r["d_err"], bie=r["bie"],
                             iL2=r["iL2"], iters="--", wall=r["wall"]))

        all_rows[n] = rows

        # Per-level console summary
        print(f"\n  {'Method':<20} {'d_err':>8}  {'BIE':>8}  "
              f"{'iL2':>8}  {'iters':>6}  {'wall':>7}")
        for r in rows:
            de_s = "--" if isinstance(r["d_err"], str) else f"{r['d_err']:.4f}"
            it_s = r["iters"] if isinstance(r["iters"], str) else str(r["iters"])
            print(f"  {r['label']:<20} {de_s:>8}  {r['bie']:>8.2e}  "
                  f"{r['iL2']:>8.2e}  {it_s:>6}  {r['wall']:>7.1f}s")

        # Free large matrices before next level
        del V_h, W_tilde; gc.collect()

    # ── 6. Save results ────────────────────────────────────────────────────
    np.save(os.path.join(DATA_DIR, "all_rows.npy"), all_rows)
    np.save(os.path.join(DATA_DIR, "gmres_data.npy"), gmres_data)

    # ── 7. Tables ──────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("TABLES")
    print(f"{'='*68}")
    make_tables(all_rows, TAB_DIR)

    # ── 8. Figures ─────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("FIGURES")
    print(f"{'='*68}")
    fig_gmres_convergence(gmres_data,
                          os.path.join(FIG_DIR, "gmres_convergence.png"))
    fig_accuracy_vs_time(all_rows,
                         os.path.join(FIG_DIR, "accuracy_vs_time.png"))

    # ── 9. Final summary ───────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("FINAL SUMMARY")
    print(f"{'='*68}")
    for n in LEVELS:
        print(f"\nKoch({n})  Nq={LEVEL_CFG[n]['n_per_edge']*3*(4**(n-1))*LEVEL_CFG[n]['p_gl']}")
        print(f"  {'Method':<20} {'d_err':>8}  {'BIE':>8}  "
              f"{'iL2':>8}  {'iters':>6}  {'wall':>7}")
        for r in all_rows[n]:
            de_s = "--" if isinstance(r["d_err"], str) else f"{r['d_err']:.4f}"
            it_s = r["iters"] if isinstance(r["iters"], str) else str(r["iters"])
            print(f"  {r['label']:<20} {de_s:>8}  {r['bie']:>8.2e}  "
                  f"{r['iL2']:>8.2e}  {it_s:>6}  {r['wall']:>7.1f}s")

    print(f"\n  BINN data: loaded from thesis runs (NOT retrained)")
    print(f"  Outputs → experiments/ex_solver_comparison/")
    print(f"    figures/: gmres_convergence.png, accuracy_vs_time.png")
    print(f"    tables/:  solver_comparison.{{tex,csv}}")
    print(f"    data/:    sigma_LU_koch{{1,2}}.npy")


if __name__ == "__main__":
    main()
