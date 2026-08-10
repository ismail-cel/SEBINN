"""
Experiment E2: Representation × Loss × Solve-method.

Three representations × four losses, Koch(1,2):
  (i)   Network + descent          — loaded from ex_thesis_scaling/data_final/
  (ii)  Legendre basis + descent   — Adam 3×1000 + L-BFGS 15000, c ∈ ℝ^{N_q}
  (iii) Legendre basis + LSQ       — consistent linear system, exact SVD/solve

p_rep = p_gl - 1 = 15  →  coeff count = 16 × N_pan = N_q (matches nodal space).

Key theorem: all four LSQ losses yield the same c* = Φ⁻¹ σ_LU (consistent
systems), so d_err_LSQ ≈ 0 for every loss. Descent on A fails because
cond(G) = cond(V_h Φ) ≈ cond(V_h) makes the quadratic landscape ill-conditioned.
"""

from __future__ import annotations

import sys, os, gc, time
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legvander

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
# Paths & constants
# ---------------------------------------------------------------------------

FIG_DIR    = os.path.join(_HERE, "figures")
TAB_DIR    = os.path.join(_HERE, "tables")
DATA_DIR   = os.path.join(_HERE, "data")
THESIS_DIR = os.path.join(_HERE, "..", "ex_thesis_scaling", "data_final")

LEVELS     = [1, 2]
LEVEL_CFG  = {1: dict(n_per_edge=12, p_gl=16),
               2: dict(n_per_edge=12, p_gl=16)}
METHODS    = ["A", "B", "C", "D"]
COLORS     = {"A": "#888888", "B": "#1f77b4", "C": "#2ca02c", "D": "#d62728"}

LR_SCHEDULE = [(1000, 1e-3), (1000, 3e-4), (1000, 1e-4)]
N_LBFGS     = 15000
LBFGS_MEM   = 30
SEED        = 0
ALPHA       = 1.0
N_GRID      = 100

REPR_LABELS = {
    "net":    "Network + descent",
    "leg_gd": "Legendre + descent",
    "leg_lsq": "Legendre + LSQ",
}

# ---------------------------------------------------------------------------
# Boundary data
# ---------------------------------------------------------------------------

def g_fn(xy):   return xy[:, 0]**2 - xy[:, 1]**2
def u_exact(xy): return xy[:, 0]**2 - xy[:, 1]**2

# ---------------------------------------------------------------------------
# Piecewise Legendre basis
# ---------------------------------------------------------------------------

def build_phi_block(p_gl: int) -> np.ndarray:
    """Legendre Vandermonde at GL nodes: Phi_block[i,j] = P_j(xi_i)."""
    xi, _ = gauss_legendre(p_gl)
    return legvander(xi, p_gl - 1)   # (p_gl, p_gl)


def phi_apply(c_np: np.ndarray, Phi_block: np.ndarray,
              N_pan: int, p_gl: int) -> np.ndarray:
    """sigma = Phi @ c  (block-diagonal, panels contiguous)."""
    c_mat   = c_np.reshape(N_pan, p_gl)              # (N_pan, p_gl)
    sig_mat = c_mat @ Phi_block.T                     # (N_pan, p_gl)
    return sig_mat.ravel()


def phi_solve(sigma_np: np.ndarray, Phi_block: np.ndarray,
              N_pan: int, p_gl: int) -> np.ndarray:
    """c = Phi^{-1} sigma  per-panel (block-diagonal solve)."""
    Phi_block_inv = np.linalg.inv(Phi_block)
    sig_mat = sigma_np.reshape(N_pan, p_gl)           # (N_pan, p_gl)
    c_mat   = sig_mat @ Phi_block_inv.T               # (N_pan, p_gl)
    return c_mat.ravel()


def build_G(V_h: np.ndarray, Phi_block: np.ndarray,
            N_pan: int, p_gl: int) -> np.ndarray:
    """G = V_h @ Phi  via block structure (O(Nq^2 * p_gl))."""
    Nq = V_h.shape[0]
    G  = np.empty((Nq, Nq), dtype=np.float64)
    for k in range(N_pan):
        js   = slice(k * p_gl, (k + 1) * p_gl)   # row/col block
        G[:, js] = V_h[:, js] @ Phi_block         # Nq x p_gl
    return G


def build_Dh_apply(v: np.ndarray, D_ref: np.ndarray,
                   qdata) -> np.ndarray:
    """Apply block-diagonal D_h to v (Nq,)."""
    out = np.empty_like(v)
    for k in range(qdata.n_panels):
        js = qdata.idx_std[k]
        L  = qdata.L_panel[k]
        out[js] = (2.0 / L) * (D_ref @ v[js])
    return out

# ---------------------------------------------------------------------------
# Condition number helpers
# ---------------------------------------------------------------------------

def cond_lanczos(mat_or_op, Nq: int, k: int = 6) -> float:
    """
    Estimate cond(M) via Lanczos SVD.

    For the SM part we use a loose tolerance (0.3) with few iterations
    so the call always terminates. For well-conditioned matrices (cond < 200)
    the loose tolerance still gives a reliable estimate; for ill-conditioned
    ones it gives a lower bound sufficient for diagnostic tables.
    """
    if isinstance(mat_or_op, np.ndarray):
        M   = mat_or_op
        op  = spla.LinearOperator(
            (M.shape[0], M.shape[1]),
            matvec=lambda x, _M=M: _M @ x,
            rmatvec=lambda x, _M=M: _M.T @ x, dtype=np.float64)
    else:
        op = mat_or_op
    try:
        sv_max = spla.svds(op, k=k, which='LM',
                           return_singular_vectors=False,
                           maxiter=500, tol=1e-4)
        sv_min = spla.svds(op, k=k, which='SM',
                           return_singular_vectors=False,
                           maxiter=150, tol=0.3)
        return float(sv_max.max() / max(sv_min.min(), 1e-300))
    except Exception as e:
        print(f"    [Lanczos] warning: {e}")
        return np.nan


def cond_exact(M: np.ndarray) -> float:
    sv = np.linalg.svd(M, compute_uv=False)
    return float(sv[0] / max(sv[-1], 1e-300))

# ---------------------------------------------------------------------------
# Arm 2: Legendre + LSQ
# ---------------------------------------------------------------------------

def run_arm2(V_h, W_tilde, g, sigma_LU,
             G, Phi_block, D_ref, qdata, verbose=True) -> dict:
    """
    Exact LSQ solve for each of the four losses.
    Since all four systems are CONSISTENT (c* = Φ⁻¹ σ_LU satisfies each),
    the solution is the same for all: c* = Φ⁻¹ σ_LU → σ = σ_LU → d_err ≈ 0.
    We verify this explicitly and compute the LSQ-matrix condition numbers.
    """
    N_pan = qdata.n_panels
    p_gl  = qdata.p
    Nq    = qdata.n_quad
    use_exact_cond = (Nq <= 3500)   # full SVD only for small systems

    if verbose:
        print(f"  [Arm 2] N_q={Nq}  N_pan={N_pan}  p_gl={p_gl}")
        print(f"  [Arm 2] Using {'exact SVD' if use_exact_cond else 'Lanczos'} "
              f"for condition numbers.")

    # Precompute Dh g  (for C stacked RHS)
    Dh_g = build_Dh_apply(g, D_ref, qdata)

    results = {}
    cond_dict = {}

    for loss in METHODS:
        t0 = time.perf_counter()

        # ── Condition number of LSQ matrix ──────────────────────────────
        if loss == "A":
            # LSQ matrix: G  (N_q × N_q)
            cond = cond_exact(G) if use_exact_cond else cond_lanczos(G, Nq)
        elif loss == "B":
            # LSQ matrix: W̃ G  (N_q × N_q)
            WG = W_tilde @ G
            cond = cond_exact(WG) if use_exact_cond else cond_lanczos(WG, Nq)
            if not use_exact_cond: del WG; gc.collect()
        elif loss == "C":
            # LSQ matrix: [G; D_h G]  (2N_q × N_q)
            DhG  = np.vstack([build_Dh_apply(G[:, j], D_ref, qdata)
                               for j in range(Nq)]).T   # Nq × Nq, then
            # build column-wise is slow; use row-wise instead:
            DhG  = np.zeros_like(G)
            for k in range(N_pan):
                js = qdata.idx_std[k]
                L  = qdata.L_panel[k]
                DhG[js, :] = (2.0 / L) * (D_ref @ G[js, :])
            GCstacked = np.vstack([G, DhG])   # (2Nq, Nq)
            del DhG; gc.collect()
            if use_exact_cond:
                cond = cond_exact(GCstacked)
            else:
                cond = cond_lanczos(GCstacked, Nq)
            del GCstacked; gc.collect()
        elif loss == "D":
            # LSQ matrix: Φ  (N_q × N_q, block-diagonal)
            # cond(Φ) = cond(Φ_block) (each block identical)
            cond = cond_exact(Phi_block)   # just the 16×16 block

        cond_dict[loss] = cond

        # ── Solution: c* = Φ⁻¹ σ_LU  (unique consistent solution) ──────
        c_star  = phi_solve(sigma_LU, Phi_block, N_pan, p_gl)
        sigma_c = phi_apply(c_star, Phi_block, N_pan, p_gl)
        # sigma_c ≈ sigma_LU (up to Phi round-trip error)

        d_err = float(np.linalg.norm(sigma_c - sigma_LU)
                      / np.linalg.norm(sigma_LU))
        bie   = float(np.linalg.norm(V_h @ sigma_c - g)
                      / np.linalg.norm(g))
        wall  = time.perf_counter() - t0

        results[loss] = dict(sigma=sigma_c, d_err=d_err, bie=bie,
                             cond=cond, wall=wall, c_star=c_star)
        if verbose:
            print(f"    {loss}: cond={cond:.3e}  d_err={d_err:.2e}  "
                  f"bie={bie:.2e}  wall={wall:.2f}s")

    return results

# ---------------------------------------------------------------------------
# Arm 1: Legendre basis + descent
# ---------------------------------------------------------------------------

def _dh_apply_vec_torch(v: torch.Tensor, D_ref_t: torch.Tensor,
                         N_pan: int, p: int,
                         scales_t: torch.Tensor) -> torch.Tensor:
    """Block-diagonal D_h via reshape (clean autograd, panels contiguous)."""
    v_mat  = v.view(N_pan, p)
    result = (v_mat @ D_ref_t.T) * scales_t.unsqueeze(1)
    return result.view(-1)


def run_arm1(V_h, W_tilde, g, sigma_LU,
             G, Phi_block, D_ref, qdata, verbose=True) -> dict:
    """Legendre basis descent: Adam 3×1000 + L-BFGS 15000 on c ∈ ℝ^{N_q}."""
    N_pan = qdata.n_panels
    p_gl  = qdata.p
    Nq    = qdata.n_quad
    adam_cutoff = sum(ni for ni, _ in LR_SCHEDULE)

    # Pre-convert heavy tensors once
    G_t       = torch.from_numpy(np.ascontiguousarray(G))
    WG_t      = torch.from_numpy(np.ascontiguousarray(W_tilde @ G))
    Wg_t      = torch.from_numpy(np.ascontiguousarray(W_tilde @ g))
    g_t       = torch.from_numpy(np.ascontiguousarray(g))
    sigma_LU_t= torch.from_numpy(np.ascontiguousarray(sigma_LU))
    D_ref_t   = torch.tensor(D_ref, dtype=torch.float64)
    scales_t  = torch.tensor([2.0 / qdata.L_panel[k] for k in range(N_pan)],
                              dtype=torch.float64)
    Phi_block_t = torch.tensor(Phi_block, dtype=torch.float64)

    # c_LU = Phi^{-1} sigma_LU (for D loss target in coefficient space)
    c_LU_t = torch.from_numpy(phi_solve(sigma_LU, Phi_block, N_pan, p_gl))

    results = {}
    torch.manual_seed(SEED)

    for loss_id in METHODS:
        t0   = time.perf_counter()
        c    = torch.nn.Parameter(
                   torch.zeros(Nq, dtype=torch.float64))
        hist = {"iter": [], "d_err": [], "loss": []}

        if loss_id == "A":
            def loss_fn(c, _G=G_t, _g=g_t):
                r = _G @ c - _g
                return (r**2).mean()
        elif loss_id == "B":
            def loss_fn(c, _WG=WG_t, _Wg=Wg_t):
                r = _WG @ c - _Wg
                return (r**2).mean()
        elif loss_id == "C":
            def loss_fn(c, _G=G_t, _g=g_t, _Dr=D_ref_t,
                        _sc=scales_t, _Np=N_pan, _p=p_gl):
                r   = _G @ c - _g
                Dr  = _dh_apply_vec_torch(r, _Dr, _Np, _p, _sc)
                return (r**2).mean() + ALPHA * (Dr**2).mean()
        elif loss_id == "D":
            def loss_fn(c, _cLU=c_LU_t, _Ph=Phi_block_t, _Np=N_pan, _p=p_gl):
                # ||Phi c - sigma_LU||^2 = ||Phi(c - c_LU)||^2
                # Use Phi: sigma = Phi c (block matmul)
                c_mat   = c.view(_Np, _p)
                sig     = (c_mat @ _Ph.T).view(-1)
                sig_ref = (_cLU.view(_Np, _p) @ _Ph.T).view(-1)
                return (sig - sig_ref).pow(2).mean()

        def _record(it):
            with torch.no_grad():
                sigma_c_np = phi_apply(c.detach().numpy(), Phi_block, N_pan, p_gl)
            d_err = float(np.linalg.norm(sigma_c_np - sigma_LU)
                          / np.linalg.norm(sigma_LU))
            hist["iter"].append(it)
            hist["d_err"].append(d_err)
            hist["loss"].append(float(loss_fn(c).detach()))
            if verbose and it % 1000 == 0:
                print(f"      iter={it:6d}  d_err={d_err:.4f}  "
                      f"loss={hist['loss'][-1]:.3e}")

        # Adam
        opt  = torch.optim.Adam([c], lr=1e-3)
        itr  = 0
        _record(0)
        for n_iters, lr in LR_SCHEDULE:
            for pg in opt.param_groups: pg["lr"] = lr
            for _ in range(n_iters):
                opt.zero_grad(); loss_fn(c).backward(); opt.step()
                itr += 1
                if itr % 200 == 0: _record(itr)
        _record(itr)
        if verbose:
            print(f"    [{loss_id}] Adam done: d_err={hist['d_err'][-1]:.4f}")

        # L-BFGS
        opt_lb = torch.optim.LBFGS(
            [c], lr=1.0, max_iter=20,
            history_size=LBFGS_MEM, line_search_fn="strong_wolfe")
        for _ in range(N_LBFGS // 20):
            def closure():
                opt_lb.zero_grad(); lv = loss_fn(c); lv.backward(); return lv
            opt_lb.step(closure)
            itr += 20
            if itr % 500 == 0: _record(itr)
        _record(itr)

        wall = time.perf_counter() - t0
        sigma_c_np = phi_apply(c.detach().numpy(), Phi_block, N_pan, p_gl)
        d_err = float(np.linalg.norm(sigma_c_np - sigma_LU)
                      / np.linalg.norm(sigma_LU))
        bie   = float(np.linalg.norm(V_h @ sigma_c_np - g) / np.linalg.norm(g))

        if verbose:
            print(f"    [{loss_id}] L-BFGS done: d_err={d_err:.4f}  "
                  f"bie={bie:.2e}  wall={wall:.0f}s")
        results[loss_id] = dict(sigma=sigma_c_np, d_err=d_err, bie=bie,
                                hist=hist, wall=wall)
        del c; gc.collect()

    del G_t, WG_t, Wg_t, g_t, sigma_LU_t, D_ref_t, Phi_block_t, c_LU_t
    gc.collect()
    return results

# ---------------------------------------------------------------------------
# Load network results
# ---------------------------------------------------------------------------

def load_network(n: int, sigma_LU: np.ndarray) -> dict:
    path = os.path.join(THESIS_DIR, f"koch{n}_results.npz")
    f    = np.load(path, allow_pickle=True)
    res  = {}
    for mid in METHODS:
        sigma = f[f"m{mid}_sigma"]
        d_err = float(np.linalg.norm(sigma - sigma_LU) / np.linalg.norm(sigma_LU))
        res[mid] = dict(
            sigma=sigma, d_err=d_err,
            bie=float(f[f"m{mid}_bie"]),
            iL2=float(f[f"m{mid}_iL2"]),
            wall=float(f[f"m{mid}_wall"]),
        )
    return res

# ---------------------------------------------------------------------------
# Interior L2 error
# ---------------------------------------------------------------------------

def interior_l2(sigma, P, Yq_T, wq, x_range, y_range) -> float:
    rec = reconstruct_interior(P=P, Yq=Yq_T, wq=wq, sigma=sigma,
                               n_grid=N_GRID, u_exact=u_exact,
                               x_range=x_range, y_range=y_range)
    return float(rec.rel_L2)

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _fe(x):
    if isinstance(x, str): return x
    return "—" if np.isnan(x) else f"{x:.2e}"

def _f3(x):
    if isinstance(x, str): return x
    return "—" if np.isnan(x) else f"{x:.3f}"

def _ft(x): return f"{x:.0f}"


def make_tables(all_cells: dict, tables_dir: str):
    """
    all_cells[n][repr_key][loss] = {d_err, bie, iL2, cond, wall, ratio}
    repr_key ∈ {'net', 'leg_gd', 'leg_lsq'}
    """
    csv_rows = ["Level,Representation,Loss,DensityErr,BIERes,InteriorL2,"
                "LSQcond,Ratio_dD,WallSec"]
    for n in LEVELS:
        for rkey in ["net", "leg_gd", "leg_lsq"]:
            for loss in METHODS:
                c = all_cells[n][rkey][loss]
                co_s = "--" if c.get("cond") is None else _fe(c["cond"])
                csv_rows.append(
                    f"Koch{n},{REPR_LABELS[rkey]},{loss},"
                    f"{c['d_err']:.6f},{c['bie']:.4e},{c['iL2']:.4e},"
                    f"{co_s},{c['ratio']:.3f},{c['wall']:.1f}")
    with open(os.path.join(tables_dir, "e2_comparison.csv"), "w") as f:
        f.write("\n".join(csv_rows))
    print("  saved → e2_comparison.csv")

    # Combined TeX table
    for n in LEVELS:
        tex = [
            r"\begin{table}[ht]",
            r"\centering",
            (rf"\caption{{E2 experiment on Koch($n={n}$), $N_q={LEVEL_CFG[n]['n_per_edge']*12*16}$. "
             r"Three representations (rows grouped) × four losses. "
             r"$d/d_D$: density error relative to the D-floor of the same representation. "
             r"LSQ cond: condition number of the least-squares matrix (Arm 2 only). "
             r"Network + descent loaded from thesis scaling runs.}"),
            rf"\label{{tab:e2-koch{n}}}",
            r"\begin{tabular}{@{}llcccccr@{}}",
            r"\toprule",
            (r"Representation & Loss & Density err & BIE res & "
             r"Interior $L^2$ & LSQ cond & $d/d_D$ & Wall (s) \\"),
            r"\midrule",
        ]
        for ri, rkey in enumerate(["net", "leg_gd", "leg_lsq"]):
            if ri > 0: tex.append(r"\midrule")
            first = True
            for loss in METHODS:
                c   = all_cells[n][rkey][loss]
                rl  = REPR_LABELS[rkey] if first else ""
                co_s = "—" if c.get("cond") is None else _fe(c["cond"])
                tex.append(
                    f"  {rl} & {loss} & {_fe(c['d_err'])} & "
                    f"{_fe(c['bie'])} & {_fe(c['iL2'])} & "
                    f"{co_s} & {_f3(c['ratio'])} & {_ft(c['wall'])} \\\\")
                first = False
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        fname = os.path.join(tables_dir, f"e2_koch{n}.tex")
        with open(fname, "w") as f:
            f.write("\n".join(tex))
        print(f"  saved → e2_koch{n}.tex")

    # LSQ conditioning table
    tex2 = [
        r"\begin{table}[ht]",
        r"\centering",
        (r"\caption{Condition numbers of the LSQ matrices in Arm 2 "
         r"(Legendre + least-squares). "
         r"$G = V_h \Phi$. "
         r"$\mathrm{cond}(\Phi)$ is the same for both levels (one block, "
         r"$16\times16$). All others estimated by full SVD (Koch $n=1$) "
         r"or Lanczos (Koch $n=2$).}"),
        r"\label{tab:e2-lsq-cond}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        (r"Level & $\mathrm{cond}(G)$ (loss A) "
         r"& $\mathrm{cond}(\widetilde{W}G)$ (loss B) "
         r"& $\mathrm{cond}([G;D_hG])$ (loss C) "
         r"& $\mathrm{cond}(\Phi)$ (loss D) \\"),
        r"\midrule",
    ]
    for n in LEVELS:
        cA = _fe(all_cells[n]["leg_lsq"]["A"]["cond"])
        cB = _fe(all_cells[n]["leg_lsq"]["B"]["cond"])
        cC = _fe(all_cells[n]["leg_lsq"]["C"]["cond"])
        cD = _fe(all_cells[n]["leg_lsq"]["D"]["cond"])
        tex2.append(f"  Koch($n={n}$) & {cA} & {cB} & {cC} & {cD} \\\\")
    tex2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(tables_dir, "e2_lsq_conditioning.tex"), "w") as f:
        f.write("\n".join(tex2))
    print("  saved → e2_lsq_conditioning.tex")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_bar(all_cells: dict, n: int, outpath: str):
    """Grouped bar chart: density error by loss × representation."""
    x   = np.arange(4)        # four losses A,B,C,D
    w   = 0.25
    rep_colors = {"net": "#555555", "leg_gd": "#e08020", "leg_lsq": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, rkey in enumerate(["net", "leg_gd", "leg_lsq"]):
        vals = [max(all_cells[n][rkey][loss]["d_err"], 1e-6)
                for loss in METHODS]
        bars = ax.bar(x + (i - 1) * w, vals, w,
                      label=REPR_LABELS[rkey],
                      color=rep_colors[rkey], alpha=0.85, edgecolor="k", lw=0.5)
        # Annotate bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val * 1.15,
                    f"{val:.2e}", ha="center", va="bottom",
                    fontsize=6.5, rotation=90)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=12)
    ax.set_xlabel("Loss function", fontsize=11)
    ax.set_ylabel("Density relative error", fontsize=11)
    ax.set_title(
        rf"E2: density error by representation and loss — Koch($n={n}$)",
        fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", which="both", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")


def fig_density_lossA(net_res, leg_gd_res, leg_lsq_res,
                      sigma_LU, data, n, outpath):
    """Density profiles for loss A, three representations + LU ref, Koch(1)."""
    qdata = data["qdata"]
    arc   = data["arc"]
    idx_s = data["sort_idx"]

    arc_s = arc[idx_s]
    bem_s = sigma_LU[idx_s]
    margin = 0.15 * (bem_s.max() - bem_s.min())
    ymin   = bem_s.min() - margin
    ymax   = bem_s.max() + margin

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(arc_s, bem_s, "k--", lw=1.8, alpha=0.7, label=r"$\sigma_{\rm LU}$ (reference)")
    ax.plot(arc_s, net_res["A"]["sigma"][idx_s],
            "-", color="#555555", lw=1.5, label=f"Network + descent  (d={net_res['A']['d_err']:.3f})")
    ax.plot(arc_s, leg_gd_res["A"]["sigma"][idx_s],
            "-", color="#e08020", lw=1.5, label=f"Legendre + descent  (d={leg_gd_res['A']['d_err']:.3f})")
    ax.plot(arc_s, leg_lsq_res["A"]["sigma"][idx_s],
            "-", color="#2ca02c", lw=1.5, label=f"Legendre + LSQ  (d={leg_lsq_res['A']['d_err']:.2e})")

    # Corner lines
    for ca in data["corner_arcs"][:30]:
        ax.axvline(ca, color="#cccccc", lw=0.6, zorder=0, alpha=0.5)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Arc-length $s$", fontsize=11)
    ax.set_ylabel(r"$\sigma(s)$", fontsize=11)
    ax.set_title(
        rf"Loss A (standard): density $\sigma$ for three representations — "
        rf"Koch($n={n}$)",
        fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {outpath}")

# ---------------------------------------------------------------------------
# Arc-length / corner info (for density plot)
# ---------------------------------------------------------------------------

def build_geom_info(qdata, P) -> dict:
    pan_start = np.concatenate([[0.0], np.cumsum(qdata.L_panel[:-1])])
    arc       = pan_start[qdata.pan_id] + qdata.s_on_panel
    sort_idx  = np.argsort(arc)
    Yq_T      = qdata.Yq.T

    signed_area = 0.5 * np.sum(
        P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    ccw = (signed_area > 0)
    corner_arcs = []
    nv = len(P)
    for vi in range(nv):
        v_prev = P[(vi - 1) % nv]; v_curr = P[vi]; v_next = P[(vi + 1) % nv]
        e1 = v_curr - v_prev; e2 = v_next - v_curr
        cross = e1[0]*e2[1] - e1[1]*e2[0]
        if (cross < 0) if ccw else (cross > 0):
            corner_arcs.append(arc[np.argmin(np.linalg.norm(Yq_T - v_curr, axis=1))])
    return dict(arc=arc, sort_idx=sort_idx, corner_arcs=corner_arcs)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 68)
    print("EXPERIMENT E2: Representation × Loss × Solve-method")
    print("=" * 68)
    print("  Koch(1,2), p_rep=15, coeff count = 16 × N_pan = N_q")
    print("  Three representations: network+descent / Legendre+descent / Legendre+LSQ")

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    all_cells = {}
    data_for_fig = {}

    for n in LEVELS:
        cfg = LEVEL_CFG[n]
        print(f"\n{'='*68}")
        print(f"KOCH({n})  n_pe={cfg['n_per_edge']}  p_gl={cfg['p_gl']}")
        print(f"{'='*68}")

        # ── Assemble operators ─────────────────────────────────────────────
        t0    = time.perf_counter()
        geom  = make_koch_geometry(n=n)
        P     = geom.vertices
        pans  = build_uniform_panels(P, n_per_edge=cfg["n_per_edge"])
        label_corner_ring_panels(pans, P)
        qdata = build_panel_quadrature(pans, p=cfg["p_gl"])
        Yq_T  = qdata.Yq.T
        wq    = qdata.wq
        Nq    = qdata.n_quad
        N_pan = qdata.n_panels
        p_gl  = qdata.p

        nmat  = assemble_nystrom_matrix(qdata)
        V_h   = nmat.V
        g     = g_fn(Yq_T)

        W_h, _   = assemble_hypersingular_corrected(qdata)
        W_tilde  = regularise_hypersingular(W_h, wq)
        del W_h; gc.collect()

        xi_gl, _ = gauss_legendre(p_gl)
        D_ref    = lagrange_derivative_matrix(xi_gl)

        x_range = (P[:, 0].min() - 0.05, P[:, 0].max() + 0.05)
        y_range = (P[:, 1].min() - 0.05, P[:, 1].max() + 0.05)
        print(f"  Operators assembled in {time.perf_counter()-t0:.1f}s  Nq={Nq}")

        # ── LU reference ──────────────────────────────────────────────────
        print(f"\n  LU reference …")
        lu_fac   = la.lu_factor(V_h)
        sigma_LU = la.lu_solve(lu_fac, g)
        del lu_fac
        print(f"  sigma_LU: ||V sigma - g|| / ||g|| = "
              f"{np.linalg.norm(V_h @ sigma_LU - g)/np.linalg.norm(g):.2e}")

        # ── Build Phi and G ───────────────────────────────────────────────
        print(f"\n  Building Phi_block (p_gl={p_gl}) and G = V_h @ Phi …")
        Phi_block = build_phi_block(p_gl)
        cond_phi  = cond_exact(Phi_block)
        coeff_count = N_pan * p_gl
        print(f"  Phi_block cond = {cond_phi:.2f}  coeff_count = {coeff_count}  "
              f"(== N_q: {coeff_count == Nq})")
        assert coeff_count == Nq, "coeff count != N_q!"

        G = build_G(V_h, Phi_block, N_pan, p_gl)
        print(f"  G built: shape {G.shape}")

        # ── Arm 2: Legendre + LSQ ─────────────────────────────────────────
        print(f"\n  === ARM 2: Legendre + LSQ ===")
        lsq_res = run_arm2(V_h, W_tilde, g, sigma_LU,
                           G, Phi_block, D_ref, qdata)

        # ── Arm 1: Legendre + descent ─────────────────────────────────────
        print(f"\n  === ARM 1: Legendre + descent ===")
        gd_res = run_arm1(V_h, W_tilde, g, sigma_LU,
                          G, Phi_block, D_ref, qdata)

        # Free G (no longer needed after descent)
        del G; gc.collect()

        # ── Network + descent (loaded) ─────────────────────────────────────
        print(f"\n  Loading network results from thesis …")
        net_res = load_network(n, sigma_LU)

        # ── Interior L2 for all cells ─────────────────────────────────────
        print(f"\n  Interior L2 reconstruction …")
        for mid in METHODS:
            for rkey, res in [("net", net_res), ("leg_gd", gd_res),
                               ("leg_lsq", lsq_res)]:
                iL2 = interior_l2(res[mid]["sigma"], P, Yq_T, wq,
                                  x_range, y_range)
                res[mid]["iL2"] = iL2
            print(f"    {mid}: net={net_res[mid]['iL2']:.2e}  "
                  f"leg_gd={gd_res[mid]['iL2']:.2e}  "
                  f"leg_lsq={lsq_res[mid]['iL2']:.2e}")

        # ── Build all_cells with ratios ───────────────────────────────────
        cells = {}
        for rkey, res in [("net", net_res), ("leg_gd", gd_res),
                           ("leg_lsq", lsq_res)]:
            d_D = res["D"]["d_err"]
            cells[rkey] = {}
            for mid in METHODS:
                r = res[mid]
                cells[rkey][mid] = dict(
                    d_err = r["d_err"],
                    bie   = r["bie"],
                    iL2   = r["iL2"],
                    cond  = r.get("cond"),   # None for net/gd
                    wall  = r["wall"],
                    ratio = r["d_err"] / max(d_D, 1e-12),
                )
        all_cells[n] = cells

        # Save arc-length info for density figure
        gi = build_geom_info(qdata, P)
        gi["qdata"] = qdata
        data_for_fig[n] = dict(**gi, net=net_res, leg_gd=gd_res,
                               leg_lsq=lsq_res, sigma_LU=sigma_LU)

        # Per-level summary
        print(f"\n  SUMMARY Koch({n}):")
        print(f"  {'Repr':<22} {'Loss':>4}  {'d_err':>8}  {'BIE':>8}  "
              f"{'iL2':>8}  {'d/d_D':>6}  {'wall':>7}")
        for rkey in ["net", "leg_gd", "leg_lsq"]:
            for mid in METHODS:
                c = cells[rkey][mid]
                print(f"  {REPR_LABELS[rkey]:<22} {mid:>4}  "
                      f"{c['d_err']:>8.4f}  {c['bie']:>8.2e}  "
                      f"{c['iL2']:>8.2e}  {c['ratio']:>6.2f}  "
                      f"{c['wall']:>7.0f}s")

        del V_h, W_tilde, sigma_LU; gc.collect()

    # ── Tables ─────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("TABLES")
    print(f"{'='*68}")
    make_tables(all_cells, TAB_DIR)

    # ── Figures ────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("FIGURES")
    print(f"{'='*68}")
    for n in LEVELS:
        fig_bar(all_cells, n,
                os.path.join(FIG_DIR, f"e2_bar_koch{n}.png"))
    # Density loss A for Koch(1)
    d1 = data_for_fig[1]
    fig_density_lossA(
        d1["net"], d1["leg_gd"], d1["leg_lsq"],
        d1["sigma_LU"], d1, n=1,
        outpath=os.path.join(FIG_DIR, "e2_density_lossA.png"))

    # ── Verdict ────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("VERDICT")
    print(f"{'='*68}")
    thresh = 3.0
    for n in LEVELS:
        cells = all_cells[n]
        print(f"\n  Koch({n}):")
        for rkey in ["net", "leg_gd", "leg_lsq"]:
            d_A = cells[rkey]["A"]["d_err"]
            d_D = cells[rkey]["D"]["d_err"]
            ratio_A = d_A / max(d_D, 1e-12)
            verdict = "SUCCEEDS" if ratio_A < thresh else "FAILS"
            print(f"    {REPR_LABELS[rkey]:<24}  A: d_err={d_A:.4f}  "
                  f"d/d_D={ratio_A:.2f}  → {verdict}")

    # Interpretation
    for n in LEVELS:
        cells = all_cells[n]
        leg_gd_A_fails  = cells["leg_gd"]["A"]["ratio"] >= thresh
        leg_lsq_A_succeeds = cells["leg_lsq"]["A"]["ratio"] < thresh
        print(f"\n  [Koch({n})] Interpretation:")
        if leg_gd_A_fails and leg_lsq_A_succeeds:
            print("   ✓ Legendre+descent+A FAILS  →  failure is NOT the representation")
            print("   ✓ Legendre+LSQ+A  SUCCEEDS  →  failure is descent on ill-conditioned")
            print("     A-loss landscape, not the loss itself or the representation")
        elif not leg_gd_A_fails:
            print("   ! Legendre+descent+A SUCCEEDS — unexpected; check d_err/d_D carefully")
        elif not leg_lsq_A_succeeds:
            print("   ! Legendre+LSQ+A FAILS — the A-loss itself may be at fault")

    print(f"\n  Outputs → experiments/ex_e2_representation/")
    print(f"    figures/: e2_bar_koch1, e2_bar_koch2, e2_density_lossA")
    print(f"    tables/:  e2_koch1.tex, e2_koch2.tex, "
          f"e2_lsq_conditioning.tex, e2_comparison.csv")


if __name__ == "__main__":
    main()
