"""
Continuation script: load Koch(1) results from npz, then run Koch(2) + Koch(3),
then generate all figures and tables.
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..")
sys.path.insert(0, _ROOT)

# Re-use everything from scaling_study
from scaling_study import (
    SEED, N_PER_EDGE, P_GL, ALPHA, HIDDEN_WIDTH, N_HIDDEN,
    LR_SCHEDULE, N_LBFGS, LBFGS_MEM, LOG_EVERY, N_GRID_KOCH2,
    LEVELS, METHODS, METHOD_NAMES,
    g_fn, u_exact_fn,
    build_shared_init,
    setup_level, run_level, interior_for_level,
    save_level_npz,
    fig_convergence, fig_density,
    fig_interior_solutions, fig_interior_errors,
    fig_scaling_conditioning, fig_scaling_density_error,
    make_table_main, make_table_conditioning,
    print_summary,
)

import numpy as np
import torch


def load_level1_specs(data_dir: str) -> tuple[dict, dict]:
    """Load Koch(1) spectral specs and results from the saved npz."""
    npz = np.load(os.path.join(data_dir, "koch1_results.npz"), allow_pickle=True)

    specs1 = {
        "n": int(npz["n"]),
        "Nq": int(npz["Nq"]),
        "cond_V": float(npz["cond_V"]),
        "cond_eig_WV": float(npz["cond_eig_WV"]),
        "cond_svd_WV": float(npz["cond_svd_WV"]),
        "non_norm_WV": float(npz["non_norm_WV"]),
    }

    # Reconstruct per-method result dicts (history stored as arrays)
    results1 = {}
    for mid in METHODS:
        p = f"m{mid}_"
        hist_iter = npz[p + "hist_iter"].tolist()
        hist_loss = npz[p + "hist_loss"].tolist()
        hist_derr = npz[p + "hist_derr"].tolist()
        results1[mid] = {
            "sigma":        npz[p + "sigma"],
            "d_err":        float(npz[p + "d_err"]),
            "bie":          float(npz[p + "bie"]),
            "iL2":          float(npz[p + "iL2"]),
            "wall":         float(npz[p + "wall"]),
            "lbfgs_reason": "converges" if float(npz[p + "d_err"]) < 0.15 else "stalls",
            "hist": {
                "iter":              hist_iter,
                "loss":              hist_loss,
                "density_reldiff":   hist_derr,
                "adam_cutoff":       sum(n for n, _ in LR_SCHEDULE),
                "lbfgs_ratio":       1.0,
                "lbfgs_converged":   float(npz[p + "d_err"]) < 0.15,
                "lbfgs_reason":      "converges" if float(npz[p + "d_err"]) < 0.15 else "stalls",
            },
        }

    return specs1, results1


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    fig_dir   = os.path.join(_HERE, "figures")
    table_dir = os.path.join(_HERE, "tables")
    data_dir  = os.path.join(_HERE, "data")

    print("\n" + "=" * 72)
    print("THESIS SCALING STUDY — CONTINUATION FROM KOCH(2)")
    print("=" * 72)

    # Shared initial weights (must match original run)
    init_state = build_shared_init(SEED)
    adam_cutoff = sum(n for n, _ in LR_SCHEDULE)

    # Load Koch(1) from disk
    print("\n  Loading Koch(1) results from npz …")
    specs1, results1 = load_level1_specs(data_dir)
    print(f"  Koch(1): A={results1['A']['d_err']:.4f}  B={results1['B']['d_err']:.4f}"
          f"  C={results1['C']['d_err']:.4f}  D={results1['D']['d_err']:.4f}")

    all_results = {1: results1}
    specs       = [specs1]

    # Also rebuild data1 just for the density figure (we need arc etc.)
    # Actually, figures for Koch(1) are already saved — skip.

    # -----------------------------------------------------------------------
    # Koch(2)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  STEP 3: Koch(2) — assembly + 4 training runs + interior")
    print("=" * 72)

    data2 = setup_level(2, verbose=True)
    specs.append({
        "n": 2, "Nq": data2["Nq"],
        "cond_V": data2["cond_V"],
        "cond_eig_WV": data2["cond_eig_WV"],
        "cond_svd_WV": data2["cond_svd_WV"],
        "non_norm_WV": data2["non_norm_WV"],
    })

    results2 = run_level(data2, init_state, verbose=True)
    all_results[2] = results2
    save_level_npz(2, data2, results2, data_dir)

    print("\n  FIG 2: convergence_koch2.png")
    fig_convergence(results2, adam_cutoff, 2,
                    os.path.join(fig_dir, "convergence_koch2.png"))
    print("  FIG 4: density_koch2.png")
    fig_density(results2, data2, 2,
                os.path.join(fig_dir, "density_koch2.png"))

    grids2 = interior_for_level(data2, results2)
    print("  FIG 7: interior_koch2_solutions.png")
    fig_interior_solutions(grids2, data2,
                           os.path.join(fig_dir, "interior_koch2_solutions.png"))
    print("  FIG 8: interior_koch2_errors.png")
    fig_interior_errors(grids2, data2,
                        os.path.join(fig_dir, "interior_koch2_errors.png"))

    # -----------------------------------------------------------------------
    # Koch(3)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  STEP 4: Koch(3) — assembly + 4 training runs (SLOW)")
    print("=" * 72)

    data3 = setup_level(3, verbose=True)
    specs.append({
        "n": 3, "Nq": data3["Nq"],
        "cond_V": data3["cond_V"],
        "cond_eig_WV": data3["cond_eig_WV"],
        "cond_svd_WV": data3["cond_svd_WV"],
        "non_norm_WV": data3["non_norm_WV"],
    })

    results3 = run_level(data3, init_state, verbose=True)
    all_results[3] = results3
    save_level_npz(3, data3, results3, data_dir)

    # -----------------------------------------------------------------------
    # Scaling figures (all 3 levels)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  STEP 5: Scaling figures and tables")
    print("=" * 72)

    print("  FIG 5: scaling_conditioning.png")
    fig_scaling_conditioning(specs,
                             os.path.join(fig_dir, "scaling_conditioning.png"))

    print("  FIG 6: scaling_density_error.png")
    fig_scaling_density_error(all_results, specs,
                              os.path.join(fig_dir, "scaling_density_error.png"))

    # -----------------------------------------------------------------------
    # Tables
    # -----------------------------------------------------------------------
    print("  TABLE 1: main_comparison.{tex,csv}")
    make_table_main(all_results, specs, table_dir)

    print("  TABLE 2: conditioning.{tex,csv}")
    make_table_conditioning(specs, table_dir)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print_summary(all_results, specs)

    print(f"\n  All outputs written to {_HERE}/")
    print(f"    figures/ : 8 figures (300 dpi)")
    print(f"    tables/  : main_comparison.{{tex,csv}}, conditioning.{{tex,csv}}")
    print(f"    data/    : koch{{2,3}}_results.npz")


if __name__ == "__main__":
    main()
