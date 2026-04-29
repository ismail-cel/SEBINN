"""
Full diagnostic suite for the corrected hypersingular assembly.

Compares assemble_hypersingular_direct (Hadamard-only) vs
assemble_hypersingular_corrected (full panel, MATLAB port).

Koch(1), n_per_edge=12, p=16 (Nq=2304) — standard experiment config.
"""

import sys
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from src.boundary.polygon import make_koch_geometry
from src.boundary.panels import build_uniform_panels, label_corner_ring_panels
from src.quadrature.panel_quad import build_panel_quadrature
from src.quadrature.nystrom import assemble_nystrom_matrix
from src.quadrature.hypersingular import (
    assemble_hypersingular_direct,
    assemble_hypersingular_corrected,
    regularise_hypersingular,
    compute_panel_normals,
)


def _assemble_K(qdata):
    """Double-layer K_h[i,j] = (1/2π)(y_j-x_i)·n_j/|x_i-y_j|² * w_j."""
    Yq  = qdata.Yq
    wq  = qdata.wq
    Nq  = qdata.n_quad
    normals_pan, _ = compute_panel_normals(qdata)
    node_n = np.empty((Nq, 2))
    for pid in range(qdata.n_panels):
        node_n[qdata.idx_std[pid], :] = normals_pan[pid]
    d_x  = Yq[0, :][None, :] - Yq[0, :][:, None]
    d_y  = Yq[1, :][None, :] - Yq[1, :][:, None]
    rho2 = np.where(d_x**2 + d_y**2 > 1e-30, d_x**2 + d_y**2, 1.0)
    d_dot_n = d_x * node_n[:, 0][None, :] + d_y * node_n[:, 1][None, :]
    K = (1.0 / (2.0 * np.pi)) * d_dot_n / rho2
    np.fill_diagonal(K, 0.0)
    return K * wq[None, :]

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
geom   = make_koch_geometry(n=1)
P      = geom.vertices
panels = build_uniform_panels(P, n_per_edge=12)
label_corner_ring_panels(panels, P)
qdata  = build_panel_quadrature(panels, p=16)
nmat   = assemble_nystrom_matrix(qdata)

V_h  = nmat.V
K_h  = _assemble_K(qdata)
wq   = qdata.wq
Nq   = qdata.n_quad

print(f"Koch(1): n_per_edge=12, p=16, Nq={Nq}, Npan={qdata.n_panels}")
print("Assembling operators...")

# Old W_h (Hadamard diagonal-only self-panel correction)
W_old, _   = assemble_hypersingular_direct(qdata)
WT_old     = regularise_hypersingular(W_old, wq)

# New W_h (full panel correction, MATLAB port)
W_new, _   = assemble_hypersingular_corrected(qdata)
WT_new     = regularise_hypersingular(W_new, wq)

I   = np.eye(Nq)
K2  = K_h @ K_h

print("Operators assembled. Running diagnostics...\n")

# ---------------------------------------------------------------------------
# 1. CALDERÓN IDENTITY: VW = (1/4)I - K²
# ---------------------------------------------------------------------------
print("=" * 70)
print("1. CALDERÓN IDENTITY:  VW = (1/4)I - K²")
print("=" * 70)

ref = 0.25 * I - K2

VW_old = V_h @ W_old
VW_new = V_h @ W_new

rel_old = np.linalg.norm(VW_old - ref, "fro") / np.linalg.norm(ref, "fro")
rel_new = np.linalg.norm(VW_new - ref, "fro") / np.linalg.norm(ref, "fro")

print(f"  OLD (Hadamard only):   rel_error = {rel_old:.4e}")
print(f"  NEW (full correction): rel_error = {rel_new:.4e}")
print(f"  Target: < 0.1")
print(f"  Improvement: {rel_old/max(rel_new,1e-15):.2f}×")

# ---------------------------------------------------------------------------
# 2. EIGENVALUE ANALYSIS OF W̃V
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. EIGENVALUE ANALYSIS OF W̃V")
print("=" * 70)

for label, WT in [("OLD", WT_old), ("NEW", WT_new)]:
    ev = np.linalg.eigvals(WT @ V_h)
    ab = np.abs(ev)
    print(f"\n  [{label}] Eigenvalues of W̃V:")
    print(f"    min|λ|    = {ab.min():.6f}")
    print(f"    max|λ|    = {ab.max():.6f}")
    print(f"    median|λ| = {np.median(ab):.6f}  (target: 0.250)")
    print(f"    cond_eig  = {ab.max()/ab.min():.2f}")
    print(f"    max|imag| = {np.abs(np.imag(ev)).max():.3e}")

# ---------------------------------------------------------------------------
# 3. SINGULAR VALUE ANALYSIS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. SINGULAR VALUE ANALYSIS OF W̃V")
print("=" * 70)

for label, WT in [("OLD", WT_old), ("NEW", WT_new)]:
    WV  = WT @ V_h
    sv  = np.linalg.svd(WV, compute_uv=False)
    cnd = sv[0] / sv[-1]
    non = (np.linalg.norm(WV.T @ WV - WV @ WV.T)
           / np.linalg.norm(WV) ** 2)
    print(f"  [{label}]  cond_svd = {cnd:.1f},  Hessian cond ≈ {cnd**2:.0f},  non-normality = {non:.3e}")

# Reference for context
sv_V = np.linalg.svd(V_h, compute_uv=False)
print(f"  [V only] cond_svd = {sv_V[0]/sv_V[-1]:.0f},  Hessian cond ≈ {(sv_V[0]/sv_V[-1])**2:.2e}")

# ---------------------------------------------------------------------------
# 4. DEFINITENESS OF W̃
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. DEFINITENESS OF W̃")
print("=" * 70)

for label, WT in [("OLD", WT_old), ("NEW", WT_new)]:
    Ws  = 0.5 * (WT + WT.T)
    evs = np.linalg.eigvalsh(Ws)
    n_n = int((evs < 0).sum())
    print(f"\n  [{label}]  n_neg = {n_n},  λ_min = {evs.min():.4f},  λ_max = {evs.max():.4f}")
    neg = evs[evs < 0]
    if len(neg):
        print(f"    Negative: count={len(neg)}, min={neg.min():.4f}, max={neg.max():.4f}, median={np.median(neg):.4f}")

# Spectral flip on NEW
Ws_new = 0.5 * (WT_new + WT_new.T)
evs_new, Q_new = np.linalg.eigh(Ws_new)
n_neg_new = int((evs_new < 0).sum())
W_plus = Q_new @ np.diag(np.abs(evs_new)) @ Q_new.T
W_plus = 0.5 * (W_plus + W_plus.T)
ev_WpV = np.linalg.eigvals(W_plus @ V_h)
ab_WpV = np.abs(ev_WpV)
sv_WpV = np.linalg.svd(W_plus @ V_h, compute_uv=False)
print(f"\n  After spectral flip of NEW W̃:")
print(f"    cond_eig(W̃_+ V) = {ab_WpV.max()/ab_WpV.min():.2f}  (was {np.abs(np.linalg.eigvals(WT_new@V_h)).max()/np.abs(np.linalg.eigvals(WT_new@V_h)).min():.2f} before flip)")
print(f"    cond_svd(W̃_+ V) = {sv_WpV[0]/sv_WpV[-1]:.2f}")
print(f"    (if cond_eig ≈ unchanged: spectral flip preserves conditioning)")

# ---------------------------------------------------------------------------
# 5. RIGHT-PRECONDITIONING VIABILITY: B = -VW̃
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. RIGHT-PRECONDITIONING OPERATOR B = -VW̃")
print("=" * 70)

for label, WT in [("OLD", WT_old), ("NEW", WT_new)]:
    B   = -V_h @ WT
    sv  = np.linalg.svd(B, compute_uv=False)
    ev  = np.linalg.eigvals(B)
    ab  = np.abs(ev)
    non = np.linalg.norm(B.T @ B - B @ B.T) / np.linalg.norm(B) ** 2
    print(f"  [{label}]  cond_eig = {ab.max()/ab.min():.1f},  cond_svd = {sv[0]/sv[-1]:.1f},  non-norm = {non:.3e}")
    print(f"           Hessian cond (loss landscape) = {(sv[0]/sv[-1])**2:.0f}")

# ---------------------------------------------------------------------------
# 6. CALDERÓN AT MATRIX LEVEL: B_nys vs B_direct
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("6. CALDERÓN AT MATRIX LEVEL")
print("=" * 70)

B_direct = 0.25 * I - K2   # exact Calderón rhs (= -B since B = -VW̃)

for label, WT in [("OLD", WT_old), ("NEW", WT_new)]:
    B   = -V_h @ WT
    # B ≈ -(1/4)I + K² = -B_direct, so B + B_direct ≈ 0
    rd  = np.linalg.norm(B + B_direct, "fro") / np.linalg.norm(B_direct, "fro")
    print(f"  [{label}]  ||B_nys + B_direct|| / ||B_direct|| = {rd:.4e}")
    print(f"           (target < 0.1; measures how well -VW̃ ≈ (1/4)I - K²)")

# ---------------------------------------------------------------------------
# 7. SUMMARY TABLE
# ---------------------------------------------------------------------------
ev_old = np.linalg.eigvals(WT_old @ V_h); ab_old = np.abs(ev_old)
ev_new = np.linalg.eigvals(WT_new @ V_h); ab_new = np.abs(ev_new)
sv_old = np.linalg.svd(WT_old @ V_h, compute_uv=False)
sv_new = np.linalg.svd(WT_new @ V_h, compute_uv=False)
non_old = (np.linalg.norm((WT_old@V_h).T@(WT_old@V_h) - (WT_old@V_h)@(WT_old@V_h).T)
           / np.linalg.norm(WT_old@V_h)**2)
non_new = (np.linalg.norm((WT_new@V_h).T@(WT_new@V_h) - (WT_new@V_h)@(WT_new@V_h).T)
           / np.linalg.norm(WT_new@V_h)**2)

Ws_old = 0.5*(WT_old+WT_old.T); evs_o = np.linalg.eigvalsh(Ws_old)
n_neg_old = int((evs_o < 0).sum())

B_old_sv = np.linalg.svd(-V_h@WT_old, compute_uv=False)
B_new_sv = np.linalg.svd(-V_h@WT_new, compute_uv=False)
B_old = -V_h @ WT_old; rd_old = np.linalg.norm(B_old+B_direct,"fro")/np.linalg.norm(B_direct,"fro")
B_new = -V_h @ WT_new; rd_new = np.linalg.norm(B_new+B_direct,"fro")/np.linalg.norm(B_direct,"fro")

print(f"\n{'='*70}")
print(f"SUMMARY TABLE — corrected vs old hypersingular (Koch(1), Nq={Nq})")
print(f"{'='*70}")
print(f"{'Metric':<35s} | {'OLD':>12s} | {'CORRECTED':>12s} | {'Target':>10s}")
print(f"{'-'*35}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

rows = [
    ("Calderón rel_error",         f"{rel_old:.3e}",    f"{rel_new:.3e}",   "< 0.1"),
    ("cond_eig(W̃V)",               f"{ab_old.max()/ab_old.min():.2f}",
                                    f"{ab_new.max()/ab_new.min():.2f}",       "O(1)"),
    ("median|λ(W̃V)|",              f"{np.median(ab_old):.4f}",
                                    f"{np.median(ab_new):.4f}",               "0.250"),
    ("cond_svd(W̃V)",               f"{sv_old[0]/sv_old[-1]:.1f}",
                                    f"{sv_new[0]/sv_new[-1]:.1f}",            "O(1)"),
    ("non-normality(W̃V)",          f"{non_old:.3e}",    f"{non_new:.3e}",   "< 0.01"),
    ("n_neg eigenvalues of W̃",     f"{n_neg_old}",      f"{n_neg_new}",     "0"),
    ("cond_svd(-VW̃)",              f"{B_old_sv[0]/B_old_sv[-1]:.1f}",
                                    f"{B_new_sv[0]/B_new_sv[-1]:.1f}",        "O(1)"),
    ("Hessian cond(-VW̃)²",        f"{(B_old_sv[0]/B_old_sv[-1])**2:.0f}",
                                    f"{(B_new_sv[0]/B_new_sv[-1])**2:.0f}",   "1"),
    ("||B_nys+B_direct||/||B_dir||",f"{rd_old:.3e}",    f"{rd_new:.3e}",    "< 0.1"),
]
for name, old, new, tgt in rows:
    print(f"{name:<35s} | {old:>12s} | {new:>12s} | {tgt:>10s}")

print(f"{'='*70}")

# Verdict
ce = ab_new.max() / ab_new.min()
cs = sv_new[0] / sv_new[-1]
if rel_new < 0.1 and cs < 50:
    verdict = "Corrected W̃ is viable for preconditioning! Proceed to training."
elif rel_new < 0.5 and cs < 100:
    verdict = "Significant improvement but Calderón identity incomplete. Bilinear form may still work."
else:
    verdict = "Calderón identity still does not hold well. Stick with V⁻¹."

print(f"\nVERDICT: {verdict}")
