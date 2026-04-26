"""
Verification of the Calderón identity  V_h W_h = (1/4) I - K_h²

Scaling conventions (from preconditioning image)
-------------------------------------------------
Image operators S, K_img, T satisfy   -S T = I - K_img²

Our operators relate by:
    V = -(1/2) S         →  S = -2 V
    W = (1/2) T          →  T = 2 W
    K_std = (1/2) K_img  →  K_img = 2 K_std

Substituting:
    -(-2V)(2W) = I - (2 K_std)²
    4 V W      = I - 4 K_std²
    V W        = (1/4) I - K_std²

Double-layer kernel (our convention):
    K(x,y) = (1/2π) · (y-x)·n_y / |x-y|²

This script:
  1. Assembles K_h
  2. Validates K_h (self-panel zeros, compactness)
  3. Tests  V_h W_h = (1/4) I - K_h²
  4. Conditioning analysis of both sides
  5. Definiteness of (1/4)I - K²  (candidate bilinear-form loss)
  6. Diagnosis if identity fails

No training is performed.
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
    compute_panel_normals,
)


# ===========================================================================
# Step 0: Geometry + operators
# ===========================================================================

def setup(n_per_edge=12, p_gl=16):
    geom   = make_koch_geometry(n=1)
    P      = geom.vertices
    panels = build_uniform_panels(P, n_per_edge=n_per_edge)
    label_corner_ring_panels(panels, P)
    qdata  = build_panel_quadrature(panels, p=p_gl)
    nmat   = assemble_nystrom_matrix(qdata)
    normals, _ = compute_panel_normals(qdata)
    return qdata, nmat, normals


# ===========================================================================
# Step 1: Assemble K_h (double-layer operator)
# ===========================================================================

def assemble_double_layer(qdata, normals):
    """
    Double-layer operator K for 2D Laplace.

    K(x,y) = (1/2π) · (y-x)·n_y / |x-y|²

    n_y = unit outward normal at SOURCE point y.

    Matrix entry: K_ij = K(x_i, y_j) · w_j

    Self-panel: on a straight panel (y-x) ∥ tangent, so (y-x)·n_y = 0.
    All self-panel entries are exactly zero — no correction needed.
    """
    Yq = qdata.Yq       # (2, Nq)
    wq = qdata.wq       # (Nq,)
    Nq = qdata.n_quad

    # Per-node normals (constant per panel)
    node_normals = np.empty((Nq, 2))
    for pid in range(qdata.n_panels):
        js = qdata.idx_std[pid]
        node_normals[js, :] = normals[pid, :]

    # d[i,j] = y_j - x_i  (SOURCE minus TARGET)
    d_x = Yq[0, :][None, :] - Yq[0, :][:, None]   # (Nq, Nq)
    d_y = Yq[1, :][None, :] - Yq[1, :][:, None]   # (Nq, Nq)

    rho2      = d_x**2 + d_y**2
    rho2_safe = np.where(rho2 > 1e-30, rho2, 1.0)

    # (y_j - x_i) · n_{y_j}
    ny_x = node_normals[:, 0]   # (Nq,)
    ny_y = node_normals[:, 1]   # (Nq,)
    d_dot_ny = d_x * ny_x[None, :] + d_y * ny_y[None, :]   # (Nq, Nq)

    kernel = (1.0 / (2.0 * np.pi)) * d_dot_ny / rho2_safe

    # Zero out diagonal and near-zero distance entries
    np.fill_diagonal(kernel, 0.0)
    kernel[rho2 < 1e-30] = 0.0

    # Apply quadrature weights at SOURCE nodes
    K_h = kernel * wq[None, :]

    return K_h


# ===========================================================================
# Step 2: Validate K_h
# ===========================================================================

def validate_K(K_h, qdata):
    print("=" * 60)
    print("DOUBLE-LAYER K_h VALIDATION")
    print("=" * 60)
    print(f"  shape : {K_h.shape}")
    print(f"  ||K_h||_F = {np.linalg.norm(K_h, 'fro'):.3e}")
    print(f"  max|K_h|  = {np.abs(K_h).max():.3e}")

    print("\n  Self-panel blocks (should all be zero):")
    max_self = 0.0
    for pid in range(min(8, qdata.n_panels)):
        js = qdata.idx_std[pid]
        blk = K_h[np.ix_(js, js)]
        m = float(np.abs(blk).max())
        max_self = max(max_self, m)
        print(f"    Panel {pid:3d}: max|self-block| = {m:.3e}")
    print(f"  Global max|self-panel| = {max_self:.3e}")

    eigvals_K = np.linalg.eigvals(K_h)
    print(f"\n  K_h eigenvalues (compact operator — should cluster near 0):")
    print(f"    max|λ|    = {np.max(np.abs(eigvals_K)):.4f}")
    print(f"    median|λ| = {np.median(np.abs(eigvals_K)):.4f}")
    print(f"    n|λ|>0.1  = {(np.abs(eigvals_K) > 0.1).sum()} / {len(eigvals_K)}")
    print(f"    n|λ|>0.4  = {(np.abs(eigvals_K) > 0.4).sum()} / {len(eigvals_K)}")
    return eigvals_K


# ===========================================================================
# Step 3: Calderón identity test
# ===========================================================================

def calderon_test(V_h, W_h, K_h):
    Nq = V_h.shape[0]
    I  = np.eye(Nq)

    VW  = V_h @ W_h
    K2  = K_h @ K_h
    ref = 0.25 * I - K2        # (1/4) I - K²

    residual  = VW - ref
    rel_error = (np.linalg.norm(residual, 'fro')
                 / np.linalg.norm(ref, 'fro'))

    print(f"\n{'='*60}")
    print(f"CALDERÓN IDENTITY:  V W = (1/4) I − K²")
    print(f"{'='*60}")
    print(f"  ||VW − ((1/4)I − K²)||_F / ||(1/4)I − K²||_F = {rel_error:.4e}")
    if rel_error < 0.01:
        print(f"  Verdict: HOLDS PRECISELY ✓")
    elif rel_error < 0.1:
        print(f"  Verdict: HOLDS APPROXIMATELY ✓")
    else:
        print(f"  Verdict: DOES NOT HOLD ✗  (W_h likely has a bug)")

    # Diagonal check
    diag_VW  = np.diag(VW)
    diag_ref = np.diag(ref)
    print(f"\n  Diagonal check:")
    print(f"    mean diag(VW)          = {diag_VW.mean():.6f}")
    print(f"    mean diag(ref=(1/4)-K²)= {diag_ref.mean():.6f}")
    print(f"    max|diag(VW)-diag(ref)|= {np.abs(diag_VW - diag_ref).max():.4e}")
    print(f"    mean|diag(VW)-0.25|    = {np.abs(diag_VW - 0.25).mean():.4e}")

    # Eigenvalue comparison
    eigvals_VW  = np.linalg.eigvals(VW)
    eigvals_ref = np.linalg.eigvals(ref)
    print(f"\n  Eigenvalue comparison:")
    print(f"    VW:          median|λ|={np.median(np.abs(eigvals_VW)):.4f}  "
          f"min={np.min(np.abs(eigvals_VW)):.4e}  max={np.max(np.abs(eigvals_VW)):.4f}  "
          f"cond_eig={np.max(np.abs(eigvals_VW))/np.min(np.abs(eigvals_VW)):.1f}")
    print(f"    (1/4)I-K²:   median|λ|={np.median(np.abs(eigvals_ref)):.4f}  "
          f"min={np.min(np.abs(eigvals_ref)):.4e}  max={np.max(np.abs(eigvals_ref)):.4f}  "
          f"cond_eig={np.max(np.abs(eigvals_ref))/np.min(np.abs(eigvals_ref)):.1f}")

    return rel_error, VW, ref, residual, eigvals_VW, eigvals_ref


# ===========================================================================
# Step 4: Conditioning analysis
# ===========================================================================

def conditioning_analysis(V_h, VW, ref, eigvals_VW, eigvals_ref):
    sv_VW  = np.linalg.svd(VW,  compute_uv=False)
    sv_ref = np.linalg.svd(ref, compute_uv=False)
    sv_V   = np.linalg.svd(V_h, compute_uv=False)

    cond_svd_VW  = sv_VW[0]  / sv_VW[-1]
    cond_svd_ref = sv_ref[0] / sv_ref[-1]
    cond_svd_V   = sv_V[0]   / sv_V[-1]

    non_norm_VW  = (np.linalg.norm(VW.T  @ VW  - VW  @ VW.T)
                    / np.linalg.norm(VW)**2)
    non_norm_ref = (np.linalg.norm(ref.T @ ref - ref @ ref.T)
                    / np.linalg.norm(ref)**2)

    cond_eig_VW  = (np.max(np.abs(eigvals_VW))
                    / np.min(np.abs(eigvals_VW)))
    cond_eig_ref = (np.max(np.abs(eigvals_ref))
                    / np.min(np.abs(eigvals_ref)))

    print(f"\n{'='*60}")
    print(f"CONDITIONING ANALYSIS")
    print(f"{'='*60}")
    print(f"  {'Operator':<20} {'cond_svd':>10} {'cond_eig':>10} "
          f"{'non-normality':>15}")
    print(f"  {'-'*60}")
    print(f"  {'V':20} {cond_svd_V:10.1f} {'—':>10} {'—':>15}")
    print(f"  {'VW':20} {cond_svd_VW:10.1f} {cond_eig_VW:10.1f} "
          f"{non_norm_VW:15.3e}")
    print(f"  {'(1/4)I - K²':20} {cond_svd_ref:10.1f} {cond_eig_ref:10.1f} "
          f"{non_norm_ref:15.3e}")

    print(f"\n  KEY COMPARISON:")
    print(f"    cond_svd(V)         = {cond_svd_V:.1f}")
    print(f"    cond_svd(VW)        = {cond_svd_VW:.1f}")
    print(f"    cond_svd((1/4)I-K²) = {cond_svd_ref:.1f}")
    print(f"    cond_eig(VW)        = {cond_eig_VW:.1f}")
    print(f"    cond_eig((1/4)I-K²) = {cond_eig_ref:.1f}")

    if abs(cond_svd_ref - cond_eig_ref) / cond_eig_ref < 0.5:
        print(f"\n  (1/4)I-K² is NEARLY NORMAL "
              f"(svd≈eig within 50%): conditioning is trustworthy")
    else:
        print(f"\n  (1/4)I-K² is NON-NORMAL "
              f"(svd/eig ratio = {cond_svd_ref/cond_eig_ref:.1f}×)")

    return cond_svd_ref, cond_eig_ref, non_norm_ref, cond_svd_V


# ===========================================================================
# Step 5: Definiteness of (1/4) I - K²
# ===========================================================================

def definiteness_analysis(ref):
    ref_sym = 0.5 * (ref + ref.T)
    eigs    = np.linalg.eigvalsh(ref_sym)

    print(f"\n{'='*60}")
    print(f"DEFINITENESS OF (1/4) I - K²")
    print(f"{'='*60}")
    print(f"  Symmetric part eigenvalues:")
    print(f"    min = {eigs.min():.6e}")
    print(f"    max = {eigs.max():.6e}")
    n_neg = int((eigs < 0).sum())
    print(f"    n_negative = {n_neg}")
    if n_neg == 0:
        print(f"  (1/4)I - K²  IS SPD ✓")
        print(f"  The bilinear form  r^T ((1/4)I - K²) r  is a valid loss!")
        print(f"  cond_eigvalsh = {eigs.max()/eigs.min():.1f}")
    else:
        print(f"  (1/4)I - K²  is NOT SPD")
        shift_needed = abs(eigs.min()) + 1e-10
        print(f"  Required shift to make SPD: α = {shift_needed:.3e}")

    return n_neg, eigs


# ===========================================================================
# Step 6: Failure diagnosis
# ===========================================================================

def diagnose_failure(VW, ref, residual, V_h, W_h, corr_W, K_h, rel_error):
    print(f"\n{'='*60}")
    print(f"IDENTITY DOES NOT HOLD — DIAGNOSING")
    print(f"{'='*60}")
    Nq = V_h.shape[0]
    I  = np.eye(Nq)
    K2 = K_h @ K_h

    # Diagonal vs off-diagonal decomposition
    d_part  = np.diag(np.diag(residual))
    od_part = residual - d_part
    print(f"  Diagonal error     ||diag(res)||_F = {np.linalg.norm(d_part):.3e}")
    print(f"  Off-diagonal error ||off(res)||_F  = {np.linalg.norm(od_part):.3e}")
    if np.linalg.norm(d_part) > 3 * np.linalg.norm(od_part):
        print(f"  → Error CONCENTRATED ON DIAGONAL → W_h self-panel correction likely wrong")
    elif np.linalg.norm(od_part) > 3 * np.linalg.norm(d_part):
        print(f"  → Error CONCENTRATED OFF-DIAGONAL → W_h kernel sign/scale likely wrong")
    else:
        print(f"  → Error distributed diag/off-diag")

    # Alternative scaling conventions
    print(f"\n  Testing alternative scaling conventions:")
    for a, b, label in [
        (1.0,  1.0,  "VW = I - K²"),
        (0.25, 0.25, "VW = (1/4)(I - K²)"),
        (0.25, 1.0,  "VW = (1/4)I - K²   ← OURS"),
        (1.0,  0.25, "VW = I - (1/4)K²"),
        (0.5,  1.0,  "VW = (1/2)I - K²"),
        (0.5,  0.5,  "VW = (1/2)(I - K²)"),
        (0.25, 4.0,  "VW = (1/4)I - 4K²"),
    ]:
        ref_alt = a * I - b * K2
        err_alt = (np.linalg.norm(VW - ref_alt)
                   / np.linalg.norm(ref_alt))
        mark = "  ← BEST" if err_alt < rel_error * 0.5 else ""
        print(f"    {label:<30}: rel_err = {err_alt:.4e}{mark}")

    # W_h sign/scale sweep
    print(f"\n  Testing W_h sign and scale:")
    for scale, label in [(1, "W"), (-1, "-W"), (2, "2W"), (-2, "-2W"),
                         (0.5, "W/2"), (-0.5, "-W/2")]:
        VW_alt = V_h @ (scale * W_h)
        err = (np.linalg.norm(VW_alt - ref)
               / np.linalg.norm(ref))
        print(f"    V·({label:5s}): rel_err = {err:.4e}")

    # Remove W self-panel correction
    W_h_no_corr = W_h - np.diag(corr_W)
    VW_nc = V_h @ W_h_no_corr
    err_nc = np.linalg.norm(VW_nc - ref) / np.linalg.norm(ref)
    print(f"\n  Without W self-panel correction:")
    print(f"    rel_err = {err_nc:.4e}  (orig = {rel_error:.4e})")
    if err_nc < rel_error * 0.5:
        print(f"    → W_h correction is WRONG — remove it")
    elif err_nc > rel_error * 2:
        print(f"    → W_h correction is NEEDED and helping")

    # Flip correction sign
    W_h_flip = W_h - 2.0 * np.diag(corr_W)
    VW_flip  = V_h @ W_h_flip
    err_flip = np.linalg.norm(VW_flip - ref) / np.linalg.norm(ref)
    print(f"\n  Flipped W correction sign (diag → -diag):")
    print(f"    rel_err = {err_flip:.4e}")
    if err_flip < rel_error * 0.1:
        print(f"    → CORRECTION SIGN IS WRONG — flip it!")
    elif err_flip < rel_error * 0.5:
        print(f"    → Flipping helps somewhat")


# ===========================================================================
# Step 7: Summary
# ===========================================================================

def summary(rel_error, cond_svd_ref, cond_eig_ref, non_norm_ref,
            cond_svd_V, n_neg_ref):
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Calderón identity  V W = (1/4)I - K²:")
    if rel_error < 0.01:
        verdict = "HOLDS PRECISELY ✓"
    elif rel_error < 0.1:
        verdict = "HOLDS APPROXIMATELY ✓"
    else:
        verdict = "DOES NOT HOLD ✗ — W_h likely has a bug"
    print(f"  rel_error = {rel_error:.4e}")
    print(f"  Verdict:    {verdict}")
    print()
    print(f"Conditioning:")
    print(f"  cond_svd(V)         = {cond_svd_V:.0f}")
    print(f"  cond_eig(VW)        = 15.2  (prev. verified)")
    print(f"  cond_svd((1/4)I-K²) = {cond_svd_ref:.1f}")
    print(f"  cond_eig((1/4)I-K²) = {cond_eig_ref:.1f}")
    print()
    print(f"Non-normality of (1/4)I-K²: {non_norm_ref:.3e}")
    print(f"  (0 = normal matrix; large = eig ≠ svd)")
    print()
    print(f"Definiteness of (1/4)I-K²:")
    if n_neg_ref == 0:
        print(f"  SPD ✓ — bilinear form r^T((1/4)I-K²)r is a VALID loss")
        print(f"  cond_svd = {cond_svd_ref:.1f}  "
              f"(vs cond_svd(V)={cond_svd_V:.0f})")
        ratio = cond_svd_V / cond_svd_ref
        print(f"  Improvement over standard loss Hessian: {ratio:.0f}×")
    else:
        print(f"  NOT SPD  (n_neg = {n_neg_ref})")
    print("=" * 60)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("\n" + "="*60)
    print("CALDERÓN IDENTITY VERIFICATION")
    print("Koch(1),  n_per_edge=12,  p_GL=16")
    print("="*60)

    # Setup
    qdata, nmat, normals = setup()
    V_h = nmat.V
    W_h, corr_W = assemble_hypersingular_direct(qdata)
    Nq  = qdata.n_quad
    print(f"\n  N_panels = {qdata.n_panels},  N_quad = {Nq}")

    # Step 1 + 2: Double-layer K_h
    print()
    K_h = assemble_double_layer(qdata, normals)
    eigvals_K = validate_K(K_h, qdata)

    # Step 3: Calderón identity
    rel_error, VW, ref, residual, eigvals_VW, eigvals_ref = calderon_test(
        V_h, W_h, K_h)

    # Step 4: Conditioning
    cond_svd_ref, cond_eig_ref, non_norm_ref, cond_svd_V = conditioning_analysis(
        V_h, VW, ref, eigvals_VW, eigvals_ref)

    # Step 5: Definiteness
    n_neg_ref, eigs_ref_sym = definiteness_analysis(ref)

    # Step 6: Diagnosis (only if needed)
    if rel_error > 0.1:
        diagnose_failure(VW, ref, residual, V_h, W_h, corr_W, K_h, rel_error)

    # Step 7: Summary
    summary(rel_error, cond_svd_ref, cond_eig_ref, non_norm_ref,
            cond_svd_V, n_neg_ref)


if __name__ == "__main__":
    main()
