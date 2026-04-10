"""
SE-BINN residual and weighted MSE loss.

MATLAB reference
----------------
residual_vector_from_state   lines 1094-1160
modelLossBINN_state          lines 1083-1092

The MATLAB residual is:

    Vstd  = A * sigma_std          (matrix-vector product, differentiable)
    Vsig  = Vstd + corr * sigma_c  (add self-panel correction)
    res   = Vsig - f               (BIE residual at collocation points)

    loss  = sum(wCol * res^2) / sum(wCol)

SE-BINN change: sigma_std and sigma_c are no longer the raw network output.
They are the enriched densities:

    sigma_std = net(Yq) + gamma * sigma_s_q      (at quadrature nodes)
    sigma_c   = net(Xc) + gamma * sigma_s_c      (at collocation nodes)

Everything else — the matrix product, the correction, the weighted MSE —
is identical to the MATLAB baseline.

Debug quantities (MATLAB dbgCore)
----------------------------------
Returned as a plain dict for logging:
  mean_abs_Vstd  : mean|A * sigma_std|
  mean_abs_corr  : mean|corr * sigma_c|
  mean_abs_res   : mean|residual|
  mse_scaled     : mean(res^2)  (scaled)
  mse_unscaled   : mean(res/eq_scale)^2
"""

from __future__ import annotations

import torch

from ..models.sebinn import SEBINNModel
from .operator import OperatorState


def residual_vector(
    model: SEBINNModel,
    op: OperatorState,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the BIE residual vector at all collocation points.

    MATLAB: residual_vector_from_state (lines 1094-1160).

    The near-panel refinement branch (useNearPanelRefine) is intentionally
    omitted here: the operator state already incorporates the refinement
    into the pre-assembled matrix A via assemble_nystrom_matrix.  Dynamic
    near-panel correction inside the loss would require per-iteration kernel
    evaluations that are expensive and were disabled in the MATLAB runs
    (cfg.useNearPanelRefine = false by default).

    Parameters
    ----------
    model : SEBINNModel
    op    : OperatorState

    Returns
    -------
    res  : Tensor (Nb,)  — BIE residual, differentiable w.r.t. model params
    dbg  : dict          — diagnostic scalars (detached)
    """
    # --- Evaluate enriched density at quadrature and collocation nodes ---
    # sigma_std = sigma_w(Yq) + gamma * sigma_s_q
    sigma_std = model(op.Yq, op.sigma_s_q)    # (Nq,)

    # sigma_c   = sigma_w(Xc) + gamma * sigma_s_c
    sigma_c = model(op.Xc, op.sigma_s_c)      # (Nb,)

    # --- BIE matrix-vector product ---
    # Vstd[k] = sum_j A[k,j] * sigma_std[j]   (MATLAB: A_dl * sigma_std)
    Vstd = op.A @ sigma_std                    # (Nb,)

    # --- Self-panel correction ---
    # Vsig[k] = Vstd[k] + corr[k] * sigma_c[k]
    Vsig = Vstd + op.corr * sigma_c            # (Nb,)

    # --- BIE residual ---
    res = Vsig - op.f                          # (Nb,)

    # --- Debug quantities (detached) ---
    with torch.no_grad():
        dbg = {
            "mean_abs_Vstd": float((op.A @ sigma_std.detach()).abs().mean()),
            "mean_abs_corr": float((op.corr * sigma_c.detach()).abs().mean()),
            "mean_abs_res":  float(res.detach().abs().mean()),
        }

    return res, dbg


def sebinn_loss(
    model: SEBINNModel,
    op: OperatorState,
) -> tuple[torch.Tensor, dict]:
    """
    Weighted MSE loss over BIE collocation residuals.

    MATLAB: modelLossBINN_state (lines 1083-1092).

        loss = sum(wCol * res^2) / sum(wCol)

    Parameters
    ----------
    model : SEBINNModel
    op    : OperatorState

    Returns
    -------
    loss : Tensor ()   — scalar, differentiable
    dbg  : dict        — diagnostic scalars (detached)
    """
    res, dbg = residual_vector(model, op)

    res2 = res ** 2                                        # (Nb,)
    loss = (op.wCol * res2).sum() / op.wCol_sum            # scalar

    # Unscaled MSE (removes eq_scale from the diagnostic)
    with torch.no_grad():
        eq_scale = max(op.eq_scale, 1e-14)
        res_unscaled = res.detach() / eq_scale
        dbg["mse_scaled"]   = float(res2.detach().mean())
        dbg["mse_unscaled"] = float((res_unscaled ** 2).mean())
        dbg["loss"]         = float(loss.detach())
        dbg["gamma"]        = model.gamma_value()

    return loss, dbg
