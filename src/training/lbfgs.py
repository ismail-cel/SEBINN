"""
L-BFGS refinement with Armijo backtracking for SE-BINN.

MATLAB reference
----------------
run_lbfgs_refinement   lines 1162-1285
lbfgs_two_loop         lines 1287-1315
lbfgs_line_search      lines 1317-1378
loss_and_grad_from_theta lines 1431-1437

The algorithm is standard limited-memory BFGS (Nocedal & Wright §7.4) with
Armijo backtracking.  Curvature pairs are maintained as explicit Python lists
of (s, y) tensors, matching MATLAB's column-append/drop logic on S and Y.

SE-BINN vs MATLAB
-----------------
MATLAB net_to_vector / vector_to_net   →   model.to_vector() / model.from_vector()
MATLAB loss_and_grad_from_theta        →   _loss_and_grad() (below)

Everything else — two-loop recursion, line search, stopping criteria — is
translated directly from the MATLAB reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch

from ..models.sebinn import SEBINNModel
from .loss import sebinn_loss
from .operator import OperatorState


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LBFGSConfig:
    """
    Configuration for the L-BFGS refinement.

    MATLAB cfg equivalents are noted in comments.
    """
    max_iters:           int         = 3500   # cfg.lbfgsMaxIters
    grad_tol:            float       = 1e-8   # cfg.lbfgsGradTol
    step_tol:            float       = 1e-12  # cfg.lbfgsStepTol
    step_tol_needs_grad: bool        = True   # cfg.lbfgsStepTolNeedsGrad
    memory:              int         = 10     # cfg.lbfgsMemory
    log_every:           int         = 25     # cfg.lbfgsLogEvery
    alpha0:              float       = 1e-1   # cfg.lbfgsAlpha0
    alpha_fallback:      List[float] = field(
        default_factory=lambda: [1e-2, 1e-3]
    )                                         # cfg.lbfgsAlphaFallback
    armijo_c1:           float       = 1e-4   # cfg.lbfgsArmijoC1
    backtrack_beta:      float       = 0.5    # cfg.lbfgsBacktrackBeta
    max_backtrack:       int         = 20     # cfg.lbfgsMaxBacktrack


@dataclass
class LBFGSResult:
    """
    Output of run_lbfgs.

    Attributes
    ----------
    loss_hist     : list of float  — loss at each completed iteration
    grad_hist     : list of float  — gradient norm at each completed iteration
    reason        : str            — stopping reason string
    n_iters       : int            — iterations completed
    n_ls_failures : int            — Armijo line-search failures
    """
    loss_hist:     List[float]
    grad_hist:     List[float]
    reason:        str
    n_iters:       int
    n_ls_failures: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _loss_and_grad(
    theta: torch.Tensor,
    model: SEBINNModel,
    op:    OperatorState,
    loss_fn: Callable = sebinn_loss,
) -> Tuple[float, torch.Tensor]:
    """
    Load theta into model, evaluate loss, return (f, g).

    MATLAB: loss_and_grad_from_theta (lines 1431-1437).

    After the call the model holds theta.  If the step is rejected the caller
    must restore the previous theta via model.from_vector(theta_old).

    Parameters
    ----------
    theta   : Tensor (D,)  — no grad required
    model   : SEBINNModel  — modified in place
    op      : OperatorState
    loss_fn : callable (model, op) -> (loss, dbg)

    Returns
    -------
    f : float
    g : Tensor (D,)  detached gradient w.r.t. theta
    """
    model.from_vector(theta)

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    loss, _ = loss_fn(model, op)
    loss.backward()

    grads = []
    for p in model.parameters():
        g_p = p.grad
        grads.append(
            g_p.detach().view(-1) if g_p is not None
            else torch.zeros(p.numel(), dtype=p.dtype, device=p.device)
        )
    g = torch.cat(grads)
    return float(loss.detach()), g


def _two_loop(
    g:        torch.Tensor,
    S:        List[torch.Tensor],
    Y:        List[torch.Tensor],
    rho_list: List[float],
) -> torch.Tensor:
    """
    L-BFGS two-loop recursion (Nocedal & Wright Algorithm 7.4).

    MATLAB: lbfgs_two_loop (lines 1287-1315).

    Parameters
    ----------
    g        : Tensor (D,)
    S        : list of Tensor (D,)  — s vectors, oldest first
    Y        : list of Tensor (D,)  — y vectors, oldest first
    rho_list : list of float        — 1 / (y^T s), oldest first

    Returns
    -------
    Hg : Tensor (D,)  — H^{-1} g
    """
    m = len(S)
    q = g.clone()
    alphas = [0.0] * m

    for i in range(m - 1, -1, -1):
        a = rho_list[i] * float(S[i].dot(q))
        q = q - a * Y[i]
        alphas[i] = a

    # Scaling: H_0 = gamma * I,  gamma = s_{m-1}^T y_{m-1} / ||y_{m-1}||^2
    if m > 0:
        sy = float(S[-1].dot(Y[-1]))
        yy = float(Y[-1].dot(Y[-1]))
        gamma = sy / max(yy, 1e-20)
        if not (gamma > 0 and torch.isfinite(torch.tensor(gamma))):
            gamma = 1.0
    else:
        gamma = 1.0

    r = gamma * q
    for i in range(m):
        beta = rho_list[i] * float(Y[i].dot(r))
        r = r + S[i] * (alphas[i] - beta)

    return r


def _armijo_line_search(
    theta:        torch.Tensor,
    f:            float,
    g:            torch.Tensor,
    p:            torch.Tensor,
    model:        SEBINNModel,
    op:           OperatorState,
    cfg:          LBFGSConfig,
    alpha_starts: List[float],
    loss_fn:      Callable = sebinn_loss,
) -> Tuple[float, torch.Tensor, float, torch.Tensor, bool]:
    """
    Armijo backtracking line search.

    MATLAB: lbfgs_line_search (lines 1317-1378).

    Returns
    -------
    alpha     : accepted step size (0.0 if not accepted)
    theta_new : Tensor (D,)
    f_new     : float
    g_new     : Tensor (D,)
    accepted  : bool
    """
    c1   = cfg.armijo_c1
    beta = cfg.backtrack_beta

    gtp = float(g.dot(p))
    if gtp >= 0:
        p   = -g
        gtp = float(g.dot(p))

    theta_new = theta
    f_new     = f
    g_new     = g
    accepted  = False
    alpha_out = 0.0

    for alpha_try in alpha_starts:
        for _ in range(cfg.max_backtrack):
            theta_try = theta + alpha_try * p
            try:
                f_try, g_try = _loss_and_grad(theta_try, model, op, loss_fn)
            except Exception:
                alpha_try *= beta
                if alpha_try < cfg.step_tol:
                    break
                continue

            if not (
                torch.isfinite(torch.tensor(f_try))
                and g_try.isfinite().all()
            ):
                alpha_try *= beta
                if alpha_try < cfg.step_tol:
                    break
                continue

            if f_try <= f + c1 * alpha_try * gtp:
                accepted  = True
                theta_new = theta_try
                f_new     = f_try
                g_new     = g_try
                alpha_out = alpha_try
                break

            alpha_try *= beta
            if alpha_try < cfg.step_tol:
                break

        if accepted:
            break

    if not accepted:
        # Restore old parameters so model stays consistent
        model.from_vector(theta)

    return alpha_out, theta_new, f_new, g_new, accepted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_lbfgs(
    model:   SEBINNModel,
    op:      OperatorState,
    cfg:     LBFGSConfig,
    verbose: bool = True,
    loss_fn: Optional[Callable] = None,
) -> LBFGSResult:
    """
    L-BFGS refinement with Armijo backtracking.

    MATLAB: run_lbfgs_refinement (lines 1162-1285).

    The model is modified in place and left at the last accepted parameters.

    Parameters
    ----------
    model   : SEBINNModel
    op      : OperatorState
    cfg     : LBFGSConfig
    verbose : bool
    loss_fn : callable (model, op) -> (loss, dbg) or None
        If None, defaults to sebinn_loss.  Pass make_loss_fn(...) for the
        corner-penalty variant.

    Returns
    -------
    LBFGSResult
    """
    if loss_fn is None:
        loss_fn = sebinn_loss

    theta = model.to_vector().clone()
    f, g  = _loss_and_grad(theta, model, op, loss_fn)

    max_iters = cfg.max_iters
    loss_hist: List[float] = [0.0] * max_iters
    grad_hist: List[float] = [0.0] * max_iters

    S:        List[torch.Tensor] = []
    Y:        List[torch.Tensor] = []
    rho_list: List[float]        = []

    ls_failures = 0
    reason      = "maxIters"
    k_final     = 0

    for k in range(max_iters):
        k_final = k + 1
        g_norm  = float(g.norm())
        loss_hist[k] = f
        grad_hist[k] = g_norm

        # --- Stopping: non-finite ---
        if not (
            torch.isfinite(torch.tensor(f))
            and torch.isfinite(torch.tensor(g_norm))
        ):
            reason = "nonFinite"
            if verbose:
                print(
                    f"LBFGS {k+1:4d} | loss={f:.3e} | "
                    f"gradNorm={g_norm:.3e} | stop={reason}"
                )
            break

        # --- Stopping: gradient tolerance ---
        if g_norm < cfg.grad_tol:
            reason = "gradTol"
            if verbose:
                print(
                    f"LBFGS {k+1:4d} | loss={f:.3e} | "
                    f"gradNorm={g_norm:.3e} | stop={reason}"
                )
            break

        # --- Search direction ---
        p = -_two_loop(g, S, Y, rho_list) if S else -g
        if float(p.dot(g)) >= -1e-16:
            p = -g

        # --- Line search ---
        alpha_starts = [cfg.alpha0] + list(cfg.alpha_fallback)
        alpha, theta_new, f_new, g_new, accepted = _armijo_line_search(
            theta, f, g, p, model, op, cfg, alpha_starts, loss_fn
        )

        if not accepted:
            ls_failures += 1
            # Reset L-BFGS memory; retry steepest descent
            S, Y, rho_list = [], [], []
            p = -g
            alpha, theta_new, f_new, g_new, accepted = _armijo_line_search(
                theta, f, g, p, model, op, cfg, list(cfg.alpha_fallback), loss_fn
            )
            if not accepted:
                reason = "near-stationary / line-search stalled"
                if verbose:
                    print(
                        f"LBFGS {k+1:4d} | loss={f:.3e} | "
                        f"gradNorm={g_norm:.3e} | stop={reason}"
                    )
                break

        s         = theta_new - theta
        step_norm = float(s.norm())

        if (k + 1) % cfg.log_every == 0 or k == 0 or k + 1 == max_iters:
            if verbose:
                print(
                    f"LBFGS {k+1:4d} | loss={f_new:.3e} | "
                    f"gradNorm={float(g_new.norm()):.3e} | "
                    f"alpha={alpha:.2e} | stepNorm={step_norm:.2e}"
                )

        # --- Step tolerance ---
        if step_norm <= cfg.step_tol * max(1.0, float(theta.norm())):
            theta = theta_new
            f     = f_new
            g     = g_new
            loss_hist[k] = f
            grad_hist[k] = float(g.norm())
            if cfg.step_tol_needs_grad and float(g.norm()) >= cfg.grad_tol:
                S, Y, rho_list = [], [], []
                if verbose:
                    print(
                        f"LBFGS {k+1:4d} | small step, resetting memory | "
                        f"loss={f:.3e} | gradNorm={float(g.norm()):.3e}"
                    )
                continue
            else:
                reason = "stepTol"
                if verbose:
                    print(
                        f"LBFGS {k+1:4d} | loss={f:.3e} | "
                        f"gradNorm={float(g.norm()):.3e} | stop={reason}"
                    )
                break

        # --- Curvature pair update ---
        y  = g_new - g
        ys = float(y.dot(s))
        if ys > 1e-12 * max(1.0, float(s.norm()) * float(y.norm())):
            if len(S) == cfg.memory:
                S.pop(0); Y.pop(0); rho_list.pop(0)
            S.append(s.clone())
            Y.append(y.clone())
            rho_list.append(1.0 / ys)

        theta = theta_new
        f     = f_new
        g     = g_new

    loss_hist = loss_hist[:k_final]
    grad_hist = grad_hist[:k_final]

    if verbose:
        print(
            f"LBFGS done | iters={k_final} | reason={reason} | "
            f"ls_failures={ls_failures}"
        )

    return LBFGSResult(
        loss_hist=loss_hist,
        grad_hist=grad_hist,
        reason=reason,
        n_iters=k_final,
        n_ls_failures=ls_failures,
    )
