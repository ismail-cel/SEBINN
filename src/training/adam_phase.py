"""
Adam training phase loop for SE-BINN.

MATLAB reference
----------------
run_static_training_case   lines 589-695  (Adam inner loop: lines 619-641)

The MATLAB code runs multiple Adam phases sequentially, passing the accumulated
trailing averages (moment states) across phases so the global step counter and
moment estimates are preserved.  This is reproduced here by keeping a single
torch.optim.Adam instance and updating its learning rate at phase boundaries.

Phase schedule
--------------
Each phase is defined by (n_iters, lr).  The Adam moment states are NOT reset
between phases — the optimizer instance is shared.  The global iteration
counter advances monotonically, matching MATLAB's 'it' variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from ..models.sebinn import SEBINNModel
from .loss import sebinn_loss
from .operator import OperatorState


@dataclass
class AdamConfig:
    """
    Configuration for the multi-phase Adam loop.

    Attributes
    ----------
    phase_iters : list of int
        Iterations per phase.  MATLAB: cfg.adamPhaseIters = [500].
    phase_lrs   : list of float
        Learning rate per phase.  Must be the same length as phase_iters.
        MATLAB: cfg.adamPhaseLr = [1e-4].
    log_every   : int
        Print diagnostics every this many global iterations.
        MATLAB: cfg.logEvery = 50.
    betas : (float, float)
        Adam beta1, beta2.  PyTorch default (0.9, 0.999) matches MATLAB.
    eps   : float
        Adam epsilon.  PyTorch default 1e-8 matches MATLAB.
    """
    phase_iters: List[int]   = field(default_factory=lambda: [500])
    phase_lrs:   List[float] = field(default_factory=lambda: [1e-4])
    log_every:   int         = 50
    betas:       tuple       = (0.9, 0.999)
    eps:         float       = 1e-8

    def __post_init__(self):
        if len(self.phase_iters) != len(self.phase_lrs):
            raise ValueError(
                "phase_iters and phase_lrs must have the same length, "
                f"got {len(self.phase_iters)} and {len(self.phase_lrs)}"
            )


@dataclass
class AdamResult:
    """
    Output of run_adam_phases.

    Attributes
    ----------
    loss_hist  : list of float  — loss at every global iteration
    dbg_hist   : list of dict   — diagnostics at every logged iteration
    final_loss : float          — loss at the last iteration
    n_iters    : int            — total iterations completed
    """
    loss_hist:  List[float]
    dbg_hist:   List[dict]
    final_loss: float
    n_iters:    int


def _fmt_gamma(g) -> str:
    """Format gamma for logging — handles scalar or list."""
    if isinstance(g, (list, tuple)):
        return "[" + ", ".join(f"{v:.4f}" for v in g) + "]"
    try:
        return f"{float(g):.4f}"
    except (TypeError, ValueError):
        return str(g)


def run_adam_phases(
    model:   SEBINNModel,
    op:      OperatorState,
    cfg:     AdamConfig,
    verbose: bool = True,
    loss_fn: Optional[Callable] = None,
) -> AdamResult:
    """
    Run Adam in multiple phases, preserving moment states across phases.

    MATLAB: Adam inner loop in run_static_training_case (lines 619-641).

    Parameters
    ----------
    model   : SEBINNModel   — modified in place
    op      : OperatorState — fixed; not rebuilt between phases
    cfg     : AdamConfig
    verbose : bool
    loss_fn : callable (model, op) -> (loss, dbg) or None
        If None, defaults to sebinn_loss.  Pass make_loss_fn(...) for the
        corner-penalty variant.

    Returns
    -------
    AdamResult
    """
    if loss_fn is None:
        loss_fn = sebinn_loss

    total_iters = sum(cfg.phase_iters)
    n_phases    = len(cfg.phase_iters)

    loss_hist: List[float] = [0.0] * total_iters
    dbg_hist:  List[dict]  = []

    # Single Adam optimizer.  lr is updated at phase boundaries but the moment
    # states (exp_avg, exp_avg_sq) persist — matching MATLAB's trailingAvg /
    # trailingAvgSq being passed into each adamupdate call.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.phase_lrs[0],
        betas=cfg.betas,
        eps=cfg.eps,
    )

    global_it = 0  # 0-based; MATLAB 'it' is 1-based

    for i_phase, (n_it, lr) in enumerate(zip(cfg.phase_iters, cfg.phase_lrs)):
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if verbose:
            print(
                f"phase {i_phase+1}/{n_phases} | iters={n_it} | "
                f"lr={lr:.1e} | Nb={op.n_colloc}"
            )

        for j in range(n_it):
            optimizer.zero_grad()
            loss, dbg = loss_fn(model, op)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.detach())
            loss_hist[global_it] = loss_val

            log_this = (
                verbose and (
                    global_it == 0
                    or j == 0
                    or (global_it + 1) % cfg.log_every == 0
                    or global_it == total_iters - 1
                )
            )
            if log_this:
                dbg_hist.append({**dbg, "iter": global_it + 1})
                print(
                    f"  Adam {global_it+1:4d} | loss={loss_val:.3e} "
                    f"| mse_unsc={dbg.get('mse_unscaled', float('nan')):.3e} "
                    f"| gamma={_fmt_gamma(dbg.get('gamma', float('nan')))} "
                    f"| phase={i_phase+1}"
                )

            global_it += 1

    final_loss = loss_hist[-1] if loss_hist else float("nan")

    if verbose:
        print(
            f"Adam done | total_iters={global_it} | final_loss={final_loss:.3e}"
        )

    return AdamResult(
        loss_hist=loss_hist,
        dbg_hist=dbg_hist,
        final_loss=final_loss,
        n_iters=global_it,
    )
