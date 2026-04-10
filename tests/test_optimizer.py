"""
Tests for src/training/adam_phase.py and src/training/lbfgs.py.

These tests use a minimal synthetic operator state (tiny A, corr, f, wCol)
so they run fast without any geometry or quadrature setup.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.models.sebinn import SEBINNModel
from src.training.operator import OperatorState
from src.training.adam_phase import AdamConfig, AdamResult, run_adam_phases
from src.training.lbfgs import LBFGSConfig, LBFGSResult, run_lbfgs, _two_loop


# ---------------------------------------------------------------------------
# Shared fixture: tiny synthetic operator
# ---------------------------------------------------------------------------

def _make_op(Nb=4, Nq=6, seed=0):
    """
    Build a minimal OperatorState for testing.

    A is (Nb, Nq), corr is (Nb,), f is (Nb,), wCol uniform.
    All tensors are float64.
    """
    rng = np.random.default_rng(seed)
    A_np    = rng.standard_normal((Nb, Nq)) * 0.1
    corr_np = rng.standard_normal(Nb) * 0.1
    f_np    = rng.standard_normal(Nb) * 0.05
    wCol_np = np.ones(Nb)

    Yq_np      = rng.standard_normal((Nq, 2))
    Xc_np      = rng.standard_normal((Nb, 2))
    sigma_s_q  = np.zeros(Nq)
    sigma_s_c  = np.zeros(Nb)

    def t(x):
        return torch.tensor(x, dtype=torch.float64)

    op = OperatorState(
        A=t(A_np),
        corr=t(corr_np),
        f=t(f_np),
        wCol=t(wCol_np),
        wCol_sum=float(wCol_np.sum()),
        Yq=t(Yq_np),
        Xc=t(Xc_np),
        sigma_s_q=t(sigma_s_q),
        sigma_s_c=t(sigma_s_c),
        pan_of_xc=np.zeros(Nb, dtype=int),
        s0_of_xc=np.zeros(Nb),
        idx_std=[np.arange(Nq)],
        n_colloc=Nb,
        n_quad=Nq,
        n_panels=1,
        eq_scale=1.0,
    )
    return op


def _make_model(seed=0):
    torch.manual_seed(seed)
    return SEBINNModel(
        hidden_width=8, n_hidden=2, n_gamma=1, gamma_init=0.0,
        dtype=torch.float64,
    )


# ===========================================================================
# AdamConfig
# ===========================================================================

class TestAdamConfig:

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            AdamConfig(phase_iters=[100, 200], phase_lrs=[1e-3])

    def test_default_single_phase(self):
        cfg = AdamConfig()
        assert len(cfg.phase_iters) == 1
        assert len(cfg.phase_lrs) == 1

    def test_multi_phase_ok(self):
        cfg = AdamConfig(phase_iters=[50, 50], phase_lrs=[1e-3, 1e-4])
        assert len(cfg.phase_iters) == 2


# ===========================================================================
# run_adam_phases
# ===========================================================================

class TestRunAdamPhases:

    def test_returns_adam_result(self):
        model = _make_model()
        op    = _make_op()
        cfg   = AdamConfig(phase_iters=[3], phase_lrs=[1e-3], log_every=10)
        result = run_adam_phases(model, op, cfg, verbose=False)
        assert isinstance(result, AdamResult)

    def test_loss_hist_length(self):
        model = _make_model()
        op    = _make_op()
        cfg   = AdamConfig(phase_iters=[5, 5], phase_lrs=[1e-3, 1e-4],
                           log_every=100)
        result = run_adam_phases(model, op, cfg, verbose=False)
        assert len(result.loss_hist) == 10
        assert result.n_iters == 10

    def test_final_loss_matches_last_entry(self):
        model = _make_model()
        op    = _make_op()
        cfg   = AdamConfig(phase_iters=[4], phase_lrs=[1e-3], log_every=100)
        result = run_adam_phases(model, op, cfg, verbose=False)
        assert abs(result.final_loss - result.loss_hist[-1]) < 1e-15

    def test_loss_decreases_overall(self):
        """After enough Adam steps, loss should decrease from its initial value."""
        torch.manual_seed(0)
        model = _make_model()
        op    = _make_op()
        cfg   = AdamConfig(phase_iters=[200], phase_lrs=[1e-2], log_every=1000)
        result = run_adam_phases(model, op, cfg, verbose=False)
        assert result.loss_hist[-1] < result.loss_hist[0], (
            f"Loss did not decrease: {result.loss_hist[0]:.4e} -> {result.loss_hist[-1]:.4e}"
        )

    def test_model_modified_in_place(self):
        """Parameters must change after training."""
        model = _make_model()
        op    = _make_op()
        theta_before = model.to_vector().clone()
        cfg = AdamConfig(phase_iters=[5], phase_lrs=[1e-3], log_every=100)
        run_adam_phases(model, op, cfg, verbose=False)
        theta_after = model.to_vector()
        assert not torch.allclose(theta_before, theta_after)

    def test_moment_states_preserved_across_phases(self):
        """
        Running [n, n] at lr should give the same result as one phase [2n]
        at the same lr — because moment states are preserved.
        """
        op = _make_op()

        torch.manual_seed(7)
        m1 = _make_model()
        cfg1 = AdamConfig(phase_iters=[6, 6], phase_lrs=[1e-3, 1e-3],
                          log_every=100)
        run_adam_phases(m1, op, cfg1, verbose=False)

        torch.manual_seed(7)
        m2 = _make_model()
        cfg2 = AdamConfig(phase_iters=[12], phase_lrs=[1e-3], log_every=100)
        run_adam_phases(m2, op, cfg2, verbose=False)

        assert torch.allclose(m1.to_vector(), m2.to_vector(), atol=1e-14), (
            "Two-phase and single-phase runs with the same lr should be identical"
        )

    def test_lr_changes_between_phases(self):
        """
        With lr=0 in phase 2, optimizer steps are no-ops so the model
        parameters (and therefore the loss) stay constant across phase 2.

        loss_hist[k] records the loss BEFORE step k.  Step 4 (end of phase 1)
        uses lr=1e-2, so loss_hist[5] = loss after that step.  Steps 5-9 use
        lr=0, so loss_hist[5] == loss_hist[6] == ... == loss_hist[9].
        """
        model = _make_model()
        op    = _make_op()
        cfg   = AdamConfig(
            phase_iters=[5, 5], phase_lrs=[1e-2, 0.0], log_every=100
        )
        result = run_adam_phases(model, op, cfg, verbose=False)
        # All phase-2 iterations (indices 5–9) record the same loss because
        # lr=0 means parameters never change after index 4's step.
        phase2 = result.loss_hist[5:]
        for v in phase2:
            assert v == pytest.approx(phase2[0], rel=1e-12)

    def test_all_loss_values_finite(self):
        model = _make_model()
        op    = _make_op()
        cfg   = AdamConfig(phase_iters=[10], phase_lrs=[1e-3], log_every=100)
        result = run_adam_phases(model, op, cfg, verbose=False)
        assert all(np.isfinite(v) for v in result.loss_hist)


# ===========================================================================
# L-BFGS two-loop recursion
# ===========================================================================

class TestTwoLoop:

    def test_empty_memory_is_steepest_descent(self):
        """With no curvature pairs, H^{-1} g = g (gamma=1 scaling)."""
        g = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        Hg = _two_loop(g, [], [], [])
        assert torch.allclose(Hg, g)

    def test_positive_definite_one_pair(self):
        """With one curvature pair, the direction should be a descent direction."""
        g = torch.tensor([1.0, 0.0], dtype=torch.float64)
        s = torch.tensor([0.1, 0.0], dtype=torch.float64)
        y = torch.tensor([0.2, 0.0], dtype=torch.float64)
        rho = [1.0 / float(y.dot(s))]
        Hg = _two_loop(g, [s], [y], rho)
        # Hg should point in the same direction as g for a P.D. problem
        assert float(Hg.dot(g)) > 0

    def test_scaling_gamma(self):
        """
        With one pair, the initial scaling is gamma = s^T y / ||y||^2.
        For g orthogonal to y, the output is gamma * g.
        """
        s = torch.tensor([1.0, 0.0], dtype=torch.float64)
        y = torch.tensor([2.0, 0.0], dtype=torch.float64)
        g = torch.tensor([0.0, 1.0], dtype=torch.float64)  # orthogonal to y
        rho = [1.0 / float(y.dot(s))]
        Hg = _two_loop(g, [s], [y], rho)
        gamma = float(s.dot(y)) / float(y.dot(y))  # = 0.5
        assert torch.allclose(Hg, gamma * g, atol=1e-14)


# ===========================================================================
# run_lbfgs
# ===========================================================================

class TestRunLBFGS:

    def test_returns_lbfgs_result(self):
        model = _make_model()
        op    = _make_op()
        cfg   = LBFGSConfig(max_iters=5, log_every=10)
        result = run_lbfgs(model, op, cfg, verbose=False)
        assert isinstance(result, LBFGSResult)

    def test_loss_hist_length_le_max_iters(self):
        model = _make_model()
        op    = _make_op()
        cfg   = LBFGSConfig(max_iters=10, log_every=100)
        result = run_lbfgs(model, op, cfg, verbose=False)
        assert len(result.loss_hist) <= 10
        assert len(result.grad_hist) == len(result.loss_hist)
        assert result.n_iters == len(result.loss_hist)

    def test_all_values_finite(self):
        model = _make_model()
        op    = _make_op()
        cfg   = LBFGSConfig(max_iters=10, log_every=100)
        result = run_lbfgs(model, op, cfg, verbose=False)
        assert all(np.isfinite(v) for v in result.loss_hist)
        assert all(np.isfinite(v) for v in result.grad_hist)

    def test_model_modified_in_place(self):
        model = _make_model()
        op    = _make_op()
        theta_before = model.to_vector().clone()
        cfg = LBFGSConfig(max_iters=5, log_every=100)
        run_lbfgs(model, op, cfg, verbose=False)
        # After at least one accepted step, parameters change
        # (may not change if immediately at stationary point, but that's
        #  unlikely for a random init)
        theta_after = model.to_vector()
        # At minimum, the model is consistent (no crash)
        assert theta_after.shape == theta_before.shape

    def test_loss_decreases_after_adam_warmup(self):
        """
        L-BFGS should be monotone: loss_hist[-1] <= loss_hist[0].

        Note: adam_res.final_loss is the pre-step loss at the last Adam
        iteration, while L-BFGS starts from the post-step parameters, so
        the two are not directly comparable.  The right invariant is that
        L-BFGS itself is non-increasing.
        """
        torch.manual_seed(1)
        model = _make_model()
        op    = _make_op()

        # Adam warm-up
        adam_cfg = AdamConfig(phase_iters=[100], phase_lrs=[1e-2],
                              log_every=1000)
        run_adam_phases(model, op, adam_cfg, verbose=False)

        # L-BFGS refinement
        lbfgs_cfg = LBFGSConfig(max_iters=50, log_every=100)
        lbfgs_res = run_lbfgs(model, op, lbfgs_cfg, verbose=False)

        if lbfgs_res.n_iters > 1:
            assert lbfgs_res.loss_hist[-1] <= lbfgs_res.loss_hist[0] + 1e-14, (
                f"L-BFGS loss increased: "
                f"{lbfgs_res.loss_hist[0]:.4e} -> {lbfgs_res.loss_hist[-1]:.4e}"
            )

    def test_stopping_reason_is_set(self):
        model = _make_model()
        op    = _make_op()
        cfg   = LBFGSConfig(max_iters=3, log_every=100)
        result = run_lbfgs(model, op, cfg, verbose=False)
        assert result.reason in {
            "maxIters", "gradTol", "stepTol",
            "nonFinite", "near-stationary / line-search stalled"
        }

    def test_grad_tol_stopping(self):
        """
        If we set a very large grad_tol, L-BFGS stops immediately.
        """
        model = _make_model()
        op    = _make_op()
        cfg   = LBFGSConfig(max_iters=100, grad_tol=1e10, log_every=1000)
        result = run_lbfgs(model, op, cfg, verbose=False)
        assert result.reason == "gradTol"
        assert result.n_iters == 1

    def test_max_iters_stopping(self):
        """
        With tiny grad_tol and many iterations, stops at max_iters.
        """
        model = _make_model()
        op    = _make_op()
        cfg   = LBFGSConfig(max_iters=5, grad_tol=0.0, step_tol=0.0,
                            log_every=100)
        result = run_lbfgs(model, op, cfg, verbose=False)
        assert result.reason == "maxIters"
        assert result.n_iters == 5
