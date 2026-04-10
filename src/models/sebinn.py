"""
SE-BINN combined model: sigma = sigma_w(y) + gamma * sigma_s(y).

MATLAB counterpart
------------------
The MATLAB BINN evaluates:
    sigma = forward(net, Yq)          # only sigma_w; no enrichment

SE-BINN replaces this with:
    sigma = sigma_w_net(Yq) + gamma * sigma_s_precomputed

The network and gamma are both trainable; sigma_s is a fixed buffer.

Mathematical role
-----------------
    sigma(y) = sigma_w(y) + gamma * sigma_s(y)

where:
  - sigma_w: nn.Sequential (4 x 80, tanh) approximates the smooth remainder
  - gamma:   GammaParameter (scalar or per-corner) learned during training
  - sigma_s: precomputed fixed ndarray from SingularEnrichment.precompute()

The split is preserved explicitly in the forward pass so that:
  - sigma_w and gamma * sigma_s can be inspected separately
  - the singular component is never mixed into network weights
  - gamma can be read off after training as a diagnostic

Design notes
------------
- sigma_s values are registered as non-trainable buffers, one per node set
  (quadrature nodes Yq, collocation nodes Xc, etc.).  Buffers move to the
  correct device automatically.
- register_sigma_s() must be called before any forward pass that uses that
  node set.
- forward(y, sigma_s_vals) takes the precomputed sigma_s tensor as an
  explicit argument so the model is stateless with respect to node sets.
  This avoids the need to store one buffer per node set inside the module
  and keeps the interface clean for the training operator.
- Parameter flattening for L-BFGS follows the MATLAB net_to_vector /
  vector_to_net pattern, using PyTorch's
  torch.nn.utils.parameters_to_vector / vector_to_parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .sigma_w_net import build_sigma_w_network
from .gamma import GammaParameter


class SEBINNModel(nn.Module):
    """
    Combined SE-BINN model.

    sigma(y) = sigma_w_net(y) + gamma * sigma_s(y)

    Parameters
    ----------
    hidden_width : int
        Width of hidden layers in sigma_w_net.  Default 80.
    n_hidden : int
        Number of hidden layers.  Default 4.
    n_gamma : int
        Number of gamma parameters (1 or n_singular_corners).  Default 1.
    gamma_init : float
        Initial value for gamma.  Default 0.0.
    dtype : torch.dtype
        Floating-point precision.  Default float64 (matches BEM assembly).
    """

    def __init__(
        self,
        hidden_width: int = 80,
        n_hidden: int = 4,
        n_gamma: int = 1,
        gamma_init: float = 0.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dtype = dtype

        self.sigma_w_net = build_sigma_w_network(hidden_width, n_hidden).to(dtype)
        self.gamma_module = GammaParameter(n_gamma=n_gamma, init_value=gamma_init)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        y: torch.Tensor,
        sigma_s_vals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the full enriched density sigma at points y.

        Parameters
        ----------
        y : Tensor, shape (N, 2)
            Boundary (or interior) evaluation points.
        sigma_s_vals : Tensor, shape (N,) or (N, n_gamma)
            Precomputed sigma_s values at y.
            - shape (N,)         when n_gamma == 1 (shared gamma)
            - shape (N, n_gamma) when per-corner gammas are used

        Returns
        -------
        sigma : Tensor, shape (N,)
            Full density sigma = sigma_w + gamma * sigma_s.
        """
        sigma_w = self.sigma_w_net(y).squeeze(-1)   # (N,)
        gamma = self.gamma_module()                  # () or (n_gamma,)

        if self.gamma_module.n_gamma == 1:
            singular = gamma * sigma_s_vals          # (N,)
        else:
            # sigma_s_vals: (N, n_gamma), gamma: (n_gamma,)
            singular = (sigma_s_vals * gamma).sum(dim=-1)  # (N,)

        return sigma_w + singular

    def sigma_w(self, y: torch.Tensor) -> torch.Tensor:
        """Evaluate only the smooth part sigma_w(y). Shape (N,)."""
        return self.sigma_w_net(y).squeeze(-1)

    def singular_part(self, sigma_s_vals: torch.Tensor) -> torch.Tensor:
        """
        Evaluate only the singular part gamma * sigma_s.

        Parameters
        ----------
        sigma_s_vals : Tensor, shape (N,) or (N, n_gamma)

        Returns
        -------
        Tensor, shape (N,)
        """
        gamma = self.gamma_module()
        if self.gamma_module.n_gamma == 1:
            return gamma * sigma_s_vals
        return (sigma_s_vals * gamma).sum(dim=-1)

    # ------------------------------------------------------------------
    # Parameter utilities (for L-BFGS)
    # MATLAB: net_to_vector / vector_to_net (lines 1380-1414)
    # ------------------------------------------------------------------

    def to_vector(self) -> torch.Tensor:
        """
        Flatten all trainable parameters into a single 1-D tensor.

        MATLAB: net_to_vector — concatenates all Learnables into a column
        vector theta.  Here we use PyTorch's parameters_to_vector which
        does the same in parameter-iteration order.

        Returns
        -------
        theta : Tensor, shape (n_params,)
            Detached copy.
        """
        return parameters_to_vector(self.parameters()).detach().clone()

    def from_vector(self, theta: torch.Tensor) -> None:
        """
        Load parameters from a flat vector (in-place).

        MATLAB: vector_to_net — splits theta back into each Learnable.

        Parameters
        ----------
        theta : Tensor, shape (n_params,)
        """
        vector_to_parameters(theta, self.parameters())

    def n_params(self) -> int:
        """Total number of trainable scalar parameters."""
        return int(sum(p.numel() for p in self.parameters()))

    def gamma_value(self) -> float | list[float]:
        """Return current gamma as Python scalar(s). Useful for logging."""
        return self.gamma_module.item()
