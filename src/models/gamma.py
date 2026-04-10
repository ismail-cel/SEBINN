"""
Trainable singular amplitude coefficient gamma.

MATLAB counterpart
------------------
None.  gamma is the key SE-BINN addition that does not exist in the
MATLAB BINN baseline.

Mathematical role
-----------------
The density ansatz is:

    sigma(y) = sigma_w(y) + gamma * sigma_s(y)

gamma is a scalar stress-intensity-type coefficient that the optimizer
learns jointly with the network weights for sigma_w.

For a domain with multiple singular corners, there are two choices:
  - One shared gamma (per_corner=False): all corners share one amplitude.
    Appropriate for symmetric geometries such as Koch(1).
  - Per-corner gammas (per_corner=True): each corner has its own gamma_c.
    Appropriate for asymmetric domains or when corners have different angles.

Design notes
------------
- GammaParameter is a thin nn.Module so gamma appears in model.parameters()
  and is updated by the optimizer without special-casing.
- Initialisation: gamma_0 = 0.0 is the safest start (network starts as
  plain BINN; gamma is learned from residual pressure).
- The value of gamma can be read as a plain float via gamma.item() or as
  a tensor via gamma.gamma (the Parameter itself).
- For per-corner mode, gamma.gamma is shape (n_corners,); the training
  operator handles the dot product with the per-corner sigma_s array.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GammaParameter(nn.Module):
    """
    Trainable scalar (or vector) gamma for the singular enrichment.

    Attributes
    ----------
    gamma : nn.Parameter
        Shape () for single gamma, shape (n_corners,) for per-corner mode.
    n_gamma : int
        Number of gamma parameters (1 or n_corners).
    """

    def __init__(
        self,
        n_gamma: int = 1,
        init_value: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        n_gamma : int
            1 for a shared scalar gamma; >1 for per-corner gammas.
        init_value : float
            Initial value for all gamma entries.  Default 0.0 so the
            network starts as a plain BINN.
        """
        super().__init__()
        if n_gamma == 1:
            self.gamma = nn.Parameter(
                torch.tensor(init_value, dtype=torch.float64)
            )
        else:
            self.gamma = nn.Parameter(
                torch.full((n_gamma,), init_value, dtype=torch.float64)
            )
        self.n_gamma = n_gamma

    def forward(self) -> torch.Tensor:
        """Return the gamma parameter tensor."""
        return self.gamma

    def item(self) -> float | list[float]:
        """Convenience: return gamma as Python scalar(s)."""
        if self.n_gamma == 1:
            return float(self.gamma.item())
        return self.gamma.detach().tolist()

    def extra_repr(self) -> str:
        if self.n_gamma == 1:
            return f"gamma={self.gamma.item():.6f}"
        vals = ", ".join(f"{v:.4f}" for v in self.gamma.detach().tolist())
        return f"gamma=[{vals}]"
