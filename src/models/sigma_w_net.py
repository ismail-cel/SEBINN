"""
Neural network for the smooth density remainder sigma_w.

MATLAB reference
----------------
build_pinn_network  lines 573-587

    layers = [
        featureInputLayer(2)
        fullyConnectedLayer(80)  tanhLayer
        fullyConnectedLayer(80)  tanhLayer
        fullyConnectedLayer(80)  tanhLayer
        fullyConnectedLayer(80)  tanhLayer
        fullyConnectedLayer(1)
    ]

This is a direct port to PyTorch nn.Sequential.  Architecture is
identical: 4 hidden layers of width 80, tanh activations, scalar output.

Design notes
------------
- Input is a 2D boundary point y = (y1, y2).
- Output is a scalar approximating sigma_w(y).
- The network takes batched input of shape (N, 2) and returns (N, 1).
  Callers squeeze the last dimension when needed.
- Weight initialisation uses PyTorch defaults (Kaiming uniform for Linear),
  which differ from MATLAB's dlnetwork defaults but are standard practice.
- The network is not aware of the singular correction; it only learns the
  smooth remainder sigma_w.  The full density sigma = sigma_w + gamma*sigma_s
  is assembled in SEBINNModel (sebinn.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_sigma_w_network(
    hidden_width: int = 80,
    n_hidden: int = 4,
) -> nn.Sequential:
    """
    Build the smooth-density network.

    MATLAB: build_pinn_network (lines 573-587).
    Architecture: input(2) -> [Linear(w) -> Tanh] x n_hidden -> Linear(1).

    Parameters
    ----------
    hidden_width : int
        Width of each hidden layer.  MATLAB: 80.
    n_hidden : int
        Number of hidden layers.  MATLAB: 4.

    Returns
    -------
    net : nn.Sequential
        Maps (N, 2) -> (N, 1).
    """
    layers: list[nn.Module] = []
    in_features = 2
    for _ in range(n_hidden):
        layers.append(nn.Linear(in_features, hidden_width))
        layers.append(nn.Tanh())
        in_features = hidden_width
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)
