"""Prediction heads.

Regression-first per SPEC §4 / §5. A linear head is the default;
we keep the API a `nn.Module` so an MLP head is a drop-in replacement
if future ablations require one.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearRegressionHead(nn.Module):
    """z -> y_pred (N,)."""

    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.linear(z)
        return y.squeeze(-1) if self.out_dim == 1 else y
