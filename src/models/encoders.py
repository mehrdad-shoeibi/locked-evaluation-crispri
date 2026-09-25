"""Encoders.

MLPEncoder:        3-layer (configurable) MLP with LayerNorm + ReLU.
                   Output dim = hidden_dim of the final layer.

FactorizedEncoder: Shared trunk followed by two heads producing (z_s, z_c).
                   Used by DAFR per SPEC §5.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _build_mlp_trunk(
    input_dim: int,
    hidden_dim: int,
    n_layers: int,
    activation: str = "relu",
    dropout: float = 0.0,
    layernorm: bool = True,
) -> nn.Sequential:
    """n_layers Linear blocks each followed by (optional LN) -> activation -> (optional dropout).

    Output dim is `hidden_dim`.
    """
    act_cls = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation.lower()]
    layers: list[nn.Module] = []
    in_d = input_dim
    for i in range(n_layers):
        layers.append(nn.Linear(in_d, hidden_dim))
        if layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_d = hidden_dim
    return nn.Sequential(*layers)


class MLPEncoder(nn.Module):
    """Standard encoder f(x) -> z  with output dimension = `hidden_dim`."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
        layernorm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = _build_mlp_trunk(
            input_dim, hidden_dim, n_layers,
            activation=activation, dropout=dropout, layernorm=layernorm,
        )

    @property
    def out_dim(self) -> int:
        return self.hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FactorizedEncoder(nn.Module):
    """Shared trunk + two linear projections producing (z_s, z_c).

    The split is intentionally *after* the nonlinear trunk so the trunk can
    learn a shared representation, while the factorization is a cheap linear
    decomposition trained with L_sep.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        z_s_dim: int = 128,
        z_c_dim: int = 128,
        activation: str = "relu",
        dropout: float = 0.0,
        layernorm: bool = True,
    ):
        super().__init__()
        self.trunk = _build_mlp_trunk(
            input_dim, hidden_dim, n_layers,
            activation=activation, dropout=dropout, layernorm=layernorm,
        )
        self.to_z_s = nn.Linear(hidden_dim, z_s_dim)
        self.to_z_c = nn.Linear(hidden_dim, z_c_dim)
        self.z_s_dim = z_s_dim
        self.z_c_dim = z_c_dim

    @property
    def out_dim(self) -> int:
        return self.z_s_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        z_s = self.to_z_s(h)
        z_c = self.to_z_c(h)
        return z_s, z_c
