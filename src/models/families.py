# NOTE FOR THIS PUBLIC RELEASE: the `build_iga_bundle` builder and its two
# imports were removed. They belong to a separate, unpublished project and are
# not used by any result in this paper. Nothing else in this file was changed.
# See RELEASE_NOTES.md for the hash of the original.
"""Model families (SPEC §5, locked names).

Design principle: a model is a lightweight composition of encoder + head
that exposes everything the composite loss might need. The *loss term
activation* is determined by the trainer's config, not by the model class —
so swapping DWP→SRM is just toggling the stab loss, not instantiating a
different architecture. This gives us clean controls for ablations.

All four models share MLPEncoder (SRM, SPSM) or FactorizedEncoder (DAFR);
DWP is identical to SRM/SPSM architecturally but the trainer will never
call its `context_feats` pathway.

Public output: `ModelOutput(y_pred, z, z_s, z_c)`.
    - z      : full representation (== z_s for DAFR, raw encoder output otherwise)
    - z_s, z_c: populated only for DAFR; None otherwise.

Prediction path:
    DWP/SRM/SPSM:   y = head(z)
    DAFR:           y = head(z_s)      # prediction from stable component only
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from .encoders import FactorizedEncoder, MLPEncoder
from .heads import LinearRegressionHead


@dataclass
class ModelOutput:
    y_pred: torch.Tensor
    z: torch.Tensor
    z_s: Optional[torch.Tensor] = None
    z_c: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------
class _EncoderHeadModel(nn.Module):
    """Shared base for DWP / SRM / SPSM (non-factorized)."""

    def __init__(self, encoder: MLPEncoder, head: LinearRegressionHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> ModelOutput:
        z = self.encoder(x)
        y = self.head(z)
        return ModelOutput(y_pred=y, z=z, z_s=None, z_c=None)


class DWP(_EncoderHeadModel):
    """Direct Weak Predictor. Trainer uses L_pred only."""


class SRM(_EncoderHeadModel):
    """Stable Representation Model. Trainer uses L_pred + λ_stab L_stab(z)."""


class SPSM(_EncoderHeadModel):
    """Structure-Preserving Stable Model.
    Trainer uses L_pred + λ_stab L_stab(z) + λ_rel L_rel(y_pred per ctx)."""


# ---------------------------------------------------------------------------
# DAFR
# ---------------------------------------------------------------------------
class DAFR(nn.Module):
    """Drift-Aware Factorized Representation Model.

    Architecture: FactorizedEncoder(x) -> (z_s, z_c); head(z_s) -> y_pred.
    Trainer uses L_pred + λ_stab L_stab(z_s) + λ_rel L_rel on z_s-based
    predictions + λ_sep L_sep(z_s, z_c).
    """

    def __init__(self, encoder: FactorizedEncoder, head: LinearRegressionHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> ModelOutput:
        z_s, z_c = self.encoder(x)
        y = self.head(z_s)
        return ModelOutput(y_pred=y, z=z_s, z_s=z_s, z_c=z_c)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def _mlp_encoder_from_cfg(input_dim: int, model_cfg: Mapping[str, Any]) -> MLPEncoder:
    ec = model_cfg.get("encoder", {})
    return MLPEncoder(
        input_dim=input_dim,
        hidden_dim=ec.get("hidden_dim", 256),
        n_layers=ec.get("n_layers", 3),
        activation=ec.get("activation", "relu"),
        dropout=ec.get("dropout", 0.0),
        layernorm=ec.get("layernorm", True),
    )


def _factorized_encoder_from_cfg(
    input_dim: int, model_cfg: Mapping[str, Any]
) -> FactorizedEncoder:
    ec = model_cfg.get("encoder", {})
    fz = model_cfg.get("factorization", {"z_s_dim": 128, "z_c_dim": 128})
    return FactorizedEncoder(
        input_dim=input_dim,
        hidden_dim=ec.get("hidden_dim", 256),
        n_layers=ec.get("n_layers", 3),
        z_s_dim=fz.get("z_s_dim", 128),
        z_c_dim=fz.get("z_c_dim", 128),
        activation=ec.get("activation", "relu"),
        dropout=ec.get("dropout", 0.0),
        layernorm=ec.get("layernorm", True),
    )


def _head_from_cfg(in_dim: int, model_cfg: Mapping[str, Any]) -> LinearRegressionHead:
    hc = model_cfg.get("head", {"out_dim": 1})
    return LinearRegressionHead(in_dim=in_dim, out_dim=hc.get("out_dim", 1))


def build_dwp(input_dim: int, model_cfg: Mapping[str, Any]) -> DWP:
    enc = _mlp_encoder_from_cfg(input_dim, model_cfg)
    return DWP(enc, _head_from_cfg(enc.out_dim, model_cfg))


def build_srm(input_dim: int, model_cfg: Mapping[str, Any]) -> SRM:
    enc = _mlp_encoder_from_cfg(input_dim, model_cfg)
    return SRM(enc, _head_from_cfg(enc.out_dim, model_cfg))


def build_spsm(input_dim: int, model_cfg: Mapping[str, Any]) -> SPSM:
    enc = _mlp_encoder_from_cfg(input_dim, model_cfg)
    return SPSM(enc, _head_from_cfg(enc.out_dim, model_cfg))


def build_dafr(input_dim: int, model_cfg: Mapping[str, Any]) -> DAFR:
    enc = _factorized_encoder_from_cfg(input_dim, model_cfg)
    return DAFR(enc, _head_from_cfg(enc.z_s_dim, model_cfg))


def build_model(
    name: str, input_dim: int, model_cfg: Mapping[str, Any]
) -> Union[nn.Module, Dict[str, Any]]:
    """Dispatch to a family builder.

    Returns an `nn.Module` for DWP/SRM/SPSM/DAFR.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](input_dim, model_cfg)
