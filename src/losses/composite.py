# NOTE FOR THIS PUBLIC RELEASE: the optional 'gated' stability backend was
# removed. It belonged to a separate, unpublished project and no result in this
# paper used it: every reported run uses stab_backend='mmd'. The MMD path below
# is unchanged. See RELEASE_NOTES.md for the hash of the original.
"""Composite loss.

Assembles the four locked primary terms into the total objective

    L = L_pred + lambda_stab * L_stab + lambda_rel * L_rel + lambda_sep * L_sep

with per-term enable flags. The trainer is responsible for populating the
inputs; this module only does the weighted sum and reports the components
so they can be logged per epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from .prediction import prediction_mse
from .stability_mmd import stability_mmd
from .relation_softspearman import relation_softspearman
from .separation_covpenalty import separation_covpenalty


_ALLOWED_STAB_BACKENDS = {"mmd"}


@dataclass
class LossWeights:
    lambda_stab: float = 0.0
    lambda_rel:  float = 0.0
    lambda_sep:  float = 0.0


class CompositeLoss:
    """Stateless composer. The actual computation is in the primary modules.

    Only the terms with non-zero weight (or explicit `enabled=True`) are
    evaluated; other terms are skipped and reported as 0.0 in the breakdown.
    """

    def __init__(
        self,
        weights: LossWeights,
        *,
        stab_enabled: bool = False,
        rel_enabled: bool = False,
        sep_enabled: bool = False,
        mmd_bandwidth_sq: Optional[float] = None,
        softspearman_tau: float = 0.1,
        stab_backend: str = "mmd",
        pred_loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ):
        if stab_backend not in _ALLOWED_STAB_BACKENDS:
            raise ValueError(
                f"stab_backend must be one of {sorted(_ALLOWED_STAB_BACKENDS)}, "
                f"got {stab_backend!r}"
            )
        self.weights = weights
        self.stab_enabled = stab_enabled
        self.rel_enabled = rel_enabled
        self.sep_enabled = sep_enabled
        self.mmd_bandwidth_sq = mmd_bandwidth_sq
        self.softspearman_tau = softspearman_tau
        self.stab_backend = stab_backend
        # Prediction loss is MSE by default (regression-first path); classifiers
        # can swap in prediction_bce for logit-based binary cross-entropy.
        self.pred_loss_fn: Callable[[Tensor, Tensor], Tensor] = (
            pred_loss_fn if pred_loss_fn is not None else prediction_mse
        )

    def __call__(
        self,
        *,
        y_pred: Tensor,
        y_true: Tensor,
        context_feats: Optional[Sequence[Tensor]] = None,
        context_ids: Optional[Sequence[int]] = None,
        rel_pairs: Optional[Sequence[Tuple[Tensor, Tensor, float]]] = None,
        z_s: Optional[Tensor] = None,
        z_c: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        device = y_pred.device
        l_pred = self.pred_loss_fn(y_pred, y_true)

        if self.stab_enabled and context_feats is not None and len(context_feats) >= 2:
            l_stab = stability_mmd(context_feats, bandwidth_sq=self.mmd_bandwidth_sq)
        else:
            l_stab = torch.zeros((), device=device)

        if self.rel_enabled and rel_pairs:
            l_rel = relation_softspearman(rel_pairs, tau=self.softspearman_tau)
        else:
            l_rel = torch.zeros((), device=device)

        if self.sep_enabled and z_s is not None and z_c is not None:
            l_sep = separation_covpenalty(z_s, z_c)
        else:
            l_sep = torch.zeros((), device=device)

        total = (
            l_pred
            + self.weights.lambda_stab * l_stab
            + self.weights.lambda_rel * l_rel
            + self.weights.lambda_sep * l_sep
        )
        components: Dict[str, Any] = {
            "L_pred": float(l_pred.detach()),
            "L_stab": float(l_stab.detach()),
            "L_rel":  float(l_rel.detach()),
            "L_sep":  float(l_sep.detach()),
            "L_total": float(total.detach()),
        }
        return total, components
