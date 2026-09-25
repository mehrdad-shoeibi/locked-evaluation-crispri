"""Prediction loss — MSE (locked primary per SPEC §6).

`prediction_mse` is the regression-path default.

Phase 5.5b adds `prediction_bce` for binary classification tasks.
The composite loss selects one or the other via its
optional `pred_loss_fn` parameter, so the trainers, gates, and
stability paths are untouched.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def prediction_mse(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Mean-squared error.

    Accepts (N,) or (N, 1) on either side; both are reshaped to (N,).
    Returns a 0-dim tensor.
    """
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    return torch.mean((y_pred - y_true) ** 2)


def prediction_bce(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Binary cross-entropy from raw logits (numerically stable).

    `y_pred` is treated as an unnormalized logit (same linear head the
    regression path uses — no sigmoid inside the model). `y_true` is a
    0/1 float target. Both get flattened to (N,). Used by binary
    classification pipelines; not used by any result in this paper.
    """
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1).float()
    return F.binary_cross_entropy_with_logits(y_pred, y_true)
