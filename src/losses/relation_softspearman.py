"""Relation loss — SoftSpearman on tier-weighted matched pairs.

Locked primary per SPEC §6. Approximates Spearman correlation on predictions
via a sigmoid-smoothed pairwise-comparison rank (converges to the hard rank
as tau -> 0) and is differentiable in the predictions.

API
---
    relation_softspearman(
        pairs: list[(pred_c: Tensor, pred_c': Tensor, weight: float)],
        tau: float = 0.1,
    ) -> Tensor (0-dim)

    L_rel = sum_p w_p * (1 - SoftSpearman_p) / sum_p w_p

A list of matched context pairs is passed in rather than a ragged tensor so
the caller (trainer / pair-builder) has full control over tier weights and
per-pair sizes.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import Tensor


def _soft_rank(v: Tensor, tau: float = 0.1) -> Tensor:
    """Soft rank in [0, n-1]. Differentiable in v."""
    v = v.reshape(-1)
    diff = v.unsqueeze(1) - v.unsqueeze(0)              # (n, n)
    s = torch.sigmoid(diff / max(tau, 1e-12))
    n = v.shape[0]
    eye_mask = torch.eye(n, device=v.device, dtype=torch.bool)
    s = s.masked_fill(eye_mask, 0.0)
    return s.sum(dim=1)


def _soft_spearman(a: Tensor, b: Tensor, tau: float = 0.1) -> Tensor:
    ra = _soft_rank(a, tau=tau)
    rb = _soft_rank(b, tau=tau)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = torch.sqrt((ra * ra).sum() * (rb * rb).sum()).clamp_min(1e-12)
    return (ra * rb).sum() / denom


def relation_softspearman(
    pairs: Sequence[Tuple[Tensor, Tensor, float]],
    tau: float = 0.1,
) -> Tensor:
    """Weighted (1 - SoftSpearman) across matched pairs.

    Each pair is (pred_c, pred_c', weight). Predictions must be same-length
    1D (the caller aligns them via the matched-pair index set).
    """
    if not pairs:
        return torch.zeros(())
    device = pairs[0][0].device
    num = torch.zeros((), device=device)
    den = 0.0
    for a, b, w in pairs:
        corr = _soft_spearman(a, b, tau=tau)
        num = num + float(w) * (1.0 - corr)
        den += float(w)
    return num / max(den, 1e-12)
