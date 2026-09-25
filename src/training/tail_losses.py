"""Phase 8C — tail-weighted MSE prediction losses.

Both factories return a callable with the standard
`pred_loss_fn(y_pred, y_true) -> Tensor` signature so they slot into
the existing `CompositeLoss` without trainer changes.

The weights depend on the *training* y distribution (frozen at
construction time) so that train and val/test rows experience the
same weighting scheme — no leakage from val labels into training.

  rank-weighted:
      w(y) = 1 + gamma * |F_train(y) - 0.5|
      where F_train is the empirical CDF of train y values
      (computed once at construction).

  quintile-weighted:
      w(y) = q_weights[quintile(y; train_edges)]
      where train_edges are the four 0.2/0.4/0.6/0.8 quantiles of
      train y. Defaults: q_weights = (2, 1, 1, 1, 2) — extra weight
      on Q1 and Q5; an alternative (3, 1.5, 1, 1.5, 3) emphasizes
      tails more aggressively while still soft-weighting the inner
      quintiles.

Numerics: weighted-MSE returns
    (sum_i w_i * (y_pred_i - y_true_i)^2) / sum_i w_i
i.e. weighted *mean* (not weighted sum), so loss magnitude stays
comparable to plain MSE and existing learning-rate / regularizer
choices remain reasonable.

Implementation note: torch.searchsorted is used for rank lookup —
O(log n) per element on the cached sorted train tensor; effectively
free per batch.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor


def _cache_train_y(train_y: Sequence[float], device: str = "cpu") -> Tensor:
    """Return a sorted-ascending 1-D tensor of train y values."""
    arr = np.asarray(train_y, dtype=np.float32).reshape(-1)
    arr = np.sort(arr)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def _weighted_mse(y_pred: Tensor, y_true: Tensor, w: Tensor) -> Tensor:
    """Sum-normalized weighted MSE: Σ w_i (y_pred_i - y_true_i)^2 / Σ w_i."""
    diff_sq = (y_pred - y_true) ** 2
    num = (w * diff_sq).sum()
    den = w.sum()
    # nan_to_num guard: if the batch happens to have zero total weight
    # (degenerate; shouldn't happen with the factories below), fall back
    # to plain mean.
    return torch.where(den > 0, num / den, diff_sq.mean())


def make_rank_tail_weighted_mse(
    train_y: Sequence[float],
    gamma: float = 1.0,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Rank-based weighting: w(y) = 1 + gamma * |F_train(y) - 0.5|.

    For y at the median of train, w = 1.
    For y at the train min/max, w = 1 + gamma/2 (so gamma=4 gives
    roughly 1 → 3 weight at the tails).
    """
    if gamma < 0:
        raise ValueError(f"gamma must be ≥ 0, got {gamma}")
    sorted_y_cpu = _cache_train_y(train_y, device="cpu")
    n = sorted_y_cpu.numel()

    def fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        if sorted_y_cpu.device != y_true.device:
            srt = sorted_y_cpu.to(y_true.device)
        else:
            srt = sorted_y_cpu
        # right=True ensures ranks are in (0, 1]; right=False would push
        # exact-equal-min values to rank 0. Either is fine; we use the
        # right-search convention so rank ∈ [1, n] / n ∈ (0, 1].
        ranks = torch.searchsorted(srt, y_true.detach(), right=True).float() / float(n)
        ranks = torch.clamp(ranks, 1.0 / n, 1.0)
        w = 1.0 + gamma * torch.abs(ranks - 0.5)
        return _weighted_mse(y_pred, y_true, w)

    fn.gamma = gamma  # type: ignore[attr-defined]
    fn.scheme = "rank"  # type: ignore[attr-defined]
    return fn


_DEFAULT_QUINTILE_WEIGHTS = (2.0, 1.0, 1.0, 1.0, 2.0)


def make_quintile_tail_weighted_mse(
    train_y: Sequence[float],
    q_weights: Sequence[float] = _DEFAULT_QUINTILE_WEIGHTS,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Bin y into 5 quintiles by train CDF; weight per quintile."""
    if len(q_weights) != 5:
        raise ValueError(f"q_weights must have 5 entries, got {len(q_weights)}")
    if any(float(w) < 0 for w in q_weights):
        raise ValueError("q_weights must all be ≥ 0")
    train_y_arr = np.asarray(train_y, dtype=np.float32).reshape(-1)
    edges = np.quantile(train_y_arr, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    edges_cpu = torch.as_tensor(edges, dtype=torch.float32, device="cpu")
    weights_cpu = torch.as_tensor(q_weights, dtype=torch.float32, device="cpu")

    def fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        if edges_cpu.device != y_true.device:
            edges_d = edges_cpu.to(y_true.device)
            weights_d = weights_cpu.to(y_true.device)
        else:
            edges_d = edges_cpu
            weights_d = weights_cpu
        # bucketize returns 0..len(edges) for each value (so 0..4 here)
        idx = torch.bucketize(y_true.detach(), edges_d)
        w = weights_d[idx]
        return _weighted_mse(y_pred, y_true, w)

    fn.q_weights = tuple(float(w) for w in q_weights)  # type: ignore[attr-defined]
    fn.scheme = "quintile"  # type: ignore[attr-defined]
    return fn


__all__ = [
    "make_rank_tail_weighted_mse",
    "make_quintile_tail_weighted_mse",
]
