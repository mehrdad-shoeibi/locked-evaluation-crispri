"""Predictive metrics (SPEC §7, group 1).

Regression-first (all synthetic + real targets are continuous per SPEC §4).
Inputs are 1D NumPy arrays of matched length.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _flat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination.

    Returns NaN when `y_true` is constant (undefined).
    """
    yt, yp = _flat(y_true), _flat(y_pred)
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    ss_res = float(np.sum((yt - yp) ** 2))
    return 1.0 - ss_res / ss_tot


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation. NaN if either vector has zero variance."""
    a, b = _flat(y_true), _flat(y_pred)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    if denom == 0.0:
        return float("nan")
    return float((a @ b) / denom)


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks (1-indexed) — matches scipy.stats.rankdata(..., 'average')."""
    x = _flat(x)
    order = np.argsort(x, kind="mergesort")
    n = x.shape[0]
    ranks = np.empty(n, dtype=float)
    # Find groups of ties in sorted order; assign average rank
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0   # 1-indexed average
        ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation with tie-correct average ranking."""
    return pearson(_ranks(y_true), _ranks(y_pred))


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int | float) -> float:
    """Normalized Discounted Cumulative Gain at top-k (by predicted score).

    `k` may be an int (absolute) or a float in (0, 1] (fraction of N).
    Gain is the raw `y_true` value; items selected by descending `y_pred`.
    Normalized by the ideal top-k ordering of `y_true`.

    If `y_true` is everywhere <= 0 we shift by its min so gains are non-negative
    (NDCG is ill-defined on negative relevance).
    """
    yt, yp = _flat(y_true), _flat(y_pred)
    n = yt.shape[0]
    if n == 0:
        return float("nan")
    if isinstance(k, float) and 0 < k <= 1:
        k_abs = max(1, int(round(k * n)))
    else:
        k_abs = min(int(k), n)
    shift = max(0.0, -float(yt.min()))
    yt_ = yt + shift
    discounts = 1.0 / np.log2(np.arange(2, 2 + k_abs))
    # DCG: top k by yp
    idx = np.argsort(-yp, kind="mergesort")[:k_abs]
    dcg = float((yt_[idx] * discounts).sum())
    # IDCG: top k by yt
    idx_ideal = np.argsort(-yt_, kind="mergesort")[:k_abs]
    idcg = float((yt_[idx_ideal] * discounts).sum())
    if idcg == 0.0:
        return float("nan")
    return dcg / idcg


def predictive_bundle(y_true: np.ndarray, y_pred: np.ndarray,
                      ndcg_k: int | float = 0.1) -> dict:
    """Convenience: all group-1 metrics in one call."""
    return {
        "R2": r2(y_true, y_pred),
        "Pearson": pearson(y_true, y_pred),
        "Spearman": spearman(y_true, y_pred),
        "NDCG": ndcg_at_k(y_true, y_pred, ndcg_k),
    }
