"""Relation metrics (SPEC §7, group 4).

All operate on already-computed predictions (or feature-importance vectors)
and matched-pair index sets. Pure NumPy.

Metrics:
  - rank_consistency            : Spearman of predictions over matched pairs
                                  across all context pairs (mean, tier-weighted).
  - pairwise_order_agreement    : fraction of matched-pair orderings preserved
                                  across two contexts.
  - feature_importance_stability: Spearman of per-context feature importance
                                  rankings, averaged across context pairs.
  - stable_predictor_recovery   : |cos(w_learned, w_0)|  (synthetic only).
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np

from .predictive import spearman, _ranks


def rank_consistency(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray, float]],
) -> float:
    """Mean Spearman correlation across matched context pairs.

    `pairs` is a list of (pred_c, pred_c', weight) triples. Weights are used
    as tier weights (Tier 1 / 2 / 3). Higher = more consistent ranking.
    Returns NaN if no valid pairs.
    """
    num = 0.0
    den = 0.0
    for a, b, w in pairs:
        s = spearman(a, b)
        if np.isnan(s):
            continue
        num += float(w) * s
        den += float(w)
    return num / den if den > 0 else float("nan")


def rank_consistency_by_tier(
    pairs_by_tier: Mapping[int, Sequence[Tuple[np.ndarray, np.ndarray, float]]],
) -> dict:
    """Break down rank-consistency by tier (1, 2, 3) as required by SPEC §4
    fallback-tier logging. Input: {tier -> list of (a, b, w)}.

    Returns dict with keys 'tier_1', 'tier_2', 'tier_3', 'combined'.
    """
    out = {}
    all_pairs: list = []
    for tier in (1, 2, 3):
        tier_pairs = pairs_by_tier.get(tier, [])
        out[f"tier_{tier}"] = rank_consistency(tier_pairs) if tier_pairs else float("nan")
        all_pairs.extend(tier_pairs)
    out["combined"] = rank_consistency(all_pairs) if all_pairs else float("nan")
    return out


def pairwise_order_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of i<j pairs with sign(a_i - a_j) == sign(b_i - b_j).

    Ignores tied pairs in both (neither agreement nor disagreement counted).
    Related to Kendall's tau but simpler and more interpretable.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = a.shape[0]
    if n < 2:
        return float("nan")
    # Pairwise sign matrices (upper triangle)
    da = np.sign(a[:, None] - a[None, :])
    db = np.sign(b[:, None] - b[None, :])
    iu = np.triu_indices(n, k=1)
    sa, sb = da[iu], db[iu]
    # Exclude ties in either
    mask = (sa != 0) & (sb != 0)
    if mask.sum() == 0:
        return float("nan")
    return float((sa[mask] == sb[mask]).mean())


def feature_importance_stability(
    importances_by_context: Mapping[int, np.ndarray],
) -> float:
    """Mean Spearman across all unordered pairs of per-context feature-importance
    vectors. Each vector has shape (d_features,); higher = more stable ranking
    of feature importance across contexts.

    A common choice for 'importance' is |corr(feature_j, y)| per context; the
    caller computes this upstream and passes the result.
    """
    ctxs = list(importances_by_context.keys())
    if len(ctxs) < 2:
        return float("nan")
    vals = []
    for i in range(len(ctxs)):
        for j in range(i + 1, len(ctxs)):
            s = spearman(importances_by_context[ctxs[i]], importances_by_context[ctxs[j]])
            if not np.isnan(s):
                vals.append(s)
    return float(np.mean(vals)) if vals else float("nan")


def per_context_feature_importance_abscorr(
    X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """|Pearson(feature_j, y)| per feature. Helper for callers who don't
    bring their own importance scores.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    num = np.abs(Xc.T @ yc)
    denom = np.sqrt((Xc * Xc).sum(axis=0) * (yc @ yc) + 1e-12)
    return np.where(denom > 0, num / denom, 0.0)


def stable_predictor_recovery(w_learned: np.ndarray, w_0: np.ndarray) -> float:
    """|cos(w_learned, w_0)| in [0, 1]. 1.0 = perfectly recovered direction."""
    a = np.asarray(w_learned, dtype=float).reshape(-1)
    b = np.asarray(w_0, dtype=float).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(abs((a @ b) / (na * nb)))
