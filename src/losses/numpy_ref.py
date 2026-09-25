"""Framework-free NumPy reference implementations of the four loss terms.

These are NOT the production training path — they are used by unit tests to
verify directional behavior and, when torch is available, to cross-check the
torch primary backends produce identical numeric values.

All functions are pure, take NumPy arrays, and return scalar floats.

Locked primaries (SPEC §6):
  - prediction: MSE
  - stability:  MMD^2 with Gaussian kernel, median-heuristic bandwidth
  - relation:   SoftSpearman loss (soft-ranking Spearman)
  - separation: normalized cross-covariance Frobenius penalty
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def mse_np(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    return float(np.mean((y_pred - y_true) ** 2))


# ---------------------------------------------------------------------------
# Stability: MMD^2 with Gaussian kernel (median heuristic)
# ---------------------------------------------------------------------------
def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # (n, d), (m, d) -> (n, m) squared distances
    aa = (a * a).sum(axis=1, keepdims=True)         # (n, 1)
    bb = (b * b).sum(axis=1, keepdims=True).T       # (1, m)
    return np.maximum(aa + bb - 2.0 * a @ b.T, 0.0)


def _median_bandwidth(x: np.ndarray, y: np.ndarray) -> float:
    """Median heuristic: sigma^2 = median of pairwise squared distances on [x; y]."""
    z = np.concatenate([x, y], axis=0)
    n = z.shape[0]
    if n < 2:
        return 1.0
    d2 = _pairwise_sq_dists(z, z)
    iu = np.triu_indices(n, k=1)
    med = float(np.median(d2[iu]))
    return max(med, 1e-12)


def mmd2_gaussian_np(x: np.ndarray, y: np.ndarray, bandwidth_sq: float | None = None) -> float:
    """Biased MMD^2 estimator with a Gaussian kernel k(u,v)=exp(-||u-v||^2/(2 σ²))."""
    if bandwidth_sq is None:
        bandwidth_sq = _median_bandwidth(x, y)
    denom = 2.0 * bandwidth_sq
    Kxx = np.exp(-_pairwise_sq_dists(x, x) / denom)
    Kyy = np.exp(-_pairwise_sq_dists(y, y) / denom)
    Kxy = np.exp(-_pairwise_sq_dists(x, y) / denom)
    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())


def mmd_np(
    context_feats: Sequence[np.ndarray],
    bandwidth_sq: float | None = None,
) -> float:
    """Average MMD^2 over all unordered pairs of context feature matrices.

    Each element of `context_feats` is an (n_c, d) array of representations
    for one context. Returns the mean of MMD^2 over the K*(K-1)/2 pairs.
    If K < 2, returns 0.
    """
    K = len(context_feats)
    if K < 2:
        return 0.0
    total = 0.0
    n_pairs = 0
    for i in range(K):
        for j in range(i + 1, K):
            total += mmd2_gaussian_np(
                context_feats[i], context_feats[j], bandwidth_sq=bandwidth_sq
            )
            n_pairs += 1
    return total / n_pairs


# ---------------------------------------------------------------------------
# Relation: SoftSpearman
# ---------------------------------------------------------------------------
def _soft_rank(v: np.ndarray, tau: float = 0.1) -> np.ndarray:
    """Soft ranks in [0, n-1] via sigmoid-smoothed pairwise comparisons.

    soft_rank_i = sum_{j != i} sigmoid( (v_i - v_j) / tau )

    As tau -> 0, this converges to the hard rank (0-indexed).
    Ties give exactly rank 0.5 on the pair. Differentiable in v.
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    diff = v[:, None] - v[None, :]
    s = 1.0 / (1.0 + np.exp(-diff / max(tau, 1e-12)))
    np.fill_diagonal(s, 0.0)
    return s.sum(axis=1)


def _soft_spearman_corr(a: np.ndarray, b: np.ndarray, tau: float = 0.1) -> float:
    """Pearson correlation on soft ranks = soft Spearman in [-1, 1]."""
    ra = _soft_rank(a, tau)
    rb = _soft_rank(b, tau)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    if denom < 1e-12:
        return 0.0
    return float((ra @ rb) / denom)


def softspearman_loss_np(
    pairs: Sequence[tuple[np.ndarray, np.ndarray, float]],
    tau: float = 0.1,
) -> float:
    """Weighted SoftSpearman loss across matched context pairs.

    `pairs` is a list of (pred_c, pred_c', weight) triples, where pred_c and
    pred_c' are 1D arrays of predictions on a *matched* pair set and weight
    is a tier weight in (0, 1]. Returns

        L_rel = sum_p w_p * (1 - SoftSpearman_p) / sum_p w_p

    which is in [0, 2] (with 1 being the 'uncorrelated' baseline).
    """
    if not pairs:
        return 0.0
    num = 0.0
    den = 0.0
    for a, b, w in pairs:
        corr = _soft_spearman_corr(np.asarray(a), np.asarray(b), tau=tau)
        num += w * (1.0 - corr)
        den += w
    return num / max(den, 1e-12)


# ---------------------------------------------------------------------------
# Separation: normalized cross-covariance Frobenius penalty
# ---------------------------------------------------------------------------
def _cov(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Sample cross-covariance of rows of A and rows of B.

    A: (n, p), B: (n, q)  ->  (p, q)
    """
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    n = A.shape[0]
    return (Ac.T @ Bc) / max(n - 1, 1)


def covpenalty_np(z_s: np.ndarray, z_c: np.ndarray, eps: float = 1e-8) -> float:
    """Normalized cross-covariance penalty.

        L_sep = ||Cov(z_s, z_c)||_F^2  /  ( ||Cov(z_s)||_F * ||Cov(z_c)||_F + eps )

    Returns 0 when z_s and z_c are (sample-)uncorrelated, and grows without
    bound as they become perfectly correlated relative to their own variance.
    """
    C_sc = _cov(z_s, z_c)
    C_ss = _cov(z_s, z_s)
    C_cc = _cov(z_c, z_c)
    num = float(np.sum(C_sc * C_sc))                   # ||·||_F^2
    denom = float(
        np.sqrt(np.sum(C_ss * C_ss)) * np.sqrt(np.sum(C_cc * C_cc)) + eps
    )
    return num / denom
