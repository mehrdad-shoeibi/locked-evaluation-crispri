"""Representation metrics (SPEC §7, group 3).

All metrics are pure NumPy and operate on feature arrays `Z` and labels.
Shared helpers import from the loss NumPy reference to keep the math
consistent (MMD definition must match the training-time loss).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..losses.numpy_ref import (
    _median_bandwidth,
    _pairwise_sq_dists,
    mmd2_gaussian_np,
    mmd_np,
)


# ---------------------------------------------------------------------------
# MMD — re-export (same math as training-time stability loss)
# ---------------------------------------------------------------------------
def mmd2_gaussian(x: np.ndarray, y: np.ndarray, bandwidth_sq: float | None = None) -> float:
    return mmd2_gaussian_np(x, y, bandwidth_sq=bandwidth_sq)


def mean_mmd_across_contexts(
    context_feats: Sequence[np.ndarray],
    bandwidth_sq: float | None = None,
) -> float:
    return mmd_np(context_feats, bandwidth_sq=bandwidth_sq)


# ---------------------------------------------------------------------------
# CKA
# ---------------------------------------------------------------------------
def _center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between feature matrices of aligned samples.

    X: (n, p), Y: (n, q).  Linear CKA = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F).
    Invariant to orthogonal transforms and isotropic scaling. Returns NaN if
    either feature set has zero variance.
    """
    Xc = _center(X)
    Yc = _center(Y)
    num = float(np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2)
    denom = (
        float(np.linalg.norm(Xc.T @ Xc, ord="fro"))
        * float(np.linalg.norm(Yc.T @ Yc, ord="fro"))
    )
    if denom == 0.0:
        return float("nan")
    return num / denom


def _rbf_gram(X: np.ndarray, bandwidth_sq: float | None = None) -> np.ndarray:
    if bandwidth_sq is None:
        bandwidth_sq = _median_bandwidth(X, X)
    d2 = _pairwise_sq_dists(X, X)
    return np.exp(-d2 / (2.0 * bandwidth_sq))


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Biased HSIC with Gram matrices K, L on the same n samples."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    return float((Kc * L).sum() / max(n - 1, 1) ** 2)


def rbf_cka(X: np.ndarray, Y: np.ndarray, bandwidth_sq: float | None = None) -> float:
    """RBF CKA via HSIC. Uses separate median-heuristic bandwidths for X, Y
    unless a single `bandwidth_sq` is provided (applied to both)."""
    Kx = _rbf_gram(X, bandwidth_sq)
    Ky = _rbf_gram(Y, bandwidth_sq)
    num = _hsic(Kx, Ky)
    denom = np.sqrt(max(_hsic(Kx, Kx), 0.0) * max(_hsic(Ky, Ky), 0.0))
    if denom == 0.0:
        return float("nan")
    return num / denom


# ---------------------------------------------------------------------------
# Linear probe of latent `u` from representation `z_s` (synthetic only)
# ---------------------------------------------------------------------------
def linear_probe_r2(
    Z: np.ndarray,              # (n, d_z)
    U: np.ndarray,              # (n, d_u)
    ridge: float = 1e-3,
) -> float:
    """Ridge-regress each coordinate of `U` from `Z`; return mean R^2 across
    coordinates. Evaluated in-sample (no held-out split) — this is a
    capacity / recoverability probe, not a generalization metric.
    """
    n = Z.shape[0]
    Zc = np.hstack([Z, np.ones((n, 1))])        # intercept
    A = Zc.T @ Zc + ridge * np.eye(Zc.shape[1])
    B = Zc.T @ U
    W = np.linalg.solve(A, B)
    U_hat = Zc @ W
    ss_res = ((U - U_hat) ** 2).sum(axis=0)
    ss_tot = ((U - U.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2s = np.where(ss_tot > 0, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)
    return float(np.nanmean(r2s))


def linear_probe_stable_predictor_weights(
    Z: np.ndarray, y: np.ndarray, ridge: float = 1e-3,
) -> np.ndarray:
    """Return the ridge-regressed linear weights mapping Z -> y (no intercept
    contribution on the weights; intercept is dropped).

    Used downstream for StablePredictorRecovery (cosine vs. w_0).
    """
    n, d = Z.shape
    Zc = Z - Z.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    A = Zc.T @ Zc + ridge * np.eye(d)
    w = np.linalg.solve(A, Zc.T @ yc)
    return w


# ---------------------------------------------------------------------------
# Context separability: linear classifier Acc(c | z_s)
# ---------------------------------------------------------------------------
def _multiclass_ridge_fit(Z: np.ndarray, y: np.ndarray, n_classes: int,
                          ridge: float = 1.0) -> np.ndarray:
    """One-vs-rest ridge-regressed linear classifier in closed form.

    Targets are +1 for in-class, -1 for others. Returns weight matrix W
    shape (d+1, n_classes) (last row = intercept).
    """
    n, d = Z.shape
    Zc = np.hstack([Z, np.ones((n, 1))])
    Y = -np.ones((n, n_classes))
    Y[np.arange(n), y] = 1.0
    A = Zc.T @ Zc + ridge * np.eye(d + 1)
    return np.linalg.solve(A, Zc.T @ Y)


def context_separability(
    Z: np.ndarray,              # (n, d_z)
    c: np.ndarray,              # (n,) int context labels in [0, K)
    ridge: float = 1.0,
    n_splits: int = 5,
    rng: np.random.Generator | None = None,
) -> float:
    """Cross-validated multi-class classification accuracy on (Z -> c).

    Uses a closed-form multiclass ridge classifier for determinism and
    reproducibility (no SGD). Low accuracy ≈ z is context-invariant; high
    accuracy ≈ z carries context information.

    Returns mean accuracy across stratified folds. Chance level is 1/K.
    """
    rng = rng or np.random.default_rng(0)
    c = np.asarray(c, dtype=int)
    n = Z.shape[0]
    classes = np.unique(c)
    K = classes.size
    if K < 2:
        return float("nan")
    # Remap contexts to 0..K-1
    remap = {cl: i for i, cl in enumerate(classes)}
    y = np.array([remap[v] for v in c])

    # Stratified K-fold
    folds = [[] for _ in range(n_splits)]
    for cls in range(K):
        idx_cls = np.where(y == cls)[0]
        rng.shuffle(idx_cls)
        chunks = np.array_split(idx_cls, n_splits)
        for i, ch in enumerate(chunks):
            folds[i].extend(ch.tolist())

    accs = []
    for i in range(n_splits):
        test_idx = np.array(folds[i], dtype=int)
        train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=False)
        W = _multiclass_ridge_fit(Z[train_idx], y[train_idx], K, ridge=ridge)
        Zt = np.hstack([Z[test_idx], np.ones((len(test_idx), 1))])
        pred = np.argmax(Zt @ W, axis=1)
        accs.append(float((pred == y[test_idx]).mean()))
    return float(np.mean(accs))


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------
def representation_bundle(
    context_feats: Sequence[np.ndarray],
    *,
    U_matched: np.ndarray | None = None,    # concatenated (n_total, d_u) for linear probe
    Z_matched: np.ndarray | None = None,
    context_labels: np.ndarray | None = None,
) -> dict:
    out = {"mean_MMD2": mean_mmd_across_contexts(context_feats)}
    # CKA: average pairwise linear CKA over contexts when shapes match
    # (falls back to NaN if contexts have different sample counts).
    try:
        K = len(context_feats)
        vals = []
        for i in range(K):
            for j in range(i + 1, K):
                if context_feats[i].shape[0] == context_feats[j].shape[0]:
                    vals.append(linear_cka(context_feats[i], context_feats[j]))
        out["mean_linear_CKA"] = float(np.mean(vals)) if vals else float("nan")
    except Exception:
        out["mean_linear_CKA"] = float("nan")

    if Z_matched is not None and U_matched is not None:
        out["LinearProbe_R2_u_from_z"] = linear_probe_r2(Z_matched, U_matched)
    if Z_matched is not None and context_labels is not None:
        out["ContextSeparability"] = context_separability(Z_matched, context_labels)
    return out
