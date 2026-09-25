"""Phase 9C-Lite — Magnitude-augmented features for SPSM V4.

Computes four non-leaky magnitude statistics per row from x alone:
    mean_abs(x), ||x||_2, max(|x|), std(x)

These four scalars are z-scored using *train statistics only*, then
concatenated to the existing per-row feature vector. The result has
dim = (original_d_x + 4).

Why this exists:
  Phase 9B-1 found that a 1-D scalar (mean|x|) outperforms the
  full 2000-D LogReg on Q1 vs Q5 separability. The hypothesis is that
  the SPSM encoder's MLP collapses these scalar statistics under MSE
  training. By adding them as explicit input features, we give the
  encoder direct access without needing to discover them.

Strict Phase-9C-Lite scope: this is the *only* method change. No new
loss, no aux head, no architecture change beyond input dim += 4.

API:
  augment_shards_with_magnitude(train, val, test) -> (train', val', test')
    Returns deep-copied shard lists with augmented x. Train statistics
    used for z-scoring are frozen and applied to val/test.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


def _row_magnitude_features(X: np.ndarray) -> np.ndarray:
    """X: (n, d) float — return (n, 4) of [mean|x|, ||x||_2, max|x|, std(x)]."""
    if X.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    abs_x = np.abs(X)
    out = np.empty((X.shape[0], 4), dtype=np.float32)
    out[:, 0] = abs_x.mean(axis=1)                      # mean|x|
    out[:, 1] = np.linalg.norm(X, axis=1)                # L2 norm
    out[:, 2] = abs_x.max(axis=1)                        # max|x|
    out[:, 3] = X.std(axis=1, ddof=0)                    # std (per-row population std)
    return out


@dataclass
class MagnitudeStats:
    """Frozen z-score statistics computed from training rows only."""
    mean: np.ndarray   # (4,)
    std: np.ndarray    # (4,)
    feature_names: Tuple[str, str, str, str] = (
        "mean_abs_x", "l2_norm_x", "max_abs_x", "std_x"
    )

    def transform(self, M: np.ndarray) -> np.ndarray:
        """Apply z-score using cached train statistics. Safe on empty input."""
        if M.shape[0] == 0:
            return M.astype(np.float32)
        denom = np.where(self.std > 0, self.std, 1.0)
        return ((M - self.mean) / denom).astype(np.float32)


def fit_magnitude_stats(train_shards: Sequence) -> MagnitudeStats:
    """Compute mean/std of the 4 magnitude features over ALL train rows."""
    parts = []
    for s in train_shards:
        if s.x.shape[0] == 0: continue
        parts.append(_row_magnitude_features(s.x))
    if not parts:
        raise ValueError("no training rows available")
    M = np.concatenate(parts, axis=0)
    return MagnitudeStats(
        mean=M.mean(axis=0).astype(np.float32),
        std=M.std(axis=0, ddof=0).astype(np.float32),
    )


def augment_shards_with_magnitude(
    train_shards: Sequence,
    val_shards: Sequence,
    test_shards: Sequence,
) -> Tuple[list, list, list, MagnitudeStats]:
    """Replace each shard's `x` with `concat(x, z(magnitude_features))`.

    Train stats are computed once from train and frozen for val/test.
    Returns shallow-copied shard lists (so we don't mutate caller's data).
    """
    stats = fit_magnitude_stats(train_shards)

    def _apply(shards):
        out = []
        for s in shards:
            new_s = copy.copy(s)
            if s.x.shape[0] == 0:
                new_s.x = np.zeros((0, s.x.shape[1] + 4), dtype=np.float32)
                out.append(new_s); continue
            mag = _row_magnitude_features(s.x)
            mag_z = stats.transform(mag)
            new_s.x = np.concatenate([s.x.astype(np.float32, copy=False),
                                       mag_z], axis=1)
            out.append(new_s)
        return out

    return _apply(train_shards), _apply(val_shards), _apply(test_shards), stats


__all__ = [
    "MagnitudeStats",
    "fit_magnitude_stats",
    "augment_shards_with_magnitude",
    "_row_magnitude_features",
]
