"""Phase 9A-Mini — Label Distribution Smoothing (LDS) weighted MSE.

Implements the LDS weighting scheme from Yang et al., "Delving into
Deep Imbalanced Regression" (ICML 2021):

  1. Bin the *training* labels y into K equal-width bins.
  2. Smooth the bin counts with a Gaussian kernel of std σ (in
     bin-index units).
  3. Compute effective per-bin density = smoothed_count / total.
  4. Per-bin weight = 1 / sqrt(eff_density)  (the paper's
     inverse-sqrt variant — gives less aggressive weighting than
     plain 1/density).
  5. Normalize so mean weight over training samples ≈ 1.
  6. Clip per-bin weight to [0, max_weight] for stability.

The factory closes over the train-y bin edges + per-bin weights;
the returned callable matches the standard
`pred_loss_fn(y_pred, y_true) -> Tensor` signature so it slots into
`CompositeLoss.pred_loss_fn` with no trainer changes.

Numerics: same sum-normalized weighted-mean form as `tail_losses`:
  L = Σ wᵢ (yᵢ − ŷᵢ)² / Σ wᵢ.

This keeps loss magnitude comparable to plain MSE so existing LR /
weight-decay choices remain reasonable.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor


def make_lds_weighted_mse(
    train_y: Sequence[float],
    *,
    n_bins: int = 20,
    sigma: float = 2.0,
    max_weight: float = 5.0,
    eps_density: float = 1e-12,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Build a label-density-smoothing weighted MSE loss.

    Parameters
    ----------
    train_y : 1-D collection of training labels (used only at construction).
    n_bins : number of equal-width bins between train_y.min() and max().
    sigma : Gaussian kernel std in BIN-INDEX units (per the LDS paper).
    max_weight : per-bin clip ceiling (after mean-normalization).
    eps_density : floor on smoothed density to avoid div-by-zero in
                  truly empty bins.
    """
    from scipy.ndimage import gaussian_filter1d

    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if max_weight <= 0:
        raise ValueError(f"max_weight must be > 0, got {max_weight}")

    train_y_arr = np.asarray(train_y, dtype=np.float64).reshape(-1)
    if train_y_arr.size < n_bins:
        raise ValueError(
            f"need at least n_bins={n_bins} train labels, got {train_y_arr.size}"
        )
    y_min = float(train_y_arr.min())
    y_max = float(train_y_arr.max())
    if y_max <= y_min:
        raise ValueError(f"train_y must have nonzero range; got [{y_min}, {y_max}]")
    bin_edges = np.linspace(y_min, y_max, n_bins + 1, dtype=np.float64)
    # Push the right edge slightly so y == y_max lands in bin n_bins-1
    bin_edges[-1] += 1e-9

    counts, _ = np.histogram(train_y_arr, bins=bin_edges)
    smoothed = gaussian_filter1d(counts.astype(np.float64), sigma=float(sigma),
                                 mode="reflect")
    eff_density = smoothed / max(smoothed.sum(), eps_density)
    bin_weights = 1.0 / np.sqrt(eff_density + eps_density)

    # Normalize so mean per-sample weight on TRAINING labels = 1.
    train_bin_idx = np.clip(np.digitize(train_y_arr, bin_edges[1:-1]),
                            0, n_bins - 1)
    train_per_sample_w = bin_weights[train_bin_idx]
    norm = float(train_per_sample_w.mean())
    bin_weights = bin_weights / norm
    # Clip — but keep at least 1.0 as the floor isn't enforced by the
    # original LDS recipe; we follow the paper.
    bin_weights = np.clip(bin_weights, 0.0, float(max_weight))

    bin_edges_cpu = torch.as_tensor(bin_edges, dtype=torch.float32, device="cpu")
    bin_weights_cpu = torch.as_tensor(bin_weights, dtype=torch.float32, device="cpu")
    n_bins_int = int(n_bins)

    def fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        if bin_edges_cpu.device != y_true.device:
            be = bin_edges_cpu.to(y_true.device)
            bw = bin_weights_cpu.to(y_true.device)
        else:
            be, bw = bin_edges_cpu, bin_weights_cpu
        idx = torch.bucketize(y_true.detach(), be[1:-1])
        idx = torch.clamp(idx, 0, n_bins_int - 1)
        w = bw[idx]
        diff_sq = (y_pred - y_true) ** 2
        num = (w * diff_sq).sum()
        den = w.sum()
        return torch.where(den > 0, num / den, diff_sq.mean())

    fn.scheme = "lds"  # type: ignore[attr-defined]
    fn.n_bins = n_bins_int  # type: ignore[attr-defined]
    fn.sigma = float(sigma)  # type: ignore[attr-defined]
    fn.max_weight = float(max_weight)  # type: ignore[attr-defined]
    fn.bin_weights_norm = bin_weights.tolist()  # type: ignore[attr-defined]
    fn.bin_edges = bin_edges.tolist()  # type: ignore[attr-defined]
    return fn


__all__ = ["make_lds_weighted_mse"]
