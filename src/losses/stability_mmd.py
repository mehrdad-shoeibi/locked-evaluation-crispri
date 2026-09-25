"""Stability loss — MMD^2 with Gaussian kernel, median-heuristic bandwidth.

Locked primary per SPEC §6.

API
---
    stability_mmd(context_feats, bandwidth_sq=None) -> Tensor (0-dim)

`context_feats` is a list of (n_c, d) tensors, one per context. Returns the
mean biased MMD^2 over all K*(K-1)/2 unordered pairs. If K < 2, returns 0.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import Tensor


def _pairwise_sq_dists(a: Tensor, b: Tensor) -> Tensor:
    aa = (a * a).sum(dim=1, keepdim=True)              # (n, 1)
    bb = (b * b).sum(dim=1, keepdim=True).transpose(0, 1)  # (1, m)
    d2 = aa + bb - 2.0 * a @ b.transpose(0, 1)
    return torch.clamp(d2, min=0.0)


@torch.no_grad()
def _median_bandwidth_sq(feats: Sequence[Tensor]) -> float:
    """Median of pairwise squared distances on the concatenated batch."""
    z = torch.cat(list(feats), dim=0)
    n = z.shape[0]
    if n < 2:
        return 1.0
    d2 = _pairwise_sq_dists(z, z)
    iu = torch.triu_indices(n, n, offset=1, device=z.device)
    vals = d2[iu[0], iu[1]]
    med = torch.median(vals).item()
    return max(float(med), 1e-12)


def _mmd2_gaussian(x: Tensor, y: Tensor, bandwidth_sq: float) -> Tensor:
    denom = 2.0 * bandwidth_sq
    Kxx = torch.exp(-_pairwise_sq_dists(x, x) / denom)
    Kyy = torch.exp(-_pairwise_sq_dists(y, y) / denom)
    Kxy = torch.exp(-_pairwise_sq_dists(x, y) / denom)
    return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()


def stability_mmd(
    context_feats: Sequence[Tensor],
    bandwidth_sq: Optional[float] = None,
) -> Tensor:
    """Average MMD^2 over all unordered context pairs.

    Bandwidth heuristic is computed under `no_grad`: the kernel bandwidth is
    treated as a hyperparameter of the objective, not a differentiable
    quantity, which gives cleaner gradients for the encoder.
    """
    feats = [f for f in context_feats if f.shape[0] > 0]
    K = len(feats)
    if K < 2:
        return torch.zeros((), device=feats[0].device if feats else "cpu")
    if bandwidth_sq is None:
        bandwidth_sq = _median_bandwidth_sq(feats)

    total = feats[0].new_zeros(())
    n_pairs = 0
    for i in range(K):
        for j in range(i + 1, K):
            total = total + _mmd2_gaussian(feats[i], feats[j], bandwidth_sq)
            n_pairs += 1
    return total / n_pairs
