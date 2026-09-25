"""Separation loss — normalized cross-covariance Frobenius penalty.

Locked primary per SPEC §6. DAFR only.

    L_sep = ||Cov(z_s, z_c)||_F^2  /  ( ||Cov(z_s)||_F * ||Cov(z_c)||_F + eps )

Near zero when z_s, z_c are (sample-)uncorrelated across the batch; larger
when they share covariance structure.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _cov(A: Tensor, B: Tensor) -> Tensor:
    Ac = A - A.mean(dim=0, keepdim=True)
    Bc = B - B.mean(dim=0, keepdim=True)
    n = A.shape[0]
    return (Ac.t() @ Bc) / max(n - 1, 1)


def separation_covpenalty(z_s: Tensor, z_c: Tensor, eps: float = 1e-8) -> Tensor:
    C_sc = _cov(z_s, z_c)
    C_ss = _cov(z_s, z_s)
    C_cc = _cov(z_c, z_c)
    num = (C_sc * C_sc).sum()
    denom = (
        torch.sqrt((C_ss * C_ss).sum()).clamp_min(0.0)
        * torch.sqrt((C_cc * C_cc).sum()).clamp_min(0.0)
        + eps
    )
    return num / denom
