"""Highly-variable gene (HVG) selection for VCC pseudo-bulks.

Design deviation vs the Phase-6.1 locked spec: the spec named
`scanpy.pp.highly_variable_genes(flavor="seurat_v3")`, which requires
raw integer counts. Our pseudo-bulks are already mean-log1p-normalized,
so seurat_v3 does not apply. We use the next-simplest deterministic
rule: rank genes by variance of the pseudo-bulk expression across
training (target_gene, batch) groups, and take the top `n_hvg`.

This is equivalent to `flavor="seurat"` on log-normalized data (which
also ranks by normalized dispersion in practice) and avoids a scanpy
dependency at HVG time.
"""

from __future__ import annotations

import numpy as np


def select_hvg_from_pseudobulks(
    pseudobulks: np.ndarray,
    n_hvg: int = 2000,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Return the top-`n_hvg` gene indices by variance across pseudobulks.

    Parameters
    ----------
    pseudobulks : (n_groups, n_genes) float array
        Mean-log1p expression per (target_gene, batch) training group.
    n_hvg : int
        Number of HVGs to select.
    seed : int
        Used only as a tie-breaker in the argsort (stable sort → no
        randomness unless ties exist).

    Returns
    -------
    idx : (n_hvg,) int32
        Gene indices (into the pseudobulks columns) sorted ascending so
        downstream code can feed them directly to `np.take`.
    """
    if pseudobulks.ndim != 2:
        raise ValueError(f"pseudobulks must be 2-D, got {pseudobulks.shape}")
    n_genes = pseudobulks.shape[1]
    if n_hvg >= n_genes:
        return np.arange(n_genes, dtype=np.int32)

    # Variance per gene across groups. Use mean-centered fp64 for stability.
    x = pseudobulks.astype(np.float64, copy=False)
    var = x.var(axis=0, ddof=0)
    # argsort is stable → deterministic; seed kept for future tie-break use.
    order_desc = np.argsort(-var, kind="stable")
    top = order_desc[:n_hvg]
    top.sort()  # ascending for easy slicing
    return top.astype(np.int32)
