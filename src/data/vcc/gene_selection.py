"""Phase-6.2c Innovation 5: cross-context gene selection.

Replaces "top variance across pseudo-bulks" (HVG) with a score that
favours genes whose perturbation effect is **consistent across batches**:

    score(g)  =  within_batch_var(g) / (cross_batch_var(g) + eps)

  * `within_batch_var(g)` = average across batches of the variance of
    perturbed-cell pseudo-bulk values at gene g — high if perturbations
    drive diverse responses in this gene within any single batch.
  * `cross_batch_var(g)` = variance across batches of the per-batch mean
    perturbed pseudo-bulk at gene g — high if a gene's mean shifts
    between batches (i.e. carries batch effects).

A gene with high within-batch variance and low cross-batch variance
carries clean perturbation signal that's invariant to the batch — exactly
what we want the model to learn from on a multi-batch dataset.

The selector returns top-k indices by this ratio.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def select_crossctx_genes(
    perturbed_pseudobulks: Sequence[np.ndarray],
    n_genes: int = 2000,
    *,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick the top `n_genes` by within/cross batch variance ratio.

    Parameters
    ----------
    perturbed_pseudobulks : list of length n_batches
        Each entry is `(n_groups_in_b, n_genes)` mean-log1p pseudo-bulks
        for the perturbed (target_gene, batch_b) groups in that batch.
        Empty batches (no perturbed groups) are skipped.
    n_genes : int
        Number of genes to keep. Returned indices are sorted ascending.
    eps : float
        Numerical floor on cross-batch variance to avoid divide-by-zero
        for genes that happen to be perfectly constant across batches.

    Returns
    -------
    (top_indices, scores) : ((n_genes,) int32, (total_genes,) float32)
        `top_indices` is the selected gene set (ascending).
        `scores` is the full per-gene ratio (for diagnostics).
    """
    arrays = [a for a in perturbed_pseudobulks if a is not None and a.shape[0] > 0]
    if len(arrays) < 2:
        raise ValueError(
            f"Need >= 2 non-empty per-batch pseudobulk arrays, got {len(arrays)}"
        )
    n_total_genes = arrays[0].shape[1]
    if any(a.shape[1] != n_total_genes for a in arrays):
        raise ValueError("Per-batch pseudobulks must all have the same gene dim")

    # Within-batch: variance of each gene across the perturbed groups in
    # that batch; average across batches with non-trivial perturbation
    # diversity (>= 2 groups).
    within_vars = []
    means_per_batch = np.zeros((len(arrays), n_total_genes), dtype=np.float64)
    for i, a in enumerate(arrays):
        a64 = a.astype(np.float64, copy=False)
        means_per_batch[i] = a64.mean(axis=0)
        if a64.shape[0] >= 2:
            within_vars.append(a64.var(axis=0, ddof=0))
    if not within_vars:
        raise ValueError("No batch had >= 2 perturbed groups; cannot compute within-batch variance")
    within = np.mean(np.stack(within_vars, axis=0), axis=0)  # (n_total_genes,)

    # Cross-batch: variance of per-batch means across the n_batches axis.
    cross = means_per_batch.var(axis=0, ddof=0)  # (n_total_genes,)

    score = within / (cross + eps)
    score = score.astype(np.float32, copy=False)

    if n_genes >= n_total_genes:
        return np.arange(n_total_genes, dtype=np.int32), score

    order = np.argsort(-score, kind="stable")
    top = np.sort(order[:n_genes]).astype(np.int32)
    return top, score
