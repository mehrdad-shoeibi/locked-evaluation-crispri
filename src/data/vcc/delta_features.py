"""Phase-6.2c Innovation 4: delta features.

For a perturbed pseudo-bulk in batch `b` and target gene `g`, replace
the raw mean-log1p expression vector with its difference from the
batch's non-targeting baseline:

    x_delta[g, b] = mean_log1p(perturbed_cells_g_b) − mean_log1p(NT_cells_b)

Two motivations:
  * Removes per-batch technical effects at the feature level. Context
    structure is still visible to the trainer; the encoder no longer has
    to learn to subtract batch effects itself.
  * Aligns with the simple linear-regression baseline (Ahlmann-Eltze
    2025) which works on this delta, giving every model the same
    starting condition.

Pure NumPy; called from the loader's per-batch pseudobulk path when
`feature_transform="delta"`.
"""

from __future__ import annotations

import numpy as np


def compute_delta_features(
    perturbed_x_log: np.ndarray,
    nt_baseline_log: np.ndarray,
) -> np.ndarray:
    """Subtract the per-batch NT baseline from each perturbed pseudobulk row.

    Parameters
    ----------
    perturbed_x_log : (n_groups, n_genes) float32
        Mean log1p expression of perturbed cells per (target_gene, batch).
        All rows are from the same batch.
    nt_baseline_log : (n_genes,) float32
        Mean log1p expression of non-targeting cells in the same batch.

    Returns
    -------
    (n_groups, n_genes) float32 — perturbation-effect deltas.
    """
    perturbed_x_log = np.asarray(perturbed_x_log, dtype=np.float32)
    nt_baseline_log = np.asarray(nt_baseline_log, dtype=np.float32)
    if perturbed_x_log.ndim != 2:
        raise ValueError(f"perturbed_x_log must be 2-D, got shape {perturbed_x_log.shape}")
    if nt_baseline_log.shape != (perturbed_x_log.shape[1],):
        raise ValueError(
            f"nt_baseline_log shape {nt_baseline_log.shape} does not match "
            f"perturbed_x_log gene dim {perturbed_x_log.shape[1]}"
        )
    return (perturbed_x_log - nt_baseline_log[None, :]).astype(np.float32, copy=False)
