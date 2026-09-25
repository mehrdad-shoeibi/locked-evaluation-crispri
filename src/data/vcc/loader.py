"""VCC pseudo-bulk loader with per-split caching.

Pipeline (first run)
--------------------
    1. Read `adata_Training.h5ad` (backed). Pull the 38,176 non-targeting
       cells once — this is the shared control pool for all splits, as
       established in Phase 6.0.5.
    2. For every Flex batch b (48 of them), compute:
         * `nt_mean_log1p[b]` — mean log1p(normalize_total(x, 1e4)) over
           NT cells in b. Full (18080,) vector per batch. Used both as
           the HVG-selection input and (implicitly) as the distributional
           baseline for AD on the target-gene column.
         * `nt_log1p_by_batch[b]` — per-cell log1p at ALL gene columns
           for NT cells in b. (n_nt_in_b, 18080) float32. Stored on disk
           but only referenced via target-gene column slices at AD time.
    3. Pseudo-bulk training split (perturbed cells only). For each
       (target_gene, batch) with >= min_cells_per_group cells:
         * x_full     — mean log1p expression, full 18080 genes
         * y_ad       — Anderson–Darling of perturbed vs NT log1p at the
                        target gene (single scalar)
         Concatenate `x_full` rows from training → pick top-`n_hvg`
         genes by variance → save HVG index array.
    4. Slice `x = x_full[:, hvg_idx]` for all training groups.
    5. Repeat pseudo-bulking on val + test (using the *locked* HVG
       selection from step 3 and the *same* NT baselines from step 2).
    6. Save caches and emit VccDataset-shaped objects.

Subsequent runs
---------------
    Cached artifacts are reloaded in seconds. No h5ad re-read.

On-disk cache layout at `cache_dir` (default `$IGA_DATA_ROOT/cache/vcc/`)
------------------------------------------------------------------------
    meta.json                            # config, batch_names, target_gene_registry
    hvg_genes.npy                        # (n_hvg,) int32 indices into var.index
    var_gene_names.npy                   # (18080,) str — var.index from Training
    nt_baseline/
        mean_log1p.npy                   # (48, 18080) float32
        batch_order.npy                  # (48,) str — Flex batch names in cache order
        per_batch_full_log1p/
            Flex_1_01.npy                # (n_nt_cells_in_b, 18080) float32
            ...                          # one file per batch
    shards/
        train/Flex_1_01.npz              # x, y, sid, targets
        train/... (48)
        val/... (48)
        test/... (48)

Notes on locked design decisions
-------------------------------
* D3 (shared-NT): NT is loaded from `adata_Training.h5ad` only, then
  used for both the AD baseline and the HVG-reference.
* D4 (context=batch): the loader emits exactly 48 shards per split,
  one per Flex batch, using the batch_order discovered in the NT pool.
* D5 (min_cells=10): applied after the (target_gene, batch) groupby
  inside each split.
"""

from __future__ import annotations

import os as _o, sys as _s
_r = _o.path.dirname(_o.path.abspath(__file__))
while _r != _o.path.dirname(_r) and not _o.path.exists(_o.path.join(_r, "iga_paths.py")):
    _r = _o.path.dirname(_r)
if _r not in _s.path:
    _s.path.insert(0, _r)
import iga_paths
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .hvg import select_hvg_from_pseudobulks
from .schema import VccContextShard, VccDataset, VccSplit


log = logging.getLogger(__name__)


DEFAULT_SPLITS = ("train", "val", "test")
_SPLIT_TO_FILENAME = {
    "train": "train/adata_Training.h5ad",
    "val":   "validation/adata_Validation.h5ad",
    "test":  "test/adata_Test.h5ad",
}
_NT_LABEL = "non-targeting"

# Bump whenever the y transformation, pseudo-bulking math, or shard-file
# schema changes in a way that invalidates caches built by earlier code.
# Caches whose meta.json carries a different cache_version are rebuilt
# automatically (stale shards for that split are re-derived; NT baseline
# + HVG are reused since those are version-independent).
_CACHE_VERSION = "v2_log1p_ad"

# Phase 6.2c: feature_transform + gene_selection variants.
_FEATURE_TRANSFORMS = ("raw", "delta")
_GENE_SELECTIONS = ("hvg", "crossctx", "all")


def _variant_subdir(feature_transform: str, gene_selection: str) -> str:
    """Compute the per-variant cache subdir.

    Returns "" for the (raw, hvg) default — preserves the legacy on-disk
    layout from Phase 6.1. Other combinations get
    `variants/{transform}_{selection}/`.
    """
    if feature_transform == "raw" and gene_selection == "hvg":
        return ""
    return f"variants/{feature_transform}_{gene_selection}"


def _default_cache_dir() -> Path:
    root = iga_paths.IGA_NEURIPS
    return Path(root) / "cache" / "vcc"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _sparse_to_dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def _normalize_total_log1p(
    x_raw_dense: np.ndarray, *, target_sum: float = 1e4
) -> np.ndarray:
    """Per-cell library-size normalization to `target_sum`, then log1p.

    Matches `scanpy.pp.normalize_total` + `scanpy.pp.log1p` on dense data.
    Deterministic, dtype-safe (returns float32).
    """
    x = np.asarray(x_raw_dense, dtype=np.float32)
    row_sums = x.sum(axis=1, keepdims=True)
    # Guard against zero-sum rows (shouldn't happen post-QC, but safe).
    row_sums = np.where(row_sums < 1.0, 1.0, row_sums)
    x = x * (np.float32(target_sum) / row_sums)
    return np.log1p(x, dtype=np.float32)


def _anderson_darling_stat(
    a: np.ndarray, b: np.ndarray
) -> float:
    """log1p of the scipy k-sample A–D statistic on two 1-D arrays.

    Higher = more dissimilar distributions. Returns `log1p(max(0, stat))`
    (monotone; compresses the [0, ~300] raw range down to [0, ~5.7] so MSE
    training is numerically well-conditioned). Matches Replogle's
    `y = log1p(anderson_darling_counts)` convention.

    Returns NaN on degenerate inputs (< 2 unique values in either sample,
    etc.) rather than raising — callers filter NaNs.
    """
    from scipy.stats import anderson_ksamp  # local import (scipy already in env)
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        return float("nan")
    try:
        # anderson_ksamp warns when p-value is outside its table range — we
        # only care about the statistic, so swallow the warning.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = anderson_ksamp([a, b])
        stat = float(res.statistic)
        if not np.isfinite(stat):
            return float("nan")
        # log1p of (clamped at 0) raw statistic. Replogle uses the same
        # transformation; keeps y in a well-conditioned range for MSE.
        return float(np.log1p(max(0.0, stat)))
    except (ValueError, RuntimeError):
        return float("nan")


# ---------------------------------------------------------------------------
# Target-gene registry (shared integer encoding across splits/batches)
# ---------------------------------------------------------------------------
def _load_target_gene_registry(data_root: Path) -> Dict[str, int]:
    """Build a stable `{target_gene: int_id}` map over all three splits.

    Reads the three `pert_counts_*.csv` files (sorted union of their
    `target_gene` columns), assigns integer ids starting at 0. Adds a
    final "non-targeting" entry so any future NT-aware code has an id.
    """
    import pandas as pd
    csv_paths = [
        data_root / "train" / "pert_counts_Training.csv",
        data_root / "validation" / "pert_counts_Validation.csv",
        data_root / "test" / "pert_counts_Test.csv",
    ]
    names: set = set()
    for p in csv_paths:
        df = pd.read_csv(p)
        names.update(df["target_gene"].astype(str).tolist())
    sorted_names = sorted(names)
    registry = {g: i for i, g in enumerate(sorted_names)}
    registry.setdefault(_NT_LABEL, len(registry))
    return registry


# ---------------------------------------------------------------------------
# Non-targeting baseline computation (per-batch)
# ---------------------------------------------------------------------------
def _build_nt_baselines(
    train_adata_path: Path,
    cache_dir: Path,
    *,
    verbose: bool,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str], np.ndarray]:
    """Materialize and cache the per-batch NT log1p matrices.

    Returns
    -------
    nt_mean_log1p : (n_batches, n_genes) float32 — per-batch NT mean log1p.
    nt_full_log1p : dict[batch_name -> (n_nt_in_b, n_genes) float32]
                     per-cell NT log1p used for AD.
    batch_order   : list[str] — batch_name ordering (indexes rows of
                     `nt_mean_log1p`, and is the key set of `nt_full_log1p`).
    var_gene_names : (n_genes,) str — gene symbol ordering from var.index.
    """
    import anndata as ad

    nt_root = cache_dir / "nt_baseline"
    per_batch_dir = nt_root / "per_batch_full_log1p"
    mean_path = nt_root / "mean_log1p.npy"
    batch_order_path = nt_root / "batch_order.npy"
    var_names_path = cache_dir / "var_gene_names.npy"

    # Cache hit?
    if (
        mean_path.exists()
        and batch_order_path.exists()
        and var_names_path.exists()
        and per_batch_dir.exists()
        and any(per_batch_dir.glob("*.npy"))
    ):
        if verbose:
            log.info("[vcc nt_baseline] cache hit at %s", nt_root)
        nt_mean = np.load(mean_path)
        batch_order = [str(x) for x in np.load(batch_order_path, allow_pickle=True)]
        var_names = np.load(var_names_path, allow_pickle=True)
        nt_full = {
            b: np.load(per_batch_dir / f"{b}.npy") for b in batch_order
        }
        return nt_mean, nt_full, batch_order, np.asarray(var_names)

    # Compute from scratch.
    if verbose:
        log.info("[vcc nt_baseline] computing from %s", train_adata_path)
    nt_root.mkdir(parents=True, exist_ok=True)
    per_batch_dir.mkdir(parents=True, exist_ok=True)

    a = ad.read_h5ad(train_adata_path, backed="r")
    var_names = np.asarray(a.var.index.astype(str).to_numpy())
    nt_mask_all = (a.obs["target_gene"].astype(str) == _NT_LABEL).to_numpy()
    nt_rows = np.where(nt_mask_all)[0]
    batches_nt = a.obs["batch"].astype(str).to_numpy()[nt_rows]
    batch_order = sorted(set(batches_nt.tolist()))

    n_genes = int(a.var.shape[0])
    nt_mean = np.zeros((len(batch_order), n_genes), dtype=np.float32)
    nt_full: Dict[str, np.ndarray] = {}

    t0 = time.time()
    for i, b in enumerate(batch_order):
        if verbose:
            log.info("[vcc nt_baseline]   batch %d/%d  %s",
                     i + 1, len(batch_order), b)
        idx_in_nt = np.where(batches_nt == b)[0]
        row_idx = nt_rows[idx_in_nt]
        x_raw = _sparse_to_dense(a.X[row_idx, :])
        x_log = _normalize_total_log1p(x_raw)
        nt_full[b] = x_log.astype(np.float32, copy=False)
        nt_mean[i] = x_log.mean(axis=0).astype(np.float32, copy=False)
        np.save(per_batch_dir / f"{b}.npy", nt_full[b])

    np.save(mean_path, nt_mean)
    np.save(batch_order_path, np.asarray(batch_order, dtype=object))
    np.save(var_names_path, var_names)
    if verbose:
        log.info("[vcc nt_baseline] done in %.1fs  (%d batches, %d genes)",
                 time.time() - t0, len(batch_order), n_genes)
    return nt_mean, nt_full, batch_order, var_names


def _verify_nt_shared_across_splits(data_root: Path) -> None:
    """Assert val/test NT obs_names ⊆ train NT obs_names (Phase 6.0.5)."""
    import anndata as ad

    a_train = ad.read_h5ad(data_root / "train" / "adata_Training.h5ad", backed="r")
    nt_train_mask = (a_train.obs["target_gene"].astype(str) == _NT_LABEL).to_numpy()
    nt_train_names = set(a_train.obs_names[nt_train_mask].tolist())

    for split_name, fname in [
        ("val",  "validation/adata_Validation.h5ad"),
        ("test", "test/adata_Test.h5ad"),
    ]:
        a = ad.read_h5ad(data_root / fname, backed="r")
        nt_mask = (a.obs["target_gene"].astype(str) == _NT_LABEL).to_numpy()
        nt_names = set(a.obs_names[nt_mask].tolist())
        missing = nt_names - nt_train_names
        if missing:
            raise RuntimeError(
                f"VCC non-targeting invariant violated for split={split_name}: "
                f"{len(missing)} NT cells are not in the train NT pool. "
                f"Phase 6.0.5's shared-control assumption no longer holds."
            )


# ---------------------------------------------------------------------------
# Per-split pseudo-bulk
# ---------------------------------------------------------------------------
def _pseudobulk_split(
    adata_path: Path,
    *,
    split: str,
    hvg_idx: Optional[np.ndarray],
    nt_full_log1p: Dict[str, np.ndarray],
    nt_mean_log1p: Optional[np.ndarray] = None,   # (n_batches, n_genes)
    var_names: np.ndarray,
    batch_order: List[str],
    target_gene_registry: Mapping[str, int],
    min_cells_per_group: int,
    verbose: bool,
    feature_transform: str = "raw",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute the pseudo-bulks for one split.

    If `hvg_idx is None`, returns per-batch x at FULL gene resolution
    (used on the first training-split call to drive HVG selection).
    If `hvg_idx` is provided, returns per-batch x at HVG resolution.

    Returns
    -------
    per_batch : {batch_name: {
        "x":       (n_groups_in_b, n_cols) float32,
        "y":       (n_groups_in_b,)        float32,
        "sid":     (n_groups_in_b,)        int64,
        "targets": (n_groups_in_b,)        object  (gene symbols)
    }}
    """
    import anndata as ad

    gene_idx_by_symbol = {g: i for i, g in enumerate(var_names.tolist())}

    a = ad.read_h5ad(adata_path, backed="r")
    obs_batch = a.obs["batch"].astype(str).to_numpy()
    obs_target = a.obs["target_gene"].astype(str).to_numpy()
    is_perturbed = obs_target != _NT_LABEL

    per_batch_results: Dict[str, Dict[str, np.ndarray]] = {}

    t0 = time.time()
    for i, b in enumerate(batch_order):
        batch_mask = (obs_batch == b) & is_perturbed
        pert_rows = np.where(batch_mask)[0]
        if verbose:
            log.info("[vcc pseudobulk %s]   batch %d/%d  %s  n_perturbed=%d",
                     split, i + 1, len(batch_order), b, int(pert_rows.size))
        if pert_rows.size == 0:
            per_batch_results[b] = {
                "x": np.empty((0, len(var_names) if hvg_idx is None else len(hvg_idx)),
                              dtype=np.float32),
                "y": np.empty((0,), dtype=np.float32),
                "sid": np.empty((0,), dtype=np.int64),
                "targets": np.empty((0,), dtype=object),
            }
            continue

        # Materialize the full-gene log1p for this batch's perturbed cells.
        x_raw = _sparse_to_dense(a.X[pert_rows, :])
        x_log = _normalize_total_log1p(x_raw)  # (n_pert_in_b, n_genes)

        target_in_batch = obs_target[pert_rows]
        # For each unique target, compute the pseudo-bulk + AD
        nt_b_log = nt_full_log1p[b]           # (n_nt_in_b, n_genes)

        xs: List[np.ndarray] = []
        ys: List[float] = []
        sids: List[int] = []
        tgs: List[str] = []

        for tg in np.unique(target_in_batch):
            tg_mask = target_in_batch == tg
            n_cells = int(tg_mask.sum())
            if n_cells < min_cells_per_group:
                continue
            tg_col = gene_idx_by_symbol.get(str(tg))
            if tg_col is None:
                # Target gene not in var.index — skip. (Shouldn't happen but guard.)
                log.warning("[vcc pseudobulk %s]  target %s not in var.index; skipping",
                            split, tg)
                continue
            # y = Anderson-Darling of perturbed target-gene log1p vs NT target-gene log1p
            y_ad = _anderson_darling_stat(
                x_log[tg_mask, tg_col], nt_b_log[:, tg_col]
            )
            if not np.isfinite(y_ad):
                # Degenerate group (no signal) — drop.
                continue
            # x = mean log1p expression (optionally HVG-sliced).
            # If feature_transform="delta", subtract per-batch NT baseline
            # mean before any gene-subset slicing (innovation 4).
            x_mean = x_log[tg_mask, :].mean(axis=0).astype(np.float32, copy=False)
            if feature_transform == "delta":
                if nt_mean_log1p is None:
                    raise ValueError(
                        "feature_transform='delta' requires nt_mean_log1p"
                    )
                from .delta_features import compute_delta_features
                x_mean = compute_delta_features(
                    x_mean[None, :], nt_mean_log1p[i],
                )[0]  # (n_genes,)
            if hvg_idx is not None:
                x_mean = x_mean[hvg_idx]
            xs.append(x_mean)
            ys.append(float(y_ad))
            sids.append(int(target_gene_registry[str(tg)]))
            tgs.append(str(tg))

        if xs:
            per_batch_results[b] = {
                "x": np.stack(xs, axis=0).astype(np.float32, copy=False),
                "y": np.asarray(ys, dtype=np.float32),
                "sid": np.asarray(sids, dtype=np.int64),
                "targets": np.asarray(tgs, dtype=object),
            }
        else:
            per_batch_results[b] = {
                "x": np.empty((0, len(var_names) if hvg_idx is None else len(hvg_idx)),
                              dtype=np.float32),
                "y": np.empty((0,), dtype=np.float32),
                "sid": np.empty((0,), dtype=np.int64),
                "targets": np.empty((0,), dtype=object),
            }

    if verbose:
        log.info("[vcc pseudobulk %s] done in %.1fs", split, time.time() - t0)
    return per_batch_results


def _build_shards_from_per_batch(
    per_batch: Dict[str, Dict[str, np.ndarray]],
    *,
    batch_order: List[str],
) -> List[VccContextShard]:
    shards: List[VccContextShard] = []
    for ctx_id, b in enumerate(batch_order):
        rec = per_batch.get(b)
        if rec is None or rec["y"].shape[0] == 0:
            # Keep the shard but empty — the trainer handles empty contexts
            # defensively (min_group_size drop). Track as zero-sized rather
            # than omitting so context_id space stays contiguous.
            n_cols = rec["x"].shape[1] if rec is not None else 0
            shards.append(VccContextShard(
                context_id=ctx_id,
                x=np.empty((0, n_cols), dtype=np.float32),
                y=np.empty((0,), dtype=np.float32),
                cluster_idx=np.empty((0,), dtype=np.int64),
                u=np.empty((0, 1), dtype=np.float32),
                context_name=b,
                sample_ids=np.empty((0,), dtype=object),
                sample_id_int=np.empty((0,), dtype=np.int64),
                target_genes=[],
                batch_name=b,
            ))
            continue
        n = rec["y"].shape[0]
        shards.append(VccContextShard(
            context_id=ctx_id,
            x=rec["x"],
            y=rec["y"],
            cluster_idx=np.zeros(n, dtype=np.int64),
            u=np.zeros((n, 1), dtype=np.float32),
            context_name=b,
            sample_ids=rec["targets"],
            sample_id_int=rec["sid"],
            target_genes=[str(g) for g in rec["targets"].tolist()],
            batch_name=b,
        ))
    return shards


# ---------------------------------------------------------------------------
# Cache I/O for per-split shards
# ---------------------------------------------------------------------------
def _save_shard(shard: VccContextShard, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        x=shard.x,
        y=shard.y,
        sample_id_int=shard.sample_id_int,
        sample_ids=shard.sample_ids,
        context_id=np.asarray([shard.context_id], dtype=np.int64),
        batch_name=np.asarray([shard.batch_name], dtype=object),
    )


def _load_shard(in_path: Path) -> VccContextShard:
    d = np.load(in_path, allow_pickle=True)
    n = int(d["y"].shape[0])
    batch_name = str(d["batch_name"][0])
    targets = [str(s) for s in d["sample_ids"].tolist()]
    return VccContextShard(
        context_id=int(d["context_id"][0]),
        x=np.asarray(d["x"], dtype=np.float32),
        y=np.asarray(d["y"], dtype=np.float32),
        cluster_idx=np.zeros(n, dtype=np.int64),
        u=np.zeros((n, 1), dtype=np.float32),
        context_name=batch_name,
        sample_ids=np.asarray(d["sample_ids"], dtype=object),
        sample_id_int=np.asarray(d["sample_id_int"], dtype=np.int64),
        target_genes=targets,
        batch_name=batch_name,
    )


# ---------------------------------------------------------------------------
# Cache invalidation helper (run on version bump; removes all stale shards
# so the per-split _cache_valid check triggers a fresh pseudo-bulk pass).
# ---------------------------------------------------------------------------
def _invalidate_all_shard_caches(cache_dir: Path) -> None:
    shards_root = cache_dir / "shards"
    if not shards_root.exists():
        return
    for split in DEFAULT_SPLITS:
        split_dir = shards_root / split
        if not split_dir.exists():
            continue
        for f in split_dir.iterdir():
            if f.is_file():
                f.unlink()


def _invalidate_variant_shard_caches(variant_root: Path) -> None:
    """Same as `_invalidate_all_shard_caches` but for a non-default
    variant subdirectory."""
    _invalidate_all_shard_caches(variant_root)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def _cache_valid(
    cache_dir: Path, split: str, batch_order: List[str], variant_subdir: str = ""
) -> bool:
    """Shards present on disk AND tagged with the current _CACHE_VERSION.

    `variant_subdir` is "" for the legacy (raw, hvg) layout (paths sit
    directly under cache_dir/shards/). For other variants it points
    under `cache_dir/variants/<v>/shards/`.
    """
    base = cache_dir / variant_subdir if variant_subdir else cache_dir
    split_dir = base / "shards" / split
    if not split_dir.exists():
        return False
    meta_path = base / "meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    if str(meta.get("cache_version", "")) != _CACHE_VERSION:
        return False
    for b in batch_order:
        if not (split_dir / f"{b}.npz").exists():
            return False
    return True


def load_vcc_pseudobulk(
    data_root: str | Path,
    split: VccSplit,
    *,
    n_hvg: int = 2000,
    min_cells_per_group: int = 10,
    cache_dir: Optional[str | Path] = None,
    seed: int = 42,
    verbose: bool = True,
    feature_transform: str = "raw",
    gene_selection: str = "hvg",
) -> List[VccContextShard]:
    """Return the 48-shard list for one VCC split.

    Parameters
    ----------
    feature_transform : "raw" | "delta"
        "raw" uses mean log1p(normalize(counts)); "delta" subtracts the
        per-batch NT mean (Phase 6.2c innovation 4).
    gene_selection : "hvg" | "crossctx"
        "hvg" picks top-`n_hvg` genes by variance across pseudobulks;
        "crossctx" uses within/cross batch variance ratio (Phase 6.2c
        innovation 5).

    Cache layout: NT baselines stay shared at `cache_dir/nt_baseline/`.
    Per-variant data (selected genes + shards + meta) lives under
    `cache_dir/variants/{transform}_{selection}/` for non-default
    variants, or directly under `cache_dir/` for the (raw, hvg)
    default (preserves backward-compat with Phase-6.1 cache).
    """
    if feature_transform not in _FEATURE_TRANSFORMS:
        raise ValueError(
            f"feature_transform must be one of {_FEATURE_TRANSFORMS}; "
            f"got {feature_transform!r}"
        )
    if gene_selection not in _GENE_SELECTIONS:
        raise ValueError(
            f"gene_selection must be one of {_GENE_SELECTIONS}; "
            f"got {gene_selection!r}"
        )

    data_root = Path(data_root)
    cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    variant_subdir = _variant_subdir(feature_transform, gene_selection)
    variant_root = cache_dir / variant_subdir if variant_subdir else cache_dir
    variant_root.mkdir(parents=True, exist_ok=True)

    meta_path = variant_root / "meta.json"
    # For default variant we keep the legacy filename "hvg_genes.npy" so
    # the existing Phase-6.1 cache loads unchanged. New variants use
    # "selected_genes.npy" (semantically generic since gene_selection
    # may not be HVG).
    if not variant_subdir:
        sel_genes_path = variant_root / "hvg_genes.npy"
    else:
        sel_genes_path = variant_root / "selected_genes.npy"
    var_names_path = cache_dir / "var_gene_names.npy"

    if split not in _SPLIT_TO_FILENAME:
        raise ValueError(f"split must be one of {list(_SPLIT_TO_FILENAME)}; got {split!r}")

    # Step 1: NT baselines + var.index (always cached)
    nt_mean_log1p, nt_full_log1p, batch_order, var_names = _build_nt_baselines(
        data_root / "train" / "adata_Training.h5ad",
        cache_dir,
        verbose=verbose,
    )

    # Step 2: target-gene registry
    registry = _load_target_gene_registry(data_root)

    # Step 3: gene selection (computed once from training; cached per variant).
    # HVG / crossctx selection is variant-specific. NT baseline + var.index
    # are shared across all variants.
    if sel_genes_path.exists() and meta_path.exists():
        sel_idx = np.load(sel_genes_path).astype(np.int32)
        meta = json.loads(meta_path.read_text())
        if gene_selection != "all" and int(meta.get("n_hvg", -1)) != n_hvg:
            raise RuntimeError(
                f"Cached gene set has n_hvg={meta.get('n_hvg')} but caller asked "
                f"for {n_hvg}. Delete {variant_root} and retry, or choose a matching n_hvg."
            )
        if str(meta.get("feature_transform", "raw")) != feature_transform:
            raise RuntimeError(
                f"Cached variant has feature_transform="
                f"{meta.get('feature_transform')!r} but caller asked for "
                f"{feature_transform!r}. Variant subdir mismatch."
            )
        if str(meta.get("gene_selection", "hvg")) != gene_selection:
            raise RuntimeError(
                f"Cached variant has gene_selection={meta.get('gene_selection')!r}"
                f" but caller asked for {gene_selection!r}. Variant subdir mismatch."
            )
        # Stale version → invalidate this variant's shards (per-split
        # rebuild on demand). The selected_genes file stays.
        if str(meta.get("cache_version", "")) != _CACHE_VERSION:
            if verbose:
                log.info("[vcc cache] stale version %r in %s; invalidating shard "
                         "caches for rebuild under %r",
                         meta.get("cache_version"), variant_root, _CACHE_VERSION)
            _invalidate_variant_shard_caches(variant_root)
            meta["cache_version"] = _CACHE_VERSION
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
    else:
        if verbose:
            log.info("[vcc genes] computing from training pseudo-bulks "
                     "(full-gene pass, one-time per variant)")
        # Compute full-gene training pseudobulks once to derive gene selection.
        pre_train = _pseudobulk_split(
            data_root / _SPLIT_TO_FILENAME["train"],
            split="train",
            hvg_idx=None,          # full gene resolution
            nt_full_log1p=nt_full_log1p,
            nt_mean_log1p=nt_mean_log1p,
            var_names=var_names,
            batch_order=batch_order,
            target_gene_registry=registry,
            min_cells_per_group=min_cells_per_group,
            verbose=verbose,
            feature_transform=feature_transform,
        )
        if gene_selection == "hvg":
            all_x = np.concatenate(
                [rec["x"] for rec in pre_train.values() if rec["x"].shape[0] > 0],
                axis=0,
            )
            sel_idx = select_hvg_from_pseudobulks(all_x, n_hvg=n_hvg, seed=seed)
        elif gene_selection == "crossctx":
            from .gene_selection import select_crossctx_genes
            per_batch_arrays = [
                pre_train[b]["x"] for b in batch_order
                if pre_train[b]["x"].shape[0] > 0
            ]
            sel_idx, _scores = select_crossctx_genes(
                per_batch_arrays, n_genes=n_hvg,
            )
        else:  # "all" — full gene resolution; n_hvg ignored
            sel_idx = np.arange(len(var_names), dtype=np.int32)
        np.save(sel_genes_path, sel_idx)
        if verbose:
            log.info("[vcc genes] selected %d genes via %s  (saved to %s)",
                     sel_idx.shape[0], gene_selection, sel_genes_path)
        # Materialize training shards on this same full-gene pass by slicing.
        train_split_dir = variant_root / "shards" / "train"
        train_split_dir.mkdir(parents=True, exist_ok=True)
        for ctx_id, b in enumerate(batch_order):
            rec = pre_train[b]
            if rec["x"].shape[0] > 0:
                x_sel = rec["x"][:, sel_idx].astype(np.float32, copy=False)
                shard = VccContextShard(
                    context_id=ctx_id,
                    x=x_sel,
                    y=rec["y"],
                    cluster_idx=np.zeros(rec["y"].shape[0], dtype=np.int64),
                    u=np.zeros((rec["y"].shape[0], 1), dtype=np.float32),
                    context_name=b,
                    sample_ids=rec["targets"],
                    sample_id_int=rec["sid"],
                    target_genes=[str(g) for g in rec["targets"].tolist()],
                    batch_name=b,
                )
            else:
                shard = VccContextShard(
                    context_id=ctx_id,
                    x=np.empty((0, n_hvg), dtype=np.float32),
                    y=np.empty((0,), dtype=np.float32),
                    cluster_idx=np.empty((0,), dtype=np.int64),
                    u=np.empty((0, 1), dtype=np.float32),
                    context_name=b,
                    sample_ids=np.empty((0,), dtype=object),
                    sample_id_int=np.empty((0,), dtype=np.int64),
                    target_genes=[],
                    batch_name=b,
                )
            _save_shard(shard, train_split_dir / f"{b}.npz")
        meta = {
            "cache_version": _CACHE_VERSION,
            "n_hvg": int(sel_idx.shape[0]) if gene_selection == "all" else int(n_hvg),
            "min_cells_per_group": int(min_cells_per_group),
            "seed": int(seed),
            "batch_order": batch_order,
            "target_gene_registry": registry,
            "n_genes": int(len(var_names)),
            "feature_transform": feature_transform,
            "gene_selection": gene_selection,
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

    hvg_idx = sel_idx  # alias for downstream code

    # Step 4: ensure this split's shards are cached (train already saved in the
    # HVG block above if we just computed them).
    if not _cache_valid(cache_dir, split, batch_order, variant_subdir):
        per_batch = _pseudobulk_split(
            data_root / _SPLIT_TO_FILENAME[split],
            split=split,
            hvg_idx=hvg_idx,
            nt_mean_log1p=nt_mean_log1p,
            feature_transform=feature_transform,
            nt_full_log1p=nt_full_log1p,
            var_names=var_names,
            batch_order=batch_order,
            target_gene_registry=registry,
            min_cells_per_group=min_cells_per_group,
            verbose=verbose,
        )
        split_dir = variant_root / "shards" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for ctx_id, b in enumerate(batch_order):
            rec = per_batch[b]
            if rec["y"].shape[0] > 0:
                shard = VccContextShard(
                    context_id=ctx_id,
                    x=rec["x"],
                    y=rec["y"],
                    cluster_idx=np.zeros(rec["y"].shape[0], dtype=np.int64),
                    u=np.zeros((rec["y"].shape[0], 1), dtype=np.float32),
                    context_name=b,
                    sample_ids=rec["targets"],
                    sample_id_int=rec["sid"],
                    target_genes=[str(g) for g in rec["targets"].tolist()],
                    batch_name=b,
                )
            else:
                shard = VccContextShard(
                    context_id=ctx_id,
                    x=np.empty((0, len(hvg_idx)), dtype=np.float32),
                    y=np.empty((0,), dtype=np.float32),
                    cluster_idx=np.empty((0,), dtype=np.int64),
                    u=np.empty((0, 1), dtype=np.float32),
                    context_name=b,
                    sample_ids=np.empty((0,), dtype=object),
                    sample_id_int=np.empty((0,), dtype=np.int64),
                    target_genes=[],
                    batch_name=b,
                )
            _save_shard(shard, split_dir / f"{b}.npz")

    # Step 5: load shards from cache and return
    split_dir = variant_root / "shards" / split
    shards: List[VccContextShard] = []
    for ctx_id, b in enumerate(batch_order):
        s = _load_shard(split_dir / f"{b}.npz")
        # Reassign context_id to the canonical batch_order index in case the
        # cached value is stale (shouldn't be, but cheap).
        s.context_id = ctx_id
        shards.append(s)
    if verbose:
        total = sum(s.y.shape[0] for s in shards)
        log.info("[vcc load %s]  %d shards, %d total groups", split, len(shards), total)
    return shards
