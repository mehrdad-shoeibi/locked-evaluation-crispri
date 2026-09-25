"""Virtual Cell Challenge (VCC) shard/dataset dataclasses.

Duck-types against `src.data.replogle.schema.ReplogleContextShard` so the
existing torch + bi-level trainers consume VCC pseudo-bulks unchanged.
Tier-2 (cluster) and tier-3 (u-space NN) relation matching are not
meaningful on single-cell pseudo-bulks — we ship zero placeholders so
the trainer's `_ContextDataset.concatenate` path works, and the
relation loss runs tier-1 only via `sample_id_int`.

Tier-1 alignment — "same perturbation measured in multiple contexts" —
is carried by `sample_id_int`: the target-gene integer id is shared
across all 48 Flex batches, so the batcher's `build_rel_pair_indices`
emits real cross-batch relation pairs.

Two user-facing extras beyond Replogle (optional for the trainer, used
by the paper's interpretability analyses):

  * `target_genes` — per-row gene symbols, so the result tables can
    report which perturbation each shard row corresponds to.
  * `batch_name` — the Flex batch label ("Flex_1_01"…"Flex_3_16") for
    logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np


# Public Literal type for split selection in the loader.
VccSplit = Literal["train", "val", "test"]


@dataclass
class VccContextShard:
    """One Flex batch of VCC pseudo-bulks.

    Trainer-required fields (duck-typed against ReplogleContextShard):
        context_id:     int
        x:              (n, n_hvg) float32       mean log1p HVG expression
        y:              (n,)        float32       Anderson–Darling statistic
        cluster_idx:    (n,)        int64         placeholder zeros
        u:              (n, 1)      float32       placeholder zeros

    Replogle-shape extras:
        context_name:   str             e.g. "Flex_1_01"
        sample_ids:     (n,)  object    target-gene symbols (strings)
        sample_id_int:  (n,)  int64     integer encoding of target_gene,
                                        shared across all 48 batches so
                                        tier-1 relation matching fires.
        params:         None            duck-type only.

    VCC-specific:
        target_genes:   List[str]       per-row gene symbols (same as
                                        sample_ids, listed explicitly for
                                        downstream table rendering).
        batch_name:     str             == context_name (alias, kept for
                                        call-site clarity in VCC code).
    """

    context_id: int
    x: np.ndarray
    y: np.ndarray
    cluster_idx: np.ndarray
    u: np.ndarray
    context_name: str = ""
    sample_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    sample_id_int: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    params: Optional[object] = None
    target_genes: List[str] = field(default_factory=list)
    batch_name: str = ""

    def __post_init__(self) -> None:
        n = int(self.y.shape[0])
        if self.x.shape[0] != n:
            raise ValueError(
                f"x.shape[0]={self.x.shape[0]} != y.shape[0]={n}"
            )
        if self.cluster_idx.shape[0] != n:
            raise ValueError(
                f"cluster_idx.shape[0]={self.cluster_idx.shape[0]} != n={n}"
            )
        if self.u.shape[0] != n:
            raise ValueError(f"u.shape[0]={self.u.shape[0]} != n={n}")
        if self.sample_id_int.shape[0] != n:
            raise ValueError(
                f"sample_id_int.shape[0]={self.sample_id_int.shape[0]} != n={n}"
            )
        if len(self.target_genes) != n:
            raise ValueError(
                f"len(target_genes)={len(self.target_genes)} != n={n}"
            )
        if self.x.dtype != np.float32:
            raise ValueError(f"x.dtype must be float32, got {self.x.dtype}")
        if self.y.dtype != np.float32:
            raise ValueError(f"y.dtype must be float32, got {self.y.dtype}")


@dataclass
class VccDataset:
    """Train/val/test splits (one list of 48 shards each)."""

    train: List[VccContextShard]
    val: List[VccContextShard]
    test: List[VccContextShard]
    metadata: dict = field(default_factory=dict)

    @property
    def all_contexts(self) -> List[VccContextShard]:
        return [*self.train, *self.val, *self.test]
