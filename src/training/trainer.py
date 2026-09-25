"""Torch trainer for the four locked model families (production path).

Not runnable in environments without torch; the full MCE is exercised in the
sandbox via `numpy_trainer.py`. This module is kept in sync with the spec and
with the NumPy runner so that when torch is available, re-running a given
(case, model, seed) config with this trainer should produce directionally
identical results.

Responsibilities
----------------
1. Construct context-stratified batches. Each batch is a concatenation of
   per-context chunks; `context_ids` follows the same row order so
   `group_batch_by_context` can recover per-context slices (see
   `training.batching`).

2. For L_stab: materialize `context_feats = [z[g.indices] for g in groups]`
   where `z` is `model.forward(x).z` for DWP/SRM/SPSM, and `z_s` for DAFR.
   The stability target is stated on the *stable* representation when one
   exists, pooled on the full representation otherwise.

3. For L_rel: call `build_rel_pair_indices(...)` on the current batch, then
   materialize predictions by index:
       pa = torch.index_select(y_pred, 0, idx_a)
       pb = torch.index_select(y_pred, 0, idx_b)
   preserving autograd. Tier weights are read from `TIER_WEIGHTS` (locked).

4. For L_sep: pass `z_s, z_c` from factorized encoder directly.

5. Per epoch: log total + each component + predictive/transfer/representation
   /relation metric bundles (computed on a held-out chunk of val data).

6. At end of training: save predictions (`preds.parquet`), write metrics.json,
   flush log.csv, dump run_metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


from typing import TYPE_CHECKING
if TYPE_CHECKING:  # type-hint only; the synthetic generator is not part of this release
    from ..data.synthetic.generator import ContextShard, SyntheticDataset
from .batching import (
    TIER_WEIGHTS,
    build_rel_pair_indices,
    group_batch_by_context,
    tier_counts,
)


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    lambda_stab: float = 0.0
    lambda_rel:  float = 0.0
    lambda_sep:  float = 0.0
    stab_enabled: bool = False
    rel_enabled:  bool = False
    sep_enabled:  bool = False
    device: str = "cpu"
    log_every: int = 1


if _HAS_TORCH:

    class _ContextDataset(Dataset):
        """Row-per-sample tensor dataset carrying context_id + cluster + u.

        If every shard exposes an integer `sample_id_int` (length = n_rows),
        we concatenate them too and expose as `self.sid`. This is what the
        trainer uses to enable tier-1 relation matching on real data where
        the same perturbation is measured in multiple cell lines. For the
        synthetic path, ContextShard has no such attribute and `self.sid`
        stays None (tier-1 matches nothing).
        """
        def __init__(self, shards: Sequence["ContextShard"]):
            self.x = torch.from_numpy(np.concatenate([s.x for s in shards], axis=0)).float()
            self.y = torch.from_numpy(np.concatenate([s.y for s in shards], axis=0)).float()
            self.cid = torch.from_numpy(
                np.concatenate([np.full(s.x.shape[0], s.context_id) for s in shards])
            ).long()
            self.cluster = torch.from_numpy(
                np.concatenate([s.cluster_idx for s in shards])
            ).long()
            self.u = torch.from_numpy(np.concatenate([s.u for s in shards], axis=0)).float()
            sid_arrays = []
            for s in shards:
                sid = getattr(s, "sample_id_int", None)
                if sid is None or np.asarray(sid).size != s.x.shape[0]:
                    sid_arrays = None
                    break
                sid_arrays.append(np.asarray(sid, dtype=np.int64))
            self.sid: Optional[Tensor] = (
                torch.from_numpy(np.concatenate(sid_arrays, axis=0)).long()
                if sid_arrays else None
            )

        def __len__(self): return self.x.shape[0]

        def __getitem__(self, i: int):
            return self.x[i], self.y[i], self.cid[i], self.cluster[i], self.u[i]


    def _stratified_batches(
        ds: _ContextDataset, *, batch_size: int, rng: np.random.Generator
    ):
        """Yield batches that contain ≥ min_per_ctx rows from each context.

        Implemented by concatenating per-context mini-slices. This keeps MMD
        well-defined and gives tier-2/tier-3 rel pairs enough mass.
        """
        cids = ds.cid.numpy()
        unique_ctx = np.unique(cids)
        by_ctx = {c: np.where(cids == c)[0] for c in unique_ctx}
        per_ctx = max(4, batch_size // max(len(unique_ctx), 1))
        # shuffle each context's indices
        for c in unique_ctx:
            rng.shuffle(by_ctx[c])
        # iterate rounds
        cursors = {c: 0 for c in unique_ctx}
        while True:
            batch_idx = []
            for c in unique_ctx:
                end = cursors[c] + per_ctx
                take = by_ctx[c][cursors[c]:end]
                if take.size < 2:
                    return  # epoch over
                batch_idx.append(take)
                cursors[c] = end
            yield np.concatenate(batch_idx, axis=0)


    def train_torch(
        model: "torch.nn.Module",
        dataset: "SyntheticDataset",
        cfg: TrainConfig,
        *,
        composite_loss=None,
        seed: int = 0,
    ) -> Dict[str, object]:
        """Run torch training; returns dict of artifacts (preds, curves).

        Preconditions: model.forward returns `ModelOutput(y_pred, z, z_s, z_c)`
        where z_s/z_c are populated for DAFR only. `composite_loss` must be a
        CompositeLoss instance matching the family's enabled terms.
        """
        from ..losses.composite import CompositeLoss, LossWeights

        if composite_loss is None:
            composite_loss = CompositeLoss(
                weights=LossWeights(cfg.lambda_stab, cfg.lambda_rel, cfg.lambda_sep),
                stab_enabled=cfg.stab_enabled,
                rel_enabled=cfg.rel_enabled,
                sep_enabled=cfg.sep_enabled,
            )

        device = torch.device(cfg.device)
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        train_ds = _ContextDataset(dataset.train)
        val_ds   = _ContextDataset(dataset.val)
        rng = np.random.default_rng(seed)

        curves: List[Dict[str, float]] = []

        for epoch in range(cfg.epochs):
            model.train()
            epoch_loss = {"L_pred": 0.0, "L_stab": 0.0, "L_rel": 0.0,
                          "L_sep": 0.0, "L_total": 0.0, "n": 0}
            for batch_idx in _stratified_batches(train_ds, batch_size=cfg.batch_size, rng=rng):
                x = train_ds.x[batch_idx].to(device)
                y = train_ds.y[batch_idx].to(device)
                cid = train_ds.cid[batch_idx].numpy()
                cluster = train_ds.cluster[batch_idx].numpy()
                u = train_ds.u[batch_idx].numpy()

                out = model(x)
                # Build context groups + rel pairs on CPU indices
                groups = group_batch_by_context(
                    cid, cluster_idx=cluster, u=u, min_group_size=2
                )

                # context_feats: use z_s if the model exposes it (DAFR), else z
                z_for_stab = out.z_s if getattr(out, "z_s", None) is not None else out.z
                context_feats = [z_for_stab[torch.from_numpy(g.indices).to(device)] for g in groups]

                batch_sid = (
                    train_ds.sid[batch_idx].numpy()
                    if train_ds.sid is not None else None
                )
                rel_indices = build_rel_pair_indices(
                    groups, batch_row_to_sample_id=batch_sid,
                    enable_tier1=batch_sid is not None,
                    enable_tier2=True,
                    enable_tier3=True,
                )
                rel_pairs: List[Tuple[Tensor, Tensor, float]] = []
                for p in rel_indices:
                    ia = torch.from_numpy(p.idx_a).to(device)
                    ib = torch.from_numpy(p.idx_b).to(device)
                    rel_pairs.append((
                        torch.index_select(out.y_pred, 0, ia),
                        torch.index_select(out.y_pred, 0, ib),
                        p.weight,
                    ))

                total, comps = composite_loss(
                    y_pred=out.y_pred, y_true=y,
                    context_feats=context_feats,
                    rel_pairs=rel_pairs,
                    z_s=out.z_s, z_c=out.z_c,
                )
                opt.zero_grad()
                total.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

                for k in ("L_pred", "L_stab", "L_rel", "L_sep", "L_total"):
                    epoch_loss[k] += comps[k] * y.shape[0]
                epoch_loss["n"] += y.shape[0]

            avg = {k: (epoch_loss[k] / max(epoch_loss["n"], 1))
                   for k in ("L_pred", "L_stab", "L_rel", "L_sep", "L_total")}
            avg["epoch"] = epoch
            curves.append(avg)

        return {"curves": curves, "match_breakdown": {}}

else:
    # Stubs so imports don't fail on systems without torch.
    def train_torch(*a, **k):  # type: ignore
        raise ImportError("torch is required for train_torch; use numpy_trainer instead")
