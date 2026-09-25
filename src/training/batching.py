"""Context batching and relational-pair construction (framework-agnostic).

This module owns the two pieces of trainer wiring that must be locked *once*
and reused identically by every model family so ablations are clean:

1. `group_batch_by_context(batch) -> List[ContextGroup]`

   Batches are grouped by `context_id`. Each `ContextGroup` carries row
   indices, the context id, and references to the batch's x/y slabs. The
   trainer uses these groups to:

   - build `context_feats = [z[idx] for g in groups for idx in [g.indices]]`
     (list of per-context representation tensors) for `L_stab`;
   - slice per-context predictions for `L_rel` matched pairs.

   Minimum group size is enforced via `min_group_size`. Contexts with too
   few samples in the current batch are *dropped* from stability / relation
   computation (not from the prediction loss); this is the right behavior
   for small-batch edge cases and is logged.

2. `build_rel_pairs(groups, pred_fn, u_fn, *, tiers, k_nn) -> List[RelPair]`

   Constructs matched (context_a, context_b) prediction tuples following
   the SPEC-locked three-tier fallback:

     Tier 1 (weight=1.00): exact index alignment — two contexts share the
         same sample indices in the batch (i.e. the same underlying u).
     Tier 2 (weight=0.50): partial cluster/group alignment — the cluster
         (or latent-bucket) id matches for a subset of rows in each context.
     Tier 3 (weight=0.25): nearest-neighbor in u-space — for each row in
         context a, pick its k-NN in context b (over u or a supplied
         auxiliary key) and average the prediction pairs.

   The return type is a list of `RelPair(pred_a, pred_b, weight, tier)`
   whose `pred_a`/`pred_b` are 1-D arrays/tensors of matched predictions.

   Tier weights are the SPEC-locked values (1.00, 0.50, 0.25). They are
   *not* re-derived from pair counts; they encode our trust in the quality
   of the match, not its quantity.

The functions here are pure NumPy and accept both torch tensors and numpy
arrays for `pred_fn` outputs — the trainer is responsible for keeping the
returned arrays on the correct autograd graph. For the torch trainer, we
operate on indices only (NumPy) and the trainer materializes the pair
tensors via `torch.index_select` before handing them to the loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Context grouping
# ---------------------------------------------------------------------------
@dataclass
class ContextGroup:
    """Indices of a single context within the current batch."""

    context_id: int
    indices: np.ndarray          # (n_ctx,) row indices into the batch
    cluster_idx: Optional[np.ndarray] = None   # (n_ctx,) if available
    u: Optional[np.ndarray] = None             # (n_ctx, d_u) if available


def group_batch_by_context(
    context_ids: np.ndarray,
    *,
    cluster_idx: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    min_group_size: int = 2,
) -> List[ContextGroup]:
    """Return one ContextGroup per distinct context_id, filtered by size.

    Groups with fewer than `min_group_size` rows are dropped. This matters
    for MMD (degenerate for n<2 per context) and for SoftSpearman ranks.
    """
    context_ids = np.asarray(context_ids).reshape(-1)
    groups: List[ContextGroup] = []
    for cid in np.unique(context_ids):
        idx = np.where(context_ids == cid)[0]
        if idx.size < min_group_size:
            continue
        groups.append(ContextGroup(
            context_id=int(cid),
            indices=idx,
            cluster_idx=None if cluster_idx is None else cluster_idx[idx],
            u=None if u is None else u[idx],
        ))
    return groups


# ---------------------------------------------------------------------------
# Relational pair construction
# ---------------------------------------------------------------------------
# SPEC-locked tier weights. DO NOT reweight by sample count — these encode
# match-quality, not sample mass.
TIER_WEIGHTS: Dict[int, float] = {1: 1.00, 2: 0.50, 3: 0.25}


@dataclass
class RelPairIndices:
    """Index-only specification of one matched pair between two contexts.

    The trainer materializes the prediction arrays:
        pred_a = y_pred[idx_a]
        pred_b = y_pred[idx_b]
    at training time (keeps torch autograd intact).
    """

    context_a: int
    context_b: int
    idx_a: np.ndarray   # (m,)
    idx_b: np.ndarray   # (m,) same length
    weight: float
    tier: int


def _tier1_exact(
    groups: Sequence[ContextGroup],
    batch_row_to_sample_id: Optional[np.ndarray],
) -> List[RelPairIndices]:
    """Tier 1: exact sample-id alignment between contexts.

    Requires a `sample_id` per batch row (e.g. the underlying u-row id).
    For synthetic data where each context is an independent sample of u,
    this yields no matches. For real data where the *same* perturbation is
    measured in multiple cell lines, tier-1 matches are plentiful.
    """
    if batch_row_to_sample_id is None:
        return []
    pairs: List[RelPairIndices] = []
    for i in range(len(groups)):
        sid_i = batch_row_to_sample_id[groups[i].indices]
        for j in range(i + 1, len(groups)):
            sid_j = batch_row_to_sample_id[groups[j].indices]
            # intersect
            common, ii, jj = np.intersect1d(sid_i, sid_j, return_indices=True)
            if common.size == 0:
                continue
            pairs.append(RelPairIndices(
                context_a=groups[i].context_id,
                context_b=groups[j].context_id,
                idx_a=groups[i].indices[ii],
                idx_b=groups[j].indices[jj],
                weight=TIER_WEIGHTS[1],
                tier=1,
            ))
    return pairs


def _tier2_cluster(
    groups: Sequence[ContextGroup],
    already_matched: Sequence[RelPairIndices],
) -> List[RelPairIndices]:
    """Tier 2: match rows by cluster/bucket id, excluding tier-1 pairs.

    For each unordered pair of contexts with no tier-1 match, we pair up
    rows that share cluster_idx. If context A has n_k rows in cluster k
    and B has m_k, we align the first min(n_k, m_k) of each (order is
    stable by np.where).
    """
    matched_pairs = {(p.context_a, p.context_b) for p in already_matched}
    pairs: List[RelPairIndices] = []
    for i in range(len(groups)):
        gi = groups[i]
        if gi.cluster_idx is None:
            continue
        for j in range(i + 1, len(groups)):
            gj = groups[j]
            if gj.cluster_idx is None:
                continue
            if (gi.context_id, gj.context_id) in matched_pairs:
                continue
            idx_a: list[int] = []
            idx_b: list[int] = []
            for c in np.unique(np.concatenate([gi.cluster_idx, gj.cluster_idx])):
                ai = gi.indices[gi.cluster_idx == c]
                bj = gj.indices[gj.cluster_idx == c]
                m = min(ai.size, bj.size)
                if m > 0:
                    idx_a.extend(ai[:m].tolist())
                    idx_b.extend(bj[:m].tolist())
            if not idx_a:
                continue
            pairs.append(RelPairIndices(
                context_a=gi.context_id,
                context_b=gj.context_id,
                idx_a=np.asarray(idx_a, dtype=np.int64),
                idx_b=np.asarray(idx_b, dtype=np.int64),
                weight=TIER_WEIGHTS[2],
                tier=2,
            ))
    return pairs


def _tier3_knn(
    groups: Sequence[ContextGroup],
    already_matched: Sequence[RelPairIndices],
    *,
    k_nn: int = 1,
) -> List[RelPairIndices]:
    """Tier 3: nearest-neighbor in u-space, excluding covered pairs.

    For each remaining unordered context pair, match every row in A to its
    k-NN in B by Euclidean distance in u. We emit exactly one pair per
    row-in-A (with k_nn=1) to avoid quadratic blow-up; with k_nn>1, emit
    k_nn pairs per row-in-A. Requires both groups to carry `u`.
    """
    matched = {(p.context_a, p.context_b) for p in already_matched}
    pairs: List[RelPairIndices] = []
    for i in range(len(groups)):
        gi = groups[i]
        if gi.u is None:
            continue
        for j in range(i + 1, len(groups)):
            gj = groups[j]
            if gj.u is None:
                continue
            if (gi.context_id, gj.context_id) in matched:
                continue
            # pairwise squared distances (n_a, n_b)
            aa = (gi.u * gi.u).sum(axis=1, keepdims=True)
            bb = (gj.u * gj.u).sum(axis=1, keepdims=True).T
            d2 = aa + bb - 2.0 * gi.u @ gj.u.T
            k = min(k_nn, gj.u.shape[0])
            # take top-k smallest per row
            nn_idx = np.argsort(d2, axis=1)[:, :k]           # (n_a, k)
            idx_a = np.repeat(gi.indices, k)
            idx_b = gj.indices[nn_idx.reshape(-1)]
            pairs.append(RelPairIndices(
                context_a=gi.context_id,
                context_b=gj.context_id,
                idx_a=idx_a,
                idx_b=idx_b,
                weight=TIER_WEIGHTS[3],
                tier=3,
            ))
    return pairs


def build_rel_pair_indices(
    groups: Sequence[ContextGroup],
    *,
    batch_row_to_sample_id: Optional[np.ndarray] = None,
    enable_tier1: bool = True,
    enable_tier2: bool = True,
    enable_tier3: bool = True,
    k_nn: int = 1,
) -> List[RelPairIndices]:
    """Build the full ordered tier-1 → tier-2 → tier-3 pair list.

    Returns pair *indices*. The trainer combines them with current y_pred
    to produce the tensors consumed by `relation_softspearman`. Tiers are
    applied in strict order; a pair covered by tier-i is NOT duplicated
    in tier-(i+1).
    """
    pairs: List[RelPairIndices] = []
    if enable_tier1:
        pairs.extend(_tier1_exact(groups, batch_row_to_sample_id))
    if enable_tier2:
        pairs.extend(_tier2_cluster(groups, pairs))
    if enable_tier3:
        pairs.extend(_tier3_knn(groups, pairs, k_nn=k_nn))
    return pairs


def tier_counts(pairs: Sequence[RelPairIndices]) -> Dict[int, int]:
    """{tier -> count} for logging the match breakdown."""
    out = {1: 0, 2: 0, 3: 0}
    for p in pairs:
        out[p.tier] = out.get(p.tier, 0) + 1
    return out
