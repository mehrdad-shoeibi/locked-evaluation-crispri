"""x_cell same-pass running-sum accumulator (ICLR Phase-0).

Implements the construction-matched external expression comparator x_cell
(Lock section 9) as a streaming accumulator so it can be built in the SAME pass that
reads raw counts for Geneformer tokenization (preflight section E2, NEEDS_ADAPTER),
with NO second pass over the multi-GB h5ad.

Formula (reused verbatim from MADA phasec_rpe1_full_feature_extraction_v1.py:61-64,
`cp10k_log1p_mean`), computed in float64 with a divide-by-zero guard:
    per-cell:  v = log1p(counts / per_cell_total * 1e4)
    mu_construct = mean over the construct's cells of v      (= sum / count)
    mu_NT        = mean over all global-NT cells of v
    x_cell       = mu_construct - mu_NT      (external NATIVE gene space)

The VCC-HVG symbol map / zero-fill / +/-10 clip are applied ONCE at finalize, at
adapter level, on the aggregate delta (E3), reusing the manuscript contract
(scripts/phase11b_classical_transfer/run.py:80-113). The accumulator itself is
dataset-agnostic and stays in the external native gene space.

CPU only. No Geneformer, no GPU, no labels.
"""
from __future__ import annotations
import os, pickle, tempfile
import numpy as np


# --------------------------------------------------------------------------- #
# formula
# --------------------------------------------------------------------------- #
def cp10k_log1p_rows(counts):
    """Per-cell log1p(CP10K). counts: (n, g). Returns float64 (n, g).
    Matches MADA cp10k_log1p_mean(counts) == cp10k_log1p_rows(counts).mean(axis=0)."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    return np.log1p(counts / total * 10000.0)


def _neumaier_add(s, c, x):
    """Neumaier (Kahan-Babuska) compensated vector add: (s, c) += x. Returns (s', c')."""
    t = s + x
    cond = np.abs(s) >= np.abs(x)
    c = c + np.where(cond, (s - t) + x, (x - t) + s)
    return t, c


class XCellError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# accumulator
# --------------------------------------------------------------------------- #
class XCellAccumulator:
    """Streaming per-construct + global-NT mean-of-log1p(CP10K) accumulator.

    Memory scales as (#constructs + 1) * n_genes (three running arrays each), plus the
    current chunk. A construct spread over many chunks yields the same result as if all
    its cells arrived at once.

    checkpoint_dir is a PARAMETER (constructor). No path is hard-coded: the validation
    run points it at a local dir; the real run points it at the SSD root. Passing None
    disables checkpointing.
    """

    NT_KEY = "__NT_POOL__"

    def __init__(self, n_genes, checkpoint_dir=None, checkpoint_every=1):
        if int(n_genes) <= 0:
            raise XCellError(f"n_genes must be positive, got {n_genes}")
        self.n_genes = int(n_genes)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = int(checkpoint_every)
        # per-construct: plain sum, neumaier sum, neumaier comp, count
        self._sum = {}      # cid -> (g,) float64  plain running sum
        self._nsum = {}     # cid -> (g,) float64  neumaier running sum
        self._ncomp = {}    # cid -> (g,) float64  neumaier compensation
        self._count = {}    # cid -> int
        # global NT
        self._nt_sum = np.zeros(self.n_genes, dtype=np.float64)
        self._nt_nsum = np.zeros(self.n_genes, dtype=np.float64)
        self._nt_ncomp = np.zeros(self.n_genes, dtype=np.float64)
        self._nt_count = 0
        self._consumed = set()          # authoritative set of consumed chunk ids
        self._total_cells = 0
        self._updates = 0
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            self._maybe_resume()

    # ---- internal helpers -------------------------------------------------- #
    def _ensure(self, cid):
        if cid not in self._sum:
            self._sum[cid] = np.zeros(self.n_genes, dtype=np.float64)
            self._nsum[cid] = np.zeros(self.n_genes, dtype=np.float64)
            self._ncomp[cid] = np.zeros(self.n_genes, dtype=np.float64)
            self._count[cid] = 0

    def _add(self, cid, vec, n):
        self._ensure(cid)
        self._sum[cid] = self._sum[cid] + vec
        self._nsum[cid], self._ncomp[cid] = _neumaier_add(self._nsum[cid], self._ncomp[cid], vec)
        self._count[cid] += int(n)

    def _add_nt(self, vec, n):
        self._nt_sum = self._nt_sum + vec
        self._nt_nsum, self._nt_ncomp = _neumaier_add(self._nt_nsum, self._nt_ncomp, vec)
        self._nt_count += int(n)

    # ---- public API -------------------------------------------------------- #
    def update(self, counts_chunk, construct_ids, nt_mask, chunk_id):
        """Accumulate one chunk. Idempotent per chunk_id (consumed set is authoritative)."""
        if chunk_id in self._consumed:
            return  # already accounted for; double-counting is structurally impossible
        counts_chunk = np.asarray(counts_chunk)
        construct_ids = np.asarray(construct_ids, dtype=object)
        nt_mask = np.asarray(nt_mask, dtype=bool)
        n = counts_chunk.shape[0]
        # ---- safety checks: fail loudly, never silently ----
        if counts_chunk.ndim != 2 or counts_chunk.shape[1] != self.n_genes:
            raise XCellError(f"gene-dim mismatch: chunk has shape {counts_chunk.shape}, expected (*, {self.n_genes})")
        if len(construct_ids) != n or len(nt_mask) != n:
            raise XCellError(f"length mismatch: counts n={n}, construct_ids={len(construct_ids)}, nt_mask={len(nt_mask)}")
        if not np.isfinite(counts_chunk).all():
            raise XCellError("non-finite values in counts chunk")
        if (counts_chunk < 0).any():
            raise XCellError("negative values in counts chunk")

        rows = cp10k_log1p_rows(counts_chunk)  # float64 (n, g)
        # NT
        if nt_mask.any():
            self._add_nt(rows[nt_mask].sum(axis=0), int(nt_mask.sum()))
        # perturbed, grouped by construct
        pmask = ~nt_mask
        if pmask.any():
            cids = construct_ids[pmask]
            prows = rows[pmask]
            # guard: unsafe id normalisation collapsing distinct ids to one str() key
            key_to_orig = {}
            for o in cids:
                k = str(o)
                if k in key_to_orig:
                    if not (key_to_orig[k] == o):
                        raise XCellError(f"construct id collision under str(): "
                                         f"{key_to_orig[k]!r} and {o!r} both map to {k!r}")
                else:
                    key_to_orig[k] = o
            cids_str = cids.astype(str)
            for cid in np.unique(cids_str):
                sel = cids_str == cid
                self._add(str(cid), prows[sel].sum(axis=0), int(sel.sum()))

        self._consumed.add(chunk_id)
        self._total_cells += n
        self._updates += 1
        if self.checkpoint_dir is not None and (self._updates % self.checkpoint_every == 0):
            self._save_checkpoint()

    def finalize(self, use_compensated=False):
        """Return dict: construct_id -> x_cell (g,) delta, plus mu_construct, mu_nt.
        use_compensated selects the Neumaier running sum instead of the plain one."""
        if self._nt_count == 0:
            raise XCellError("finalize() called before any NT cells were observed")
        if len(self._sum) == 0:
            raise XCellError("finalize() called before any perturbed construct was observed")
        if use_compensated:
            mu_nt = (self._nt_nsum + self._nt_ncomp) / self._nt_count
        else:
            mu_nt = self._nt_sum / self._nt_count
        mu_c, x_cell = {}, {}
        for cid, cnt in self._count.items():
            if cnt <= 0:
                raise XCellError(f"construct {cid!r} has zero accumulated cells")
            if use_compensated:
                s = self._nsum[cid] + self._ncomp[cid]
            else:
                s = self._sum[cid]
            mu = s / cnt
            mu_c[cid] = mu
            x_cell[cid] = mu - mu_nt
        return {"x_cell": x_cell, "mu_construct": mu_c, "mu_nt": mu_nt,
                "nt_count": self._nt_count, "counts": dict(self._count)}

    def compensated_max_delta(self):
        """max |plain - compensated| across all finalized means (drift measurement)."""
        d = 0.0
        mu_nt_p = self._nt_sum / self._nt_count
        mu_nt_c = (self._nt_nsum + self._nt_ncomp) / self._nt_count
        d = max(d, float(np.abs(mu_nt_p - mu_nt_c).max()))
        for cid, cnt in self._count.items():
            p = self._sum[cid] / cnt
            c = (self._nsum[cid] + self._ncomp[cid]) / cnt
            d = max(d, float(np.abs(p - c).max()))
        return d

    def get_state(self):
        """Full serialisable accumulator state (for external atomic checkpointing)."""
        return {"n_genes": self.n_genes, "sum": self._sum, "nsum": self._nsum, "ncomp": self._ncomp,
                "count": self._count, "nt_sum": self._nt_sum, "nt_nsum": self._nt_nsum,
                "nt_ncomp": self._nt_ncomp, "nt_count": self._nt_count, "consumed": set(self._consumed),
                "total_cells": self._total_cells, "updates": self._updates}

    def set_state(self, s):
        """Restore from get_state() output."""
        if int(s["n_genes"]) != self.n_genes:
            raise XCellError(f"state gene-dim {s['n_genes']} != {self.n_genes}")
        self._sum = s["sum"]; self._nsum = s["nsum"]; self._ncomp = s["ncomp"]; self._count = s["count"]
        self._nt_sum = s["nt_sum"]; self._nt_nsum = s["nt_nsum"]; self._nt_ncomp = s["nt_ncomp"]
        self._nt_count = s["nt_count"]; self._consumed = set(s["consumed"])
        self._total_cells = s["total_cells"]; self._updates = s["updates"]

    def consumed(self):
        """The authoritative set of consumed (durably committable) work-unit ids."""
        return set(self._consumed)

    def per_construct_counts(self):
        return dict(self._count)

    def trace(self):
        return {"n_genes": self.n_genes, "n_constructs": len(self._count),
                "nt_count": self._nt_count, "total_cells": self._total_cells,
                "chunks_consumed": sorted(self._consumed),
                "per_construct_counts": dict(self._count)}

    # ---- checkpoint / resume ---------------------------------------------- #
    def _ckpt_path(self):
        return os.path.join(self.checkpoint_dir, "xcell_state.pkl")

    def _save_checkpoint(self):
        state = {
            "n_genes": self.n_genes,
            "sum": self._sum, "nsum": self._nsum, "ncomp": self._ncomp, "count": self._count,
            "nt_sum": self._nt_sum, "nt_nsum": self._nt_nsum, "nt_ncomp": self._nt_ncomp,
            "nt_count": self._nt_count, "consumed": self._consumed,
            "total_cells": self._total_cells, "updates": self._updates,
        }
        fd, tmp = tempfile.mkstemp(dir=self.checkpoint_dir, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self._ckpt_path())  # atomic

    def _maybe_resume(self):
        p = self._ckpt_path()
        if not os.path.exists(p):
            return
        with open(p, "rb") as f:
            s = pickle.load(f)
        if int(s["n_genes"]) != self.n_genes:
            raise XCellError(f"checkpoint gene-dim {s['n_genes']} != {self.n_genes}")
        self._sum = s["sum"]; self._nsum = s["nsum"]; self._ncomp = s["ncomp"]; self._count = s["count"]
        self._nt_sum = s["nt_sum"]; self._nt_nsum = s["nt_nsum"]; self._nt_ncomp = s["nt_ncomp"]
        self._nt_count = s["nt_count"]; self._consumed = set(s["consumed"])
        self._total_cells = s["total_cells"]; self._updates = s["updates"]


# --------------------------------------------------------------------------- #
# finalize adapter: map native delta -> VCC delta-HVG namespace (E3 order)
# --------------------------------------------------------------------------- #
def map_and_clip_to_vcc_hvg(x_cell_native, native_symbols, vcc_hvg_symbols):
    """Apply the manuscript symbol map -> zero-fill -> clip +/-10 LAST, on the aggregate
    delta. Mirrors scripts/phase11b_classical_transfer/run.py:80-113 (symbol contract).
    Returns (mapped (H,) float32, n_mapped)."""
    native_symbols = np.asarray(native_symbols).astype(str)
    vcc_hvg_symbols = np.asarray(vcc_hvg_symbols).astype(str)
    sym_to_idx = {}
    for i, s in enumerate(native_symbols):
        if s not in sym_to_idx:
            sym_to_idx[s] = i
    hvg_to_native = np.array([sym_to_idx.get(s, -1) for s in vcc_hvg_symbols], dtype=np.int64)
    present = hvg_to_native >= 0
    out = np.zeros(len(vcc_hvg_symbols), dtype=np.float32)
    if present.any():
        out[np.where(present)[0]] = np.asarray(x_cell_native, dtype=np.float32)[hvg_to_native[present]]
    out = np.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
    out = np.clip(out, -10.0, 10.0)          # clip LAST, on the aggregate
    return out, int(present.sum())
