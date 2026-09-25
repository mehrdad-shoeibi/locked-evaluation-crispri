"""L10 Phase-1 (VCC) harness. Fits models. VCC-only.

Label = the manuscript's target-gene AD response magnitude: shard.y = log1p(anderson_darling_counts)
(src/data/vcc/loader.py:147,393; VCC-train mean ~4.23). NOT y_ad_distance_log1p (= log1p(mean-over-
genes AD) = the forbidden breadth endpoint; == log1p(ad_stat_mean)).

x (VCC) = delta-HVG expression from cache/vcc/variants/delta_hvg/shards (2000-d, unclipped).
m̃ (VCC) = L9 recomputation = _row_magnitude_features(x) z-scored on VCC-train (manuscript's own code).
z = 768-d Geneformer CLS delta from geneformer_row_features_full.parquet (gf_z_emb_*), joined by
(split,batch,target_gene). §6B StandardScaler fit on VCC train only, frozen.
"""
import os as _o, sys as _s
_r = _o.path.dirname(_o.path.abspath(__file__))
while _r != _o.path.dirname(_r) and not _o.path.exists(_o.path.join(_r, "iga_paths.py")):
    _r = _o.path.dirname(_r)
if _r not in _s.path:
    _s.path.insert(0, _r)
import iga_paths
import os, sys, glob, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 2)))
REPO = _ROOT
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/iclr2027_revision/pipeline")
from src.training.magnitude_augment import _row_magnitude_features, fit_magnitude_stats
import hvg_guard as G
import controls as C

ROOT = iga_paths.IGA_NEURIPS
SHARDS = ROOT + "/cache/vcc/variants/delta_hvg/shards"
ZPARQ = (iga_paths.MADA + "/cache/foundation_models/geneformer/"
         "full_feature_extraction_v2/geneformer_row_features_full.parquet")
SPLITS = {"train": "training", "val": "validation", "test": "test"}


class FeatShard:
    """Feature-shard: overrides .x and .z, delegates all context attributes (context_id, cluster_idx,
    u, y, target_genes, sample_ids, context_name, sid) to the real manuscript shard."""
    def __init__(self, real, x):
        self._r = real
        self.x = np.asarray(x, np.float32)
    def __getattr__(self, k):
        return getattr(self._r, k)


def _z_cols():
    import pyarrow.parquet as pq
    cols = [f.name for f in pq.read_schema(ZPARQ)]
    return sorted([c for c in cols if re.fullmatch(r"gf_z_emb_\d+", c)], key=lambda c: int(c.split("_")[-1]))


def load_base_shards():
    """Return {split: [real shard...]} via the manuscript loader (delta, hvg), guarded, with z attached
    per row (joined by (split,batch,target_gene) from geneformer_row_features_full.parquet)."""
    from src.data.vcc.loader import load_vcc_pseudobulk
    G.guard_load_vcc_pseudobulk_args("delta", "hvg")
    zc = _z_cols()
    zdf = pd.read_parquet(ZPARQ, columns=["split", "batch", "target_gene"] + zc)
    zdf[["split", "batch", "target_gene"]] = zdf[["split", "batch", "target_gene"]].astype(str)
    zlook = {(r.split, r.batch, r.target_gene): i for i, r in enumerate(zdf.itertuples(index=False))}
    Z = zdf[zc].to_numpy(np.float32)
    DATA = ROOT + "/data/vcc"
    out = {}
    for sp, full in SPLITS.items():
        shs = load_vcc_pseudobulk(DATA, sp if sp != "val" else "val", feature_transform="delta",
                                  gene_selection="hvg", verbose=False)
        for s in shs:
            batch = str(s.context_name); tg = np.asarray(s.target_genes).astype(str)
            s.z = np.stack([Z[zlook[(full, batch, str(t))]] for t in tg]).astype(np.float32)
        out[sp] = shs
    return out


# ---- feature assembly (train-frozen) ---------------------------------------- #
def concat(shs, attr="x"):
    return np.concatenate([getattr(s, attr) for s in shs if s.y.shape[0] > 0], 0)

def concat_y(shs):
    return np.concatenate([s.y for s in shs if s.y.shape[0] > 0], 0)

def fit_mtilde_stats(train_shs):
    class _S:
        def __init__(s, x): s.x = x
    return fit_magnitude_stats([_S(concat(train_shs, "x"))])

def mtilde(shs, stats):
    """Per-shard train-zscored m̃ (4-D), returned as a list aligned to shards."""
    return [stats.transform(_row_magnitude_features(s.x)) for s in shs]


# ---- metrics (manuscript convention) ---------------------------------------- #
def r2_pooled(y, yh):
    y = np.asarray(y, float); yh = np.asarray(yh, float)
    sst = ((y - y.mean()) ** 2).sum()
    return float("nan") if sst == 0 else float(1.0 - ((y - yh) ** 2).sum() / sst)

def spearman_avgtie(y, yh):
    if np.std(y) == 0 or np.std(yh) == 0:
        return float("nan")
    r, _ = spearmanr(y, yh)
    return float(r) if r is not None else float("nan")


# ---- SPSM training (reuse manuscript trainer) ------------------------------- #
SPSM_CFG = {"epochs": 25, "batch_size": 64, "lr": 1e-3, "lambda_stab": 0.5, "lambda_rel": 0.5,
            "inner_steps_per_outer": 1, "device": "cuda", "pred_loss": "mse",
            "weight_decay": 1e-5, "grad_clip": 1.0}   # Lock §11 SPSM V4; warmup OFF

class Trio:
    def __init__(s, tr, va, te): s.train, s.val, s.test = tr, va, te

def train_spsm(trio_shards, seed):
    from src.training.torch_backend import train_family_torch
    return train_family_torch("SPSM", trio_shards, seed=seed, cfg=dict(SPSM_CFG))

def make_feature_shards(shs, feat_list):
    """Wrap each real shard with an overridden .x (keeps context_id/cluster_idx/u/y for SPSM)."""
    return [FeatShard(s, feat) for s, feat in zip(shs, feat_list)]
