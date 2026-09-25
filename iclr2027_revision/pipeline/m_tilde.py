"""(b) m̃ recompute over the authoritative delta_hvg artifact + external oracle.

m̃ = four magnitude scalars [mean|x|, ||x||_2 (RAW, per code — M3), max|x|, std(x)], z-scored
on VCC TRAIN stats only, computed over `selected_genes.npy` (Lock §6, §17, §18).

Source of m̃ is the MANUSCRIPT's own code, imported (not reimplemented):
  src/training/magnitude_augment.py :: _row_magnitude_features / fit_magnitude_stats / MagnitudeStats

Externals use `x_release` (NOT `x_cell`): m̃ is the manuscript's magnitude feature, defined over the
released harmonized profile. No m̃(x_cell) variant exists in the Lock (verified L9). Non-finite /
zero-fill / clip follows the manuscript contract (scripts/phase11b_classical_transfer/run.py:80-113):
symbol-map -> zero-fill missing HVG -> nan_to_num(nan=0, posinf=10, neginf=-10) -> clip ±10.
"""
import os as _o, sys as _s
_r = _o.path.dirname(_o.path.abspath(__file__))
while _r != _o.path.dirname(_r) and not _o.path.exists(_o.path.join(_r, "iga_paths.py")):
    _r = _o.path.dirname(_r)
if _r not in _s.path:
    _s.path.insert(0, _r)
import iga_paths
import os, sys, glob
import numpy as np, pandas as pd, anndata as ad

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 2)))
REPO = _ROOT
sys.path.insert(0, REPO)
from src.training.magnitude_augment import _row_magnitude_features, fit_magnitude_stats, MagnitudeStats

ROOT = iga_paths.IGA_NEURIPS
DELTA_HVG_SHARDS = ROOT + "/cache/vcc/variants/delta_hvg/shards"
MAG_STATS_CSV = ROOT + "/results/phase9c_verify/magnitude_feature_stats.csv"
FEATURES = ["mean_abs_x", "l2_norm_x", "max_abs_x", "std_x"]  # order of _row_magnitude_features cols
NB = {"RPE1": ROOT + "/data/replogle/raw/rpe1_normalized_bulk_01.h5ad",
      "K562": ROOT + "/data/replogle/raw/K562_essential_normalized_bulk_01.h5ad"}


# ---- VCC delta-HVG x per split (cached authoritative shards) ---------------- #
def load_vcc_x(split):
    files = sorted(glob.glob(f"{DELTA_HVG_SHARDS}/{split}/*.npz"))
    if not files:
        raise FileNotFoundError(f"no delta_hvg shards for split {split!r}")
    xs, ids = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        xs.append(np.asarray(z["x"], dtype=np.float32))
        ids.extend([str(s) for s in z["sample_ids"]])
    return np.concatenate(xs, axis=0), np.array(ids)


# ---- external x_release in VCC-HVG namespace (manuscript contract) ---------- #
def build_x_release_mapped(screen, vcc_hvg_symbols):
    """Return (X_mapped (n,2000) float32, n_nonfinite_repaired, obs_ids). Mirrors
    run.py:80-113: map by symbol -> zero-fill missing -> nan_to_num -> clip ±10."""
    a = ad.read_h5ad(NB[screen])
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    X = np.asarray(X, dtype=np.float32)
    sym = a.var.gene_name.astype(str).to_numpy()
    obs_ids = a.obs.index.astype(str).to_numpy()
    sym_to_idx = {}
    for i, s in enumerate(sym):
        if s not in sym_to_idx:
            sym_to_idx[s] = i
    hvg_to_src = np.array([sym_to_idx.get(str(s), -1) for s in vcc_hvg_symbols], dtype=np.int64)
    present = hvg_to_src >= 0
    out = np.zeros((X.shape[0], len(vcc_hvg_symbols)), dtype=np.float32)
    out[:, np.where(present)[0]] = X[:, hvg_to_src[present]]
    n_nonfinite = int((~np.isfinite(out)).sum())            # repaired by nan_to_num below
    out = np.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
    out = np.clip(out, -10.0, 10.0)
    return out, n_nonfinite, obs_ids


# ---- m̃ recompute + z-score (train-only) ----------------------------------- #
def m_tilde_raw(X):
    """4 magnitude scalars via the manuscript's own function (raw ||x||_2, M3)."""
    return _row_magnitude_features(np.asarray(X, dtype=np.float32))


def m_tilde_eq1(X):
    """Eq.1 variant with ||x||_2/sqrt(d) on column 1 (for the M3 pre-z-score comparison)."""
    M = m_tilde_raw(X).copy()
    M[:, 1] = M[:, 1] / np.sqrt(X.shape[1])
    return M


def fit_train_zscore(train_X):
    """Fit MagnitudeStats (mean/std, ddof=0, zero-var guard) on VCC train m̃ only."""
    class _S:  # minimal shard shim so we can reuse fit_magnitude_stats
        def __init__(self, x): self.x = x
    return fit_magnitude_stats([_S(np.asarray(train_X, dtype=np.float32))])


# ---- aggregate-stat oracle vs the manuscript's stored reference ------------- #
def aggregate_stats(M):
    """Per-feature mean/std/min/max over rows (ddof=0), matching verify_magnitude.py."""
    return {"mean": M.mean(0), "std": M.std(0, ddof=0), "min": M.min(0), "max": M.max(0)}


def load_reference_stats():
    return pd.read_csv(MAG_STATS_CSV)
