"""(d) §6B FM representation-block preprocessing — TRAIN_STANDARDIZE.

sklearn StandardScaler fitted on VCC TRAIN ONLY, frozen, applied unchanged to validation,
test, RPE1, K562-essential. z gets its own scaler; C1 its own; C2 its own; C3 uses the SAME
scaler fitted on unshuffled z. No unit-L2 normalisation anywhere; no scaler ever fit on val,
test, or either external screen. (Lock §6B.)
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_block_scaler(train_block):
    """Fit a StandardScaler on VCC-train rows only (Lock §6B). Returns the frozen scaler.
    StandardScaler's zero-variance guard maps near-constant coords' scale_ -> 1."""
    X = np.asarray(train_block, dtype=np.float64)
    return StandardScaler(with_mean=True, with_std=True).fit(X)


def near_constant_count(scaler, tol=1e-12):
    """Number of coordinates StandardScaler treated as (near-)constant (var_ <= tol).
    For these, scale_ is set to 1 by StandardScaler's guard rather than a tiny std."""
    return int((np.asarray(scaler.var_) <= tol).sum())


def apply_scaler(scaler, block):
    """Apply a frozen train-fitted scaler to any split/screen (never re-fit)."""
    return scaler.transform(np.asarray(block, dtype=np.float64)).astype(np.float32)
