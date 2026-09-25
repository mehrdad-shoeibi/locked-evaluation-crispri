"""(g) FM head builders — construct (do NOT fit). Lock §11.

Classical factories are App. F verbatim (scripts/phase11_strong_baselines/run.py:123-143); that
script is import-safe (has __main__ guard) but its factories are trivial sklearn constructors, so
they are reproduced verbatim here with provenance to avoid a heavy import. SPSM is built via the
manuscript's own builder (src/models/families.build_model) with ONLY the input width adapted
(SPSM_V4_ADAPTATION = INPUT_WIDTH_ONLY): 768 for z, 772 for [z; m̃]. Representation width = 256.
"""
import sys
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 2)))
REPO = _ROOT
sys.path.insert(0, REPO)

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


# ---- App. F classical factories (verbatim, run.py:123-143) ----------------- #
def make_ridge(seed=0, scale=True):
    return ("Ridge", Ridge(alpha=1.0, random_state=seed), scale)   # scale -> StandardScaler at fit


def make_rf(seed=0):
    return ("RandomForest", RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=2,
        n_jobs=8, random_state=seed,
    ), False)


def make_hgb(seed=0):
    return ("HistGradientBoosting", HistGradientBoostingRegressor(
        max_iter=500, max_depth=6, learning_rate=0.05,
        l2_regularization=0.0, random_state=seed,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
    ), False)


def make_ridge_scaler():
    """The StandardScaler App. F fits for Ridge (with_mean/with_std)."""
    return StandardScaler(with_mean=True, with_std=True)


# ---- SPSM V4 (manuscript builder; input width only) ------------------------ #
def build_spsm(input_dim):
    """build_model('SPSM', input_dim, {}) — everything inherited except input width.
    768 for z, 772 for [z; m̃]. Representation width is 256 (Lock §11 amended; M6)."""
    from src.models.families import build_model
    if input_dim not in (768, 772):
        raise ValueError(f"SPSM input width must be 768 (z) or 772 ([z; m̃]); got {input_dim}")
    return build_model("SPSM", input_dim, model_cfg={})


def spsm_rep_width(model):
    """The SPSM encoder's representation width (out_dim). Must be 256, not 128 (M6)."""
    return int(model.encoder.out_dim)
