"""Zero-shot external scoring wiring (Phase-4 prep; build only). Lock §22.

A VCC-fit head is applied to an external feature block with NO target-side calibration: no refit, no
per-screen rescaling, no label-derived transform. External features are prepared exactly as in VCC
(the frozen §6B StandardScaler for the z block; the frozen VCC-train magnitude z-score for m̃). This
module only wires prediction + the external primary metric (Spearman vs the harmonized endpoint); the
paired Δρ cluster bootstrap lives in bootstrap_wiring.external_paired_delta_rho.
"""
import numpy as np
from scipy.stats import spearmanr


def zeroshot_predict(model, X_external):
    """Apply an already-fitted VCC model to an external block. No calibration of any kind."""
    return np.asarray(model.predict(np.asarray(X_external)))


def external_spearman(y_pred, y_endpoint):
    """External primary metric: Spearman rho (average-tie) vs the harmonized endpoint (Lock §20)."""
    yp, ye = np.asarray(y_pred, float), np.asarray(y_endpoint, float)
    if np.std(yp) == 0 or np.std(ye) == 0:
        return float("nan")
    r, _ = spearmanr(yp, ye)
    return float(r) if r is not None else float("nan")
