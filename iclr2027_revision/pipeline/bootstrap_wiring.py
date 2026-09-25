"""(f) Bootstrap wiring — REUSE, do not reimplement. Lock §21.

The manuscript's bootstrap scripts are NOT import-safe: `scripts/bootstrap_ci/run_bootstrap.py`
runs the full bootstrap and calls `COMP_DIR.mkdir(...)` at module import; `scripts/v4/.../run_arm1_v2.py`
and `run_arm2.py` likewise execute + write on import. So the verified pure functions are lifted
VERBATIM here (byte-identical bodies) with provenance + SHA rather than reimplemented, and the
call-site pattern is reproduced exactly. No new bootstrap algorithm is introduced.

Provenance (sha256):
  VCC row-level paired ΔR²  : scripts/bootstrap_ci/run_bootstrap.py
      historical 644dac6fe9e1dca9692e5e62df11952eca4f13e81c4539c8e69c27fc2a50c794;
      current    98556b76977d7a5a6cd6ff42e40200e8a4b9b06e63386f418eafc0515bc0040d
      (B=10000, seed=42, pct 2.5/97.5). The audited difference between these two revisions
      affected path resolution and import infrastructure and did not change the statistical
      procedure; B, the RNG seed and the percentile interval are unchanged.
  External paired Δρ (RPE1) : scripts/v4/arm1_rpe1_harmonized_v2/run_arm1_v2.py
      a8ad9fa104cc3976ec710b2cdf4ea8ea9ef0569a7c9748c3cb27d0b811772610  (B=10000, seed=20260901)
  External paired Δρ (K562) : scripts/v4/arm2_k562_harmonized_v1/run_arm2.py
      b6261bcbc901420222afbcafcf840069ad7f05c531564fb84a7ff8feb068eac2  (B=10000, seed=20260902)
"""
import numpy as np
from scipy.stats import spearmanr, rankdata

VCC_B, VCC_SEED = 10000, 42
EXT_B = 10000
EXT_SEED = {"RPE1": 20260901, "K562": 20260902}


# ===== VCC row-level paired ΔR²  (run_bootstrap.py:30-52, 171-194 verbatim) ===== #
def r2_vec(y_true, y_pred):
    """Pooled R²; if denom=0, return NaN (skipped)."""            # run_bootstrap.py:30-35
    sst = ((y_true - y_true.mean()) ** 2).sum()
    if sst == 0:
        return np.nan
    return 1.0 - ((y_true - y_pred) ** 2).sum() / sst


def spear_vec(y_true, y_pred):                                     # run_bootstrap.py:37-41
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    r, _ = spearmanr(y_true, y_pred)
    return float(r) if r is not None else np.nan


def _summarize(deltas):                                           # run_bootstrap.py:43-52 (CI part)
    arr = np.asarray(deltas)
    nan_skipped = int(np.isnan(arr).sum())
    arr = arr[~np.isnan(arr)]
    sign_stab_pos = float((arr > 0).mean()) if arr.size else float("nan")
    return {"median": float(np.median(arr)), "ci_lo": float(np.percentile(arr, 2.5)),
            "ci_hi": float(np.percentile(arr, 97.5)),
            "sign_stab": float(max(sign_stab_pos, 1.0 - sign_stab_pos)),
            "n": int(arr.size), "nan_skipped": nan_skipped}


def vcc_paired_delta_r2(y, preds_a, preds_b, B=VCC_B, seed=VCC_SEED):
    """Row-level paired ΔR² bootstrap: ONE resample index per replicate applied to BOTH members
    (run_bootstrap.py:177-187). preds_a/preds_b: (n,) or list of per-seed (n,) arrays; if per-seed,
    ΔR² is averaged across seeds per replicate. Returns _summarize(deltas)."""
    y = np.asarray(y, float)
    pa = [np.asarray(p, float) for p in (preds_a if isinstance(preds_a, (list, tuple)) else [preds_a])]
    pb = [np.asarray(p, float) for p in (preds_b if isinstance(preds_b, (list, tuple)) else [preds_b])]
    n = y.shape[0]
    rng = np.random.default_rng(seed)
    deltas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        da = np.mean([r2_vec(y_b, p[idx]) for p in pa])
        db = np.mean([r2_vec(y_b, p[idx]) for p in pb])
        deltas[b] = da - db
    return _summarize(deltas)


# ===== External paired Δρ cluster bootstrap (run_arm1_v2.py:157-170,208-212 verbatim) ===== #
def _colcorr(Rm, r):                                             # run_arm1_v2.py:160-163
    Rc = Rm - Rm.mean(0); rc = r - r.mean(); num = Rc.T @ rc; den = np.sqrt((Rc * Rc).sum(0) * (rc @ rc))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def external_paired_delta_rho(P, y, cluster_ids, object_cols, B=EXT_B, seed=20260901):
    """Paired target-gene CLUSTER bootstrap of Δρ (run_arm1_v2.py:156-170). P: (n_rows, n_pred_cols)
    predictions; y: (n_rows,) harmonized endpoint; cluster_ids: (n_rows,) target-gene id per row;
    object_cols: dict object_name -> list of column indices in P. Returns (rho_boot, contrast_fn):
    rho_boot is (B, n_objects); contrast_fn(a,b) gives the paired percentile-95 CI of rho(a)-rho(b)."""
    P = np.asarray(P, float); y = np.asarray(y, float)
    onames = list(object_cols)
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    G = len(uniq)
    tgt_positions = [np.where(inv == g)[0] for g in range(G)]
    order = np.concatenate(tgt_positions)
    tlen = np.array([len(p) for p in tgt_positions])
    tstart = np.concatenate([[0], np.cumsum(tlen)[:-1]])

    def gather(tid):                                             # run_arm1_v2.py:157-159
        lens = tlen[tid]; total = int(lens.sum())
        seg = np.repeat(tstart[tid], lens) + (np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens))
        return order[seg]

    rho_b = np.full((B, len(onames)), np.nan)
    rng = np.random.default_rng(seed)
    for b in range(B):                                           # run_arm1_v2.py:165-170
        tid = rng.integers(0, G, G); rows = gather(tid)
        Rm = rankdata(P[rows], method="average", axis=0); ry = rankdata(y[rows], method="average")
        ch = _colcorr(Rm, ry)
        for oi, name in enumerate(onames):
            rho_b[b, oi] = np.mean(ch[object_cols[name]])

    def contrast(a, b):                                          # run_arm1_v2.py:208-212
        d = rho_b[:, onames.index(a)] - rho_b[:, onames.index(b)]; dv = d[np.isfinite(d)]
        return dict(ci_low=float(np.percentile(dv, 2.5)), ci_high=float(np.percentile(dv, 97.5)),
                    ci_median=float(np.median(dv)), n_valid=int(dv.size), n_skipped=int(B - dv.size))

    return rho_b, contrast
