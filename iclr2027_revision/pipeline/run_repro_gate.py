"""L10 Part A — reproduction gate. Refit the manuscript's own baselines on its own features/HVG/labels
and reproduce Table 1: MAG-LINEAR +0.208, RF[x;m̃] +0.370, MAG-SPSM +0.225±0.004, SPSM +0.082±0.010.
Deterministic |Δ|<=0.005; stochastic: manuscript value within mean±2SD."""
import os, sys, json, time
import numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 2)))
sys.path.insert(0, os.path.join(_ROOT, "iclr2027_revision/pipeline"))
import phase1_harness as H
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
t0 = time.time()
base = H.load_base_shards()
tr, va, te = base["train"], base["val"], base["test"]
y_tr, y_te = H.concat_y(tr), H.concat_y(te)
print(f"[data] train={len(y_tr)} val={sum(s.y.shape[0] for s in va)} test={len(y_te)} | "
      f"y_train mean={y_tr.mean():.3f} (expect ~4.23) sd={y_tr.std():.3f}", flush=True)

stats = H.fit_mtilde_stats(tr)
mt_tr = np.concatenate(H.mtilde(tr, stats), 0); mt_te = np.concatenate(H.mtilde(te, stats), 0)
x_tr, x_te = H.concat(tr, "x"), H.concat(te, "x")
xm_tr = np.concatenate([x_tr, mt_tr], 1); xm_te = np.concatenate([x_te, mt_te], 1)
R = {}

# --- MAG-LINEAR: Ridge(alpha=1)+StandardScaler on 4-D m̃ (deterministic) ---
sc = StandardScaler().fit(mt_tr)
rg = Ridge(alpha=1.0, random_state=0).fit(sc.transform(mt_tr), y_tr)
mag_lin = H.r2_pooled(y_te, rg.predict(sc.transform(mt_te)))
R["MAG_LINEAR"] = {"test_r2": mag_lin, "manuscript": 0.208, "delta": abs(mag_lin - 0.208),
                   "pass": abs(mag_lin - 0.208) <= 0.005}
print(f"[MAG-LINEAR] test R2={mag_lin:+.4f} vs +0.208 (|Δ|={abs(mag_lin-0.208):.4f})", flush=True)

# --- RF[x;m̃]: seeds {0,1,2} ---
rf_r2 = []
for s in [0, 1, 2]:
    rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=2, n_jobs=8, random_state=s).fit(xm_tr, y_tr)
    rf_r2.append(H.r2_pooled(y_te, rf.predict(xm_te)))
rf_r2 = np.array(rf_r2)
R["RF_X_MTILDE"] = {"test_r2_mean": float(rf_r2.mean()), "test_r2_sd": float(rf_r2.std(ddof=1)),
                    "per_seed": rf_r2.tolist(), "manuscript": 0.370,
                    "pass": bool(abs(rf_r2.mean() - 0.370) <= 2 * rf_r2.std(ddof=1) or (rf_r2.min() <= 0.370 <= rf_r2.max()))}
print(f"[RF x;m̃] test R2={rf_r2.mean():+.4f}±{rf_r2.std(ddof=1):.4f} vs +0.370", flush=True)

# --- SPSM (x) and MAG-SPSM ([x;m̃]): seeds {0,1,2} ---
def spsm_test_r2(feat_tr_shards, feat_va_shards, feat_te_shards, seed):
    res = H.train_spsm(H.Trio(feat_tr_shards, feat_va_shards, feat_te_shards), seed)
    yh = np.concatenate([res.predict(s.x) for s in feat_te_shards if s.y.shape[0] > 0])
    return H.r2_pooled(y_te, yh)

mt_va = H.mtilde(va, stats)
xm_tr_sh = H.make_feature_shards(tr, [np.concatenate([s.x, m], 1) for s, m in zip(tr, H.mtilde(tr, stats))])
xm_va_sh = H.make_feature_shards(va, [np.concatenate([s.x, m], 1) for s, m in zip(va, mt_va)])
xm_te_sh = H.make_feature_shards(te, [np.concatenate([s.x, m], 1) for s, m in zip(te, H.mtilde(te, stats))])

for arm, (trs, vas, tes), ref, sd_ref in [
    ("SPSM", (tr, va, te), 0.082, 0.010),
    ("MAG_SPSM", (xm_tr_sh, xm_va_sh, xm_te_sh), 0.225, 0.004)]:
    r2s = []
    for s in [0, 1, 2]:
        ts = time.time(); r2 = spsm_test_r2(trs, vas, tes, s)
        r2s.append(r2); print(f"[{arm} seed {s}] test R2={r2:+.4f}  ({time.time()-ts:.0f}s)", flush=True)
    r2s = np.array(r2s)
    R[arm] = {"test_r2_mean": float(r2s.mean()), "test_r2_sd": float(r2s.std(ddof=1)),
              "per_seed": r2s.tolist(), "manuscript": ref,
              "pass": bool(abs(r2s.mean() - ref) <= 2 * r2s.std(ddof=1) or (r2s.min() <= ref <= r2s.max()))}
    print(f"[{arm}] test R2={r2s.mean():+.4f}±{r2s.std(ddof=1):.4f} vs {ref:+.3f}±{sd_ref:.3f}", flush=True)

R["GATE"] = all(R[k]["pass"] for k in ["MAG_LINEAR", "RF_X_MTILDE", "SPSM", "MAG_SPSM"])
os.makedirs(os.path.join(_ROOT, "iclr2027_revision/phase1"), exist_ok=True)
json.dump(R, open(os.path.join(_ROOT, "iclr2027_revision/phase1/repro_gate.json"), "w"), indent=2, default=str)
print("REPRO " + json.dumps(R, default=str), flush=True)
print(f"[done] {time.time()-t0:.0f}s | GATE={'PASS' if R['GATE'] else 'FAIL'}", flush=True)
