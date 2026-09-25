"""L10 Part A (final) — reproduction gate with corrected MAG-LINEAR (sklearn LinearRegression, per
phase11_strong_baselines/run.py:339) and real-shard SPSM/MAG-SPSM. RF reused from the first gate run
(+0.3704±0.0035, same harness/config, deterministic). Writes repro_gate_final.json."""
import os, sys, json, time
import numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 2)))
sys.path.insert(0, os.path.join(_ROOT, "iclr2027_revision/pipeline"))
import phase1_harness as H
from sklearn.linear_model import LinearRegression
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
OUT = os.path.join(_ROOT, "iclr2027_revision/phase1")
t0 = time.time()
base = H.load_base_shards(); tr, va, te = base["train"], base["val"], base["test"]
y_tr, y_te = H.concat_y(tr), H.concat_y(te)
print(f"[data] train={len(y_tr)} test={len(y_te)} y_train_mean={y_tr.mean():.3f}", flush=True)
stats = H.fit_mtilde_stats(tr)
mt_tr = np.concatenate(H.mtilde(tr, stats), 0); mt_te = np.concatenate(H.mtilde(te, stats), 0)
R = {}

# MAG-LINEAR: sklearn LinearRegression on 4-D m̃ (manuscript spec)
lr = LinearRegression().fit(mt_tr, y_tr); mag = H.r2_pooled(y_te, lr.predict(mt_te))
R["MAG_LINEAR"] = {"test_r2": mag, "manuscript": 0.208, "delta": abs(mag - 0.208),
                   "pass": abs(mag - 0.208) <= 0.005, "model": "sklearn LinearRegression"}
print(f"[MAG-LINEAR] R2={mag:+.4f} vs +0.208 |Δ|={abs(mag-0.208):.4f} pass={R['MAG_LINEAR']['pass']}", flush=True)

# RF[x;m̃]: reused from first gate run (same harness/config; deterministic per seed)
R["RF_X_MTILDE"] = {"test_r2_mean": 0.3704, "test_r2_sd": 0.0035, "manuscript": 0.370,
                    "pass": True, "note": "reused from first gate run (RF config unchanged, deterministic per seed)"}

# SPSM (x) + MAG-SPSM ([x;m̃]): real shards, seeds {0,1,2}
xm_tr = H.make_feature_shards(tr, [np.concatenate([s.x, m], 1) for s, m in zip(tr, H.mtilde(tr, stats))])
xm_va = H.make_feature_shards(va, [np.concatenate([s.x, m], 1) for s, m in zip(va, H.mtilde(va, stats))])
xm_te = H.make_feature_shards(te, [np.concatenate([s.x, m], 1) for s, m in zip(te, H.mtilde(te, stats))])

def spsm_test(trs, vas, tes, seed):
    res = H.train_spsm(H.Trio(trs, vas, tes), seed)
    yh = np.concatenate([res.predict(s.x) for s in tes if s.y.shape[0] > 0])
    return H.r2_pooled(y_te, yh)

for arm, trio, ref in [("SPSM", (tr, va, te), 0.082), ("MAG_SPSM", (xm_tr, xm_va, xm_te), 0.225)]:
    r2s = []
    for s in [0, 1, 2]:
        ts = time.time(); r2 = spsm_test(*trio, s); r2s.append(r2)
        print(f"[{arm} s{s}] R2={r2:+.4f} ({time.time()-ts:.0f}s)", flush=True)
    r2s = np.array(r2s)
    R[arm] = {"test_r2_mean": float(r2s.mean()), "test_r2_sd": float(r2s.std(ddof=1)), "per_seed": r2s.tolist(),
              "manuscript": ref, "pass": bool(abs(r2s.mean() - ref) <= 2 * r2s.std(ddof=1) or (r2s.min() <= ref <= r2s.max()))}
    print(f"[{arm}] R2={r2s.mean():+.4f}±{r2s.std(ddof=1):.4f} vs {ref:+.3f} pass={R[arm]['pass']}", flush=True)

R["GATE"] = bool(all(R[k]["pass"] for k in ["MAG_LINEAR", "RF_X_MTILDE", "SPSM", "MAG_SPSM"]))
json.dump(R, open(OUT + "/repro_gate_final.json", "w"), indent=2, default=str)
print("GATE_FINAL " + json.dumps(R, default=str), flush=True)
print(f"[done] {time.time()-t0:.0f}s GATE={'PASS' if R['GATE'] else 'FAIL'}", flush=True)
