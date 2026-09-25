"""L10 Parts C+D — validation-only selection (7 arms x 4 families x seeds{0,1,2}) then single test.
Each (arm,family,seed) is trained ONCE; val AND test predictions are produced from that one model
(test preds persisted but NOT used for selection). Selection uses VCC validation R2 ONLY, per arm.
frozen_selection.json (timestamped) is written BEFORE the test table is assembled. §6B scaler on train."""
import os, sys, json, time
import numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 2)))
sys.path.insert(0, os.path.join(_ROOT, "iclr2027_revision/pipeline"))
import phase1_harness as H
import fm_preprocess as PP
import controls as C
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
OUT = os.path.join(_ROOT, "iclr2027_revision/phase1")
os.makedirs(OUT + "/preds", exist_ok=True)
SEEDS = [0, 1, 2]; FAMILIES = ["Ridge", "RF", "HistGB", "SPSM"]; ARMS = ["z", "zm", "C1", "C2", "C3", "C4", "C5"]
t0 = time.time()
base = H.load_base_shards(); tr, va, te = base["train"], base["val"], base["test"]
y = {"train": H.concat_y(tr), "val": H.concat_y(va), "test": H.concat_y(te)}
SS = {"train": tr, "val": va, "test": te}
sizes = {k: [s.y.shape[0] for s in SS[k]] for k in SS}
z_scaler = PP.fit_block_scaler(H.concat(tr, "z")); m_stats = H.fit_mtilde_stats(tr)
c1_scaler = PP.fit_block_scaler(C.c1_features(len(y["train"]), "train"))
Rmat = C.c2_projection_matrix(); c2_scaler = PP.fit_block_scaler(C.c2_features(H.concat(tr, "x"), Rmat))
print(f"[data] tr={len(y['train'])} va={len(y['val'])} te={len(y['test'])} nearconst_z={PP.near_constant_count(z_scaler)}", flush=True)

def feats(arm, split):
    shs = SS[split]; sp = split
    z = PP.apply_scaler(z_scaler, H.concat(shs, "z"))
    if arm == "z": return z
    if arm == "zm": return np.concatenate([z, np.concatenate(H.mtilde(shs, m_stats), 0)], 1)
    if arm == "C1": return PP.apply_scaler(c1_scaler, C.c1_features(z.shape[0], sp))
    if arm == "C2": return PP.apply_scaler(c2_scaler, C.c2_features(H.concat(shs, "x"), Rmat))
    if arm == "C3": return C.c3_features(z, sp)
    if arm == "C4": return C.c4_features(z, np.concatenate(H.mtilde(shs, m_stats), 0), sp)
    if arm == "C5": return C.c5_features(z, sp)

def to_shards(feat, split):
    out = []; i = 0
    for n in sizes[split]:
        out.append(feat[i:i + n]); i += n
    return H.make_feature_shards(SS[split], out)

def train_once_predict(arm, fam, seed):
    """Train on TRAIN once; return (val_pred, test_pred)."""
    Xtr, Xva, Xte = feats(arm, "train"), feats(arm, "val"), feats(arm, "test")
    if fam == "Ridge":
        sc = StandardScaler().fit(Xtr); m = Ridge(alpha=1.0, random_state=seed).fit(sc.transform(Xtr), y["train"])
        return m.predict(sc.transform(Xva)), m.predict(sc.transform(Xte))
    if fam == "RF":
        m = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=2, n_jobs=8, random_state=seed).fit(Xtr, y["train"])
        return m.predict(Xva), m.predict(Xte)
    if fam == "HistGB":
        m = HistGradientBoostingRegressor(max_iter=500, max_depth=6, learning_rate=0.05, l2_regularization=0.0,
              random_state=seed, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20).fit(Xtr, y["train"])
        return m.predict(Xva), m.predict(Xte)
    if fam == "SPSM":
        res = H.train_spsm(H.Trio(to_shards(Xtr, "train"), to_shards(Xva, "val"), to_shards(Xte, "test")), seed)
        vp = np.concatenate([res.predict(s.x) for s in to_shards(Xva, "val") if s.y.shape[0] > 0])
        tp = np.concatenate([res.predict(s.x) for s in to_shards(Xte, "test") if s.y.shape[0] > 0])
        return vp, tp

# ---- train all; store val+test preds (test not used for selection) ----
val_tbl = {}; valpred = {}; testpred = {}
for arm in ARMS:
    val_tbl[arm] = {}
    for fam in FAMILIES:
        vr2 = []
        for seed in SEEDS:
            ts = time.time(); vp, tp = train_once_predict(arm, fam, seed)
            valpred[(arm, fam, seed)] = vp; testpred[(arm, fam, seed)] = tp
            vr2.append(H.r2_pooled(y["val"], vp))
            print(f"[{arm}/{fam}/s{seed}] valR2={vr2[-1]:+.4f} ({time.time()-ts:.0f}s)", flush=True)
        vr2 = np.array(vr2)
        val_tbl[arm][fam] = {"mean": float(vr2.mean()), "sd": float(vr2.std(ddof=1)), "per_seed": vr2.tolist()}
    json.dump(val_tbl, open(OUT + "/validation_table.json", "w"), indent=2)

# ---- select per arm on validation R2 (independently), freeze BEFORE test table ----
selection = {}
for arm in ARMS:
    ranked = sorted(FAMILIES, key=lambda f: val_tbl[arm][f]["mean"], reverse=True)
    best, runner = ranked[0], ranked[1]
    margin = val_tbl[arm][best]["mean"] - val_tbl[arm][runner]["mean"]
    selection[arm] = {"selected": best, "runner_up": runner, "margin": float(margin),
                      "best_val_mean": val_tbl[arm][best]["mean"], "seed_sd": val_tbl[arm][best]["sd"],
                      "margin_inside_seed_sd": bool(margin < val_tbl[arm][best]["sd"])}
frozen = {"selection": selection, "criterion": "VCC validation R2, independently per arm",
          "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seeds": SEEDS, "families": FAMILIES}
json.dump(frozen, open(OUT + "/frozen_selection.json", "w"), indent=2)
print("FROZEN " + json.dumps(frozen, default=str), flush=True)

# ---- assemble TEST table from persisted preds of the frozen selection ----
np.save(OUT + "/preds/y_test.npy", y["test"])
test_tbl = {}
for arm in ARMS:
    fam = selection[arm]["selected"]
    P = np.stack([testpred[(arm, fam, s)] for s in SEEDS])
    np.save(OUT + f"/preds/test_{arm}.npy", P)
    r2 = [H.r2_pooled(y["test"], P[i]) for i in range(3)]; sp = [H.spearman_avgtie(y["test"], P[i]) for i in range(3)]
    test_tbl[arm] = {"family": fam, "r2_mean": float(np.mean(r2)), "r2_sd": float(np.std(r2, ddof=1)),
                     "spearman_mean": float(np.mean(sp)), "spearman_sd": float(np.std(sp, ddof=1)),
                     "r2_per_seed": r2, "spearman_per_seed": sp}
    print(f"[TEST {arm}/{fam}] R2={np.mean(r2):+.4f}±{np.std(r2,ddof=1):.4f} rho={np.mean(sp):+.4f}", flush=True)
json.dump(test_tbl, open(OUT + "/test_table.json", "w"), indent=2)
print("TESTDONE " + json.dumps(test_tbl, default=str), flush=True)
print(f"[done] {time.time()-t0:.0f}s", flush=True)
