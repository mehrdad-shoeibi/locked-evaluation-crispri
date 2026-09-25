"""Phase-3 crash tests (Part D). Runs the runner as subprocesses with injected crashes,
restarts, and verifies: orphan handling, no double-count, previous-checkpoint validity,
and restart determinism. Small deterministic subset (4 units)."""
import subprocess, os, sys, json, pickle, shutil
import numpy as np, pandas as pd

PY = sys.executable
RUN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "iclr2027_revision/pipeline/phase3_runner.py"))
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "iclr2027_revision/smoke/phase3_scratch"))
COMMON = ["--mode", "dry", "--subset-n", "400", "--unit-size", "100", "--gpu", "1"]
TOL = 1e-5

def run(out, extra):
    r = subprocess.run([PY, RUN, "--out-root", out] + COMMON + extra, capture_output=True, text=True)
    return r.returncode

def load_final(out):
    z = np.load(out + "/z/z_k562_1903.npy")
    x = np.load(out + "/x_cell/x_cell_k562_1903_mapped.npy")
    man = pd.read_parquet(out + "/z/k562_construct_index.parquet")["construct"].astype(str).tolist()
    st = pickle.load(open(out + "/checkpoints/xcell_state.pkl", "rb"))
    return z, x, man, st["count"]

def cmp_final(a, b):
    za, xa, ma, ca = load_final(a); zb, xb, mb, cb = load_final(b)
    assert ma == mb, "construct sets differ"
    zd = float(np.abs(za - zb).max()); xd = float(np.abs(xa - xb).max())
    cnt_match = (ca == cb)
    return zd, xd, cnt_match

res = {}
# reference (uninterrupted)
assert run(BASE + "/ref", ["--ckpt-cells", "200"]) == 0, "ref run failed"

# ---- Crash test 1: embedding committed, accumulator NOT checkpointed ----
shutil.rmtree(BASE + "/ct1", ignore_errors=True)
rc_crash = run(BASE + "/ct1", ["--ckpt-cells", "200", "--crash-after-unit", "2", "--crash-mode", "after_embed_before_ckpt"])
orphan_exists = os.path.exists(BASE + "/ct1/emb_units/unit_00002.npy")
st_after_crash = pickle.load(open(BASE + "/ct1/checkpoints/xcell_state.pkl", "rb"))
consumed_after_crash = sorted(st_after_crash["consumed"])
rc_restart = run(BASE + "/ct1", ["--ckpt-cells", "200"])
zd, xd, cnt_match = cmp_final(BASE + "/ct1", BASE + "/ref")
# no duplicate cell ids across units
allids = []
for p in sorted(__import__("glob").glob(BASE + "/ct1/emb_units/unit_*.parquet")):
    allids += pd.read_parquet(p)["cell_id"].astype(str).tolist()
dup = len(allids) != len(set(allids))
res["ct1"] = {"crash_rc": rc_crash, "orphan_artifact_existed": orphan_exists,
              "consumed_after_crash": consumed_after_crash, "restart_rc": rc_restart,
              "z_max_diff": zd, "x_max_diff": xd, "per_construct_counts_match": bool(cnt_match),
              "no_duplicate_cells": (not dup),
              "PASS": bool(rc_crash == 137 and orphan_exists and rc_restart == 0 and zd <= TOL and xd <= TOL and cnt_match and not dup)}

# ---- Crash test 2: crash DURING checkpoint write; previous checkpoint must survive ----
shutil.rmtree(BASE + "/ct2", ignore_errors=True)
rc_crash2 = run(BASE + "/ct2", ["--ckpt-cells", "100", "--crash-after-unit", "2", "--crash-mode", "during_checkpoint"])
cur_ok = False; consumed2 = None
try:
    st2 = pickle.load(open(BASE + "/ct2/checkpoints/xcell_state.pkl", "rb")); cur_ok = True; consumed2 = sorted(st2["consumed"])
except Exception:
    cur_ok = False
rc_restart2 = run(BASE + "/ct2", ["--ckpt-cells", "100"])
zd2, xd2, cnt2 = cmp_final(BASE + "/ct2", BASE + "/ref")
res["ct2"] = {"crash_rc": rc_crash2, "prev_checkpoint_valid": cur_ok, "consumed_before_restart": consumed2,
              "restart_rc": rc_restart2, "z_max_diff": zd2, "x_max_diff": xd2, "per_construct_counts_match": bool(cnt2),
              "PASS": bool(rc_crash2 == 137 and cur_ok and rc_restart2 == 0 and zd2 <= TOL and xd2 <= TOL and cnt2)}

# ---- Crash test 3: restart determinism (unit membership + per-cell emb within 1e-5) ----
def unit_emb(out, u):
    arr = np.load(f"{out}/emb_units/unit_{u:05d}.npy")
    ix = pd.read_parquet(f"{out}/emb_units/unit_{u:05d}.parquet")
    return {c: arr[k] for k, c in enumerate(ix.cell_id.astype(str))}
# compare unit 0 (never re-embedded in ct1) and unit 2 (re-embedded after restart in ct1) vs ref
det_max = 0.0; membership_ok = True
for u in [0, 1, 2, 3]:
    a = unit_emb(BASE + "/ref", u); b = unit_emb(BASE + "/ct1", u)
    if set(a) != set(b): membership_ok = False; continue
    for c in a: det_max = max(det_max, float(np.abs(a[c] - b[c]).max()))
res["ct3"] = {"unit_membership_identical": membership_ok, "percell_emb_max_diff": det_max,
              "PASS": bool(membership_ok and det_max <= TOL)}

print(json.dumps(res, indent=2))
print("ALL_CRASH_TESTS_PASS:", all(res[k]["PASS"] for k in res))
