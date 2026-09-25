"""H-FM4 — TRAIN/VALIDATION-ONLY SUPPORTING DIAGNOSTIC runner.

Scope
-----
VCC TRAIN + VCC VALIDATION only. No VCC test. No external screen. No GPU. No FM inference.
The original test-level H-FM4 statement is NOT resolved by this runner (execution note §17).

Governing authority
-------------------
  experiment_specs/option_b_geneformer_experiment_lock.md          §13 controls, §6B scalers, §18 H-FM4
  experiment_specs/part1_scientific.md                             H-FM4 §, object table :56
  experiment_specs/implementation_preflight.md                     §D2.2 alpha grid :201-209
  experiment_specs/hfm4_execution_note.md                          D1/D2/D3A/D3B/D4, §18/§19/§20
  experiment_specs/lock_amendment_A10_phaseB_numerical_baseline.md §1-§6 (supersedes note §10/§13
                                                                    operationally: LOAD-ONLY scalers)

Frozen code paths reused rather than reimplemented
--------------------------------------------------
  pipeline/phaseB_preprocessing.py  thread policy, content-projection digests, load_scalers (A10 §4)
  pipeline/fm_preprocess.py:25-27   apply_scaler  (§6B frozen apply path: float64 transform -> float32)
  pipeline/controls.py:37-62        C2 projection / C3 per-split permutation (lock §13)
  pipeline/phase1_harness.py:92-95  r2_pooled     (verbatim; the manuscript's own validation R2)
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, subprocess, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, _HERE)
sys.path.insert(0, REPO)

import phaseB_preprocessing as PB           # noqa: E402
import fm_preprocess as PP                  # noqa: E402
import controls as C                        # noqa: E402
from phaseB_preprocessing import PhaseBError  # noqa: E402

# ---------------------------------------------------------------- A10 §4 instrumentation
from sklearn.preprocessing import StandardScaler  # noqa: E402
FIT_COUNTS = {"fit": 0, "fit_transform": 0}
_A10_MSG = ("A10 §4: Phase-B scientific runners must LOAD canonical scaler state, never refit "
            "(StandardScaler.%s is forbidden).")


def _guard_fit(self, X, y=None, **kw):
    FIT_COUNTS["fit"] += 1
    raise PhaseBError(_A10_MSG % "fit")


def _guard_fit_transform(self, X, y=None, **kw):
    FIT_COUNTS["fit_transform"] += 1
    raise PhaseBError(_A10_MSG % "fit_transform")


StandardScaler.fit = _guard_fit
StandardScaler.fit_transform = _guard_fit_transform

from sklearn.linear_model import Ridge  # noqa: E402

# ---------------------------------------------------------------- frozen constants
ALPHA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)     # preflight :201-206 — exactly these seven
COMPONENTS = ("mean_abs_x", "l2_norm_x", "max_abs_x", "std_x")  # note §5, magnitude_augment.py:35-45
ARMS = ("z", "ceiling_x", "C2", "C3")                           # lock:392-393 subject/ceiling/floor1/floor2
REPEATS = (1, 2, 3)                                             # deterministic re-executions, NOT seeds
RIDGE_RANDOM_STATE = 0                     # App F factory shape (fm_heads.py:22); inert for dense solver='auto'

SPLITS = {                                  # ALLOWLIST — train/val only. 'test' and externals are absent.
    "train": {"shard_dir": "train", "bridge_split": "training",  "rows": 5740, "shards": 48,
              "mtilde": "iclr2027_revision/phase0/m_tilde/m_tilde_VCC_train.npz",
              "mtilde_sha": "ea7784132fbf639c521d53dd01a26c6c72f16e9cdd7e756559ba473ef219ad23"},
    "val":   {"shard_dir": "val",   "bridge_split": "validation", "rows": 2052, "shards": 48,
              "mtilde": "iclr2027_revision/phase0/m_tilde/m_tilde_VCC_val.npz",
              "mtilde_sha": "4f0e7d9999cebb89e1739d8d140a9d2a19df926a2d0912856fa7477384dfd763"},
}
FORBIDDEN_SPLIT_TOKENS = ("test", "testing", "RPE1", "K562", "rpe1", "k562", "external")
VCC_TEST_ROWS_REFERENCE = 3972              # guard/reference only — never loaded

SCALER_SHA = {                              # PHASEB_NUMERICAL_BASELINE_MANIFEST.json::canonical_scalers
    "z":         "ebf55308f3c6e62e43cd8b9aba084413549036c0f7e23a987ee6b8b390672e0e",
    "c2":        "2f6ae19992c7ec7422a3b4a8fe4fea21750a40fb862ecc15acc177d60593600d",
    "ceiling_x": "827077e84b40844ccd5aa30fb78b8b0c2571fe98c96d4e1bd9b62bd914892766",
}
C2MAT_SHA = "e5bf7ce0f07cd68d73c41529b7f01441b1b5a6f9193d663bc28bc6f974d8f41a"   # lock:316-318 artifact

PROTECTED_ROOTS = [                          # execution note §18
    "/home/mehrdad/Desktop/iga-neurips",
    "/media/mehrdad/MehrdadSSD/IGA_NeurIPS",
    "/media/mehrdad/MehrdadSSD/IGA_ICLR2027_GENEFORMER",
    "/media/mehrdad/MehrdadSSD/MADA_data",
    REPO,
]

ACCESS_LOG = []


def _record(path, role, split, sha=None):
    ACCESS_LOG.append({"path": _sanitize(path), "role": role, "split": split,
                       "read_only": True, "sha256": sha})


def _sanitize(p):
    p = os.path.abspath(p)
    root = _STORAGE[0] if _STORAGE else None
    if root and p.startswith(root):
        return "<B-STORAGE>" + p[len(root):]
    if p.startswith(REPO):
        return os.path.relpath(p, REPO)
    return p


_STORAGE = []


# ---------------------------------------------------------------- r2 (phase1_harness.py:92-95 verbatim)
def r2_pooled(y, yh):
    y = np.asarray(y, float); yh = np.asarray(yh, float)
    sst = ((y - y.mean()) ** 2).sum()
    return float("nan") if sst == 0 else float(1.0 - ((y - yh) ** 2).sum() / sst)


# ---------------------------------------------------------------- output isolation (note §18)
def resolve_out_root(raw):
    p = os.path.realpath(os.path.abspath(os.path.expanduser(raw)))
    for prot in PROTECTED_ROOTS:
        pr = os.path.realpath(prot)
        if p == pr or p.startswith(pr + os.sep):
            raise PhaseBError(f"output root is inside protected root {prot}: {p}")
    if os.path.exists(p):
        raise FileExistsError(f"output root already exists (no reuse, no overwrite): {p}")
    os.makedirs(p, exist_ok=False)
    return p


# ---------------------------------------------------------------- canonical inputs (note §11 D3A / §12 D3B)
def canonical_inputs(storage_root, split):
    """TRAIN or VAL only. Returns (X, Z, C2, prov). y / label columns are never requested."""
    if split in FORBIDDEN_SPLIT_TOKENS:
        raise PhaseBError(f"FORBIDDEN SPLIT REQUESTED: {split!r} — H-FM4 is train/validation only")
    if split not in SPLITS:
        raise PhaseBError(f"split {split!r} not in allowlist {sorted(SPLITS)}")
    import pyarrow.parquet as pq
    cfg = SPLITS[split]
    dhvg = os.path.join(storage_root, PB.DHVG_REL)
    sdir = os.path.join(dhvg, "shards", cfg["shard_dir"])
    if os.path.realpath(sdir).rstrip("/").endswith(("/test", "/testing")):
        raise PhaseBError("refusing to open a VCC-test shard directory")
    files = sorted(glob.glob(os.path.join(sdir, "*.npz")))
    if len(files) != cfg["shards"]:
        raise PhaseBError(f"{split}: {len(files)} shards != {cfg['shards']}")

    xp = PB._Proj(); xp.tok("CONTENT_PROJECTION_DIGEST_SPEC_VERSION"); xp.tok(PB.DIGEST_SPEC_VERSION)
    xp.tok("logical_artifact"); xp.tok(f"vcc_delta_hvg_{cfg['shard_dir']}_shards_x_projection")
    keys, xs = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        b = str(d["batch_name"][0]); sid = [str(s) for s in d["sample_ids"]]
        x = np.asarray(d["x"], dtype=np.float32)            # 'y' is never read from the shard
        xp.tok("shard_name"); xp.tok(os.path.basename(f))
        xp.tok("batch_name"); xp.tok(b); xp.seq("sample_ids", sid); xp.arr("x", x)
        keys += [(cfg["bridge_split"], b, s) for s in sid]; xs.append(x)
        _record(f, "delta_hvg shard x (key 'x' only)", split)
    X = np.concatenate(xs, axis=0)
    if X.shape != (cfg["rows"], 2000) or X.dtype != np.float32:
        raise PhaseBError(f"{split} x shape/dtype {X.shape}/{X.dtype}")

    ZC = [f"gf_z_emb_{i}" for i in range(768)]
    t = pq.read_table(PB.BRIDGE, columns=["split", "batch", "target_gene"] + ZC)  # label never requested
    _record(PB.BRIDGE, "z bridge (key cols + 768 gf_z_emb_* only)", split)
    sp = [str(v) for v in t.column("split").to_pylist()]
    ba = [str(v) for v in t.column("batch").to_pylist()]
    tg = [str(v) for v in t.column("target_gene").to_pylist()]
    sel = [i for i, s in enumerate(sp) if s == cfg["bridge_split"]]
    bk = [(sp[i], ba[i], tg[i]) for i in sel]
    dup_shard = len(keys) - len(set(keys)); dup_bridge = len(bk) - len(set(bk))
    if dup_shard: raise PhaseBError(f"{split}: {dup_shard} duplicate shard keys")
    if dup_bridge: raise PhaseBError(f"{split}: {dup_bridge} duplicate bridge keys")
    missing = sorted(set(keys) - set(bk)); extra = sorted(set(bk) - set(keys))
    if missing: raise PhaseBError(f"{split}: {len(missing)} keys missing from bridge")
    if extra: raise PhaseBError(f"{split}: {len(extra)} extra bridge keys")

    Zsrc = np.column_stack([np.asarray(t.column(c).to_numpy(zero_copy_only=False)) for c in ZC])[sel]
    Z32 = Zsrc.astype(np.float32)                            # D3B historical lossy round-trip
    pos = {k: i for i, k in enumerate(bk)}
    Z = np.ascontiguousarray(Z32[np.array([pos[k] for k in keys], dtype=np.int64)])
    if Z.shape != (cfg["rows"], 768): raise PhaseBError(f"{split} z shape {Z.shape}")

    zp = PB._Proj(); zp.tok("CONTENT_PROJECTION_DIGEST_SPEC_VERSION"); zp.tok(PB.DIGEST_SPEC_VERSION)
    zp.tok("logical_artifact"); zp.tok(f"vcc_z_bridge_{cfg['shard_dir']}_projection")
    zp.tok("row_order"); zp.tok("D3A_canonical_shard_key_order")
    zp.seq("split", [k[0] for k in keys]); zp.seq("batch", [k[1] for k in keys])
    zp.seq("target_gene", [k[2] for k in keys]); zp.arr("z_float32_canonical", Z)

    got = PB.sha_file(PB.C2MAT)
    if got != C2MAT_SHA:
        raise PhaseBError(f"frozen C2 projection matrix SHA-256 mismatch: {got}")
    _record(PB.C2MAT, "frozen C2 projection matrix R (2000x768)", "ALL", got)
    R = np.load(PB.C2MAT)
    if R.shape != (2000, 768) or R.dtype != np.float32:
        raise PhaseBError("C2 matrix shape/dtype")
    C2 = C.c2_features(X, R)                                 # controls.py:44-49 — float32 GEMM
    cp = PB._Proj(); cp.tok("CONTENT_PROJECTION_DIGEST_SPEC_VERSION"); cp.tok(PB.DIGEST_SPEC_VERSION)
    cp.tok("logical_artifact"); cp.tok(f"vcc_c2_{cfg['shard_dir']}_projection"); cp.arr("c2_float32", C2)

    prov = {"split": split, "bridge_split": cfg["bridge_split"], "shard_dir": cfg["shard_dir"],
            "SHARDS": len(files), "ROWS": len(keys),
            "SHARD_KEYS_UNIQUE": True, "BRIDGE_KEYS_UNIQUE": True, "KEY_SET_EQUAL": True,
            "DUPLICATE_KEYS": 0, "MISSING_KEYS": 0, "EXTRA_KEYS": 0,
            "PHYSICAL_BRIDGE_SEQUENCE_EQUAL": bool(bk == keys),
            "X_PROJECTION_SHA256": xp.hex(), "Z_PROJECTION_SHA256": zp.hex(),
            "C2_PROJECTION_SHA256": cp.hex(), "C2_MATRIX_SHA256": got}
    return X, Z, C2, keys, prov


def load_mtilde(split, n_rows):
    cfg = SPLITS[split]
    p = os.path.join(REPO, cfg["mtilde"])
    got = PB.sha_file(p)
    if got != cfg["mtilde_sha"]:
        raise PhaseBError(f"{split} m_tilde SHA-256 mismatch: {got}")
    d = np.load(p)
    M = np.asarray(d["m_zscored"])                            # note §5 — the frozen m~ object
    if M.shape != (n_rows, 4):
        raise PhaseBError(f"{split} m_zscored shape {M.shape}")
    _record(p, "m_tilde targets (m_zscored, 4 components)", split, got)
    return M


# ---------------------------------------------------------------- selection (note §8 D1 / §9 D2)
def select_alpha(alphas, r2s):
    finite = [(a, r) for a, r in zip(alphas, r2s) if np.isfinite(r)]
    if not finite:
        return {"selected_alpha": None, "selected_validation_r2": None, "tie_applied": False,
                "tied_alphas": [], "status": "PROBE_DEGENERATE", "n_finite": 0}
    best = max(r for _, r in finite)                # exact float64 comparison; no rounding, no tolerance
    tied = [a for a, r in finite if r == best]      # exact equality only
    sel = max(tied)                                 # D1: largest alpha wins an exact tie
    return {"selected_alpha": float(sel), "selected_validation_r2": float(best),
            "tie_applied": len(tied) > 1, "tied_alphas": [float(a) for a in tied],
            "status": "OK", "n_finite": len(finite)}


# ---------------------------------------------------------------- synthetic fail-closed tests (§23)
def guard_tests(storage_root, out_root):
    T = []

    def rec(name, expected, fn):
        try:
            fn(); observed, ok = "NO EXCEPTION RAISED", False
        except BaseException as e:
            observed, ok = f"{type(e).__name__}: {e}", True
        T.append({"test": name, "expected": expected, "observed": observed,
                  "result": "PASS" if ok else "FAIL"})

    def t_thread():
        keep = os.environ["OMP_NUM_THREADS"]
        os.environ["OMP_NUM_THREADS"] = "4"
        try: PB.verify_thread_policy()
        finally: os.environ["OMP_NUM_THREADS"] = keep
    rec("thread_policy_failure", "PhaseBError: thread variable not 1", t_thread)

    rec("wrong_scaler_sha", "PhaseBError: scaler artifact SHA-256 mismatch",
        lambda: PB.load_scalers(("z",), expected_sha={"z": "0" * 64}))
    rec("split_test_request", "PhaseBError before any test file is opened",
        lambda: canonical_inputs(storage_root, "test"))
    rec("split_external_request", "PhaseBError before any external file is opened",
        lambda: canonical_inputs(storage_root, "RPE1"))
    rec("scaler_refit_forbidden_api", "PhaseBError from phaseB_preprocessing.refit_forbidden",
        lambda: PB.refit_forbidden())
    rec("standardscaler_fit_instrumentation",
        "PhaseBError from the patched StandardScaler.fit, and the counter increments",
        lambda: StandardScaler().fit(np.arange(12, dtype=np.float64).reshape(6, 2)))
    rec("standardscaler_fit_transform_instrumentation",
        "PhaseBError from the patched StandardScaler.fit_transform",
        lambda: StandardScaler().fit_transform(np.arange(12, dtype=np.float64).reshape(6, 2)))

    def _keycheck(shard_keys, bridge_keys):
        if len(shard_keys) != len(set(shard_keys)):
            raise PhaseBError("duplicate shard keys")
        if set(shard_keys) - set(bridge_keys):
            raise PhaseBError("keys missing from bridge")
        if set(bridge_keys) - set(shard_keys):
            raise PhaseBError("extra bridge keys")
    rec("duplicate_keys", "PhaseBError: duplicate shard keys",
        lambda: _keycheck([("s", "b", "g1"), ("s", "b", "g1")], [("s", "b", "g1"), ("s", "b", "g2")]))
    rec("missing_keys", "PhaseBError: keys missing from bridge",
        lambda: _keycheck([("s", "b", "g1"), ("s", "b", "g9")], [("s", "b", "g1"), ("s", "b", "g2")]))
    rec("existing_output_dir", "FileExistsError: output root already exists",
        lambda: resolve_out_root(out_root))

    # selection-rule tests (no exception expected — check the returned decision)
    mixed = select_alpha(ALPHA_GRID, [float("nan"), 0.5, float("inf"), 0.7, float("-inf"),
                                      0.6, float("nan")])
    T.append({"test": "nonfinite_mixed", "expected": "finite max 0.7 at alpha=1 selected; 4 non-finite ineligible",
              "observed": f"selected_alpha={mixed['selected_alpha']} r2={mixed['selected_validation_r2']} "
                          f"n_finite={mixed['n_finite']} status={mixed['status']}",
              "result": "PASS" if (mixed["selected_alpha"] == 1.0 and mixed["n_finite"] == 3
                                   and mixed["status"] == "OK") else "FAIL"})
    tie = select_alpha(ALPHA_GRID, [0.1, 0.42, 0.2, 0.42, 0.3, 0.42, 0.05])
    T.append({"test": "exact_tie_largest_alpha",
              "expected": "alphas {1e-2, 1, 100} exactly tied at 0.42 -> select 100.0",
              "observed": f"selected_alpha={tie['selected_alpha']} tie_applied={tie['tie_applied']} "
                          f"tied_alphas={tie['tied_alphas']}",
              "result": "PASS" if (tie["selected_alpha"] == 100.0 and tie["tie_applied"]
                                   and tie["tied_alphas"] == [0.01, 1.0, 100.0]) else "FAIL"})
    allnf = select_alpha(ALPHA_GRID, [float("nan")] * 3 + [float("inf"), float("-inf")] + [float("nan")] * 2)
    T.append({"test": "all_nonfinite_probe_degenerate",
              "expected": "selected_alpha=None, status=PROBE_DEGENERATE",
              "observed": f"selected_alpha={allnf['selected_alpha']} status={allnf['status']}",
              "result": "PASS" if (allnf["selected_alpha"] is None
                                   and allnf["status"] == "PROBE_DEGENERATE") else "FAIL"})
    return T


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True, help="explicit NEW directory; no default, no reuse")
    a = ap.parse_args()

    t_start = time.gmtime()
    utc_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", t_start)

    thread_policy = PB.verify_thread_policy()          # A10 §2 — before any numerical work
    import sklearn, scipy, pyarrow, joblib, threadpoolctl
    env = {"python": sys.version.split()[0], "python_full": sys.version.replace("\n", " "),
           "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
           "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__,
           "pyarrow": pyarrow.__version__, "executable": sys.executable,
           "torch": None, "cuda": None, "gpu_model": None, "driver": None,
           "thread_env": {v: os.environ.get(v, "UNSET") for v in PB.THREAD_VARS},
           "threadpool_info": thread_policy["THREAD_POLICY_REALIZED"],
           "hostname_PRIVATE_LOCAL_PROVENANCE_ONLY": os.uname().nodename}
    if env["python"] != "3.13.12":
        raise PhaseBError(f"pinned interpreter 3.13.12 required; got {env['python']}")

    storage = PB._storage_root() if hasattr(PB, "_storage_root") else None
    if storage is None:
        import iga_paths
        storage = iga_paths.STORAGE
    _STORAGE.append(storage)

    out = resolve_out_root(a.out_root)
    print(f"[out ] {out}", flush=True)

    # ---- synthetic fail-closed gates BEFORE any scientific input -------------------
    gt = guard_tests(storage, out)
    for r in gt:
        print(f"[guard] {r['result']:4s} {r['test']}: {r['observed']}", flush=True)
    if any(r["result"] != "PASS" for r in gt):
        raise PhaseBError("synthetic fail-closed guard test failed — real H-FM4 not executed")
    instrumentation_proof = dict(FIT_COUNTS)
    FIT_COUNTS["fit"] = 0; FIT_COUNTS["fit_transform"] = 0     # reset for the real run
    json.dump({"tests": gt, "instrumentation_proof_counts": instrumentation_proof},
              open(os.path.join(out, "hfm4_guard_tests.json"), "w"), indent=2)

    git = {"sha": subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"]).decode().strip(),
           "branch": subprocess.check_output(["git", "-C", REPO, "branch", "--show-current"]).decode().strip(),
           "dirty": bool(subprocess.check_output(["git", "-C", REPO, "status", "--porcelain",
                                                  "--untracked-files=no"]).decode().strip())}
    run_id = f"hfm4_{utc_start.replace('-', '').replace(':', '')}_{git['sha'][:12]}"

    # ---- canonical inputs, both authorized splits ---------------------------------
    data, prov = {}, {}
    for s in ("train", "val"):
        X, Z, C2, keys, pv = canonical_inputs(storage, s)
        M = load_mtilde(s, pv["ROWS"])
        data[s] = {"x": X, "z": Z, "c2": C2, "m": M, "keys": keys}
        prov[s] = pv
        print(f"[data] {s}: rows={pv['ROWS']} x={X.shape} z={Z.shape} c2={C2.shape} m={M.shape}", flush=True)

    # ---- scalers: LOAD ONLY (A10 §4) ----------------------------------------------
    sc = PB.load_scalers(("z", "c2", "ceiling_x"), expected_sha=SCALER_SHA)
    for n in ("z", "c2", "ceiling_x"):
        p = os.path.join(PB.BASELINE_DIR, f"phaseB_{n}_scaler_state.npz")
        _record(p, f"canonical Phase-B {n} scaler state (LOAD ONLY)", "train-fitted", SCALER_SHA[n])
    scaler_report = {n: {"artifact": os.path.relpath(
        os.path.join(PB.BASELINE_DIR, f"phaseB_{n}_scaler_state.npz"), REPO),
        "sha256": SCALER_SHA[n], "n_features_in_": int(sc[n].n_features_in_),
        "n_samples_seen_": int(sc[n].n_samples_seen_),
        "state_digests": {f: hashlib.sha256(np.ascontiguousarray(getattr(sc[n], f)).tobytes(order="C")).hexdigest()
                          for f in ("mean_", "scale_", "var_")}} for n in sc}
    scaler_report["C3_scaler"] = "REUSES the loaded z scaler (lock:171) — no C3 scaler exists"
    scaler_report["ceiling_scaler_persisted_phase0_counterpart"] = "NONE (execution note §10 D3E)"

    # ---- build the four probe inputs ----------------------------------------------
    c3_seeds = {}
    feats = {}
    for s in ("train", "val"):
        d = data[s]
        zs = PP.apply_scaler(sc["z"], d["z"])            # fm_preprocess.py:27 frozen apply path
        feats[s] = {
            "z": zs,
            "ceiling_x": PP.apply_scaler(sc["ceiling_x"], d["x"]),
            "C2": PP.apply_scaler(sc["c2"], d["c2"]),
            "C3": C.c3_features(zs, s),                  # controls.py:60-62 on canonical-order scaled z
        }
        c3_seeds[s] = {"split_index": C.SPLIT_INDEX[s], "seed": C.C3_SEED + C.SPLIT_INDEX[s]}
        perm = C.c3_permutation(d["z"].shape[0], s)
        assert sorted(perm.tolist()) == list(range(d["z"].shape[0]))
        assert feats[s]["C3"].shape == zs.shape
        assert np.array_equal(np.sort(feats[s]["C3"], axis=0), np.sort(zs, axis=0))

    # ---- PRE-RUN FREEZE, before the first real Ridge fit ---------------------------
    freeze = {
        "HFM4_RESULT_OBSERVED_BEFORE_FREEZE": "NO",
        "REAL_HFM4_RIDGE_FIT_COUNT_BEFORE_FREEZE": 0,
        "VCC_TEST_ACCESSED_BEFORE_FREEZE": "NO",
        "EXTERNAL_ACCESSED_BEFORE_FREEZE": "NO",
        "run_id": run_id, "utc_freeze": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "utc_start": utc_start, "git": git, "output_root": out, "command": sys.argv,
        "scientific_status": "TRAIN_VALIDATION_ONLY_SUPPORTING_DIAGNOSTIC",
        "governing_documents": GOV_DOCS(),
        "environment": env, "environment_lock": ENV_LOCK(),
        "thread_policy": {"requested": env["thread_env"],
                          "realized_num_threads": [p.get("num_threads") for p in env["threadpool_info"]],
                          "check": thread_policy["THREAD_POLICY_CHECK"]},
        "authorized_splits": ["VCC train", "VCC validation"],
        "forbidden_splits": ["VCC test", "RPE1", "K562", "any external screen"],
        "expected_rows": {"train": 5740, "validation": 2052,
                          "vcc_test_reference_never_loaded": VCC_TEST_ROWS_REFERENCE},
        "canonical_key": "(split, batch_name, sample_id) == bridge (split, batch, target_gene)",
        "canonical_row_order": "sorted(glob(shards/<split>/*.npz)) then stored sample_ids order (note §11 D3A)",
        "row_order_validation": prov,
        "probe_inputs": list(ARMS), "targets": list(COMPONENTS), "target_object": "m_zscored",
        "alpha_grid": list(ALPHA_GRID),
        "ridge": {"estimator": "sklearn.linear_model.Ridge", "random_state": RIDGE_RANDOM_STATE,
                  "note": "random_state is inert for dense solver='auto'; not a randomness mechanism"},
        "feature_dtype_path": "fm_preprocess.apply_scaler: float64 transform -> float32 (frozen §6B apply path)",
        "target_dtype": "as stored in m_zscored (float32), unmodified",
        "selection_metric": "validation R2 = phase1_harness.r2_pooled (verbatim)",
        "c2_matrix_sha256": C2MAT_SHA, "c2_dim": "2000 -> 768", "c2_seed_of_record": 20270102,
        "c3_seed_rule": "default_rng(20270103 + split_index), controls.py:53-57; train=0, val=1",
        "c3_seeds": c3_seeds,
        "scalers": scaler_report,
        "scaler_rule": "A10 §4 LOAD ONLY — load, verify SHA, verify state, transform; never fit",
        "STANDARD_SCALER_FIT_CALL_COUNT_EXPECTED": 0,
        "tie_rule": "D1 — exact float64 equality; select the LARGEST tied alpha; no rounding/tolerance",
        "nonfinite_rule": "D2 — non-finite validation R2 is ineligible; all seven non-finite -> "
                          "selected_alpha=null, PROBE_DEGENERATE, no fallback",
        "ci_rule": "D4 — NO H-FM4 confidence interval", "bootstrap_rule": "D4 — NO H-FM4 bootstrap",
        "pvalue_rule": "D4 — NO p-value", "threshold_rule": "no absolute recoverability threshold (lock:393)",
        "reporting_rule": "all 4 arms x 4 components = 16 cells reported; no composite; no cherry-pick",
        "repeat_rule": "REPEAT_1..3 are deterministic re-executions for determinism verification, "
                       "NOT seeds; selection uses the single deterministic value (REPEAT_1)",
        "guard_tests": gt,
        "standardscaler_instrumentation_proof": instrumentation_proof,
    }
    json.dump(freeze, open(os.path.join(out, "HFM4_PRE_RUN_FREEZE.json"), "w"), indent=2, default=str)
    write_freeze_md(os.path.join(out, "HFM4_PRE_RUN_FREEZE.md"), freeze)
    print("[freeze] written before the first real Ridge fit", flush=True)

    # ---- REAL H-FM4 ---------------------------------------------------------------
    fits = {"unique": 0, "total": 0}
    results = []
    for arm in ARMS:
        Xtr, Xva = feats["train"][arm], feats["val"][arm]
        for k, comp in enumerate(COMPONENTS):
            ytr = data["train"]["m"][:, k]
            yva = data["val"]["m"][:, k]
            per_rep = {r: [] for r in REPEATS}
            for r in REPEATS:
                for al in ALPHA_GRID:
                    mdl = Ridge(alpha=al, random_state=RIDGE_RANDOM_STATE).fit(Xtr, ytr)
                    per_rep[r].append(r2_pooled(yva, mdl.predict(Xva)))
                    fits["total"] += 1
                    if r == 1:
                        fits["unique"] += 1
            sel = select_alpha(ALPHA_GRID, per_rep[1])
            ident = all(per_rep[r] == per_rep[1] or
                        all((x == y) or (np.isnan(x) and np.isnan(y))
                            for x, y in zip(per_rep[r], per_rep[1])) for r in REPEATS)
            sel_r2 = {r: (None if sel["selected_alpha"] is None
                          else per_rep[r][ALPHA_GRID.index(sel["selected_alpha"])]) for r in REPEATS}
            vals = [sel_r2[r] for r in REPEATS]
            results.append({
                "probe_input": arm, "component": comp,
                "candidate_alphas": list(ALPHA_GRID),
                "candidate_validation_r2": {f"repeat_{r}": per_rep[r] for r in REPEATS},
                "candidate_finite_status": {f"repeat_{r}": [bool(np.isfinite(v)) for v in per_rep[r]]
                                            for r in REPEATS},
                "selected_alpha": sel["selected_alpha"], "tie_applied": sel["tie_applied"],
                "tied_alphas": sel["tied_alphas"], "n_finite_candidates": sel["n_finite"],
                "probe_degenerate": sel["status"] == "PROBE_DEGENERATE", "status": sel["status"],
                "validation_r2_repeat_1": vals[0], "validation_r2_repeat_2": vals[1],
                "validation_r2_repeat_3": vals[2],
                "mean_validation_r2": (None if vals[0] is None else float(np.mean(vals))),
                "sd_validation_r2_ddof1": (None if vals[0] is None else float(np.std(vals, ddof=1))),
                "repeats_identical": bool(ident),
            })
            print(f"[hfm4] {arm:10s} {comp:11s} alpha={sel['selected_alpha']} "
                  f"valR2={sel['selected_validation_r2']} tie={sel['tie_applied']} {sel['status']}", flush=True)

    if FIT_COUNTS["fit"] or FIT_COUNTS["fit_transform"]:
        raise PhaseBError(f"HARD FAIL — StandardScaler fit counts {FIT_COUNTS}")
    if len(results) != 16:
        raise PhaseBError(f"expected 16 cells, produced {len(results)}")

    # ---- persist ------------------------------------------------------------------
    utc_end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res_doc = {"run_id": run_id, "scientific_status": "TRAIN_VALIDATION_ONLY_SUPPORTING_DIAGNOSTIC",
               "utc_start": utc_start, "utc_end": utc_end, "git": git,
               "alpha_grid": list(ALPHA_GRID), "probe_inputs": list(ARMS),
               "components": list(COMPONENTS), "cells": results,
               "ALL_16_CELLS_REPORTED": len(results) == 16,
               "ANY_PROBE_DEGENERATE": any(c["probe_degenerate"] for c in results),
               "ANY_EXACT_TIE": any(c["tie_applied"] for c in results),
               "ANY_NONFINITE_CANDIDATE": any(not all(c["candidate_finite_status"]["repeat_1"])
                                              for c in results),
               "DETERMINISTIC_REPEAT_VALUES_IDENTICAL": all(c["repeats_identical"] for c in results),
               "HFM4_INTERVAL_COMPUTED": "NO", "HFM4_BOOTSTRAP_COMPUTED": "NO",
               "HFM4_PVALUE_COMPUTED": "NO", "ABSOLUTE_THRESHOLD_USED": "NO"}
    json.dump(res_doc, open(os.path.join(out, "hfm4_results.json"), "w"), indent=2)

    with open(os.path.join(out, "hfm4_results.csv"), "w") as f:
        f.write("probe_input,component,selected_alpha,tie_applied,tied_alphas,"
                "validation_r2_repeat_1,validation_r2_repeat_2,validation_r2_repeat_3,"
                "mean_validation_r2,sd_validation_r2_ddof1,status\n")
        for c in results:
            ta = "|".join(f"{x:g}" for x in c["tied_alphas"])
            def fmt(v): return "" if v is None else repr(v)
            f.write(f"{c['probe_input']},{c['component']},"
                    f"{'' if c['selected_alpha'] is None else repr(c['selected_alpha'])},"
                    f"{str(c['tie_applied']).upper()},{ta},"
                    f"{fmt(c['validation_r2_repeat_1'])},{fmt(c['validation_r2_repeat_2'])},"
                    f"{fmt(c['validation_r2_repeat_3'])},{fmt(c['mean_validation_r2'])},"
                    f"{fmt(c['sd_validation_r2_ddof1'])},{c['status']}\n")

    json.dump({"ACCESSES": ACCESS_LOG,
               "VCC_TEST_SCIENTIFIC_INPUT_OPENED": "NO",
               "EXTERNAL_SCIENTIFIC_INPUT_OPENED": "NO",
               "PHASE4_EXT_PREDICTIONS_OPENED": "NO",
               "PROJECT_A_CONTENT_READ": "NO",
               "VCC_TEST_FEATURE_OPEN_COUNT": 0, "VCC_TEST_TARGET_OPEN_COUNT": 0,
               "VCC_TEST_PATH_OPEN_COUNT": 0,
               "Y_OR_LABEL_COLUMN_REQUESTED": "NO",
               "note": "every scientific path opened by this runner is listed above"},
              open(os.path.join(out, "hfm4_input_access_manifest.json"), "w"), indent=2)

    json.dump(env, open(os.path.join(out, "hfm4_environment.json"), "w"), indent=2, default=str)

    man = {"run_id": run_id, "utc_start": utc_start, "utc_end": utc_end, "git": git,
           "command": sys.argv, "output_root": out,
           "scientific_status": "TRAIN_VALIDATION_ONLY_SUPPORTING_DIAGNOSTIC",
           "governing_documents": GOV_DOCS(), "environment_lock": ENV_LOCK(),
           "environment": env, "thread_policy": freeze["thread_policy"],
           "canonical_row_order": freeze["canonical_row_order"], "canonical_key": freeze["canonical_key"],
           "row_order_validation": prov, "split_counts": {"training": 5740, "validation": 2052},
           "alpha_grid": list(ALPHA_GRID), "scalers": scaler_report,
           "scaler_provenance": "A10 LOAD-ONLY; artifact SHA-256 verified against "
                                "PHASEB_NUMERICAL_BASELINE_MANIFEST.json",
           "c2_matrix_sha256": C2MAT_SHA, "c3_seeds": c3_seeds,
           "STANDARD_SCALER_FIT_CALL_COUNT_REAL_RUN": FIT_COUNTS["fit"],
           "STANDARD_SCALER_FIT_TRANSFORM_CALL_COUNT_REAL_RUN": FIT_COUNTS["fit_transform"],
           "UNIQUE_SCIENTIFIC_ALPHA_EVALUATIONS": fits["unique"],
           "TOTAL_RIDGE_FIT_CALLS": fits["total"],
           "M_TILDE_TRAIN_LOADED": "YES", "M_TILDE_VALIDATION_LOADED": "YES",
           "M_TILDE_TEST_LOADED": "NO", "M_TILDE_EXTERNAL_LOADED": "NO",
           "VCC_TEST_ACCESSED": "NO", "EXTERNAL_OUTCOMES_ACCESSED": "NO",
           "FM_INFERENCE_PERFORMED": "NO", "GPU_WORKLOAD": "NO",
           "HFM4_INTERVAL_COMPUTED": "NO", "HFM4_BOOTSTRAP_COMPUTED": "NO",
           "HFM4_PVALUE_COMPUTED": "NO",
           "ORIGINAL_TEST_LEVEL_HFM4": "UNRESOLVED (execution note §17)",
           "RUN_STATUS": "COMPLETE"}
    for fn in ("HFM4_PRE_RUN_FREEZE.json", "HFM4_PRE_RUN_FREEZE.md", "hfm4_results.json",
               "hfm4_results.csv", "hfm4_guard_tests.json", "hfm4_input_access_manifest.json",
               "hfm4_environment.json"):
        man.setdefault("output_sha256", {})[fn] = PB.sha_file(os.path.join(out, fn))
    json.dump(man, open(os.path.join(out, "hfm4_run_manifest.json"), "w"), indent=2, default=str)
    print(f"[done] {utc_end}  unique_alpha_evals={fits['unique']} total_ridge_fits={fits['total']} "
          f"scaler_fit_calls={FIT_COUNTS['fit']}", flush=True)


def GOV_DOCS():
    d = {}
    for rel in ("iclr2027_revision/experiment_specs/option_b_geneformer_experiment_lock.md",
                "iclr2027_revision/experiment_specs/part1_scientific.md",
                "iclr2027_revision/experiment_specs/implementation_preflight.md",
                "iclr2027_revision/experiment_specs/hfm4_execution_note.md",
                "iclr2027_revision/experiment_specs/lock_amendment_A10_phaseB_numerical_baseline.md",
                "iclr2027_revision/pipeline/phaseB_preprocessing.py",
                "iclr2027_revision/pipeline/run_hfm4_tv.py"):
        p = os.path.join(REPO, rel)
        if os.path.exists(p):
            d[rel] = PB.sha_file(p)
    return d


def ENV_LOCK():
    d = {}
    for rel in ("iclr2027_revision/environment/phaseB/phaseB_requirements_locked.txt",
                "iclr2027_revision/environment/phaseB/phaseB_environment_README.md",
                "iclr2027_revision/provenance/PHASEB_NUMERICAL_BASELINE_MANIFEST.json"):
        d[rel] = PB.sha_file(os.path.join(REPO, rel))
    return d


def write_freeze_md(path, fr):
    L = ["# H-FM4 PRE-RUN FREEZE", "",
         "**Written before the first Ridge fit on scientific data.**", "",
         f"- run_id: `{fr['run_id']}`",
         f"- utc_freeze: {fr['utc_freeze']}   (utc_start {fr['utc_start']})",
         f"- git: `{fr['git']['sha']}` on `{fr['git']['branch']}` (dirty={fr['git']['dirty']})",
         f"- output_root: `{fr['output_root']}`",
         f"- scientific status: **{fr['scientific_status']}**", "",
         "## Pre-freeze declarations", "",
         "```", f"HFM4_RESULT_OBSERVED_BEFORE_FREEZE       = {fr['HFM4_RESULT_OBSERVED_BEFORE_FREEZE']}",
         f"REAL_HFM4_RIDGE_FIT_COUNT_BEFORE_FREEZE = {fr['REAL_HFM4_RIDGE_FIT_COUNT_BEFORE_FREEZE']}",
         f"VCC_TEST_ACCESSED_BEFORE_FREEZE         = {fr['VCC_TEST_ACCESSED_BEFORE_FREEZE']}",
         f"EXTERNAL_ACCESSED_BEFORE_FREEZE         = {fr['EXTERNAL_ACCESSED_BEFORE_FREEZE']}", "```", "",
         "## Governing documents (SHA-256)", ""]
    for k, v in fr["governing_documents"].items():
        L.append(f"- `{k}`  `{v}`")
    L += ["", "## Environment lock (SHA-256)", ""]
    for k, v in fr["environment_lock"].items():
        L.append(f"- `{k}`  `{v}`")
    e = fr["environment"]
    L += ["", "## Environment", "", "```",
          f"python={e['python']} numpy={e['numpy']} scipy={e['scipy']} sklearn={e['sklearn']}",
          f"joblib={e['joblib']} threadpoolctl={e['threadpoolctl']} pyarrow={e['pyarrow']}",
          f"torch={e['torch']} cuda={e['cuda']} gpu={e['gpu_model']}",
          f"thread_env={e['thread_env']}",
          f"threadpool_num_threads={fr['thread_policy']['realized_num_threads']} "
          f"check={fr['thread_policy']['check']}", "```", "",
          "## Scope", "", "```",
          f"authorized splits = {fr['authorized_splits']}",
          f"forbidden splits  = {fr['forbidden_splits']}",
          f"expected rows     = {fr['expected_rows']}",
          f"canonical key     = {fr['canonical_key']}",
          f"canonical order   = {fr['canonical_row_order']}",
          f"probe inputs      = {fr['probe_inputs']}",
          f"targets           = {fr['targets']} (object {fr['target_object']})",
          f"alpha grid        = {fr['alpha_grid']}",
          f"C2 matrix sha256  = {fr['c2_matrix_sha256']}  ({fr['c2_dim']}, seed {fr['c2_seed_of_record']})",
          f"C3 seed rule      = {fr['c3_seed_rule']}",
          f"C3 seeds          = {fr['c3_seeds']}", "```", "",
          "## Scaler rule", "", "```", fr["scaler_rule"],
          f"STANDARD_SCALER_FIT_CALL_COUNT_EXPECTED = {fr['STANDARD_SCALER_FIT_CALL_COUNT_EXPECTED']}"]
    for n in ("z", "c2", "ceiling_x"):
        s = fr["scalers"][n]
        L.append(f"{n:10s} {s['sha256']}  n_features_in_={s['n_features_in_']} "
                 f"n_samples_seen_={s['n_samples_seen_']}")
    L += [fr["scalers"]["C3_scaler"], fr["scalers"]["ceiling_scaler_persisted_phase0_counterpart"], "```", "",
          "## Result-handling rules", "", "```",
          f"tie       : {fr['tie_rule']}", f"non-finite: {fr['nonfinite_rule']}",
          f"CI        : {fr['ci_rule']}", f"bootstrap : {fr['bootstrap_rule']}",
          f"p-value   : {fr['pvalue_rule']}", f"threshold : {fr['threshold_rule']}",
          f"reporting : {fr['reporting_rule']}", f"repeats   : {fr['repeat_rule']}", "```", "",
          "## Synthetic fail-closed guard tests", "",
          "| test | expected | observed | result |", "|---|---|---|---|"]
    for t in fr["guard_tests"]:
        L.append(f"| {t['test']} | {t['expected']} | `{str(t['observed'])[:150]}` | **{t['result']}** |")
    L += ["", f"StandardScaler instrumentation proof (counters after the deliberate synthetic calls): "
              f"`{fr['standardscaler_instrumentation_proof']}` — reset to 0 before the real run.", ""]
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
