"""Phase-B canonical preprocessing — FIT ONCE, LOAD THEREAFTER.

Governed by lock amendment A10 (prospective, post-unblinding numerical-execution rule).

Scientific feature construction is unchanged and is inherited from the frozen record:
  canonical TRAIN row order  lock/D3A via pipeline/m_tilde.py:38-47 and pipeline/phase1_harness.py:60,69
  dtype path                 D3B via pipeline/fm_preprocess.py:15-16
  C2 construction            lock:316-318 via pipeline/controls.py:37-49
  scaler scope               lock:167-171 (fit on VCC train only)

This module NEVER: loads y or any label column, reads validation/test/external data, loads m_tilde,
imports or fits any predictive model, computes any metric, or touches a GPU.

Future Phase-B scientific runners MUST call load_scalers() and MUST NOT call fit_baseline().
"""
from __future__ import annotations
import os, sys, glob, json, hashlib
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
BASELINE_DIR = os.path.join(REPO, "iclr2027_revision", "provenance", "phaseB_baseline")
DHVG_REL = "IGA_NeurIPS/cache/vcc/variants/delta_hvg"
BRIDGE = os.path.join(REPO, "iclr2027_revision/phase0/bridge/vcc_z_label_bridge.parquet")
C2MAT = os.path.join(REPO, "iclr2027_revision/phase0/controls/c2_projection_2000x768.npy")
C2MAT_SHA = "e5bf7ce0f07cd68d73c41529b7f01441b1b5a6f9193d663bc28bc6f974d8f41a"
SPLIT = "training"
TRAIN_SHARDS, TRAIN_ROWS = 48, 5740
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
DIGEST_SPEC_VERSION = 1


class PhaseBError(RuntimeError):
    pass


# ---------------- thread policy (A10 §5.1/§5.2) ---------------- #
def verify_thread_policy():
    """Must run before any numerical work. Aborts if the realized pool violates the policy."""
    requested = {v: os.environ.get(v, "UNSET") for v in THREAD_VARS}
    bad = [v for v, s in requested.items() if s != "1"]
    if bad:
        raise PhaseBError(f"thread variables not set to 1 before interpreter start: {bad}")
    import threadpoolctl
    realized = threadpoolctl.threadpool_info()
    over = [p for p in realized if p.get("num_threads") not in (1, None)]
    if over:
        raise PhaseBError(f"realized threadpool violates policy: "
                          f"{[(p.get('prefix'), p.get('num_threads')) for p in over]}")
    return {"THREAD_POLICY_REQUESTED": requested, "THREAD_POLICY_REALIZED": realized,
            "THREAD_POLICY_CHECK": "PASS"}


# ---------------- digests (content-projection spec v1) ---------------- #
def _u64(n): return int(n).to_bytes(8, "little")

class _Proj:
    def __init__(self): self.h = hashlib.sha256()
    def tok(self, s):
        b = str(s).encode("utf-8"); self.h.update(_u64(len(b))); self.h.update(b)
    def arr(self, name, a):
        self.tok(name); self.tok(str(a.dtype)); self.tok(",".join(map(str, a.shape)))
        b = np.ascontiguousarray(a).tobytes(order="C"); self.h.update(_u64(len(b))); self.h.update(b)
    def seq(self, name, vals):
        self.tok(name); self.tok("utf8_sequence"); self.tok(len(vals))
        for v in vals:
            if v is None: raise PhaseBError(f"missing identifier in {name}")
            self.tok(v)
    def hex(self): return self.h.hexdigest()

def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

def _state_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes(order="C")).hexdigest()


# ---------------- canonical TRAIN inputs ---------------- #
def canonical_train_inputs(storage_root):
    """Return (x, z, C2, R, provenance). TRAIN only. No y, no labels, no m_tilde."""
    import pyarrow.parquet as pq
    dhvg = os.path.join(storage_root, DHVG_REL)
    files = sorted(glob.glob(os.path.join(dhvg, "shards", "train", "*.npz")))
    if len(files) != TRAIN_SHARDS:
        raise PhaseBError(f"TRAIN_SHARDS {len(files)} != {TRAIN_SHARDS}")
    npz = _Proj(); npz.tok("CONTENT_PROJECTION_DIGEST_SPEC_VERSION"); npz.tok(DIGEST_SPEC_VERSION)
    npz.tok("logical_artifact"); npz.tok("vcc_delta_hvg_train_shards_x_projection")
    keys, xs = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        b = str(z["batch_name"][0]); sid = [str(s) for s in z["sample_ids"]]
        x = np.asarray(z["x"], dtype=np.float32)          # y is never read
        npz.tok("shard_name"); npz.tok(os.path.basename(f))
        npz.tok("batch_name"); npz.tok(b); npz.seq("sample_ids", sid); npz.arr("x", x)
        keys += [(SPLIT, b, s) for s in sid]; xs.append(x)
    X = np.concatenate(xs, axis=0)
    if X.shape != (TRAIN_ROWS, 2000) or X.dtype != np.float32:
        raise PhaseBError(f"x shape/dtype {X.shape}/{X.dtype}")
    ZC = [f"gf_z_emb_{i}" for i in range(768)]
    t = pq.read_table(BRIDGE, columns=["split", "batch", "target_gene"] + ZC)   # label never requested
    sp = [str(v) for v in t.column("split").to_pylist()]
    ba = [str(v) for v in t.column("batch").to_pylist()]
    tg = [str(v) for v in t.column("target_gene").to_pylist()]
    sel = [i for i, s in enumerate(sp) if s == SPLIT]
    bk = [(sp[i], ba[i], tg[i]) for i in sel]
    if len(set(keys)) != len(keys): raise PhaseBError("shard keys not unique")
    if len(set(bk)) != len(bk):     raise PhaseBError("bridge train keys not unique")
    if set(keys) != set(bk):        raise PhaseBError("key sets differ")
    Zsrc = np.column_stack([np.asarray(t.column(c).to_numpy(zero_copy_only=False)) for c in ZC])[sel]
    Z32 = Zsrc.astype(np.float32)                          # D3B historical lossy round-trip
    pos = {k: i for i, k in enumerate(bk)}
    Zc = np.ascontiguousarray(Z32[np.array([pos[k] for k in keys], dtype=np.int64)])
    if Zc.shape != (TRAIN_ROWS, 768): raise PhaseBError(f"z shape {Zc.shape}")
    bp = _Proj(); bp.tok("CONTENT_PROJECTION_DIGEST_SPEC_VERSION"); bp.tok(DIGEST_SPEC_VERSION)
    bp.tok("logical_artifact"); bp.tok("vcc_z_bridge_train_projection")
    bp.tok("row_order"); bp.tok("D3A_canonical_shard_key_order")
    bp.seq("split", [k[0] for k in keys]); bp.seq("batch", [k[1] for k in keys])
    bp.seq("target_gene", [k[2] for k in keys]); bp.arr("z_float32_canonical", Zc)
    if sha_file(C2MAT) != C2MAT_SHA: raise PhaseBError("frozen C2 matrix SHA-256 mismatch")
    R = np.load(C2MAT)
    if R.shape != (2000, 768) or R.dtype != np.float32: raise PhaseBError("C2 matrix shape/dtype")
    C2 = (X @ R).astype(np.float32)                        # float32 GEMM, controls.py:49
    cp = _Proj(); cp.tok("CONTENT_PROJECTION_DIGEST_SPEC_VERSION"); cp.tok(DIGEST_SPEC_VERSION)
    cp.tok("logical_artifact"); cp.tok("vcc_c2_train_projection"); cp.arr("c2_float32", C2)
    prov = {
        "canonical_row_order": "sorted(glob(shards/train/*.npz)) then stored sample_ids order",
        "canonical_key": "(split, batch_name, sample_id) == bridge (split, batch, target_gene)",
        "TRAIN_SHARDS": len(files), "TRAIN_ROWS": len(keys),
        "SHARD_KEYS_UNIQUE": True, "BRIDGE_TRAIN_KEYS_UNIQUE": True, "KEY_SET_EQUAL": True,
        "MISSING_KEYS": 0, "EXTRA_KEYS": 0,
        "PHYSICAL_BRIDGE_SEQUENCE_EQUAL": bool(bk == keys),
        "PHASEB_X_TRAIN_PROJECTION_SHA256": npz.hex(),
        "PHASEB_Z_TRAIN_PROJECTION_SHA256": bp.hex(),
        "PHASEB_C2_PROJECTION_MATRIX_SHA256": C2MAT_SHA,
        "PHASEB_C2_TRAIN_PROJECTION_SHA256": cp.hex(),
        "dtype_path": {"x": "float32 stored -> float64 at fit",
                       "z": "bridge double -> float32 cast -> canonical reorder -> float64 at fit",
                       "C2": "float32 x @ float32 R -> float32 -> float64 at fit"},
    }
    return X, Zc, C2, R, prov


# ---------------- one-time baseline fit ---------------- #
def _fit(block):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler(with_mean=True, with_std=True).fit(np.asarray(block, dtype=np.float64))

def _save(sc, path, name, fit_input_digest):
    n = sc.n_samples_seen_
    np.savez(path, mean_=np.asarray(sc.mean_, dtype=np.float64),
             scale_=np.asarray(sc.scale_, dtype=np.float64),
             var_=np.asarray(sc.var_, dtype=np.float64),
             n_samples_seen_=np.asarray(int(np.asarray(n).item()), dtype=np.int64))
    return {"name": name, "artifact": os.path.relpath(path, REPO),
            "artifact_sha256": sha_file(path),
            "n_features": int(sc.mean_.shape[0]),
            "n_samples_seen_": int(np.asarray(n).item()),
            "fit_input_digest": fit_input_digest,
            "state_digests": {f: _state_digest(getattr(sc, f)) for f in ("mean_", "scale_", "var_")}}

def fit_baseline(storage_root, out_dir=BASELINE_DIR):
    """ONE-TIME baseline fit. Not callable by scientific runners (see load_scalers)."""
    tp = verify_thread_policy()
    X, Z, C2, R, prov = canonical_train_inputs(storage_root)
    os.makedirs(out_dir, exist_ok=True)
    rec = {}
    for name, block in (("z", Z), ("c2", C2), ("ceiling_x", X)):
        sc = _fit(block)
        rec[name] = _save(sc, os.path.join(out_dir, f"phaseB_{name}_scaler_state.npz"), name,
                          _state_digest(np.asarray(block, dtype=np.float64)))
    return {"thread_policy": tp, "inputs": prov, "scalers": rec}


# ---------------- load-only path for future runners ---------------- #
def load_scalers(names=("z", "c2", "ceiling_x"), out_dir=BASELINE_DIR, expected_sha=None):
    """The ONLY path a Phase-B scientific runner may use. Returns fitted StandardScaler objects
    rehydrated from version-independent state arrays. Never fits."""
    from sklearn.preprocessing import StandardScaler
    out = {}
    for name in names:
        p = os.path.join(out_dir, f"phaseB_{name}_scaler_state.npz")
        if not os.path.exists(p): raise PhaseBError(f"missing canonical scaler artifact: {p}")
        got = sha_file(p)
        if expected_sha and expected_sha.get(name) and expected_sha[name] != got:
            raise PhaseBError(f"{name} scaler artifact SHA-256 mismatch: {got}")
        d = np.load(p)
        sc = StandardScaler(with_mean=True, with_std=True)
        sc.mean_, sc.scale_, sc.var_ = d["mean_"], d["scale_"], d["var_"]
        sc.n_samples_seen_ = int(d["n_samples_seen_"])
        sc.n_features_in_ = int(sc.mean_.shape[0])
        out[name] = sc
    return out


def refit_forbidden(*_a, **_k):
    raise PhaseBError("A10: Phase-B scientific runners must LOAD canonical scaler state, never refit.")
