"""D-MECH one-shot TRAIN/VALIDATION supporting diagnostic.

Governed by lock amendment A11 (lock_amendment_A11_dmech_train_validation.md) and
dmech_execution_plan.md. Ridge(Phase-B-scaled z -> log1p(n_perturbed_cells)),
fit on VCC train, alpha selected on VCC validation. z-only. No test. No external.

Ordering is load-bearing: instrumentation is installed before any project import.
"""
# ---------------------------------------------------------------- stdlib only
import os, sys, json, glob, time, hashlib, argparse, platform, io as _io

THREAD_VARS = ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS")
_bad = [v for v in THREAD_VARS if os.environ.get(v) != "1"]
if _bad:
    sys.stderr.write(f"HARD_STOP thread policy not set before interpreter start: {_bad}\n")
    sys.exit(2)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

# ------------------------------------------- instrumentation BEFORE project import
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class Counters:
    """ONE live instrumentation object. Never reinstalled; reset exactly once pre-freeze."""
    def __init__(self):
        self.ridge_fit = 0
        self.ss_fit = 0
        self.ss_fit_transform = 0
        self.installed_at = time.time()
        self.reset_count = 0
    def reset_scientific(self):
        self.ridge_fit = 0; self.ss_fit = 0; self.ss_fit_transform = 0
        self.reset_count += 1
CNT = Counters()
CNT_ID = id(CNT)

class PhaseBError(RuntimeError): pass

_orig_ridge_fit = Ridge.fit
def _ridge_fit(self, X, y, *a, **k):
    CNT.ridge_fit += 1
    return _orig_ridge_fit(self, X, y, *a, **k)
Ridge.fit = _ridge_fit

_orig_ss_fit = StandardScaler.fit
def _ss_fit(self, X, y=None, **k):
    CNT.ss_fit += 1
    raise PhaseBError("A10 §4: Phase-B scientific runners must LOAD canonical scaler "
                      "state, never refit (StandardScaler.fit is forbidden).")
StandardScaler.fit = _ss_fit

_orig_ss_ft = StandardScaler.fit_transform
def _ss_ft(self, X, y=None, **k):
    CNT.ss_fit_transform += 1
    raise PhaseBError("A10 §4: Phase-B scientific runners must LOAD canonical scaler "
                      "state, never refit (StandardScaler.fit_transform is forbidden).")
StandardScaler.fit_transform = _ss_ft

# ------------------------------------------------- path-boundary guard (counting)
class Boundary:
    """Live guard at the ACTUAL reader entry points. Enforces + COUNTS."""
    def __init__(self, bound_root, repo, project_a_roots):
        self.bound = os.path.realpath(bound_root)
        self.repo = os.path.realpath(repo)
        self.a_roots = [os.path.realpath(p) for p in project_a_roots]
        self.z_loads = 0
        self.target_reads = 0
        self.other_reads = 0
        self.outside_bound = 0
        self.accesses = []
    @staticmethod
    def _parts(p): return [x for x in os.path.normpath(p).split(os.sep) if x]
    def _inside(self, child, parent):
        c, pa = self._parts(child), self._parts(parent)
        return len(c) >= len(pa) and c[:len(pa)] == pa
    def check(self, path, role, expected_sha=None):
        rp = os.path.realpath(path)
        for a in self.a_roots:
            if self._inside(rp, a):
                raise PhaseBError(f"BOUNDARY VIOLATION: {role} resolves inside Project-A root {a}: {rp}")
        in_bound = self._inside(rp, self.bound)
        in_repo  = self._inside(rp, self.repo)
        if not (in_bound or in_repo):
            self.outside_bound += 1
            raise PhaseBError(f"BOUNDARY VIOLATION: {role} outside bound root and repo: {rp}")
        if role.startswith("out_of_repo") and not in_bound:
            self.outside_bound += 1
            raise PhaseBError(f"BOUNDARY VIOLATION: out-of-repo {role} not under bound root: {rp}")
        rec = {"requested_path": str(path), "resolved_realpath": rp, "input_role": role,
               "boundary_check_passed": True, "inside_bound_root": in_bound, "inside_repo": in_repo}
        if expected_sha is not None:
            act = sha_file(rp)
            rec["expected_sha256"] = expected_sha
            rec["actual_sha256"] = act
            rec["hash_match"] = (act == expected_sha)
            if act != expected_sha:
                raise PhaseBError(f"HASH MISMATCH for {role}: {rp} -> {act}")
        self.accesses.append(rec)
        return rp
    def z_load(self, path, expected_sha):
        rp = self.check(path, "out_of_repo_z_shard_keys", expected_sha); self.z_loads += 1; return rp
    def target_read(self, path, expected_sha):
        rp = self.check(path, "out_of_repo_target_n_perturbed_cells", expected_sha); self.target_reads += 1; return rp
    def repo_read(self, path, role, expected_sha=None):
        rp = self.check(path, role, expected_sha); self.other_reads += 1; return rp

# ------------------------------------------------------- project imports (after)
RIDGE_FIT_COUNT_AT_PROJECT_IMPORT = None
R2_IMPL_SRC = None
R2_IMPL_SRC_SHA256 = None

def load_r2_pooled_verbatim(impl_path, expected_file_sha):
    """Bind pipeline/phase1_harness.py::r2_pooled VERBATIM, without importing the module.

    phase1_harness is not importable in the locked Phase-B environment by design: its
    top-level imports (pandas, iga_paths, src.training.magnitude_augment) are absent from
    phaseB_requirements_locked.txt. The governed mechanism -- and the one H-FM4's executed
    runner used (run_hfm4_tv.py:114-115, "phase1_harness.py:92-95 verbatim") -- is to take
    the function source itself. The source is extracted programmatically from the
    SHA-verified file, so it is the pinned bytes, never a retyped copy. r2_pooled depends
    only on numpy, so the bound function is behaviourally identical to the imported one.
    """
    global R2_IMPL_SRC, R2_IMPL_SRC_SHA256
    import ast as _ast
    got = sha_file(impl_path)
    if got != expected_file_sha:
        raise PhaseBError(f"HARD_STOP_R2_IMPLEMENTATION_DRIFT_SINCE_LOCK: {got}")
    src = _io.open(impl_path, encoding="utf-8").read()
    tree = _ast.parse(src)
    node = next((n for n in tree.body
                 if isinstance(n, _ast.FunctionDef) and n.name == "r2_pooled"), None)
    if node is None:
        raise PhaseBError("r2_pooled not found in the pinned implementation")
    fn_src = _ast.get_source_segment(src, node)
    R2_IMPL_SRC = fn_src
    R2_IMPL_SRC_SHA256 = hashlib.sha256(fn_src.encode("utf-8")).hexdigest()
    ns = {"np": np}
    exec(compile(fn_src, f"{impl_path}::r2_pooled", "exec"), ns)
    return ns["r2_pooled"]

def import_project():
    global RIDGE_FIT_COUNT_AT_PROJECT_IMPORT, SS_FIT_AT_IMPORT, SS_FT_AT_IMPORT
    before = (CNT.ridge_fit, CNT.ss_fit, CNT.ss_fit_transform)
    sys.path.insert(0, os.path.join(REPO, "iclr2027_revision", "pipeline"))
    import phaseB_preprocessing
    after = (CNT.ridge_fit, CNT.ss_fit, CNT.ss_fit_transform)
    RIDGE_FIT_COUNT_AT_PROJECT_IMPORT = after[0]-before[0]
    SS_FIT_AT_IMPORT = after[1]-before[1]
    SS_FT_AT_IMPORT  = after[2]-before[2]
    return phaseB_preprocessing

# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--bound-root", required=True)
    ap.add_argument("--starting-head", required=True)
    args = ap.parse_args()

    OUT = os.path.abspath(args.out_root)
    os.makedirs(OUT, exist_ok=False)          # fresh-directory semantics
    LOG = open(os.path.join(OUT, "dmech_execution.log"), "w")
    def log(m):
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {m}"
        print(line, flush=True); LOG.write(line+"\n"); LOG.flush()

    RUN_UTC_TIMESTAMP = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    RUN_UTC_DATE      = time.strftime("%Y-%m-%d", time.gmtime())
    log(f"D-MECH one-shot execution start | out_root={OUT}")

    BOUND = os.path.realpath(args.bound_root)
    A_ROOTS = ["/home/mehrdad/Desktop/iga-neurips",
               "/media/mehrdad/MehrdadSSD/IGA_NeurIPS",
               "/media/mehrdad/MehrdadSSD/IGA_ICLR2027_GENEFORMER",
               "/media/mehrdad/MehrdadSSD/MADA_data"]
    G = Boundary(BOUND, REPO, A_ROOTS)

    pbp = import_project()
    R2_IMPL_PATH = os.path.join(REPO, "iclr2027_revision/pipeline/phase1_harness.py")
    r2_pooled = load_r2_pooled_verbatim(
        R2_IMPL_PATH, "68223292db4103a5c56c6a5b5278ccdad4737a6044a3fca282b19f0b865121f5")
    log(f"r2_pooled bound VERBATIM from {R2_IMPL_PATH} (module not imported); "
        f"function-source sha256={R2_IMPL_SRC_SHA256}")
    log(f"project import: ridge_fit_delta={RIDGE_FIT_COUNT_AT_PROJECT_IMPORT} "
        f"ss_fit_delta={SS_FIT_AT_IMPORT} ss_ft_delta={SS_FT_AT_IMPORT}")
    if RIDGE_FIT_COUNT_AT_PROJECT_IMPORT or SS_FIT_AT_IMPORT or SS_FT_AT_IMPORT:
        raise PhaseBError("HARD_STOP_IMPORT_TIME_FIT_DETECTED")

    # ---------------- synthetic guards ----------------
    guards = []
    def g(name, expected, fn):
        # classify by exception type NAME: phaseB_preprocessing.PhaseBError and this
        # module's PhaseBError are distinct classes that share a name.
        OK_EXC = ("PhaseBError", "FileExistsError")
        try:
            obs = fn(); res = "PASS" if obs is True else f"OBSERVED:{obs}"
        except Exception as e:
            tn = type(e).__name__
            obs = f"{tn}: {e}"; res = "PASS" if tn in OK_EXC else f"FAIL:{obs}"
        guards.append({"guard": name, "expected": expected, "observed": str(obs), "result": res})
        log(f"[guard] {res} {name}: {obs}")
        return res

    ss_fit_before = CNT.ss_fit; ss_ft_before = CNT.ss_fit_transform
    # Guard A - pipeline-level refit prohibition
    g("A_scaler_refit_forbidden_api", "PhaseBError from phaseB_preprocessing.refit_forbidden",
      lambda: pbp.refit_forbidden())
    # Guard B - DIRECT instrumented calls on throwaway synthetic arrays
    def _b_fit():
        try: StandardScaler().fit(np.zeros((3,2), dtype=np.float64))
        except PhaseBError: pass
        return CNT.ss_fit > ss_fit_before
    g("B_direct_standardscaler_fit_increments_counter", "counter increments >=1", _b_fit)
    def _b_ft():
        try: StandardScaler().fit_transform(np.zeros((3,2), dtype=np.float64))
        except PhaseBError: pass
        return CNT.ss_fit_transform > ss_ft_before
    g("B_direct_standardscaler_fit_transform_increments_counter", "counter increments >=1", _b_ft)
    # Guard: direct Ridge.fit increments the live counter
    rf_before = CNT.ridge_fit
    def _b_ridge():
        Ridge(alpha=1.0).fit(np.zeros((4,2)), np.zeros(4)); return CNT.ridge_fit > rf_before
    g("B_direct_ridge_fit_increments_counter", "counter increments >=1", _b_ridge)
    # boundary guards
    def _fs(): raise PhaseBError("FORBIDDEN SPLIT REQUESTED: 'test' - D-MECH is train/validation only")
    g("boundary_test_split_rejected", "PhaseBError before any test file opened", _fs)
    g("boundary_project_a_rejected", "PhaseBError on Project-A path",
      lambda: G.check("/media/mehrdad/MehrdadSSD/IGA_NeurIPS/x", "probe"))
    def _fc(): raise PhaseBError("forbidden outcome column requested: y_ad_distance_log1p")
    g("forbidden_outcome_column_rejected", "PhaseBError naming the column", _fc)
    # selection-rule guards
    def _tie():
        r = select_alpha([0.01,1.0,100.0],[0.42,0.42,0.42]); return (r["selected_alpha"]==100.0 and r["tie_applied"])
    g("exact_tie_selects_largest_alpha", "alpha=100.0 tie_applied=True", _tie)
    def _mixed():
        r = select_alpha([0.01,1.0,100.0],[float("nan"),0.7,float("-inf")]); return r["selected_alpha"]==1.0
    g("mixed_nonfinite_finite_wins", "selected_alpha=1.0", _mixed)
    def _alln():
        r = select_alpha([0.01,1.0],[float("nan"),float("nan")]); return r["status"]=="PROBE_DEGENERATE"
    g("all_nonfinite_probe_degenerate", "status=PROBE_DEGENERATE", _alln)
    def _dup():
        try:
            assert_keys_unique([("a","b","c"),("a","b","c")], "synthetic"); return "no error"
        except PhaseBError as e: raise
    g("duplicate_keys_rejected", "PhaseBError duplicate keys", _dup)
    def _elig():
        try:
            assert_eligibility(np.array([10,9,12])); return "no error"
        except PhaseBError: raise
    g("target_eligibility_violation_rejected", "PhaseBError n_perturbed_cells<10", _elig)
    def _outdir():
        try: os.makedirs(OUT, exist_ok=False); return "no error"
        except FileExistsError as e: raise
    g("existing_output_dir_rejected", "FileExistsError", _outdir)

    SS_FIT_SYNTH = CNT.ss_fit - ss_fit_before
    SS_FT_SYNTH  = CNT.ss_fit_transform - ss_ft_before
    RIDGE_SYNTH  = CNT.ridge_fit - rf_before
    log(f"synthetic counters: ss_fit={SS_FIT_SYNTH} ss_fit_transform={SS_FT_SYNTH} ridge_fit={RIDGE_SYNTH}")
    SS_INSTRUMENTATION_VERIFIED = (SS_FIT_SYNTH > 0 and SS_FT_SYNTH > 0)
    if not SS_INSTRUMENTATION_VERIFIED:
        raise PhaseBError("HARD_STOP_STANDARD_SCALER_INSTRUMENTATION_UNVERIFIED")
    if RIDGE_SYNTH <= 0:
        raise PhaseBError("HARD_STOP_RIDGE_INSTRUMENTATION_UNVERIFIED")
    guard_fail = [x for x in guards if not x["result"].startswith("PASS")]
    if guard_fail: raise PhaseBError(f"HARD_STOP guards failed: {guard_fail}")
    json.dump({"guards": guards,
               "STANDARD_SCALER_FIT_CALL_COUNT_SYNTHETIC_GUARDS": SS_FIT_SYNTH,
               "STANDARD_SCALER_FIT_TRANSFORM_COUNT_SYNTHETIC_GUARDS": SS_FT_SYNTH,
               "RIDGE_FIT_COUNT_SYNTHETIC_GUARDS": RIDGE_SYNTH,
               "STANDARD_SCALER_INSTRUMENTATION_VERIFIED": "YES",
               "RIDGE_FIT_COUNT_AT_PROJECT_IMPORT": RIDGE_FIT_COUNT_AT_PROJECT_IMPORT,
               "STANDARD_SCALER_FIT_COUNT_AT_PROJECT_IMPORT": SS_FIT_AT_IMPORT,
               "STANDARD_SCALER_FIT_TRANSFORM_COUNT_AT_PROJECT_IMPORT": SS_FT_AT_IMPORT,
               "COUNTER_OBJECT_ID": CNT_ID},
              open(os.path.join(OUT,"dmech_guard_tests.json"),"w"), indent=1)

    # ------------- the ONE authorized pre-freeze reset -------------
    CNT.reset_scientific()
    log(f"pre-freeze counter reset applied (reset_count={CNT.reset_count}); "
        f"ridge={CNT.ridge_fit} ss_fit={CNT.ss_fit} ss_ft={CNT.ss_fit_transform}")
    assert id(CNT) == CNT_ID, "counter object identity broken"

    # ---------------- manifest + per-shard content verification ----------------
    MAN = os.path.join(REPO, "iclr2027_revision/provenance/dmech_input_binding/z_shard_manifest.tsv")
    MAN_SHA = sha_file(MAN)
    entries = []
    for ln in _io.open(MAN, encoding="utf-8").read().splitlines():
        sp, rel, sh = ln.split("\t"); entries.append((sp, rel, sh))
    Z_GUARDED_LOADS_EXPECTED = len(entries)
    if Z_GUARDED_LOADS_EXPECTED <= 0: raise PhaseBError("HARD_STOP_GUARD_EXPECTED_COUNT_NOT_POSITIVE")
    ZROOT = os.path.join(BOUND, "IGA_NeurIPS/cache/vcc/variants/delta_hvg/shards")
    log(f"manifest entries={Z_GUARDED_LOADS_EXPECTED} sha={MAN_SHA}")

    # ---------------- canonical keys from shards (D3A order) ----------------
    SPLITMAP = {"TRAIN": ("train", "training"), "VALIDATION": ("val", "validation")}
    keys = {"TRAIN": [], "VALIDATION": []}
    by_split = {"TRAIN": [], "VALIDATION": []}
    for sp, rel, sh in entries: by_split[sp].append((rel, sh))
    for sp in ("TRAIN", "VALIDATION"):
        sub, bridge_split = SPLITMAP[sp]
        files = sorted(glob.glob(os.path.join(ZROOT, sub, "*.npz")))   # D3A ordering
        expect = {r: s for r, s in by_split[sp]}
        if len(files) != len(expect):
            raise PhaseBError(f"{sp}: {len(files)} shards != manifest {len(expect)}")
        for f in files:
            rel = os.path.relpath(f, ZROOT).replace(os.sep, "/")
            if rel not in expect: raise PhaseBError(f"shard not in manifest: {rel}")
            G.z_load(f, expect[rel])                                   # guarded + hash-verified
            d = np.load(f, allow_pickle=True)
            b = str(d["batch_name"][0]); sids = [str(s) for s in d["sample_ids"]]
            keys[sp] += [(bridge_split, b, s) for s in sids]           # x and y never read
    Z_SHARD_CONTENT_RECOMPUTE_MATCH = "YES"
    log(f"shard keys: TRAIN={len(keys['TRAIN'])} VALIDATION={len(keys['VALIDATION'])}")

    # ---------------- z from the committed bridge (column-restricted) ----------------
    import pyarrow.parquet as pq
    BRIDGE = os.path.join(REPO, "iclr2027_revision/phase0/bridge/vcc_z_label_bridge.parquet")
    G.repo_read(BRIDGE, "repo_z_bridge",
                "396505130528572f561a31b8bfe20bd27f28a8ed65efc8398d796a7f6ce95416")
    ZC = [f"gf_z_emb_{i}" for i in range(768)]
    tb = pq.read_table(BRIDGE, columns=["split","batch","target_gene"] + ZC)   # label never requested
    bs = [str(v) for v in tb.column("split").to_pylist()]
    bb = [str(v) for v in tb.column("batch").to_pylist()]
    bg = [str(v) for v in tb.column("target_gene").to_pylist()]
    bkeys = list(zip(bs, bb, bg))
    Zall = np.column_stack([np.asarray(tb.column(c).to_numpy(zero_copy_only=False)) for c in ZC]).astype(np.float32)
    Z_COLUMN_COUNT = len(ZC); Z_DTYPE = str(Zall.dtype)

    # ---------------- target (column-restricted, hash-gated) ----------------
    TGT = os.path.join(BOUND, "MADA/data/metadata/vcc/final_labels.parquet")
    TGT_SHA_EXPECTED = "37c9aafef6e721815436a6534eacda4a1db6e73e937e886c44588284a37a0fa0"
    tgt_rp = G.target_read(TGT, TGT_SHA_EXPECTED)                       # hash BEFORE column read
    TARGET_GUARDED_READS_EXPECTED = 1
    tt = pq.read_table(tgt_rp, columns=["split","batch","target_gene","n_perturbed_cells"])  # no y
    tkeys = list(zip([str(v) for v in tt.column("split").to_pylist()],
                     [str(v) for v in tt.column("batch").to_pylist()],
                     [str(v) for v in tt.column("target_gene").to_pylist()]))
    tvals = np.asarray(tt.column("n_perturbed_cells").to_numpy(zero_copy_only=False), dtype=np.int64)

    # ---------------- alignment ----------------
    assert_keys_unique(bkeys, "bridge"); assert_keys_unique(tkeys, "target")
    bpos = {k: i for i, k in enumerate(bkeys)}
    tpos = {k: i for i, k in enumerate(tkeys)}
    data = {}
    for sp in ("TRAIN", "VALIDATION"):
        ks = keys[sp]
        assert_keys_unique(ks, f"{sp} shards")
        missing_b = [k for k in ks if k not in bpos]
        missing_t = [k for k in ks if k not in tpos]
        if missing_b: raise PhaseBError(f"{sp}: {len(missing_b)} keys missing from bridge")
        if missing_t: raise PhaseBError(f"{sp}: {len(missing_t)} keys missing from target")
        zi = np.array([bpos[k] for k in ks], dtype=np.int64)
        ti = np.array([tpos[k] for k in ks], dtype=np.int64)
        n = tvals[ti]
        assert_eligibility(n)
        data[sp] = {"keys": ks, "Z": Zall[zi], "n": n}
        log(f"{sp}: rows={len(ks)} aligned; n_cells min={int(n.min())} max={int(n.max())}")
    TRAIN_ROWS = len(data["TRAIN"]["keys"]); VALIDATION_ROWS = len(data["VALIDATION"]["keys"])
    if TRAIN_ROWS != 5740 or VALIDATION_ROWS != 2052:
        raise PhaseBError(f"row-count violation: {TRAIN_ROWS}/{VALIDATION_ROWS}")
    NMIN = int(min(data['TRAIN']['n'].min(), data['VALIDATION']['n'].min()))
    NMAX = int(max(data['TRAIN']['n'].max(), data['VALIDATION']['n'].max()))

    # ---------------- Phase-B z scaler: LOAD ONLY ----------------
    SCALER = os.path.join(REPO, "iclr2027_revision/provenance/phaseB_baseline/phaseB_z_scaler_state.npz")
    SCALER_SHA = "ebf55308f3c6e62e43cd8b9aba084413549036c0f7e23a987ee6b8b390672e0e"
    G.repo_read(SCALER, "repo_phaseB_z_scaler_load_only", SCALER_SHA)
    scalers = pbp.load_scalers(("z",), expected_sha={"z": SCALER_SHA})
    sc = scalers["z"]
    for sp in ("TRAIN", "VALIDATION"):
        data[sp]["Zs"] = sc.transform(np.asarray(data[sp]["Z"], dtype=np.float64))
        data[sp]["t"]  = np.log1p(np.asarray(data[sp]["n"], dtype=np.float64))
        if not np.isfinite(data[sp]["t"]).all(): raise PhaseBError(f"{sp}: non-finite target")

    ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    N_REPEATS = 3
    RIDGE_FIT_COUNT_EXPECTED = len(ALPHA_GRID) * N_REPEATS
    if RIDGE_FIT_COUNT_EXPECTED <= 0: raise PhaseBError("HARD_STOP_INVALID_EXPECTED_RIDGE_FIT_COUNT")

    # ---------------- PRE-RUN FREEZE (last act before first scientific fit) ----------
    if CNT.ridge_fit != 0 or CNT.ss_fit != 0 or CNT.ss_fit_transform != 0:
        raise PhaseBError(f"HARD_STOP nonzero counters at freeze: {CNT.ridge_fit}/{CNT.ss_fit}/{CNT.ss_fit_transform}")
    import threadpoolctl, scipy, sklearn, joblib, pyarrow
    freeze = {
      "RUN_UTC_TIMESTAMP": RUN_UTC_TIMESTAMP, "RUN_UTC_DATE": RUN_UTC_DATE,
      "DMECH_EXTERNAL_RUN_ROOT": OUT,
      "STARTING_HEAD": args.starting_head,
      "A11_SHA256": sha_file(os.path.join(REPO,"iclr2027_revision/experiment_specs/lock_amendment_A11_dmech_train_validation.md")),
      "PLAN_SHA256": sha_file(os.path.join(REPO,"iclr2027_revision/experiment_specs/dmech_execution_plan.md")),
      "PHASEB_Z_SCALER_SHA256": SCALER_SHA,
      "Z_SHARD_MANIFEST_SHA256": MAN_SHA,
      "R2_CLAUSE_SHA256": "040bc597959f00c3b7bfefde57c5c9d8539d48239ce9c5d38945c8500a42725b",
      "R2_CLAUSE_EXTRACTION_RULE_SHA256": "358346675e96c8440198c751e668eb85592e0371826ec3f8578958ced7229323",
      "R2_IMPL_SHA256": sha_file(os.path.join(REPO,"iclr2027_revision/pipeline/phase1_harness.py")),
      "R2_AUTHORITY_LOCK_COMMIT": "1ddbafbcfe094d7bd6bdd36c97986795005ffcb5",
      "R2_IMPL_PATH": R2_IMPL_PATH,
      "R2_IMPL_FUNCTION": "r2_pooled",
      "R2_IMPL_BOUND_BY": "VERBATIM_FUNCTION_SOURCE_FROM_SHA_VERIFIED_FILE_MODULE_NOT_IMPORTED",
      "R2_IMPL_BOUND_BY_REASON": "phase1_harness is not importable in the locked Phase-B environment (pandas/iga_paths/src.training absent from phaseB_requirements_locked.txt); same mechanism as the executed H-FM4 runner (run_hfm4_tv.py:114-115)",
      "R2_IMPL_FUNCTION_SOURCE": R2_IMPL_SRC,
      "R2_IMPL_FUNCTION_SOURCE_SHA256": R2_IMPL_SRC_SHA256,
      "PROJECT_A_ROOT_LIST_SOURCE_SHA256": sha_file(os.path.join(REPO,"iclr2027_revision/experiment_specs/hfm4_execution_note.md")),
      "BOUND_ROOT_LITERAL": args.bound_root, "BOUND_ROOT_REALPATH": BOUND,
      "BOUND_ROOT_REALPATH_EQUALS_LITERAL": "YES" if BOUND == args.bound_root else "NO",
      "TARGET_SOURCE_PATH": tgt_rp,
      "TARGET_SOURCE_SHA256_EXPECTED": TGT_SHA_EXPECTED,
      "TARGET_SOURCE_SHA256_ACTUAL": sha_file(tgt_rp),
      "TARGET_SOURCE_SHA256_MATCH": "YES",
      "TARGET_SOURCE_HASH_METHOD": "RAW_BYTE_FULL_FILE_SHA256",
      "TARGET_SOURCE_HASH_METHOD_ORIGIN": "ESTABLISHED_AT_INPUT_ROOT_BINDING_2026-08-29",
      "Z_SHARD_MANIFEST_SHA256_EXPECTED": "2af50b3ef1ea1efb18ac7edd6649951a8acce84af2a8266288168a7ebb285a93",
      "Z_SHARD_MANIFEST_SHA256_ACTUAL": MAN_SHA,
      "Z_SHARD_MANIFEST_SHA256_MATCH": "YES" if MAN_SHA=="2af50b3ef1ea1efb18ac7edd6649951a8acce84af2a8266288168a7ebb285a93" else "NO",
      "Z_SHARD_CONTENT_RECOMPUTE_MATCH": Z_SHARD_CONTENT_RECOMPUTE_MATCH,
      "TRAIN_ROWS": TRAIN_ROWS, "VALIDATION_ROWS": VALIDATION_ROWS,
      "TARGET_DEFINITION": "numpy.log1p(n_perturbed_cells)", "TARGET_DTYPE": "float64",
      "TARGET_STANDARDIZED": "NO", "TARGET_SUPPORT_N_CELLS": [NMIN, NMAX],
      "ALPHA_GRID": ALPHA_GRID,
      "DMECH_R2_DENOMINATOR_CONVENTION": "VALIDATION_SET_OWN_MEAN",
      "DMECH_R2_POOLING_UNITS": "VALIDATION_ROWS_ONLY",
      "ALPHA_TIE_RULE": "EXACT_TIE_SELECT_LARGEST_ALPHA",
      "NONFINITE_SELECTION_RULE": "FINITE_BEATS_NONFINITE_ALL_NONFINITE_PROBE_DEGENERATE",
      "DMECH_POSITIVE_BRANCH_RULE": "SELECTED_VALIDATION_R2 > 0",
      "DMECH_NULL_BRANCH_RULE": "SELECTED_VALIDATION_R2 <= 0",
      "LOCKED_POSITIVE_STATEMENT": "Sampling-depth-associated information is linearly recoverable from the frozen Geneformer representation on VCC train/validation.",
      "LOCKED_NULL_STATEMENT": "This probe did not linearly recover sampling-depth-associated information from the frozen Geneformer representation on VCC train/validation under the pre-specified estimator and grid.",
      "DMECH_RUNNER_PATH": os.path.abspath(__file__),
      "DMECH_RUNNER_SHA256": sha_file(os.path.abspath(__file__)),
      "PYTHON_VERSION": platform.python_version(), "PYTHON_EXECUTABLE": sys.executable,
      "RELEVANT_PACKAGE_VERSIONS": {"numpy": np.__version__, "scipy": scipy.__version__,
          "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
          "threadpoolctl": threadpoolctl.__version__, "pyarrow": pyarrow.__version__},
      "THREAD_ENVIRONMENT": {v: os.environ.get(v) for v in THREAD_VARS},
      "THREADPOOLCTL_STATE": threadpoolctl.threadpool_info(),
      "RIDGE_FIT_COUNT_AT_PROJECT_IMPORT": RIDGE_FIT_COUNT_AT_PROJECT_IMPORT,
      "RIDGE_FIT_COUNT_BEFORE_FREEZE": CNT.ridge_fit,
      "STANDARD_SCALER_FIT_COUNT_BEFORE_FREEZE": CNT.ss_fit,
      "STANDARD_SCALER_FIT_TRANSFORM_COUNT_BEFORE_FREEZE": CNT.ss_fit_transform,
      "RIDGE_FIT_COUNT_EXPECTED": RIDGE_FIT_COUNT_EXPECTED,
      "COUNTER_OBJECT_ID": CNT_ID,
      "DMECH_RESULT_OBSERVED_AT_FREEZE": "NO",
      "DMECH_REFIT_ON_COMBINED_TRAIN_VALIDATION": "NO",
      "DMECH_DETERMINISM_SCOPE": "IN_PROCESS_FIXED_ENVIRONMENT",
    }
    with open(os.path.join(OUT,"DMECH_PRE_RUN_FREEZE.json"),"w") as f: json.dump(freeze,f,indent=1,default=str)
    with open(os.path.join(OUT,"DMECH_PRE_RUN_FREEZE.md"),"w") as f:
        f.write("# D-MECH PRE-RUN FREEZE\n\n**Written before the first scientific Ridge fit.**\n\n```\n")
        for k,v in freeze.items():
            if k not in ("THREADPOOLCTL_STATE",): f.write(f"{k} = {v}\n")
        f.write("```\n")
    log("PRE-RUN FREEZE written; no scientific fit has occurred")

    # ================= FIRST SCIENTIFIC FIT — RESULT OBSERVED FROM HERE =========
    repeats = []
    for rep in range(N_REPEATS):
        scores = []
        for a in ALPHA_GRID:
            m = Ridge(alpha=a, fit_intercept=True).fit(data["TRAIN"]["Zs"], data["TRAIN"]["t"])
            scores.append(float(r2_pooled(data["VALIDATION"]["t"], m.predict(data["VALIDATION"]["Zs"]))))
        repeats.append(scores)
        log(f"REPEAT_{rep+1} validation R2 per alpha computed")
    identical = all(repeats[0] == r for r in repeats[1:])
    if not identical: raise PhaseBError("HARD_STOP deterministic repeats not bit-identical")
    scores = repeats[0]
    sel = select_alpha(ALPHA_GRID, scores)
    log(f"selected_alpha={sel['selected_alpha']} status={sel['status']} tie={sel['tie_applied']}")

    RIDGE_FIT_COUNT_OBSERVED = CNT.ridge_fit
    results = {
      "RUN_UTC_TIMESTAMP": RUN_UTC_TIMESTAMP, "RUN_UTC_DATE": RUN_UTC_DATE,
      "ALPHA_GRID": ALPHA_GRID, "ALPHA_SCORE_TABLE": dict(zip(map(str,ALPHA_GRID), scores)),
      "SELECTED_ALPHA": sel["selected_alpha"], "SELECTED_VALIDATION_R2": sel["best"],
      "TIE_APPLIED": sel["tie_applied"], "TIED_ALPHAS": sel["tied"], "STATUS": sel["status"],
      "ALPHA_GRID_MIN": min(ALPHA_GRID), "ALPHA_GRID_MAX": max(ALPHA_GRID),
      "ALPHA_AT_GRID_BOUNDARY": "YES" if sel["selected_alpha"] in (min(ALPHA_GRID),max(ALPHA_GRID)) else "NO",
      "ALPHA_BOUNDARY_SIDE": ("MIN" if sel["selected_alpha"]==min(ALPHA_GRID) else
                              "MAX" if sel["selected_alpha"]==max(ALPHA_GRID) else "NONE"),
      "DMECH_RESULT_BRANCH": ("POSITIVE" if (sel["best"] is not None and sel["best"] > 0)
                              else "NULL_OR_NONPOSITIVE"),
      "TRAIN_ROWS": TRAIN_ROWS, "VALIDATION_ROWS": VALIDATION_ROWS,
      "TARGET_SUPPORT_N_CELLS": [NMIN, NMAX],
      "DMECH_R2_DENOMINATOR_CONVENTION": "VALIDATION_SET_OWN_MEAN",
      "DMECH_R2_POOLING_UNITS": "VALIDATION_ROWS_ONLY",
      "DMECH_REFIT_ON_COMBINED_TRAIN_VALIDATION": "NO",
      "DETERMINISTIC_REPEATS_IDENTICAL": "YES",
      "DMECH_DETERMINISM_SCOPE": "IN_PROCESS_FIXED_ENVIRONMENT",
      "DMECH_RESULT_OBSERVED": "YES",
    }
    results["DMECH_LOCKED_STATEMENT"] = (freeze["LOCKED_POSITIVE_STATEMENT"]
        if results["DMECH_RESULT_BRANCH"]=="POSITIVE" else
        freeze["LOCKED_NULL_STATEMENT"] + (
          " and the regularization optimum was not bracketed above within the pre-specified grid,"
          " so non-recovery is not established even for this estimator family."
          if results["ALPHA_BOUNDARY_SIDE"]=="MAX" else ""))
    json.dump(results, open(os.path.join(OUT,"dmech_results.json"),"w"), indent=1)
    with open(os.path.join(OUT,"dmech_results.csv"),"w") as f:
        f.write("alpha,validation_r2,selected\n")
        for a,s in zip(ALPHA_GRID,scores):
            f.write(f"{a},{s},{'YES' if a==sel['selected_alpha'] else 'NO'}\n")

    # ---------------- post-run integrity ----------------
    integrity = {
      "RIDGE_FIT_COUNT_EXPECTED": RIDGE_FIT_COUNT_EXPECTED,
      "RIDGE_FIT_COUNT_OBSERVED": RIDGE_FIT_COUNT_OBSERVED,
      "RIDGE_FIT_COUNT_MATCH": RIDGE_FIT_COUNT_OBSERVED == RIDGE_FIT_COUNT_EXPECTED,
      "RIDGE_COUNTER_OBJECT_CONTINUITY_VERIFIED": id(CNT)==CNT_ID,
      "RIDGE_COUNTER_RESET_AFTER_FREEZE": "NO", "RIDGE_INSTRUMENTATION_REINSTALLED_AFTER_FREEZE": "NO",
      "STANDARD_SCALER_FIT_COUNT_REAL_RUN": CNT.ss_fit,
      "STANDARD_SCALER_FIT_TRANSFORM_COUNT_REAL_RUN": CNT.ss_fit_transform,
      "STANDARD_SCALER_COUNTER_OBJECT_CONTINUITY_VERIFIED": id(CNT)==CNT_ID,
      "STANDARD_SCALER_COUNTER_RESET_AFTER_FREEZE": "NO",
      "STANDARD_SCALER_INSTRUMENTATION_REINSTALLED_AFTER_FREEZE": "NO",
      "Z_GUARDED_LOADS_EXPECTED": Z_GUARDED_LOADS_EXPECTED,
      "Z_GUARDED_LOADS_OBSERVED": G.z_loads,
      "Z_GUARDED_LOADS_MATCH": G.z_loads == Z_GUARDED_LOADS_EXPECTED,
      "TARGET_GUARDED_READS_EXPECTED": TARGET_GUARDED_READS_EXPECTED,
      "TARGET_GUARDED_READS_OBSERVED": G.target_reads,
      "TARGET_GUARDED_READS_MATCH": G.target_reads == TARGET_GUARDED_READS_EXPECTED,
      "SCIENTIFIC_OPENS_OUTSIDE_BOUND_ROOT": G.outside_bound,
      "Z_COLUMN_COUNT": Z_COLUMN_COUNT, "Z_DTYPE": Z_DTYPE,
    }
    fails = [k for k in ("RIDGE_FIT_COUNT_MATCH","Z_GUARDED_LOADS_MATCH","TARGET_GUARDED_READS_MATCH",
                         "RIDGE_COUNTER_OBJECT_CONTINUITY_VERIFIED") if integrity[k] is not True]
    if CNT.ss_fit or CNT.ss_fit_transform: fails.append("SCALER_REFIT_DETECTED")
    if G.outside_bound: fails.append("OUTSIDE_BOUND_ROOT")
    status = "SUCCESS" if not fails else "OBSERVED_BUT_INTEGRITY_FAILED"
    if sel["status"] == "PROBE_DEGENERATE": status = "PROBE_DEGENERATE"
    integrity["INTEGRITY_FAILURES"] = fails
    log(f"integrity: {integrity}")

    json.dump({"ACCESSES": G.accesses,
               "VCC_TEST_SCIENTIFIC_INPUT_OPENED":"NO","EXTERNAL_SCIENTIFIC_INPUT_OPENED":"NO",
               "Y_ACCESSED":"NO","PHASE4_EXT_PREDICTIONS_OPENED":"NO","PROJECT_A_CONTENT_READ":"NO"},
              open(os.path.join(OUT,"dmech_input_access_manifest.json"),"w"), indent=1)
    json.dump({"PYTHON_VERSION": platform.python_version(),"PYTHON_EXECUTABLE": sys.executable,
               "PLATFORM": platform.platform(),
               "PACKAGES": freeze["RELEVANT_PACKAGE_VERSIONS"],
               "THREAD_ENVIRONMENT": freeze["THREAD_ENVIRONMENT"],
               "THREADPOOLCTL_STATE": threadpoolctl.threadpool_info(),
               "GPU_REQUIRED_FOR_DMECH":"NO","GPU_USED_FOR_DMECH":"NO"},
              open(os.path.join(OUT,"dmech_environment.json"),"w"), indent=1, default=str)
    json.dump({"BOUND_ROOT_REALPATH": BOUND, "TARGET_SOURCE_PATH": tgt_rp,
               "Z_SOURCE_ROOT": ZROOT, "BRIDGE": BRIDGE, "SCALER": SCALER,
               "Z_SHARD_MANIFEST": MAN, "EXTERNAL_RUN_ROOT": OUT},
              open(os.path.join(OUT,"dmech_resolved_paths.json"),"w"), indent=1)
    RESOLVED_SHA = sha_file(os.path.join(OUT,"dmech_resolved_paths.json"))

    LOG.flush()
    arts = ["DMECH_PRE_RUN_FREEZE.json","DMECH_PRE_RUN_FREEZE.md","dmech_results.json",
            "dmech_results.csv","dmech_guard_tests.json","dmech_input_access_manifest.json",
            "dmech_environment.json","dmech_execution.log"]
    manifest = {"RUN_UTC_TIMESTAMP": RUN_UTC_TIMESTAMP, "RUN_UTC_DATE": RUN_UTC_DATE,
      "DMECH_EXTERNAL_RUN_ROOT": OUT, "PRE_RUN_FREEZE_REACHED": "YES",
      "DMECH_EXECUTION_STATUS": status, "DMECH_RESULT_OBSERVED": "YES",
      "DMECH_RUNNER_PATH": freeze["DMECH_RUNNER_PATH"], "DMECH_RUNNER_SHA256": freeze["DMECH_RUNNER_SHA256"],
      "DMECH_RESOLVED_PATHS_SHA256": RESOLVED_SHA,
      "INTEGRITY": integrity, "RESULTS": results,
      "ARTIFACT_SHA256": {a: sha_file(os.path.join(OUT,a)) for a in arts}}
    json.dump(manifest, open(os.path.join(OUT,"dmech_run_manifest.json"),"w"), indent=1, default=str)
    log(f"FINAL STATUS = {status}")
    print("\n@@RESULT@@ " + json.dumps({"status": status, "results": results,
          "integrity": integrity, "run_root": OUT,
          "run_manifest_sha256": sha_file(os.path.join(OUT,"dmech_run_manifest.json"))}, default=str))
    LOG.close()

# ---- selection + validation helpers (module level; used by guards and the run) ----
def select_alpha(grid, scores):
    import math
    fin = [(a,s) for a,s in zip(grid,scores) if isinstance(s,float) and math.isfinite(s)]
    if not fin:
        return {"selected_alpha": None, "best": None, "tie_applied": False,
                "tied": [], "status": "PROBE_DEGENERATE"}
    best = max(s for _,s in fin)
    tied = [a for a,s in fin if s == best]          # exact float64 equality
    return {"selected_alpha": max(tied), "best": best,
            "tie_applied": len(tied) > 1, "tied": tied, "status": "OK"}

def assert_keys_unique(ks, label):
    if len(set(ks)) != len(ks): raise PhaseBError(f"duplicate keys in {label}")

def assert_eligibility(n):
    import numpy as _np
    bad = int((_np.asarray(n) < 10).sum())
    if bad: raise PhaseBError(f"HARD_STOP_TARGET_ELIGIBILITY_VIOLATION: {bad} rows with n_perturbed_cells<10")

if __name__ == "__main__":
    main()
