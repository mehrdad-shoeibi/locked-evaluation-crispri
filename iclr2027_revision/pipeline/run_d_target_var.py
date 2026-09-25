"""D-TARGET-VAR — target-structure diagnostic, governed by lock amendment A12.

One-way variance components of t = log1p(n_perturbed_cells) by target_gene and by
batch, within VCC TRAIN and VALIDATION separately. Four cells. CPU only.

Loads only: split, batch, target_gene, n_perturbed_cells.
No z. No model. No estimator import. No test split. No external data. No randomness.
"""
import os, sys, json, time, hashlib, argparse, platform
import numpy as np

BOUND_ROOT_LITERAL = "/media/mehrdad/MehrdadSSD/ICLR2027_2026-08-26_DATA"
TARGET_REL   = "MADA/data/metadata/vcc/final_labels.parquet"
TARGET_SHA   = "37c9aafef6e721815436a6534eacda4a1db6e73e937e886c44588284a37a0fa0"
PROJECT_A_ROOTS = ["/home/mehrdad/Desktop/iga-neurips",
                   "/media/mehrdad/MehrdadSSD/IGA_NeurIPS",
                   "/media/mehrdad/MehrdadSSD/IGA_ICLR2027_GENEFORMER",
                   "/media/mehrdad/MehrdadSSD/MADA_data"]
ALLOWED_SPLITS = {"TRAIN": "training", "VALIDATION": "validation"}
EXPECTED_ROWS  = {"TRAIN": 5740, "VALIDATION": 2052}
FORBIDDEN_SPLIT_TOKENS = {"test", "TEST", "rpe1", "RPE1", "k562", "K562"}


class DTVError(RuntimeError): pass


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def _parts(p): return [x for x in os.path.normpath(p).split(os.sep) if x]

def _inside(child, parent):
    c, pa = _parts(child), _parts(parent)
    return len(c) >= len(pa) and c[:len(pa)] == pa


# --------------------------------------------------------------- the estimator
def one_way(t, groups):
    """Frozen A12 §6-§8 one-way method-of-moments variance components.

    t      : float64 values
    groups : array of group labels, same length
    Returns the locked field set. ICC and VAR_BETWEEN are never clipped.
    """
    t = np.asarray(t, dtype=np.float64)
    N = int(t.size)
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    k = int(uniq.size)
    m = np.bincount(inv).astype(np.float64)                  # group sizes m_i
    gsum = np.bincount(inv, weights=t)
    gmean = gsum / m
    tbar = float(t.mean())

    SSB = float(np.sum(m * (gmean - tbar) ** 2))
    SSW = float(np.sum((t - gmean[inv]) ** 2))
    TOTAL_SS = float(np.sum((t - tbar) ** 2))
    SS_RESIDUAL = float(abs(TOTAL_SS - (SSB + SSW)))

    MSB = SSB / (k - 1) if k > 1 else float("nan")
    MSW = SSW / (N - k) if N > k else float("nan")
    sum_m2 = float(np.sum(m ** 2))
    m0 = (N - sum_m2 / N) / (k - 1) if k > 1 else float("nan")

    VAR_WITHIN = MSW
    VAR_BETWEEN = (MSB - MSW) / m0                            # never clipped
    denom = VAR_BETWEEN + VAR_WITHIN
    ICC = VAR_BETWEEN / denom if denom != 0 else float("nan") # never clipped
    ETA_SQUARED = SSB / TOTAL_SS if TOTAL_SS != 0 else float("nan")

    M_BAR_ARITHMETIC = N / k
    M_BAR_SIZE_WEIGHTED = sum_m2 / N

    de_a = 1.0 + (M_BAR_ARITHMETIC - 1.0) * ICC
    de_w = 1.0 + (M_BAR_SIZE_WEIGHTED - 1.0) * ICC

    def approx_n(de):
        if not np.isfinite(de) or de <= 0: return "NOT_INTERPRETABLE"
        return N / de
    n_a, n_w = approx_n(de_a), approx_n(de_w)

    def ratio(n):
        if n == "NOT_INTERPRETABLE" or n == 0: return "NOT_INTERPRETABLE"
        return N / n
    return {
        "N_ROWS": N, "N_GROUPS": k,
        "GROUP_SIZE_MIN": int(m.min()), "GROUP_SIZE_MEDIAN": float(np.median(m)),
        "GROUP_SIZE_MAX": int(m.max()), "GROUP_SIZE_MEAN": float(m.mean()),
        "SSB": SSB, "SSW": SSW, "TOTAL_SS": TOTAL_SS, "SS_RESIDUAL": SS_RESIDUAL,
        "MSB": MSB, "MSW": MSW, "m0": m0,
        "VAR_WITHIN": VAR_WITHIN, "VAR_BETWEEN": VAR_BETWEEN,
        "ICC": ICC, "ETA_SQUARED": ETA_SQUARED,
        "M_BAR_ARITHMETIC": M_BAR_ARITHMETIC, "M_BAR_SIZE_WEIGHTED": M_BAR_SIZE_WEIGHTED,
        "DESIGN_EFFECT_APPROX_ARITH": de_a, "DESIGN_EFFECT_APPROX_WEIGHTED": de_w,
        "ICC_DESIGN_EFFECT_APPROX_N_ARITH": n_a,
        "ICC_DESIGN_EFFECT_APPROX_N_WEIGHTED": n_w,
        "N_ROWS_TO_APPROX_N_RATIO_ARITH": ratio(n_a),
        "N_ROWS_TO_APPROX_N_RATIO_WEIGHTED": ratio(n_w),
    }


# ------------------------------------------------------------------- real run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--lock-commit", required=True)
    ap.add_argument("--implementation-commit", required=True)
    args = ap.parse_args()

    OUT = os.path.abspath(args.out_root)
    os.makedirs(OUT, exist_ok=False)
    RUN_UTC = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

    guards = {}
    BOUND = os.path.realpath(BOUND_ROOT_LITERAL)
    guards["BOUND_ROOT_REALPATH_EQUALS_LITERAL"] = "YES" if BOUND == BOUND_ROOT_LITERAL else "NO"
    guards["BOUND_ROOT_IS_DIRECTORY"] = "YES" if os.path.isdir(BOUND) else "NO"
    tgt = os.path.realpath(os.path.join(BOUND, TARGET_REL))
    for a in PROJECT_A_ROOTS:
        if _inside(tgt, os.path.realpath(a)):
            raise DTVError(f"BOUNDARY VIOLATION: target inside Project-A root {a}")
    if not _inside(tgt, BOUND):
        raise DTVError("BOUNDARY VIOLATION: target outside bound root")
    guards["TARGET_REALPATH_INSIDE_PROJECT_B"] = "YES"
    guards["PROJECT_A_SCIENCE_ACCESS"] = "NO"
    actual = sha_file(tgt)
    guards["TARGET_SOURCE_SHA256_EXPECTED"] = TARGET_SHA
    guards["TARGET_SOURCE_SHA256_ACTUAL"] = actual
    guards["TARGET_SOURCE_HASH_MATCH"] = "YES" if actual == TARGET_SHA else "NO"
    if actual != TARGET_SHA:
        raise DTVError("HARD_STOP_TARGET_SOURCE_HASH_MISMATCH")

    import pyarrow.parquet as pq                              # no estimator import
    tb = pq.read_table(tgt, columns=["split", "batch", "target_gene", "n_perturbed_cells"])
    split = np.array([str(v) for v in tb.column("split").to_pylist()])
    batch = np.array([str(v) for v in tb.column("batch").to_pylist()])
    gene  = np.array([str(v) for v in tb.column("target_gene").to_pylist()])
    ncell = np.asarray(tb.column("n_perturbed_cells").to_numpy(zero_copy_only=False), dtype=np.int64)

    # bridge keys define the governed retained-row population (D-MECH eligibility)
    BRIDGE = os.path.join(REPO, "iclr2027_revision/phase0/bridge/vcc_z_label_bridge.parquet")
    bt = pq.read_table(BRIDGE, columns=["split", "batch", "target_gene"])   # NO z columns, NO label
    bkeys = set(zip([str(v) for v in bt.column("split").to_pylist()],
                    [str(v) for v in bt.column("batch").to_pylist()],
                    [str(v) for v in bt.column("target_gene").to_pylist()]))
    guards["Z_ACCESS"] = "NO"
    guards["MODEL_IMPORT"] = "NO"
    guards["GPU_USE"] = "NO"
    guards["RANDOM_SAMPLING"] = "NO"

    cells, pops = {}, {}
    for LABEL, sval in ALLOWED_SPLITS.items():
        if sval in FORBIDDEN_SPLIT_TOKENS: raise DTVError(f"forbidden split {sval}")
        sel = np.array([(s == sval) and ((s, b, g) in bkeys)
                        for s, b, g in zip(split, batch, gene)])
        n = int(sel.sum()); pops[LABEL] = n
        if n != EXPECTED_ROWS[LABEL]:
            raise DTVError(f"ABORT population mismatch {LABEL}: {n} != {EXPECTED_ROWS[LABEL]}")
        nc = ncell[sel]
        if (nc < 10).any(): raise DTVError(f"{LABEL}: eligibility violation n_perturbed_cells<10")
        t = np.log1p(nc.astype(np.float64))
        if not np.isfinite(t).all(): raise DTVError(f"{LABEL}: non-finite target")
        for AX, arr, nkey in (("target_gene", gene[sel], "N_GENES"),
                              ("batch", batch[sel], "N_BATCHES")):
            r = one_way(t, arr)
            r[nkey] = r["N_GROUPS"]
            r["SPLIT"], r["GROUPING"] = LABEL, AX
            cells[f"{LABEL}/{AX}"] = r
    guards["VCC_TEST_ACCESS"] = "NO"
    guards["EXTERNAL_ACCESS"] = "NO"
    guards["TRAIN_ROWS"] = pops["TRAIN"]
    guards["VALIDATION_ROWS"] = pops["VALIDATION"]

    out = {"RUN_UTC_TIMESTAMP": RUN_UTC,
           "D_TARGET_VAR_LOCK_COMMIT": args.lock_commit,
           "D_TARGET_VAR_IMPLEMENTATION_COMMIT": args.implementation_commit,
           "LOCK_FILE": "iclr2027_revision/experiment_specs/lock_amendment_A12_d_target_var.md",
           "LOCK_SHA256": sha_file(os.path.join(REPO,
               "iclr2027_revision/experiment_specs/lock_amendment_A12_d_target_var.md")),
           "RUNNER_FILE": "iclr2027_revision/pipeline/run_d_target_var.py",
           "RUNNER_SHA256": sha_file(os.path.abspath(__file__)),
           "TARGET_SOURCE_PATH": tgt, "TARGET_SOURCE_SHA256": actual,
           "TARGET_DEFINITION": "t = numpy.log1p(n_perturbed_cells), natural log, float64",
           "ROW_KEY": "(split, batch, target_gene)",
           "TARGET_STANDARDIZED": "NO", "SPLITS_POOLED": "NO",
           "PYTHON_VERSION": platform.python_version(),
           "GUARDS": guards, "CELLS": cells}
    json.dump(out, open(os.path.join(OUT, "d_target_var_results.json"), "w"), indent=1, default=str)
    print("@@DTV@@ " + json.dumps(out, default=str))


if __name__ == "__main__":
    main()
