#!/usr/bin/env python3
"""Deterministic metadata-only reconstruction of the counterfactual non-overlap
exclusion counts (RPE1 37 / K562-essential 29).

Governed by: 37_29_reconstruction_minispec.md  (FROZEN FOR ONE-SHOT EXECUTION)

Implements ONLY the historical matching rule recovered from executed R6 Task-B code
(session record 78566c4b-1c24-4dbb-8b67-68366f6ab645.jsonl,
 2026-08-26T22:14:01.775Z / 22:14:25.261Z / 22:15:50.309Z).

NO model execution. NO training. NO prediction generation. NO metric recomputation.
NO external outcome column or value access. NO GPU. NO full-endpoint hashing.
No fallback normalization. No preprocessing search. No retry.
"""
from __future__ import annotations

import os

# ---- environment guard: set before any third-party import -------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import sys
import json
import hashlib
import datetime

import numpy as np
import pandas as pd
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
STORAGE = "/media/mehrdad/MehrdadSSD/ICLR2027_2026-08-26_DATA/IGA_NeurIPS"
REPO = "/home/mehrdad/Desktop/ICLR2027_2026-08-26/iclr2027_revision"

# ---- frozen inputs ----------------------------------------------------------
I1_H5AD = f"{STORAGE}/data/vcc/train/adata_Training.h5ad"   # digest unavailable; fingerprint asserted

PINNED_NON_ENDPOINT = {
    "I2_pert_counts": (f"{STORAGE}/data/vcc/train/pert_counts_Training.csv",
        "633d202be221418bdbac16efd8cb169666de0ef6469a4e9bfdfb64666ea81a89"),
    "I3_preflight": (f"{REPO}/experiment_specs/implementation_preflight.md",
        "468e5d97aca4ac306d5ae102aa510eda29f28a5d06d5ffbe1440a56341e4bc44"),
    "I6_A7": (f"{REPO}/experiment_specs/lock_amendment_A7.md",
        "ce728d09f69f8a6876994d2f6d61b9bd10624aa73b9b677d02e0ae30d54ae150"),
    "I7_R6": (f"{REPO}/R6_RESULTS.md",
        "398aed184872c097af34c303861be398370a93afad192404324d50ba13e15e18"),
    "I8_lock": (f"{REPO}/experiment_specs/option_b_geneformer_experiment_lock.md",
        "6321daacafcaee01a0b97253d19968218104051a9a2e044482849a0ada0265e5"),
}

# Endpoint files are NEVER hashed in full. Historical digests are provenance context only.
HISTORICAL_FULL_ENDPOINT_SHA256 = {
    "RPE1": "353c04d3ac8e3380a3addf3abc1953d30799423f7cd467006adc36800e26a126",
    "K562": "051269eeeae93c75c84ca1fc050529a096ebd15e1990cb14bb24669c1b9672ff",
}

ENDPOINT = {
    "RPE1": f"{STORAGE}/results_v4/harmonized_endpoint_v1/rpe1_endpoint_by_construct.csv",
    "K562": f"{STORAGE}/results_v4/harmonized_endpoint_k562_essential_v1/endpoint_by_construct.csv",
}

IDENTIFIER_PROJECTION_SHA256 = {
    "RPE1": "a245bf59071eb9526bc164c45f8deda04012297ca7df972d607ee047cb45b863",
    "K562": "79436dd748851bd95edbc3bff7acf2e64c5fa89192c38d610790b05680f4bed3",
}
EXPECT_ROWS = {"RPE1": 1932, "K562": 1903}

# ---- frozen rule constants (NOT tunable) ------------------------------------
NT_TOKEN = "non-targeting"
ID_COLS = ["construct", "target_symbol"]
D10 = ["ATP6V0C", "BRD9", "DNAJA3", "EPHB4", "FDPS",
       "OXA1L", "RNF20", "SALL4", "SLC39A6", "TAF13"]

EXPECT_HISTORICAL = {"RPE1": (43, 42), "K562": (33, 32)}   # validation checks
EXPECT_COUNTERFACTUAL = {"RPE1": 37, "K562": 29}           # expected historical values, not targets

FORBIDDEN_COLS = {
    "y_primary", "y_mean", "y_std", "y_p5", "y_p95", "y_iqr", "y_pooled", "y_gemgroup",
    "stored_ad", "stored_ad_log1p", "n_valid_draws", "n_eligible_gemgroups",
    "n_perturbed_pool", "n_nt_pool", "target_col", "target_ensg",
}


def die(msg: str) -> None:
    print(f"FAIL-CLOSED: {msg}", file=sys.stderr)
    sys.exit(2)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def projection_digest(rows: list) -> str:
    blob = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> int:
    prov = {
        "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": sys.version.split()[0],
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "minispec_sha256": (sha256_file(os.path.join(HERE, "37_29_reconstruction_minispec.md"))
                            if os.path.exists(os.path.join(HERE, "37_29_reconstruction_minispec.md")) else None),
        "gates": {}, "inputs": {},
    }

    # ================= PHASE A — safe metadata + identifier projections =======
    for key, (path, want) in PINNED_NON_ENDPOINT.items():
        if not os.path.exists(path):
            die(f"missing input {key}: {path}")
        got = sha256_file(path)
        prov["inputs"][key] = {"path": path, "sha256": got}
        if got != want:
            die(f"hash mismatch for {key}\n  expected {want}\n  actual   {got}")
    prov["gates"]["G5_non_endpoint_hashes"] = "PASS"

    prov["inputs"]["I1_h5ad"] = {
        "path": I1_H5AD, "sha256": None,
        "note": "whole-file digest unavailable (15.5 GB); structural fingerprint asserted instead; "
                "NOT equivalent to a cryptographic digest",
    }

    with h5py.File(I1_H5AD, "r") as f:
        tg = f["obs"]["target_gene"]
        if not isinstance(tg, h5py.Group):
            die("obs/target_gene is not categorical (categories+codes) as the historical rule assumes")
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in tg["categories"][:]])
        codes = tg["codes"][:]

    # ---- FIX 3: negative categorical code guard, BEFORE cats[codes] ---------
    if not np.issubdtype(codes.dtype, np.integer):
        die(f"G1: obs/target_gene codes are not integer-like (dtype={codes.dtype})")
    if (codes < 0).any():
        die("G1: obs/target_gene contains negative categorical codes (missing values)")
    if int(codes.max()) >= len(cats):
        die(f"G1: max code {int(codes.max())} >= len(categories) {len(cats)}")

    vals = cats[codes]

    n_cats = int(len(cats))
    n_distinct = int(len(np.unique(codes)))
    n_nt_cat = int(sum(1 for c in cats if c == NT_TOKEN))
    if n_cats != 151:      die(f"G1: expected 151 categories, got {n_cats}")
    if n_distinct != 151:  die(f"G1: expected all 151 categories used, got {n_distinct}")
    if n_nt_cat != 1:      die(f"G1: expected exactly one {NT_TOKEN!r} category, got {n_nt_cat}")
    prov["h5ad_metadata_structural_fingerprint"] = {
        "len_categories": n_cats, "n_distinct_codes": n_distinct,
        "n_nt_categories": n_nt_cat,
        "note": "NOT equivalent to a whole-file cryptographic digest",
    }
    prov["gates"]["G1_h5ad_fingerprint_and_code_guard"] = "PASS"

    RAW150 = set(v for v in vals if v != NT_TOKEN)      # exact string equality
    if len(RAW150) != 150:
        die(f"G2: |RAW150| == {len(RAW150)}, expected 150")
    prov["gates"]["G2_raw150_cardinality"] = "PASS"

    csv150 = pd.read_csv(PINNED_NON_ENDPOINT["I2_pert_counts"][0], usecols=["target_gene"])
    raw_csv = set(csv150["target_gene"].astype(str)) - {NT_TOKEN}
    if raw_csv != RAW150:
        die("G3: pert_counts set != obs set; "
            f"obs_only={sorted(RAW150 - raw_csv)} csv_only={sorted(raw_csv - RAW150)}")
    prov["gates"]["G3_raw150_csv_equivalence"] = "PASS"

    d10 = set(D10)
    if len(d10) != 10:
        die(f"G4: |D10| == {len(d10)}, expected 10")
    if not d10 <= RAW150:
        die(f"G4: D10 not a subset of RAW150: {sorted(d10 - RAW150)}")
    ELIGIBLE140 = RAW150 - d10
    if len(ELIGIBLE140) != 140:
        die(f"G4: |ELIGIBLE140| == {len(ELIGIBLE140)}, expected 140")
    prov["gates"]["G4_eligible140"] = "PASS"

    projections = {}
    for scr in ("RPE1", "K562"):
        df = pd.read_csv(ENDPOINT[scr], usecols=ID_COLS, dtype=str)   # identifier-only at parse time
        if set(df.columns) != set(ID_COLS):
            die(f"G6: {scr} columns {sorted(df.columns)} != {sorted(ID_COLS)}")
        if FORBIDDEN_COLS & set(df.columns):
            die(f"G6: forbidden column present for {scr}")
        df = df[ID_COLS]
        if len(df) != EXPECT_ROWS[scr]:
            die(f"G6: {scr} rows {len(df)} != {EXPECT_ROWS[scr]}")
        if int(df["construct"].isna().sum()) or int(df["target_symbol"].isna().sum()):
            die(f"G6: {scr} contains null identifier values")
        rows = df.values.tolist()
        dig = projection_digest(rows)
        if dig != IDENTIFIER_PROJECTION_SHA256[scr]:
            die(f"G6: {scr} identifier-projection digest mismatch\n"
                f"  expected {IDENTIFIER_PROJECTION_SHA256[scr]}\n  actual   {dig}")
        projections[scr] = df
    prov["gates"]["G6_identifier_projections"] = "PASS"
    prov["identifier_projection_sha256"] = dict(IDENTIFIER_PROJECTION_SHA256)
    prov["historical_full_endpoint_sha256"] = dict(HISTORICAL_FULL_ENDPOINT_SHA256)
    prov["full_endpoint_digest_recomputed"] = False

    # ================= PHASE B — historical validation, BOTH screens =========
    historical = {}
    for scr in ("RPE1", "K562"):
        tsym = projections[scr]["target_symbol"].to_numpy()
        con = projections[scr]["construct"].to_numpy()
        mask = np.array([s in RAW150 for s in tsym])
        historical[scr] = {
            "constructs": int(mask.sum()),
            "genes": int(len(set(tsym[mask]))),
            "excluded_targets": sorted(set(tsym[mask].tolist())),
            "excluded_constructs": sorted(con[mask].tolist()),
        }

    failures = []
    for scr in ("RPE1", "K562"):
        exp_c, exp_g = EXPECT_HISTORICAL[scr]
        if (historical[scr]["constructs"], historical[scr]["genes"]) != (exp_c, exp_g):
            failures.append(f"{scr}: got {historical[scr]['constructs']}c/{historical[scr]['genes']}g, "
                            f"recorded {exp_c}c/{exp_g}g")
    if failures:
        prov["GLOBAL_HISTORICAL_VALIDATION"] = "FAIL"
        die("G7/G8 global historical validation FAILED: " + "; ".join(failures))

    GLOBAL_HISTORICAL_VALIDATION = "PASS"
    prov["GLOBAL_HISTORICAL_VALIDATION"] = GLOBAL_HISTORICAL_VALIDATION
    prov["gates"]["G7_G8_global_historical_43_33"] = "PASS"
    print(f"GLOBAL_HISTORICAL_VALIDATION = {GLOBAL_HISTORICAL_VALIDATION}")

    # ================= PHASE C — counterfactual (only after PHASE B) =========
    counterfactual = {}
    for scr in ("RPE1", "K562"):
        tsym = projections[scr]["target_symbol"].to_numpy()
        con = projections[scr]["construct"].to_numpy()
        mask = np.array([s in ELIGIBLE140 for s in tsym])
        counterfactual[scr] = {
            "constructs": int(mask.sum()),
            "genes": int(len(set(tsym[mask]))),
            "excluded_targets": sorted(set(tsym[mask].tolist())),
            "excluded_constructs": sorted(con[mask].tolist()),
        }

    derived = {s: counterfactual[s]["constructs"] for s in counterfactual}
    agree = all(derived[s] == EXPECT_COUNTERFACTUAL[s] for s in EXPECT_COUNTERFACTUAL)

    prov.update({
        "rule_provenance": "executed R6 Task-B code, session 78566c4b-1c24-4dbb-8b67-68366f6ab645.jsonl, "
                           "2026-08-26T22:14:01.775Z / 22:14:25.261Z / 22:15:50.309Z",
        "normalization_rule": "category strings as stored, bytes decoded where necessary; exact Python "
                              "string equality; no strip, no case folding, no alias/synonym mapping, "
                              "no Ensembl remapping, no punctuation normalization",
        "nt_rule": f"exact case-sensitive equality to the single literal {NT_TOKEN!r}, applied during "
                   "set construction",
        "counting_unit": "constructs (external endpoint rows); genes reported separately via set()",
        "duplicate_policy": "constructs counted individually; not de-duplicated by gene",
        "D10": sorted(d10),
        "cardinalities": {"RAW150": len(RAW150), "D10": len(d10), "ELIGIBLE140": len(ELIGIBLE140)},
        "historical_overlap": historical,
        "counterfactual_overlap": counterfactual,
        "expected_historical": {k: {"constructs": v[0], "genes": v[1]} for k, v in EXPECT_HISTORICAL.items()},
        "expected_counterfactual": dict(EXPECT_COUNTERFACTUAL),
        "derived_counterfactual_constructs": derived,
        "matches_historical_37_29": bool(agree),
        "access_declaration": {
            "EXTERNAL_ENDPOINT_FULL_FILE_HASH_RECOMPUTED": "NO",
            "EXTERNAL_IDENTIFIER_COLUMNS_READ": ID_COLS,
            "EXTERNAL_OUTCOME_COLUMNS_READ": "NONE",
            "EXTERNAL_OUTCOME_VALUES_ACCESSED": "NONE",
            "MODEL_EXECUTION": "NONE",
            "GPU_USED": "NO",
        },
    })

    out_json = os.path.join(HERE, "37_29_reconstruction.json")
    with open(out_json, "w") as fh:
        json.dump(prov, fh, indent=2, sort_keys=True)

    lines = [
        "# 37/29 Counterfactual Exclusion Reconstruction — execution report", "",
        f"- UTC: {prov['utc']}",
        f"- Python: {prov['python']}",
        f"- script SHA-256: {prov['script_sha256']}",
        f"- mini-spec SHA-256: {prov['minispec_sha256']}",
        f"- GLOBAL_HISTORICAL_VALIDATION: {GLOBAL_HISTORICAL_VALIDATION}", "",
        "| screen | RAW150 overlap (validation) | ELIGIBLE140 overlap (reconstructed) | expected historical |",
        "|---|---|---|---|",
    ]
    for scr in ("RPE1", "K562"):
        lines.append(f"| {scr} | {historical[scr]['constructs']}c / {historical[scr]['genes']}g "
                     f"| {counterfactual[scr]['constructs']}c / {counterfactual[scr]['genes']}g "
                     f"| {EXPECT_COUNTERFACTUAL[scr]}c |")
    lines += ["", f"- matches historical 37/29: {agree}", "",
              "No outcome values, predictions, or outcome-derived metrics are recorded.",
              "FULL ENDPOINT DIGEST RECOMPUTED DURING THIS RECONSTRUCTION: NO", ""]
    with open(os.path.join(HERE, "37_29_reconstruction.md"), "w") as fh:
        fh.write("\n".join(lines))

    print(f"RAW150={len(RAW150)}  D10={len(d10)}  ELIGIBLE140={len(ELIGIBLE140)}")
    for scr in ("RPE1", "K562"):
        print(f"  {scr}: RAW150 overlap {historical[scr]['constructs']}c/{historical[scr]['genes']}g "
              f"(validated) | ELIGIBLE140 overlap {counterfactual[scr]['constructs']}c/"
              f"{counterfactual[scr]['genes']}g | expected historical {EXPECT_COUNTERFACTUAL[scr]}c")
    print("EXTERNAL_ENDPOINT_FULL_FILE_HASH_RECOMPUTED = NO")
    print(f"EXTERNAL_IDENTIFIER_COLUMNS_READ = {ID_COLS}")
    print("EXTERNAL_OUTCOME_COLUMNS_READ = NONE")
    print("EXTERNAL_OUTCOME_VALUES_ACCESSED = NONE")
    print("MODEL_EXECUTION = NONE")
    print("GPU_USED = NO")
    print(f"provenance artifact: {out_json}")

    if not agree:
        print("DISCREPANCY: derived counterfactual counts differ from the reported historical values. "
              "Per mini-spec Section 17 no preprocessing change is authorized. "
              "Returning to governance review.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
