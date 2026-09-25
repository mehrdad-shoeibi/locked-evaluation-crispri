"""(c) VCC z <-> label bridge — Lock §7.

Join MADA's cached VCC z (geneformer_row_features_full.parquet, gf_z_emb_0..767) to the
manuscript's target-gene AD labels (final_labels.parquet, y_ad_distance_log1p) on
(split, batch, target_gene). Attach the target-gene AD response-magnitude endpoint; NEVER the
mean-over-genes breadth endpoint (final_labels' `ad_stat_mean`, Spearman -0.153).

Row counts and key alignment ONLY — no statistic of y, nothing on the test split.
"""
import os as _o, sys as _s
_r = _o.path.dirname(_o.path.abspath(__file__))
while _r != _o.path.dirname(_r) and not _o.path.exists(_o.path.join(_r, "iga_paths.py")):
    _r = _o.path.dirname(_r)
if _r not in _s.path:
    _s.path.insert(0, _r)
import iga_paths
import re
import numpy as np, pandas as pd
import pyarrow.parquet as pq

Z_PARQUET = (iga_paths.MADA + "/cache/foundation_models/geneformer/"
             "full_feature_extraction_v2/geneformer_row_features_full.parquet")
LABELS_PARQUET = iga_paths.MADA + "/data/metadata/vcc/final_labels.parquet"
KEYS = ["split", "batch", "target_gene"]
LABEL_COL = "y_ad_distance_log1p"          # target-gene AD response magnitude (NOT breadth)
FORBIDDEN_LABEL = "ad_stat_mean"           # MADA mean-over-genes breadth endpoint — never attach
EXPECTED = {"total": 11764, "training": 5740, "validation": 2052, "test": 3972}


def _z_cols():
    cols = [f.name for f in pq.read_schema(Z_PARQUET)]
    zc = [c for c in cols if re.fullmatch(r"gf_z_emb_\d+", c)]
    return sorted(zc, key=lambda c: int(c.split("_")[-1]))


def build_bridge():
    """Return (bridge_df, report). bridge_df has KEYS + LABEL_COL + gf_z_emb_0..767 (z left-joined
    to the manuscript labels). report holds the integrity assertions."""
    zc = _z_cols()
    zdf = pd.read_parquet(Z_PARQUET, columns=KEYS + zc + [LABEL_COL])
    zdf[KEYS] = zdf[KEYS].astype(str)
    lab = pd.read_parquet(LABELS_PARQUET, columns=KEYS + [LABEL_COL])
    lab[KEYS] = lab[KEYS].astype(str)

    # key uniqueness on both sides
    dup_z = int(len(zdf) - zdf[KEYS].drop_duplicates().shape[0])
    dup_lab = int(len(lab) - lab[KEYS].drop_duplicates().shape[0])

    merged = zdf.merge(lab, on=KEYS, how="left", suffixes=("_z", "_lab"), indicator=True)
    matched = int((merged["_merge"] == "both").sum())
    z_orphans = int((merged["_merge"] == "left_only").sum())     # z rows with no manuscript label
    # right orphans (labels with no z) are the un-embedded constructs, expected; report separately
    lab_only = int(len(lab) - lab.merge(zdf[KEYS], on=KEYS, how="inner").shape[0])

    # label alignment (NOT a distributional statistic): the label carried in the z parquet must equal
    # the authoritative final_labels value on the join key -> confirms the correct label was attached.
    label_align_max_abs = float(np.abs(merged[LABEL_COL + "_z"].to_numpy()
                                       - merged[LABEL_COL + "_lab"].to_numpy()).max())

    Z = zdf[zc].to_numpy()
    report = {
        "rows_z": int(len(zdf)), "rows_labels": int(len(lab)),
        "matched": matched, "z_orphans": z_orphans, "label_only_rows": lab_only,
        "dup_keys_z": dup_z, "dup_keys_labels": dup_lab,
        "splits": {k: int(v) for k, v in zdf["split"].value_counts().to_dict().items()},
        "z_all_finite": bool(np.isfinite(Z).all()),
        "z_dim": int(Z.shape[1]),
        "label_alignment_max_abs_diff": label_align_max_abs,
        "label_col": LABEL_COL, "forbidden_label_not_used": FORBIDDEN_LABEL,
    }
    bridge = merged.rename(columns={LABEL_COL + "_lab": LABEL_COL})[KEYS + [LABEL_COL] + zc]
    return bridge, report


def assert_bridge(report):
    ok = (report["matched"] == EXPECTED["total"]
          and report["z_orphans"] == 0
          and report["dup_keys_z"] == 0 and report["dup_keys_labels"] == 0
          and report["splits"].get("training") == EXPECTED["training"]
          and report["splits"].get("validation") == EXPECTED["validation"]
          and report["splits"].get("test") == EXPECTED["test"]
          and report["z_all_finite"]
          and report["label_alignment_max_abs_diff"] == 0.0)
    return bool(ok)
