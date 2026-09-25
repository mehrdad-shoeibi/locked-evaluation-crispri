"""Phase 2 assembly + QC (Parts H/I/J) + Part L 26B gate. Reads the outputs of
phase2_reagg_and_oracle.py and phase2_embed_and_xcell.py, assembles z_RPE1 (1,889 reagg +
43 new) and x_cell_RPE1, runs all label-free QC, and recomputes the §26B gate with the
Phase-2 achieved rate. Writes final arrays + qc json under the SSD ICLR root."""
import os as _o, sys as _s
_r = _o.path.dirname(_o.path.abspath(__file__))
while _r != _o.path.dirname(_r) and not _o.path.exists(_o.path.join(_r, "iga_paths.py")):
    _r = _o.path.dirname(_r)
if _r not in _s.path:
    _s.path.insert(0, _r)
import iga_paths
import os, sys, json, hashlib
from datetime import datetime
import numpy as np, pandas as pd, anndata as ad

SSD = iga_paths.GENEFORMER
H5 = iga_paths.IGA_NEURIPS + "/data/replogle/raw_singlecell/rpe1_raw_singlecell_01.h5ad"
ARM = iga_paths.IGA_NEURIPS + "/results_v4/transfer_pretest_lock_v1/arm1_rpe1_construct_list.csv"
VGN = iga_paths.IGA_NEURIPS + "/cache/vcc/var_gene_names.npy"
SG = iga_paths.IGA_NEURIPS + "/cache/vcc/variants/delta_hvg/selected_genes.npy"
PB = iga_paths.IGA_NEURIPS + "/data/replogle/raw/rpe1_normalized_bulk_01.h5ad"
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 2)))
sys.path.insert(0, os.path.join(_ROOT, "iclr2027_revision/pipeline"))
from x_cell_accumulator import map_and_clip_to_vcc_hvg

man = sorted(pd.read_csv(ARM)["construct"].astype(str))
sets = json.load(open(SSD + "/manifests/rpe1_construct_sets.json"))
miss43 = set(sets["missing_43"]); cached1889 = set(sets["cached_1889"])
drv = json.load(open(SSD + "/logs/phase2_embed_xcell.json"))
reagg = json.load(open(SSD + "/scratch/reagg_oracle.json"))

# ---- assemble z_RPE1 (float32) ----
zr = np.load(SSD + "/scratch/z_reagg_1889.npz", allow_pickle=True)
z1889 = {i: v for i, v in zip(zr["ids"].astype(str), zr["z"])}
z43f = np.load(SSD + "/scratch/z43.npz", allow_pickle=True)
z43 = {i: v for i, v in zip(z43f["ids"].astype(str), z43f["z"])}
zmap = {}; zmap.update(z1889); zmap.update(z43)
assert set(zmap) == set(man), "z id set != manuscript 1932"
z_RPE1 = np.stack([zmap[c] for c in man]).astype(np.float32)          # (1932,768)

# ---- x_cell_RPE1 (mapped) ----
xm = np.load(SSD + "/x_cell/x_cell_rpe1_mapped.npz", allow_pickle=True)
xmap = {i: v for i, v in zip(xm["ids"].astype(str), xm["x"])}
assert set(xmap) == set(man)
x_RPE1 = np.stack([xmap[c] for c in man]).astype(np.float32)          # (1932,2000)

# ---- raw obs: counts + NT set (ground truth) ----
a = ad.read_h5ad(H5, backed="r")
gt_all = a.obs["gene_transcript"].astype(str).to_numpy()
gene_all = a.obs["gene"].astype(str).to_numpy()
var_symbol = a.var["gene_name"].astype(str).to_numpy()
a.file.close()
raw_counts = pd.Series(gt_all[np.isin(gt_all, man)]).value_counts().to_dict()
nt_raw_idx = np.sort(np.where(gene_all == "non-targeting")[0].astype(np.int64))

# ---- Part H: NT pool identity between z path and x_cell path ----
nt_z = np.sort(np.load(SSD + "/scratch/nt_oidx.npy"))                 # z path (cached NT embeddings)
symdiff = np.setxor1d(nt_z, nt_raw_idx)
NT_POOL_IDENTICAL = "YES" if (len(symdiff) == 0 and len(nt_z) == 11485 == len(nt_raw_idx)) else "NO"

# ---- Part I: RPE1 HVG mapping, cross-check vs pseudobulk (x_release) symbols ----
var_gene_names = np.load(VGN, allow_pickle=True).astype(str); sel_idx = np.load(SG)
vcc_hvg = var_gene_names[sel_idx]
# present indices via raw single-cell symbols
def present_set(symbols):
    s2i = {}
    for i, s in enumerate(symbols):
        s2i.setdefault(s, i)
    return {h for h, s in enumerate(vcc_hvg) if s in s2i}
pres_sc = present_set(var_symbol)
pb = ad.read_h5ad(PB, backed="r"); pb_sym = pb.var["gene_name"].astype(str).to_numpy(); pb.file.close()
pres_pb = present_set(pb_sym)
RPE1_MAPPED = len(pres_sc); RPE1_MISSING = 2000 - RPE1_MAPPED
xrelease_match = (pres_sc == pres_pb)
rpe1_vs_k562 = (RPE1_MAPPED == int(drv["mapped_hvg_count"]))  # driver used raw single-cell too
K562_MAPPED = 1430

# ---- Part J: QC assertions ----
qc = {}
qc["z_shape"] = list(z_RPE1.shape); qc["x_shape"] = list(x_RPE1.shape)
qc["z_dtype"] = str(z_RPE1.dtype); qc["x_dtype"] = str(x_RPE1.dtype)
qc["z_all_finite"] = bool(np.isfinite(z_RPE1).all()); qc["x_all_finite"] = bool(np.isfinite(x_RPE1).all())
qc["x_within_clip"] = bool((np.abs(x_RPE1) <= 10.0).all())
qc["id_sets_identical"] = bool(set(zmap) == set(xmap) == set(man))
qc["no_412_leak"] = bool(len(set(man) & (set(sets["missing_43"]) | set(sets["cached_1889"])) ^ set(man)) == 0)
# per-construct cell counts: z-path (nraw for 1889, |z43 cells| for 43) vs raw obs (ground truth)
ep = pd.read_parquet(iga_paths.MADA_DATA + "/processed/phasec/rpe1_features/full/phasec_rpe1_expression_features_full_v1.parquet")
nraw = dict(zip(ep.gene_transcript.astype(str), ep.n_perturbed_cells_raw.astype(int)))
idx43 = pd.read_parquet(SSD + "/rpe1_topup/rpe1_43_percell_index.parquet")
cnt43 = idx43.gene_transcript.astype(str).value_counts().to_dict()
zpath_cnt = {c: (nraw[c] if c in cached1889 else cnt43[c]) for c in man}
counts_match_raw = all(zpath_cnt[c] == raw_counts[c] for c in man)
qc["cell_counts_match_raw_obs"] = bool(counts_match_raw)
qc["xcell_total_cells_processed"] = int(drv["acc_trace"]["total_cells"])
qc["xcell_perturbed_equals_sum_raw"] = bool(int(drv["acc_trace"]["total_cells"]) - 11485 == sum(raw_counts.values()))
# all-zero rows
qc["z_all_zero_rows"] = int((np.abs(z_RPE1).sum(1) == 0).sum())
qc["x_all_zero_rows"] = int((np.abs(x_RPE1).sum(1) == 0).sum())
qc["row_order"] = "sorted by construct id (deterministic)"
# distributional sanity (descriptive, no outcome)
zn = np.linalg.norm(z_RPE1, axis=1); xn = np.linalg.norm(x_RPE1, axis=1)
qc["z_norm_median_p05_p95"] = [float(np.median(zn)), float(np.percentile(zn,5)), float(np.percentile(zn,95))]
qc["x_norm_median_p05_p95"] = [float(np.median(xn)), float(np.percentile(xn,5)), float(np.percentile(xn,95))]
qc["z_coord_std_median"] = float(np.median(z_RPE1.std(0))); qc["x_coord_std_median"] = float(np.median(x_RPE1.std(0)))
J_PASS = all([qc["z_shape"]==[1932,768], qc["x_shape"]==[1932,2000], qc["z_all_finite"], qc["x_all_finite"],
              qc["x_within_clip"], qc["id_sets_identical"], counts_match_raw, qc["z_all_zero_rows"]==0,
              qc["x_all_zero_rows"]==0, qc["xcell_perturbed_equals_sum_raw"]])

# ---- save final arrays + index ----
np.save(SSD + "/rpe1_topup/z_rpe1_1932.npy", z_RPE1)
np.save(SSD + "/x_cell/x_cell_rpe1_1932_mapped.npy", x_RPE1)
pd.DataFrame({"construct": man, "row": range(len(man)),
             "source": ["reagg_cached" if c in cached1889 else "new_embed_43" for c in man]}).to_parquet(SSD + "/rpe1_topup/rpe1_construct_index.parquet", index=False)

# ---- Part L: 26B gate recompute with Phase-2 achieved rate ----
rate = float(drv["achieved_cells_per_sec"])
now = datetime.now(); kill = datetime(2026, 9, 1)
proj_h = 283970 / rate / 3600
safety_h = 2 * proj_h
cal_h = (kill - now).total_seconds() / 3600
slack_days = (cal_h - safety_h) / 24
gate = "GO" if slack_days >= 5 else ("ESCALATE" if safety_h <= cal_h else "NO-GO")
# L6 alternative projection
proj_h_l6 = 283970 / 7.642 / 3600
flush_sizes = drv["flush_sizes"]
flush_comparable = (np.median(flush_sizes) >= 1000) if flush_sizes else False

res = {
    "z_RPE1_shape": list(z_RPE1.shape), "x_cell_RPE1_shape": list(x_RPE1.shape),
    "NT_POOL_IDENTICAL": NT_POOL_IDENTICAL, "nt_z": int(len(nt_z)), "nt_raw": int(len(nt_raw_idx)), "nt_symdiff": int(len(symdiff)),
    "RPE1_MAPPED_HVG_COUNT": RPE1_MAPPED, "RPE1_MISSING_HVG_COUNT": RPE1_MISSING,
    "RPE1_matches_xrelease_mapping": bool(xrelease_match), "RPE1_vs_K562_agrees": bool(RPE1_MAPPED == K562_MAPPED),
    "qc": qc, "J_PASS": bool(J_PASS),
    "achieved_rate_phase2": rate, "l6_rate": 7.642, "flush_sizes": flush_sizes, "flush_comparable_to_l6": bool(flush_comparable),
    "projected_hours_phase3": proj_h, "safety_adjusted_hours_phase3": safety_h,
    "projected_hours_phase3_l6rate": proj_h_l6,
    "assumed_start": now.isoformat(), "calendar_hours_to_kill": cal_h, "slack_days": slack_days,
    "updated_26B_gate_verdict": gate,
}
json.dump(res, open(SSD + "/logs/phase2_assemble_qc.json", "w"), indent=2, default=str)
for k in ["z_RPE1_shape","x_cell_RPE1_shape","NT_POOL_IDENTICAL","RPE1_MAPPED_HVG_COUNT","RPE1_matches_xrelease_mapping",
          "RPE1_vs_K562_agrees","J_PASS","achieved_rate_phase2","projected_hours_phase3","safety_adjusted_hours_phase3",
          "slack_days","updated_26B_gate_verdict"]:
    print(f"{k}: {res[k]}")
print("qc:", json.dumps(qc, default=str))
