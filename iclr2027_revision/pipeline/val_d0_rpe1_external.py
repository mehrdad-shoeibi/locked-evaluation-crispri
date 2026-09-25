"""PART D0 — external oracle: stream RPE1 cells through the NEW accumulator and compare
its x_cell delta to MADA's persisted x_delta (produced by the same cp10k_log1p_mean),
matched BY CONSTRUCT ID. Read-only against MADA; writes only under iclr2027_revision/."""
import os as _o, sys as _s
_r = _o.path.dirname(_o.path.abspath(__file__))
while _r != _o.path.dirname(_r) and not _o.path.exists(_o.path.join(_r, "iga_paths.py")):
    _r = _o.path.dirname(_r)
if _r not in _s.path:
    _s.path.insert(0, _r)
import iga_paths
import os, sys, json
import numpy as np, pandas as pd, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x_cell_accumulator import XCellAccumulator, cp10k_log1p_rows

FULL = iga_paths.MADA_DATA + "/processed/phasec/rpe1_features/full"
H5 = iga_paths.IGA_NEURIPS + "/data/replogle/raw_singlecell/rpe1_raw_singlecell_01.h5ad"
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 2)))
OUT = os.path.join(_ROOT, "iclr2027_revision/smoke/xcell_validation")
os.makedirs(OUT, exist_ok=True)
N_CONSTRUCTS = 20
CHUNK = 3000

z = np.load(f"{FULL}/phasec_rpe1_x_delta_full_v1.npz", allow_pickle=True)
Xd = z["x_delta"]; gt_art = np.array([str(x) for x in z["gene_transcript"]])
art_dtype = str(Xd.dtype)
g = Xd.shape[1]
ep = pd.read_parquet(f"{FULL}/phasec_rpe1_expression_features_full_v1.parquet")
nraw = dict(zip(ep.gene_transcript.astype(str), ep.n_perturbed_cells_raw.astype(int)))
nt_used = int(ep.expression_full_nt_cells_used.iloc[0])
m = pd.read_parquet(f"{FULL}/master_cell_index_v1.parquet")
m["gene_transcript"] = m.gene_transcript.astype(str); m["role"] = m.role.astype(str)

# deterministic selection: first 20 constructs (sorted) present in the artifact
sel = sorted(gt_art.tolist())[:N_CONSTRUCTS]
art_idx = {c: int(np.where(gt_art == c)[0][0]) for c in sel}

# cells: selected constructs' perturbed cells + ALL global NT
pert = m[(m.role == "perturbed") & (m.gene_transcript.isin(sel))][["orig_obs_idx", "gene_transcript"]]
nt = m[m.role == "nt"][["orig_obs_idx"]]
oidx = np.concatenate([pert.orig_obs_idx.to_numpy(), nt.orig_obs_idx.to_numpy()]).astype(np.int64)
cid = np.concatenate([pert.gene_transcript.to_numpy(), np.array(["__NT_POOL__"] * len(nt), dtype=object)])
ntm = np.concatenate([np.zeros(len(pert), bool), np.ones(len(nt), bool)])
order = np.argsort(oidx, kind="mergesort")
oidx, cid, ntm = oidx[order], cid[order].astype(object), ntm[order]
print(f"[D0] constructs={len(sel)} perturbed_cells={len(pert)} nt_cells={len(nt)} total={len(oidx)}", flush=True)

acc = XCellAccumulator(g)
with h5py.File(H5, "r") as fh:
    X = fh["X"]
    for k, a in enumerate(range(0, len(oidx), CHUNK)):
        b = min(a + CHUNK, len(oidx))
        cslice = np.sort(oidx[a:b])
        # h5py needs increasing order; oidx already globally sorted so slice is sorted
        counts = X[cslice, :].astype(np.float64)
        acc.update(counts, cid[a:b], ntm[a:b], chunk_id=k)
fin = acc.finalize()
xc = fin["x_cell"]; counts_got = fin["counts"]

# compare per construct against MADA Xd (float32), matched by construct id
raw_absdiff = []; post_absdiff = []; cellmatch = True; worst = (None, -1, -1.0)
for c in sel:
    mine = xc[c].astype(np.float64)            # native float64 delta
    mada = Xd[art_idx[c]].astype(np.float64)   # persisted float32 -> promote for diff
    d_raw = np.abs(mine - mada)
    d_post = np.abs(mine.astype(np.float32).astype(np.float64) - mada)  # both at float32
    raw_absdiff.append(d_raw); post_absdiff.append(d_post)
    if counts_got[c] != nraw[c]:
        cellmatch = False
    wm = float(d_post.max())
    if wm > worst[2]:
        worst = (c, int(np.argmax(d_post)), wm)
R = np.concatenate(raw_absdiff); P = np.concatenate(post_absdiff)
def stats(a):
    return dict(max=float(a.max()), mean=float(a.mean()), median=float(np.median(a)),
               p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)),
               n_gt_1e_12=int((a > 1e-12).sum()), n_gt_1e_10=int((a > 1e-10).sum()), n_gt_1e_8=int((a > 1e-8).sum()))
res = {
    "convention": {"mada_avg_over": "ALL cells of each construct (n_perturbed_cells_raw)",
                   "mada_nt": f"global pooled NT over all {nt_used} cells",
                   "gene_axis": f"native RPE1 var order, dim {g}",
                   "excluded_constructs": "none (all 20 conventions coincide: all-cells + global-NT + native axis)"},
    "artifact_dtype": art_dtype, "n_constructs": len(sel), "n_nt_cells": int(len(nt)),
    "cell_counts_exact": bool(cellmatch),
    "raw_float64_vs_float32": stats(R), "post_cast_float32": stats(P),
    "worst_construct": worst[0], "worst_gene_index": worst[1], "worst_post_abs": worst[2],
    "float32_eps": float(np.finfo(np.float32).eps),
}
# dtype-aware verdict (artifact is float32)
if art_dtype == "float32":
    # a "few ulp of float32": scale by the max |value| compared (values up to ~4 here)
    ulp_scale = float(np.spacing(np.abs(Xd[[art_idx[c] for c in sel]]).max().astype(np.float32)))
    res["float32_ulp_at_max_value"] = ulp_scale
    if res["post_cast_float32"]["max"] <= max(8 * ulp_scale, 1e-6):
        verdict = "EXACT"
    elif res["post_cast_float32"]["max"] <= 1e-6 and cellmatch:
        verdict = "NUMERICALLY_EQUIVALENT"
    else:
        verdict = "MISMATCH"
else:
    verdict = "EXACT" if res["raw_float64_vs_float32"]["max"] <= 1e-10 else (
        "NUMERICALLY_EQUIVALENT" if (res["raw_float64_vs_float32"]["max"] <= 1e-8 and cellmatch) else "MISMATCH")
res["EXTERNAL_ORACLE"] = verdict
json.dump(res, open(f"{OUT}/d0_rpe1_external.json", "w"), indent=2)
pd.DataFrame({"construct": sel}).to_csv(f"{OUT}/d0_selected_constructs.csv", index=False)
print("[D0] artifact_dtype:", art_dtype)
print("[D0] raw max|Δ| (f64 vs f32):", res["raw_float64_vs_float32"]["max"])
print("[D0] post-cast max|Δ| (f32 vs f32):", res["post_cast_float32"]["max"], "| float32 ulp@max:", res.get("float32_ulp_at_max_value"))
print("[D0] cell_counts_exact:", cellmatch, "| worst:", worst)
print("[D0] EXTERNAL_ORACLE =", verdict)
