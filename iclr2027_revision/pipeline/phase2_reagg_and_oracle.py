"""Phase 2, Parts C/D/E: re-aggregate the 1,889 cached RPE1 constructs' z from MADA's
cached per-cell embeddings (CPU), oracle vs MADA's cached gf_z_emb (D) and gf_mu_nt (E).
Read-only against MADA; writes only under the SSD ICLR root. LABEL-FREE."""
import os as _o, sys as _s
_r = _o.path.dirname(_o.path.abspath(__file__))
while _r != _o.path.dirname(_r) and not _o.path.exists(_o.path.join(_r, "iga_paths.py")):
    _r = _o.path.dirname(_r)
if _r not in _s.path:
    _s.path.insert(0, _r)
import iga_paths
import os, sys, json, hashlib, glob
import numpy as np, pandas as pd

FULL = iga_paths.MADA_DATA + "/processed/phasec/rpe1_features/full"
CH = FULL + "/geneformer_chunks"
SSD = iga_paths.GENEFORMER
os.makedirs(SSD + "/scratch", exist_ok=True)
sets = json.load(open(SSD + "/manifests/rpe1_construct_sets.json"))
cached_1889 = set(sets["cached_1889"])

m = pd.read_parquet(FULL + "/master_cell_index_v1.parquet")
m["gene_transcript"] = m.gene_transcript.astype(str); m["role"] = m.role.astype(str)
nch = int(m.chunk_id.max()) + 1

# ---- load all per-cell embeddings + their orig_obs_idx (Part C) ----
emb_list, oidx_list = [], []
for c in range(nch):
    emb_list.append(np.load(f"{CH}/chunk_{c:04d}/emb.npy"))                # (nc,768) f32
    ei = pd.read_parquet(f"{CH}/chunk_{c:04d}/emb_index.parquet")
    oidx_list.append(ei["orig_obs_idx"].to_numpy(np.int64))
# Keep float32: MADA aggregated the fp32 per-cell embeddings with a float32 accumulator
# (numpy .mean on a float32 array). float64 accumulation diverges by ~2.4e-4 on the large
# embedding coords (|E| up to ~4.15) — that would make RPE1/K562 z inconsistent with the
# reused VCC z (MADA float32) and the Lock §5 fp32 convention. So aggregate z in float32.
emb_all = np.concatenate(emb_list)                                          # float32 (match MADA)
oidx_all = np.concatenate(oidx_list)
assert emb_all.shape[0] == len(m) == 236335, (emb_all.shape, len(m))
order = np.argsort(oidx_all, kind="mergesort"); oidx_s = oidx_all[order]; emb_s = emb_all[order]
def emb_for(oidx):
    pos = np.searchsorted(oidx_s, oidx)
    assert np.all(oidx_s[pos] == oidx), "orig_obs_idx not found in cached embeddings"
    return emb_s[pos]

# ---- Part E: global NT mean over all 11,485 cached NT embeddings ----
nt_oidx = np.sort(m[m.role == "nt"].orig_obs_idx.to_numpy(np.int64))
assert nt_oidx.size == 11485, nt_oidx.size
mu_nt = emb_for(nt_oidx).mean(axis=0)                                       # (768,) f64
gf_mu_nt = np.load(FULL + "/phasec_rpe1_gf_mu_nt_full_v1.npy")  # float32
nt_raw = float(np.abs(mu_nt - gf_mu_nt).max())
nt_post = float(np.abs(mu_nt.astype(np.float32).astype(np.float64) - gf_mu_nt).max())
nt_verdict = "EXACT" if nt_post <= max(8*np.spacing(np.float32(np.abs(gf_mu_nt).max())), 0.0) else (
    "NUMERICALLY_EQUIVALENT" if nt_raw <= 1e-6 else "MISMATCH")
nt_hash = hashlib.sha256(nt_oidx.tobytes()).hexdigest()

# ---- Part C+D: re-aggregate 1,889 constructs' z, oracle vs MADA gf_z_emb ----
# raw per-construct cell counts (expression parquet) for the cache-covers-all assertion
ep = pd.read_parquet(FULL + "/phasec_rpe1_expression_features_full_v1.parquet")
nraw = dict(zip(ep.gene_transcript.astype(str), ep.n_perturbed_cells_raw.astype(int)))
# MADA cached z, matched by construct id
gpq = pd.read_parquet(FULL + "/phasec_rpe1_geneformer_features_full_v1.parquet")
gt_order = gpq.gene_transcript.astype(str).to_numpy()
GZ = np.load(FULL + "/phasec_rpe1_gf_z_emb_full_v1.npy")                    # (2301,768) f32
gz_by = {gt_order[i]: GZ[i] for i in range(len(gt_order))}  # float32

pert = m[m.role == "perturbed"]
by_gt = {g: sub.orig_obs_idx.to_numpy(np.int64) for g, sub in pert.groupby("gene_transcript")}
z_reagg = {}; cover_ok = True; raw_d, post_d = [], []; worst = (None, -1, -1.0)
for c in sorted(cached_1889):
    cells = by_gt[c]
    if len(cells) != nraw[c]:
        cover_ok = False
    mu = emb_for(cells).mean(axis=0)
    z = mu - mu_nt
    z_reagg[c] = z
    d_raw = np.abs(z - gz_by[c]); d_post = np.abs(z.astype(np.float32).astype(np.float64) - gz_by[c])
    raw_d.append(d_raw); post_d.append(d_post)
    if float(d_post.max()) > worst[2]:
        worst = (c, int(np.argmax(d_post)), float(d_post.max()))
R = np.concatenate(raw_d); P = np.concatenate(post_d)
def st(a): return dict(max=float(a.max()), mean=float(a.mean()), median=float(np.median(a)),
                       p95=float(np.percentile(a,95)), p99=float(np.percentile(a,99)),
                       n_gt_1e12=int((a>1e-12).sum()), n_gt_1e10=int((a>1e-10).sum()), n_gt_1e8=int((a>1e-8).sum()))
ulp = float(np.spacing(np.float32(np.abs(GZ).max())))
postmax = float(P.max())
z_verdict = "EXACT" if postmax <= max(8*ulp, 1e-6) else ("NUMERICALLY_EQUIVALENT" if (R.max()<=1e-6 and cover_ok) else "MISMATCH")

res = {
    "cache_covers_all_1889": bool(cover_ok),
    "Z_REAGG_ORACLE": z_verdict, "artifact_dtype": "float32",
    "z_raw": st(R), "z_postcast": st(P), "z_postcast_bitwise_equal_entries": int((P==0).sum()), "z_total_entries": int(P.size),
    "z_worst_construct": worst[0], "z_worst_dim": worst[1], "z_worst_post_abs": worst[2], "float32_ulp_at_max": ulp,
    "NT_EMB_ORACLE": nt_verdict, "nt_raw_max": nt_raw, "nt_post_max": nt_post, "nt_cell_count": int(nt_oidx.size), "nt_cellid_hash": nt_hash,
}
np.savez(SSD + "/scratch/z_reagg_1889.npz", ids=np.array(sorted(cached_1889)),
         z=np.stack([z_reagg[c] for c in sorted(cached_1889)]).astype(np.float64))
np.save(SSD + "/scratch/mu_nt_emb.npy", mu_nt)
np.save(SSD + "/scratch/nt_oidx.npy", nt_oidx)
json.dump(res, open(SSD + "/scratch/reagg_oracle.json", "w"), indent=2)
print("[C] cache_covers_all_1889:", cover_ok)
print("[E] NT_EMB_ORACLE:", nt_verdict, "raw", nt_raw, "post", nt_post, "| NT count", nt_oidx.size)
print("[D] Z_REAGG_ORACLE:", z_verdict, "| raw max", R.max(), "| post-cast max", postmax, "| bitwise-equal", int((P==0).sum()), "/", P.size)
print("[D] worst:", worst, "| float32 ulp@max", ulp)
