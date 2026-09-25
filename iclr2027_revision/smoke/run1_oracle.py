"""RUN 1 — RPE1 CORRECTNESS ORACLE. Read-only against MADA cache; writes only under
iclr2027_revision/. Re-embeds ~1,500 already-cached RPE1 cells (500 each from chunks
0/1/2) via the ICLR adapter and compares to the cached per-cell embeddings, matched
by cell_id."""
import os as _o, sys as _s
_r = _o.path.dirname(_o.path.abspath(__file__))
while _r != _o.path.dirname(_r) and not _o.path.exists(_o.path.join(_r, "iga_paths.py")):
    _r = _o.path.dirname(_r)
if _r not in _s.path:
    _s.path.insert(0, _r)
import iga_paths
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 2)))
os.environ["HF_HOME"] = os.path.join(_ROOT, "iclr2027_revision/smoke/hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(_ROOT, "iclr2027_revision/smoke/hf_cache/datasets")
from pathlib import Path
import numpy as np, pandas as pd, h5py, anndata as ad
sys.path.insert(0, os.path.join(_ROOT, "iclr2027_revision/smoke"))
from adapter import embed_cells

RPE1_CACHE = Path(iga_paths.MADA_DATA + "/processed/phasec/rpe1_features/full")
CHUNKS = RPE1_CACHE / "geneformer_chunks"
H5 = iga_paths.IGA_NEURIPS + "/data/replogle/raw_singlecell/rpe1_raw_singlecell_01.h5ad"
OUT = Path(os.path.join(_ROOT, "iclr2027_revision/smoke/run1_rpe1_oracle"))
OUT.mkdir(parents=True, exist_ok=True)
PER_CHUNK = 500
SEL_CHUNKS = [0, 1, 2]

print("[run1] selecting cached cells + loading cached embeddings ...", flush=True)
cached = {}          # cell_id -> (768,) cached emb
sel_orig = {}        # cell_id -> orig_obs_idx
for cid in SEL_CHUNKS:
    ei = pd.read_parquet(CHUNKS / f"chunk_{cid:04d}" / "emb_index.parquet")
    emb = np.load(CHUNKS / f"chunk_{cid:04d}" / "emb.npy")   # (5000,768) float32
    assert emb.shape[0] == len(ei)
    take = ei.index[ei["chunk_row_pos"].values < PER_CHUNK]
    for k in take:
        c = str(ei["cell_id"].iloc[k])
        cached[c] = emb[k].astype(np.float64)
        sel_orig[c] = int(ei["orig_obs_idx"].iloc[k])
print(f"[run1] selected {len(cached)} cells across chunks {SEL_CHUNKS}", flush=True)

# RPE1 var ensembl ids (var index is ENSG, matching the MADA extractor)
av = ad.read_h5ad(H5, backed="r"); var_names = np.asarray(av.var_names).astype(str); av.file.close()
assert np.mean([v.startswith("ENSG") for v in var_names[:50]]) == 1.0, "RPE1 var not ENSG"

# read raw counts for the selected cells (sorted unique orig_obs_idx)
idx_sorted = np.array(sorted(set(sel_orig.values())), dtype=np.int64)
cell_ids = [f"cell_{int(o)}" for o in idx_sorted]
assert set(cell_ids) == set(cached.keys()), "cell_id/orig mismatch"
t0 = time.time()
with h5py.File(H5, "r") as fh:
    X = fh["X"][idx_sorted, :].astype(np.float32)
print(f"[run1] read raw counts {X.shape} in {time.time()-t0:.1f}s; embedding ...", flush=True)

res = embed_cells(X, cell_ids, list(var_names), work_dir=str(OUT / "work"), prefix="rpe1_oracle")
new = res["cid2emb"]
print(f"[run1] embedded n={res['n']} tok={res['tok_sec']:.1f}s emb={res['emb_sec']:.1f}s", flush=True)

# compare matched by cell_id
diffs = []
cos = []
common = [c for c in cached if c in new]
for c in common:
    a = cached[c]; b = new[c].astype(np.float64)
    diffs.append(np.abs(b - a))
    cos.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
D = np.concatenate(diffs)            # all cell x dim abs diffs
cos = np.array(cos)
stats = {
    "n_selected": len(cached), "n_compared": len(common),
    "n_missing_in_new": int(len(cached) - len(common)),
    "chunks": SEL_CHUNKS, "per_chunk": PER_CHUNK,
    "max_abs_diff": float(D.max()), "mean_abs_diff": float(D.mean()),
    "median_abs_diff": float(np.median(D)), "p95_abs_diff": float(np.percentile(D, 95)),
    "p99_abs_diff": float(np.percentile(D, 99)),
    "n_gt_1e-6": int((D > 1e-6).sum()), "n_gt_1e-4": int((D > 1e-4).sum()), "n_gt_1e-3": int((D > 1e-3).sum()),
    "n_entries_total": int(D.size),
    "cosine_min": float(cos.min()), "cosine_median": float(np.median(cos)),
    "tok_sec": res["tok_sec"], "emb_sec": res["emb_sec"],
    "transformers": __import__("transformers").__version__,
    "torch": __import__("torch").__version__,
}
if stats["max_abs_diff"] < 1e-6:
    verdict = "EXACT"
elif stats["max_abs_diff"] < 1e-3 and stats["cosine_min"] > 0.9999:
    verdict = "NUMERICALLY_CLOSE"
else:
    verdict = "MISMATCH"
stats["ORACLE"] = verdict
json.dump(stats, open(OUT / "oracle_result.json", "w"), indent=2)
pd.DataFrame({"cell_id": sorted(cached.keys()),
             "orig_obs_idx": [sel_orig[c] for c in sorted(cached.keys())]}).to_csv(OUT / "selected_cells.csv", index=False)
print("\n==== ORACLE RESULT ====")
for k in ["n_selected","n_compared","max_abs_diff","mean_abs_diff","median_abs_diff","p95_abs_diff","p99_abs_diff",
          "n_gt_1e-6","n_gt_1e-4","n_gt_1e-3","cosine_min","cosine_median","transformers","torch","ORACLE"]:
    print(f"  {k}: {stats[k]}")
