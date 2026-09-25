"""RUN 2 — K562-ESSENTIAL THROUGHPUT + INGESTION. Read-only against MADA (model +
dicts) and the SSD raw single-cell; writes only under iclr2027_revision/. Model is
loaded ONCE and reused across chunks so per-chunk emb time excludes model load
(matching the Lock 26B projection formula, which separates one-time startup from
steady-state)."""
import os as _o, sys as _s
_r = _o.path.dirname(_o.path.abspath(__file__))
while _r != _o.path.dirname(_r) and not _o.path.exists(_o.path.join(_r, "iga_paths.py")):
    _r = _o.path.dirname(_r)
if _r not in _s.path:
    _s.path.insert(0, _r)
import iga_paths
import os, sys, json, time, pickle, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 2)))
os.environ["HF_HOME"] = os.path.join(_ROOT, "iclr2027_revision/smoke/hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(_ROOT, "iclr2027_revision/smoke/hf_cache/datasets")
from pathlib import Path
import numpy as np, pandas as pd, h5py, anndata as ad, scipy.sparse as sp

REPO = iga_paths.MADA + "/cache/foundation_models/geneformer-repo"
MODEL_DIR = REPO + "/Geneformer-V2-104M"
TOKDICT = iga_paths.MADA + "/cache/foundation_models/geneformer-repo/geneformer/token_dictionary_gc104M.pkl"
H5 = iga_paths.IGA_NEURIPS + "/data/replogle/raw_singlecell/K562_essential_raw_singlecell_01.h5ad"
ELIG = iga_paths.IGA_NEURIPS + "/results_v4/k562_essential_coverage_gate_v1/k562_essential_eligibility.csv"
OUT = Path(os.path.join(_ROOT, "iclr2027_revision/smoke/run2_k562_throughput"))
OUT.mkdir(parents=True, exist_ok=True)
N_CONSTRUCTS = 20
TARGET_TOTAL = 6000          # <= 30,000 (bounded smoke)
CHUNK = 1200
sys.path.insert(0, REPO)

# ---- decode K562 obs (old-style __categories) ----
def decode_obs(f, col):
    og = f["obs"]; node = og[col]
    if isinstance(node, h5py.Group):
        cats = node["categories"][:]; codes = node["codes"][:]
    else:
        codes = node[:]
        cats = og["__categories"][col][:] if ("__categories" in og and col in og["__categories"]) else None
        if cats is None:
            return np.array([x.decode() if isinstance(x, bytes) else x for x in codes])
    cats = np.array([c.decode() if isinstance(c, bytes) else c for c in cats])
    return cats[codes]

print("[run2] ingestion validation ...", flush=True)
f = h5py.File(H5, "r")
gene = decode_obs(f, "gene"); gt = decode_obs(f, "gene_transcript"); gem = decode_obs(f, "gem_group")
vg = f["var"]["gene_id"][:]
ensembl_ids = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in vg])
ingest = {
    "var_gene_id_is_ensembl": bool(np.mean([str(x).startswith("ENSG") for x in ensembl_ids[:200]]) == 1.0),
    "n_genes": int(len(ensembl_ids)),
    "obs_has_gene": "gene" in f["obs"], "obs_has_gene_transcript": "gene_transcript" in f["obs"],
    "obs_has_gem_group": "gem_group" in f["obs"],
    "n_cells_total": int(f["X"].shape[0]),
    "nt_pool_size": int((gene == "non-targeting").sum()),
}
# raw integer counts check on a sample block
Xsamp = f["X"][:64, :]
ingest["counts_nonneg"] = bool((Xsamp >= 0).all())
ingest["counts_integer_valued"] = bool(np.allclose(Xsamp, np.round(Xsamp)))

el = pd.read_csv(ELIG)
elig_constructs = el.loc[el["eligible"] == True, "construct"].astype(str).tolist()
elig_set = set(elig_constructs)
sel_constructs = sorted(elig_constructs)[:N_CONSTRUCTS]
ingest["constructs_join_eligible"] = bool(all(c in elig_set for c in sel_constructs))
ingest["selected_constructs"] = sel_constructs

# perturbed cell indices for selected constructs
gt_str = gt.astype(str)
pert_idx = np.where(np.isin(gt_str, sel_constructs))[0]
nt_all = np.where(gene == "non-targeting")[0]
n_nt = int(min(len(nt_all), max(1000, TARGET_TOTAL - len(pert_idx))))
nt_pick = nt_all[np.linspace(0, len(nt_all) - 1, n_nt).astype(int)]   # deterministic evenly-spaced
nt_pick = np.unique(nt_pick)
sel_idx = np.concatenate([pert_idx, nt_pick])
sel_role = np.array(["perturbed"] * len(pert_idx) + ["nt"] * len(nt_pick))
sel_gt = np.concatenate([gt_str[pert_idx], np.array(["NT_POOL"] * len(nt_pick))])
order = np.argsort(sel_idx, kind="mergesort")
sel_idx = sel_idx[order]; sel_role = sel_role[order]; sel_gt = sel_gt[order]
cell_ids = np.array([f"k562_{int(i)}" for i in sel_idx])
ingest["n_perturbed_selected"] = int(len(pert_idx)); ingest["n_nt_selected"] = int(len(nt_pick))
ingest["n_total_selected"] = int(len(sel_idx))
print(f"[run2] selected perturbed={len(pert_idx)} nt={len(nt_pick)} total={len(sel_idx)}", flush=True)
json.dump(ingest, open(OUT / "ingestion.json", "w"), indent=2, default=str)
pd.DataFrame({"cell_id": cell_ids, "orig_obs_idx": sel_idx, "role": sel_role,
             "gene_transcript": sel_gt}).to_csv(OUT / "selected_cells.csv", index=False)

# ---- load model ONCE (startup) ----
import torch
from geneformer import perturber_utils as pu, TranscriptomeTokenizer
from geneformer.emb_extractor import get_embs
tok_dict = pickle.load(open(TOKDICT, "rb"))          # gene -> id
token_gene_dict = {v: k for k, v in tok_dict.items()}
pad_token_id = tok_dict["<pad>"]
t_start = time.time()
model = pu.load_model("Pretrained", 0, MODEL_DIR, "eval")
layer_to_quant = pu.quant_layers(model) + (-1)
startup_sec = time.time() - t_start
print(f"[run2] model loaded once in {startup_sec:.1f}s; layer_to_quant={layer_to_quant}", flush=True)

def tokenize_chunk(cidx, cell_id_sub, gt_sub, role_sub, work):
    for dd in (work / "input", work / "tok"):
        if dd.exists(): shutil.rmtree(dd)
        dd.mkdir(parents=True, exist_ok=True)
    with h5py.File(H5, "r") as fh:
        X = fh["X"][np.sort(cidx), :].astype(np.float32)
    # reorder to cidx order
    o = np.argsort(cidx, kind="mergesort"); inv = np.empty_like(o); inv[o] = np.arange(len(o))
    X = X[inv]
    Xs = sp.csr_matrix(X)
    adata = ad.AnnData(X=Xs, obs=pd.DataFrame({"cell_id": cell_id_sub, "gene_transcript": gt_sub, "role": role_sub}))
    adata.var_names = [str(e) for e in ensembl_ids]; adata.var["ensembl_id"] = [str(e) for e in ensembl_ids]
    adata.obs["n_counts"] = np.asarray(Xs.sum(axis=1)).ravel()
    adata.obs_names = adata.obs["cell_id"].astype(str); adata.obs_names_make_unique()
    adata.write_h5ad(str(work / "input" / "chunk.h5ad"))
    tk = TranscriptomeTokenizer(custom_attr_name_dict={"cell_id": "cell_id", "gene_transcript": "gene_transcript", "role": "role"},
                                nproc=1, chunk_size=512, model_input_size=4096, model_version="V2")
    tk.tokenize_data(str(work / "input"), str(work / "tok"), f"chunk_{cidx.size}", file_format="h5ad")
    ds = pu.load_and_filter(None, 1, str(work / "tok" / f"chunk_{cidx.size}.dataset"))
    ds = pu.downsample_and_sort(ds, len(cidx) + 10)
    return ds

# ---- per-chunk embedding (model reused) ----
n = len(sel_idx); chunks = [(i, min(i + CHUNK, n)) for i in range(0, n, CHUNK)]
per_chunk = []; all_emb = {}; f.close()
for ci, (a, b) in enumerate(chunks):
    work = OUT / f"work_chunk{ci}"
    cidx = sel_idx[a:b]
    t0 = time.time()
    ds = tokenize_chunk(cidx, cell_ids[a:b].tolist(), sel_gt[a:b].tolist(), sel_role[a:b].tolist(), work)
    tok_sec = time.time() - t0
    t1 = time.time()
    embs = get_embs(model, ds, "cls", layer_to_quant, pad_token_id, 16, token_gene_dict, silent=True)
    torch.cuda.synchronize()
    emb_sec = time.time() - t1
    arr = embs.cpu().float().numpy()
    ids = list(ds["cell_id"])
    for k, cid in enumerate(ids): all_emb[cid] = arr[k]
    cps = arr.shape[0] / emb_sec
    per_chunk.append({"chunk": ci, "n": int(arr.shape[0]), "tok_sec": tok_sec, "emb_sec": emb_sec,
                      "cells_per_sec": cps, "finite": bool(np.isfinite(arr).all()), "shape": list(arr.shape)})
    print(f"[run2] chunk {ci}: n={arr.shape[0]} tok={tok_sec:.1f}s emb={emb_sec:.1f}s -> {cps:.2f} cells/s", flush=True)
    shutil.rmtree(work, ignore_errors=True)

# ---- determinism: re-embed first 200 cells a 2nd time ----
work = OUT / "work_det"; first200 = sel_idx[:200]
ds2 = tokenize_chunk(first200, cell_ids[:200].tolist(), sel_gt[:200].tolist(), sel_role[:200].tolist(), work)
e1 = get_embs(model, ds2, "cls", layer_to_quant, pad_token_id, 16, token_gene_dict, silent=True).cpu().float().numpy()
e2 = get_embs(model, ds2, "cls", layer_to_quant, pad_token_id, 16, token_gene_dict, silent=True).cpu().float().numpy()
det_max = float(np.abs(e1 - e2).max())
shutil.rmtree(work, ignore_errors=True)

# ---- steady state (discard chunk 0 warm-up) ----
cps_all = [c["cells_per_sec"] for c in per_chunk]
steady = [c["cells_per_sec"] for c in per_chunk[1:]] if len(per_chunk) > 1 else cps_all
tot_cells = sum(c["n"] for c in per_chunk)
tot_emb = sum(c["emb_sec"] for c in per_chunk); tot_tok = sum(c["tok_sec"] for c in per_chunk)
naive = tot_cells / (tot_emb + tot_tok)
# per-cell bytes (embeddings fp32 768 + tokenized dataset)
emb_bytes_per_cell = 768 * 4
res = {
    "ingestion": ingest,
    "startup_sec": startup_sec,
    "per_chunk": per_chunk,
    "steady_state_cells_per_sec_median": float(np.median(steady)),
    "steady_state_p05": float(np.percentile(steady, 5)),
    "steady_state_p95": float(np.percentile(steady, 95)),
    "naive_end_to_end_cells_per_sec": float(naive),
    "n_steady_chunks": len(steady), "total_cells": tot_cells,
    "determinism_max_abs_diff": det_max,
    "emb_bytes_per_cell": emb_bytes_per_cell,
    "output_shape_dim": 768, "all_finite": bool(all(c["finite"] for c in per_chunk)),
    "unique_cell_ids": int(len(all_emb)), "gpu": "cuda:1 (single, NVIDIA TITAN RTX)",
    "rate_is": "single-GPU",
    "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    "transformers": __import__("transformers").__version__, "torch": torch.__version__,
}
json.dump(res, open(OUT / "throughput_result.json", "w"), indent=2, default=str)
print("\n==== RUN2 SUMMARY ====")
print(f"  startup_sec: {startup_sec:.1f}")
print(f"  steady_state median cells/s: {res['steady_state_cells_per_sec_median']:.3f} (p05 {res['steady_state_p05']:.3f} / p95 {res['steady_state_p95']:.3f}, n_steady={len(steady)})")
print(f"  naive end-to-end cells/s: {naive:.3f}")
print(f"  determinism max|Δ|: {det_max:.3e}")
print(f"  peak VRAM alloc: {res['peak_vram_bytes']/1e9:.2f} GB  reserved: {res['peak_vram_reserved_bytes']/1e9:.2f} GB")
print(f"  all finite: {res['all_finite']}  unique cells: {res['unique_cell_ids']}")
