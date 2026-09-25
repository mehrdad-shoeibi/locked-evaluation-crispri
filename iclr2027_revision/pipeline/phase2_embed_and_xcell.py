"""Phase 2, Parts B/F/G: ONE streaming raw-count pass over all cells of the 1,932
manuscript-eligible RPE1 constructs + all 11,485 NT. Every cell -> x_cell accumulator
(native 8,749-gene space, float64). ONLY the 43 missing constructs' cells -> Geneformer
embedding branch (buffered, flushed at ~1,200 cells). The 1,889 cached constructs are
NEVER re-embedded. LABEL-FREE. Writes under the SSD ICLR root + iclr2027_revision/."""
import os as _o, sys as _s
_r = _o.path.dirname(_o.path.abspath(__file__))
while _r != _o.path.dirname(_r) and not _o.path.exists(_o.path.join(_r, "iga_paths.py")):
    _r = _o.path.dirname(_r)
if _r not in _s.path:
    _s.path.insert(0, _r)
import iga_paths
import os, sys, json, time, pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 2)))
os.environ["HF_HOME"] = os.path.join(_ROOT, "iclr2027_revision/smoke/hf_cache")
import numpy as np, pandas as pd, h5py, anndata as ad, scipy.sparse as sp
sys.path.insert(0, os.path.join(_ROOT, "iclr2027_revision/pipeline"))
from x_cell_accumulator import XCellAccumulator, cp10k_log1p_rows, map_and_clip_to_vcc_hvg

REPO = iga_paths.MADA + "/cache/foundation_models/geneformer-repo"
MODEL_DIR = REPO + "/Geneformer-V2-104M"
TOKDICT = REPO + "/geneformer/token_dictionary_gc104M.pkl"
H5 = iga_paths.IGA_NEURIPS + "/data/replogle/raw_singlecell/rpe1_raw_singlecell_01.h5ad"
ARM = iga_paths.IGA_NEURIPS + "/results_v4/transfer_pretest_lock_v1/arm1_rpe1_construct_list.csv"
SG = iga_paths.IGA_NEURIPS + "/cache/vcc/variants/delta_hvg/selected_genes.npy"
VGN = iga_paths.IGA_NEURIPS + "/cache/vcc/var_gene_names.npy"
SSD = iga_paths.GENEFORMER
WORK = os.path.join(_ROOT, "iclr2027_revision/smoke/xcell_validation/p2work")
for d in [SSD+"/rpe1_topup", SSD+"/x_cell", SSD+"/checkpoints/rpe1_xcell", SSD+"/logs", WORK]:
    os.makedirs(d, exist_ok=True)
sys.path.insert(0, REPO)
sets = json.load(open(SSD + "/manifests/rpe1_construct_sets.json"))
miss43 = set(sets["missing_43"]); cached1889 = set(sets["cached_1889"])
CHUNK = 5000; FLUSH = 1200

# ---- obs (decoded via anndata) ----
a = ad.read_h5ad(H5, backed="r")
gt_all = a.obs["gene_transcript"].astype(str).to_numpy()
gene_all = a.obs["gene"].astype(str).to_numpy()
var_ensembl = np.asarray(a.var_names).astype(str)
var_symbol = a.var["gene_name"].astype(str).to_numpy()
a.file.close()
man = set(pd.read_csv(ARM)["construct"].astype(str))
assert man == (miss43 | cached1889) and len(man) == 1932

is_nt = gene_all == "non-targeting"
is_man = np.isin(gt_all, list(man))
pass_mask = is_man | is_nt
idx = np.where(pass_mask)[0].astype(np.int64)
role = np.where(is_nt[idx], "nt", "perturbed")
cid = np.where(is_nt[idx], "__NT_POOL__", gt_all[idx]).astype(object)
needs = np.isin(gt_all[idx], list(miss43)) & (~is_nt[idx])
o = np.argsort(idx, kind="mergesort"); idx, role, cid, needs = idx[o], role[o], cid[o], needs[o]
n_nt = int((role == "nt").sum()); n_43 = int(needs.sum()); n_pert = int((role == "perturbed").sum())
print(f"[pass] total={len(idx)} perturbed={n_pert} nt={n_nt} to-embed(43)={n_43}", flush=True)
assert n_nt == 11485, n_nt
if n_43 != 8445:
    print(f"[WARN] 43-construct cell count {n_43} != expected 8445", flush=True)

# ---- Geneformer model + token dict (once) ----
import torch
from geneformer import perturber_utils as pu, TranscriptomeTokenizer
from geneformer.emb_extractor import get_embs
td = pickle.load(open(TOKDICT, "rb")); tgd = {v: k for k, v in td.items()}; pad_id = td["<pad>"]
t_load = time.time(); model = pu.load_model("Pretrained", 0, MODEL_DIR, "eval")
layer = pu.quant_layers(model) + (-1); startup = time.time() - t_load
print(f"[model] loaded in {startup:.1f}s", flush=True)

def embed(counts, cell_ids, tag):
    """Tokenize + get_embs (preloaded model). Returns cid->emb(768) f32, plus emb_seconds."""
    ind = WORK + f"/in_{tag}"; tok = WORK + f"/tok_{tag}"
    import shutil
    for dd in (ind, tok):
        shutil.rmtree(dd, ignore_errors=True); os.makedirs(dd, exist_ok=True)
    Xs = sp.csr_matrix(np.asarray(counts, dtype=np.float32))
    ada = ad.AnnData(X=Xs, obs=pd.DataFrame({"cell_id": [str(c) for c in cell_ids]}))
    ada.var_names = list(var_ensembl); ada.var["ensembl_id"] = list(var_ensembl)
    ada.obs["n_counts"] = np.asarray(Xs.sum(1)).ravel()
    ada.obs_names = ada.obs["cell_id"].astype(str); ada.obs_names_make_unique()
    ada.write_h5ad(ind + "/c.h5ad")
    tk = TranscriptomeTokenizer(custom_attr_name_dict={"cell_id": "cell_id"}, nproc=1,
                                chunk_size=512, model_input_size=4096, model_version="V2")
    tk.tokenize_data(ind, tok, tag, file_format="h5ad")
    ds = pu.load_and_filter(None, 1, tok + f"/{tag}.dataset")
    ds = pu.downsample_and_sort(ds, len(cell_ids) + 10)
    t = time.time(); embs = get_embs(model, ds, "cls", layer, pad_id, 16, tgd, silent=True)
    torch.cuda.synchronize(); dt = time.time() - t
    arr = embs.cpu().float().numpy()
    ids = list(ds["cell_id"])
    return {ids[k]: arr[k] for k in range(len(ids))}, dt

# ---- streaming pass ----
acc = XCellAccumulator(len(var_ensembl), checkpoint_dir=SSD + "/checkpoints/rpe1_xcell", checkpoint_every=5)
emb43 = {}          # cell_id(str orig_idx) -> emb(768) f32
buf_counts = []; buf_ids = []; flush_sizes = []; embed_secs = 0.0; nflush = 0
chunks = [(i, min(i + CHUNK, len(idx))) for i in range(0, len(idx), CHUNK)]
t_pass = time.time()
with h5py.File(H5, "r") as fh:
    Xd = fh["X"]
    for k, (aa, bb) in enumerate(chunks):
        cix = np.sort(idx[aa:bb])                      # sorted for h5py
        counts = Xd[cix, :].astype(np.float32)
        # rows already in cix (== idx[aa:bb] sorted) order; cid/role/needs are in idx order,
        # and idx[aa:bb] is already globally sorted so cix == idx[aa:bb]
        acc.update(counts, cid[aa:bb], (role[aa:bb] == "nt"), chunk_id=k)
        nb = needs[aa:bb]
        if nb.any():
            buf_counts.append(counts[nb]); buf_ids.extend([f"{int(x)}" for x in idx[aa:bb][nb]])
            if sum(c.shape[0] for c in buf_counts) >= FLUSH:
                C = np.concatenate(buf_counts); e, dt = embed(C, buf_ids, f"f{nflush}")
                emb43.update(e); embed_secs += dt; flush_sizes.append(C.shape[0]); nflush += 1
                print(f"[flush {nflush}] {C.shape[0]} cells {dt:.1f}s ({C.shape[0]/dt:.2f} c/s)", flush=True)
                buf_counts, buf_ids = [], []
if buf_counts:
    C = np.concatenate(buf_counts); e, dt = embed(C, buf_ids, f"f{nflush}")
    emb43.update(e); embed_secs += dt; flush_sizes.append(C.shape[0]); nflush += 1
    print(f"[flush {nflush}] {C.shape[0]} cells {dt:.1f}s", flush=True)
pass_secs = time.time() - t_pass
assert len(emb43) == n_43, (len(emb43), n_43)
achieved = n_43 / embed_secs
print(f"[pass done] {pass_secs:.1f}s | embed {embed_secs:.1f}s | achieved {achieved:.3f} c/s | flushes {nflush}", flush=True)

# ---- finalize x_cell (native 8749) + map -> VCC-HVG (2000) ----
fin = acc.finalize()
xnat = fin["x_cell"]; comp_delta = acc.compensated_max_delta()
var_gene_names = np.load(VGN, allow_pickle=True).astype(str); sel_idx = np.load(SG)
vcc_hvg = var_gene_names[sel_idx]
mapped_count = None; x_mapped = {}
for c in man:
    mm, nmap = map_and_clip_to_vcc_hvg(xnat[c], var_symbol, vcc_hvg); x_mapped[c] = mm; mapped_count = nmap

# ---- z for the 43 (float32 aggregation, matches re-agg + fp32 convention) ----
mu_nt_emb = np.load(SSD + "/scratch/mu_nt_emb.npy")   # float32
cell2gt = {f"{int(x)}": g for x, g in zip(idx[needs], gt_all[idx[needs]])}
z43 = {}
by43 = {}
for cidx, g in cell2gt.items():
    by43.setdefault(g, []).append(cidx)
for g, cids in by43.items():
    E = np.stack([emb43[c] for c in cids]).astype(np.float32)
    z43[g] = (E.mean(axis=0) - mu_nt_emb).astype(np.float32)
assert set(z43) == miss43

# ---- Part G: same-pass verification for the 43 (in-loop x_cell vs independent direct) ----
d43_cells = np.sort(idx[needs])
with h5py.File(H5, "r") as fh:
    Cd = fh["X"][d43_cells, :].astype(np.float32)
rows = cp10k_log1p_rows(Cd)
gt_d = gt_all[d43_cells]
# global NT mean over raw counts (same NT pool) for the delta
nt_oidx = np.load(SSD + "/scratch/nt_oidx.npy")
# accumulate NT raw-count mean directly for the check (bounded read already done in acc; recompute here)
mu_nt_raw = fin["mu_nt"]     # from the accumulator (all 11,485 NT), native
sp_max = 0.0
for g in miss43:
    sel = gt_d == g
    direct = rows[sel].mean(axis=0) - mu_nt_raw
    sp_max = max(sp_max, float(np.abs(xnat[g] - direct).max()))
same_pass = "YES" if sp_max <= 1e-10 else "NO"
print(f"[G] SAME_PASS_VERIFIED = {same_pass} (max|Δ|={sp_max:.3e})", flush=True)

# ---- Part B: determinism spot check (re-embed ~200 of the 43 cells) ----
spot = buf_ids[:0]  # unused
det_ids = [f"{int(x)}" for x in d43_cells[:200]]
with h5py.File(H5, "r") as fh:
    Cdet = fh["X"][np.sort(d43_cells[:200]), :].astype(np.float32)
e1, _ = embed(Cdet, det_ids, "det1"); e2, _ = embed(Cdet, det_ids, "det2")
det_max = max(float(np.abs(e1[c] - e2[c]).max()) for c in e1)
print(f"[B] determinism max|Δ| = {det_max:.3e}", flush=True)

# ---- checkpoint reload check ----
acc2 = XCellAccumulator(len(var_ensembl), checkpoint_dir=SSD + "/checkpoints/rpe1_xcell")
ckpt_ok = len(acc2.trace()["chunks_consumed"]) == len(chunks)

# ---- save per-cell embeddings for the 43 + index ----
e43_ids = sorted(emb43); E43 = np.stack([emb43[c] for c in e43_ids]).astype(np.float32)
np.save(SSD + "/rpe1_topup/rpe1_43_percell_emb.npy", E43)
pd.DataFrame({"cell_id": e43_ids, "orig_obs_idx": [int(c) for c in e43_ids],
             "gene_transcript": [cell2gt[c] for c in e43_ids]}).to_parquet(SSD + "/rpe1_topup/rpe1_43_percell_index.parquet", index=False)
# save z43 + x_cell native/mapped for assembly
np.savez(SSD + "/scratch/z43.npz", ids=np.array(sorted(z43)), z=np.stack([z43[g] for g in sorted(z43)]).astype(np.float32))
np.savez(SSD + "/x_cell/x_cell_rpe1_native.npz", ids=np.array(sorted(man)),
         x=np.stack([xnat[g] for g in sorted(man)]).astype(np.float64))
np.savez(SSD + "/x_cell/x_cell_rpe1_mapped.npz", ids=np.array(sorted(man)),
         x=np.stack([x_mapped[g] for g in sorted(man)]).astype(np.float32))

out = {"pass_secs": pass_secs, "embed_secs": embed_secs, "achieved_cells_per_sec": achieved,
       "n_embedded_new": len(miss43), "n_cells_embedded_new": n_43, "flush_count": nflush, "flush_sizes": flush_sizes,
       "startup_sec": startup, "same_pass_verified": same_pass, "same_pass_max_abs_diff": sp_max,
       "embed_determinism_max_abs_diff": det_max, "compensated_summation_max_delta": comp_delta,
       "mapped_hvg_count": int(mapped_count), "missing_hvg_count": int(len(vcc_hvg) - mapped_count),
       "checkpoint_reloaded_ok": bool(ckpt_ok), "chunks": len(chunks),
       "acc_trace": {k: acc.trace()[k] for k in ["n_genes","n_constructs","nt_count","total_cells"]}}
json.dump(out, open(SSD + "/logs/phase2_embed_xcell.json", "w"), indent=2, default=str)
print("[DONE]", json.dumps({k: out[k] for k in ["achieved_cells_per_sec","flush_count","flush_sizes","same_pass_verified","embed_determinism_max_abs_diff","compensated_summation_max_delta","mapped_hvg_count","checkpoint_reloaded_ok"]}, default=str))
