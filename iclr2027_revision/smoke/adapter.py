"""ICLR-side frozen-Geneformer extraction adapter (SMOKE ONLY).

Replicates the MADA per-cell embedding pipeline
(scripts/phasec_rpe1_full_feature_extraction_v1.py::run_geneformer) EXACTLY for the
tokenizer + EmbExtractor settings and AnnData preparation, but with EVERY write
redirected under iclr2027_revision/. MADA and MADA_data are read-only (model +
dictionaries are only READ). No training, no fine-tuning, no eval, no downloads.

Locked config (Lock section 5): Geneformer-V2-104M, model_type='Pretrained',
emb_mode='cls', cell_emb_style='mean_pool', emb_layer=-1, fp32, model_input_size=4096,
forward_batch_size=16, nproc=1, tokenizer chunk_size=512, max_ncells=n+10.
"""
import os as _o, sys as _s
_r = _o.path.dirname(_o.path.abspath(__file__))
while _r != _o.path.dirname(_r) and not _o.path.exists(_o.path.join(_r, "iga_paths.py")):
    _r = _o.path.dirname(_r)
if _r not in _s.path:
    _s.path.insert(0, _r)
import iga_paths
import os, sys, time, shutil
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

REPO = iga_paths.MADA + "/cache/foundation_models/geneformer-repo"   # READ-ONLY
MODEL_DIR = REPO + "/Geneformer-V2-104M"                                       # READ-ONLY


def embed_cells(X_counts, cell_ids, ensembl_ids, work_dir, prefix, extra_obs=None):
    """Embed raw-count cells -> per-cell frozen Geneformer CLS embeddings.

    X_counts   : (n, g) raw counts (dense ndarray or scipy sparse)
    cell_ids   : length-n list of unique str ids
    ensembl_ids: length-g list of ENSG ids for adata.var['ensembl_id']
    work_dir   : output dir (MUST be under iclr2027_revision/)
    Returns dict with: cid2emb {cell_id: (768,) float32}, emb_df, arr, tok_sec, emb_sec, n
    """
    work = Path(work_dir)
    in_dir = work / "input"; tok_dir = work / "tokenized"
    for dd in (in_dir, tok_dir):
        if dd.exists(): shutil.rmtree(dd)
        dd.mkdir(parents=True, exist_ok=True)

    Xs = sp.csr_matrix(X_counts.astype(np.float32)) if not sp.issparse(X_counts) else X_counts.astype(np.float32)
    obs = pd.DataFrame({"cell_id": [str(c) for c in cell_ids]})
    if extra_obs:
        for k, v in extra_obs.items(): obs[k] = list(v)
    adata = ad.AnnData(X=Xs, obs=obs)
    adata.var_names = [str(e) for e in ensembl_ids]
    adata.var["ensembl_id"] = [str(e) for e in ensembl_ids]
    adata.obs["n_counts"] = np.asarray(Xs.sum(axis=1)).ravel()
    adata.obs_names = adata.obs["cell_id"].astype(str); adata.obs_names_make_unique()
    adata.write_h5ad(str(in_dir / "chunk_input.h5ad"))

    sys.path.insert(0, REPO)
    from geneformer import TranscriptomeTokenizer, EmbExtractor
    attr = {"cell_id": "cell_id"}
    if extra_obs:
        for k in extra_obs: attr[k] = k

    tk = TranscriptomeTokenizer(custom_attr_name_dict=attr, nproc=1, chunk_size=512,
                                model_input_size=4096, model_version="V2")
    t0 = time.time()
    tk.tokenize_data(str(in_dir), str(tok_dir), prefix, file_format="h5ad")
    tok_sec = time.time() - t0
    tok_path = tok_dir / f"{prefix}.dataset"

    n = Xs.shape[0]
    EE = EmbExtractor(model_type="Pretrained", num_classes=0, emb_mode="cls",
                      cell_emb_style="mean_pool", max_ncells=n + 10, emb_layer=-1,
                      forward_batch_size=16, nproc=1, model_version="V2",
                      emb_label=list(attr.keys()))
    t1 = time.time()
    emb_df, emb_tensor = EE.extract_embs(model_directory=MODEL_DIR, input_data_file=str(tok_path),
                                         output_directory=str(tok_dir), output_prefix=f"{prefix}_emb",
                                         output_torch_embs=True)
    emb_sec = time.time() - t1
    arr = emb_tensor.cpu().float().numpy()
    emb_df = emb_df.reset_index(drop=True)
    cid2emb = {str(emb_df["cell_id"].iloc[k]): arr[k] for k in range(len(emb_df))}
    return dict(cid2emb=cid2emb, emb_df=emb_df, arr=arr, tok_sec=tok_sec, emb_sec=emb_sec, n=int(n))
