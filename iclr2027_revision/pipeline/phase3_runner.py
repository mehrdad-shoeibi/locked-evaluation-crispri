"""Phase 3 — K562-essential extraction runner (production, detached, resumable).

Representation production ONLY. No label/endpoint/outcome/R2/Spearman/bootstrap/model-fit.
Single GPU. Transactional per-work-unit design (C1b/C1c):
  read raw counts -> embed -> validate -> atomic-rename embedding -> apply x_cell ONCE
  -> add unit id ONCE -> checkpoint (cadence). A unit is DURABLY COMMITTED only when its
  id is in the accumulator checkpoint's `consumed`. Embedding artifacts of un-committed
  units are overwritten (not skipped, not duplicated) on rerun.

z aggregation: FLOAT32 (matches MADA/VCC/RPE1 + Lock fp32). x_cell accumulation: FLOAT64.
"""
import os as _o, sys as _s
_r = _o.path.dirname(_o.path.abspath(__file__))
while _r != _o.path.dirname(_r) and not _o.path.exists(_o.path.join(_r, "iga_paths.py")):
    _r = _o.path.dirname(_r)
if _r not in _s.path:
    _s.path.insert(0, _r)
import iga_paths
import os, sys, json, time, pickle, shutil, argparse, resource
import numpy as np, pandas as pd, h5py, anndata as ad, scipy.sparse as sp

REPO = iga_paths.MADA + "/cache/foundation_models/geneformer-repo"
MODEL_DIR = REPO + "/Geneformer-V2-104M"
TOKDICT = REPO + "/geneformer/token_dictionary_gc104M.pkl"
H5 = iga_paths.IGA_NEURIPS + "/data/replogle/raw_singlecell/K562_essential_raw_singlecell_01.h5ad"
ELIG = iga_paths.IGA_NEURIPS + "/results_v4/k562_essential_coverage_gate_v1/k562_essential_eligibility.csv"
HARM = iga_paths.IGA_NEURIPS + "/results_v4/arm2_pretest_lock_v1/evaluation_constructs.csv"
SG = iga_paths.IGA_NEURIPS + "/cache/vcc/variants/delta_hvg/selected_genes.npy"
VGN = iga_paths.IGA_NEURIPS + "/cache/vcc/var_gene_names.npy"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)                                        # geneformer (read-only import)
from x_cell_accumulator import XCellAccumulator, cp10k_log1p_rows, map_and_clip_to_vcc_hvg


def log(logf, msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(logf, "a") as f:
        f.write(line + "\n")


def decode_obs(f, col):
    og = f["obs"]; node = og[col]
    if isinstance(node, h5py.Group):
        cats = node["categories"][:]; codes = node["codes"][:]
    else:
        codes = node[:]
        cats = og["__categories"][col][:] if ("__categories" in og and col in og["__categories"]) else None
        if cats is None:
            return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in codes])
    cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
    return cats[codes]


def build_units(idx_sorted, cid_arr, nt_arr, unit_size):
    """Slice the single deterministic global ordering into work units. Constructs may span
    units. Within-unit order = the global (sorted-orig_obs_idx) order (deterministic)."""
    units = []
    for u, a in enumerate(range(0, len(idx_sorted), unit_size)):
        b = min(a + unit_size, len(idx_sorted))
        units.append({"unit_id": u, "idx": idx_sorted[a:b], "cid": cid_arr[a:b], "nt": nt_arr[a:b]})
    return units


def ckpt_paths(d):
    return d + "/xcell_state.pkl", d + "/xcell_state.prev.pkl", d + "/xcell_state.tmp"


def checkpoint_atomic(acc, ckpt_dir, crash_mode=None):
    cur, prev, tmp = ckpt_paths(ckpt_dir)
    t0 = time.time()
    with open(tmp, "wb") as f:
        pickle.dump(acc.get_state(), f, pickle.HIGHEST_PROTOCOL); f.flush(); os.fsync(f.fileno())
    if crash_mode == "during_checkpoint":
        os._exit(137)                      # crash after tmp written, before atomic install
    if os.path.exists(cur):
        shutil.copy2(cur, prev)            # retain previous known-good
    os.replace(tmp, cur)                   # atomic install
    s2 = pickle.load(open(cur, "rb"))      # reopen-validate
    assert int(s2["n_genes"]) == acc.n_genes and len(s2["consumed"]) == len(acc.consumed()), "ckpt reopen-validate failed"
    return time.time() - t0


def load_durable(acc, ckpt_dir, logf):
    cur, prev, _ = ckpt_paths(ckpt_dir)
    for p in (cur, prev):
        if os.path.exists(p):
            try:
                s = pickle.load(open(p, "rb")); acc.set_state(s)
                log(logf, f"[resume] loaded durable checkpoint {os.path.basename(p)}: "
                          f"{len(acc.consumed())} units, {acc.get_state()['total_cells']} cells")
                return acc.consumed()
            except Exception as e:
                log(logf, f"[resume] {os.path.basename(p)} unreadable ({e}); trying previous")
    log(logf, "[resume] no durable checkpoint; fresh start")
    return set()


def embed_unit(counts, cell_ids, model, layer, pad_id, tgd, var_ensembl, work):
    from geneformer import TranscriptomeTokenizer, perturber_utils as pu
    from geneformer.emb_extractor import get_embs
    import torch
    for dd in (work + "/in", work + "/tok"):
        shutil.rmtree(dd, ignore_errors=True); os.makedirs(dd, exist_ok=True)
    Xs = sp.csr_matrix(np.asarray(counts, dtype=np.float32))
    a = ad.AnnData(X=Xs, obs=pd.DataFrame({"cell_id": [str(c) for c in cell_ids]}))
    a.var_names = list(var_ensembl); a.var["ensembl_id"] = list(var_ensembl)
    a.obs["n_counts"] = np.asarray(Xs.sum(1)).ravel()
    a.obs_names = a.obs["cell_id"].astype(str); a.obs_names_make_unique()
    a.write_h5ad(work + "/in/u.h5ad")
    tk = TranscriptomeTokenizer(custom_attr_name_dict={"cell_id": "cell_id"}, nproc=1,
                                chunk_size=512, model_input_size=4096, model_version="V2")
    tk.tokenize_data(work + "/in", work + "/tok", "u", file_format="h5ad")
    ds = pu.load_and_filter(None, 1, work + "/tok/u.dataset")
    ds = pu.downsample_and_sort(ds, len(cell_ids) + 10)
    embs = get_embs(model, ds, "cls", layer, pad_id, 16, tgd, silent=True)
    torch.cuda.synchronize()
    arr = embs.cpu().float().numpy().astype(np.float32)
    ids = [str(x) for x in ds["cell_id"]]
    shutil.rmtree(work, ignore_errors=True)          # C5: delete transient per unit
    return {ids[k]: arr[k] for k in range(len(ids))}


def free_gb(path):
    s = os.statvfs(path); return s.f_bavail * s.f_frsize / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry", "prod"], required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--unit-size", type=int, default=1200)
    ap.add_argument("--subset-n", type=int, default=0)          # dry: first N intended cells
    ap.add_argument("--ckpt-cells", type=int, default=6000)
    ap.add_argument("--ckpt-minutes", type=float, default=15.0)
    ap.add_argument("--watchdog-hours", type=float, default=20.0)
    ap.add_argument("--disk-guard-gb", type=float, default=20.0)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--crash-after-unit", type=int, default=-1)
    ap.add_argument("--crash-mode", choices=["none", "after_embed_before_ckpt", "during_checkpoint"], default="none")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = args.out_root + "/hf_cache"
    OUT = args.out_root
    EMB = OUT + "/emb_units"; CK = OUT + "/checkpoints"; LOGD = OUT + "/logs"; SCR = OUT + "/scratch"
    for d in (EMB, CK, LOGD, SCR, OUT + "/z", OUT + "/x_cell", OUT + "/manifests"):
        os.makedirs(d, exist_ok=True)
    logf = LOGD + "/runner.log"
    t_start = time.time()
    log(logf, f"[start] mode={args.mode} unit_size={args.unit_size} subset_n={args.subset_n} "
              f"ckpt_cells={args.ckpt_cells} crash={args.crash_mode}@{args.crash_after_unit} gpu={args.gpu}")

    # ---- Part A assertions ----
    el = pd.read_csv(ELIG); elig = set(el.loc[el.eligible == True, "construct"].astype(str))
    harm = set(pd.read_csv(HARM)["construct"].astype(str))
    assert len(elig) == 1903, len(elig)
    assert elig == harm, f"eligible != harmonized (symdiff {len(elig ^ harm)})"
    f = h5py.File(H5, "r")
    gt = decode_obs(f, "gene_transcript"); gene = decode_obs(f, "gene")
    var_ensembl = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["var"]["gene_id"][:]])
    var_symbol = decode_obs_var(f, "gene_name")
    ngenes = f["X"].shape[1]; f.close()
    is_nt = gene == "non-targeting"; is_elig = np.isin(gt, list(elig))
    intended = np.where(is_elig | is_nt)[0].astype(np.int64)
    if args.mode == "prod":
        assert int((is_elig & ~is_nt).sum()) == 273279 and int(is_nt.sum()) == 10691 and len(intended) == 283970, \
            (int(is_elig.sum()), int(is_nt.sum()), len(intended))
    intended = np.sort(intended)
    if args.mode == "dry" and args.subset_n > 0:
        # deterministic subset guaranteed to contain NT cells (finalize needs mu_NT)
        pidx = np.where(is_elig & ~is_nt)[0]; nidx = np.where(is_nt)[0]
        npt = args.subset_n * 2 // 3; nnt = args.subset_n - npt
        intended = np.sort(np.concatenate([pidx[:npt], nidx[:nnt]]))
    cid = np.where(is_nt[intended], "__NT_POOL__", gt[intended]).astype(object)
    ntm = is_nt[intended]
    units = build_units(intended, cid, ntm, args.unit_size)
    # C1c partition assertions
    allcells = np.concatenate([u["idx"] for u in units])
    assert len(allcells) == len(intended) and len(np.unique(allcells)) == len(intended), "unit partition broken"
    assert np.array_equal(np.sort(allcells), np.sort(intended)), "union(units) != intended"
    n_units = len(units)
    log(logf, f"[units] {n_units} work units over {len(intended)} cells (unit_size {args.unit_size})")

    # ---- model + token dict (once) ----
    import torch
    from geneformer import perturber_utils as pu
    td = pickle.load(open(TOKDICT, "rb")); tgd = {v: k for k, v in td.items()}; pad_id = td["<pad>"]
    model = pu.load_model("Pretrained", 0, MODEL_DIR, "eval"); layer = pu.quant_layers(model) + (-1)
    log(logf, "[model] loaded")

    # ---- resume ----
    acc = XCellAccumulator(ngenes, checkpoint_dir=None)
    consumed = load_durable(acc, CK, logf)
    ck_write_secs = []
    cells_since_ck = 0; last_ck = time.time(); done_cells = acc.get_state()["total_cells"]
    inst_prev = time.time()

    with h5py.File(H5, "r") as fh:
        Xds = fh["X"]
        for u in units:
            if u["unit_id"] in consumed:
                continue
            for attempt in range(3):
                try:
                    cixs = np.sort(u["idx"])
                    counts = Xds[cixs, :].astype(np.float32)
                    cell_ids = [str(int(x)) for x in cixs]
                    cid2emb = embed_unit(counts, cell_ids, model, layer, pad_id, tgd, var_ensembl, SCR + f"/u{u['unit_id']}")
                    # validate embedding artifact BEFORE committing
                    ids = sorted(cid2emb)
                    arr = np.stack([cid2emb[c] for c in ids]).astype(np.float32)
                    assert arr.shape == (len(cixs), 768) and arr.dtype == np.float32 and np.isfinite(arr).all()
                    assert set(ids) == set(cell_ids), "embedding cell-id set != unit set"
                    # step 4-6: temp -> validate -> atomic rename
                    tmpn = EMB + f"/unit_{u['unit_id']:05d}.tmp.npy"; finn = EMB + f"/unit_{u['unit_id']:05d}.npy"
                    np.save(tmpn, arr)
                    ix = pd.DataFrame({"cell_id": ids, "orig_obs_idx": [int(c) for c in ids],
                                       "construct": [("__NT_POOL__" if gene[int(c)] == "non-targeting" else gt[int(c)]) for c in ids]})
                    ix.to_parquet(EMB + f"/unit_{u['unit_id']:05d}.parquet", index=False)
                    _ = np.load(tmpn)  # reopen-validate temp
                    os.replace(tmpn, finn)
                    if args.crash_mode == "after_embed_before_ckpt" and u["unit_id"] == args.crash_after_unit:
                        log(logf, f"[CRASH] after_embed_before_ckpt at unit {u['unit_id']}"); os._exit(137)
                    # step 7-8: apply x_cell + consume ONCE
                    acc.update(counts, u["cid"], u["nt"], chunk_id=u["unit_id"])
                    break
                except Exception as e:
                    log(logf, f"[retry] unit {u['unit_id']} attempt {attempt+1}/3 failed: {type(e).__name__}: {e}")
                    if attempt == 2:
                        checkpoint_atomic(acc, CK); log(logf, f"[FATAL] unit {u['unit_id']} failed 3x; exiting non-zero")
                        sys.exit(3)
                    time.sleep(2 ** attempt)
            done_cells += len(cixs); cells_since_ck += len(cixs)
            now = time.time(); inst = len(cixs) / max(1e-9, now - inst_prev); inst_prev = now
            cum = done_cells / max(1e-9, now - t_start)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            wrote_ck = ""
            if cells_since_ck >= args.ckpt_cells or (now - last_ck) >= args.ckpt_minutes * 60 or u["unit_id"] == n_units - 1:
                ws = checkpoint_atomic(acc, CK, crash_mode=(args.crash_mode if (args.crash_mode == "during_checkpoint" and u["unit_id"] == args.crash_after_unit) else None))
                ck_write_secs.append(ws); cells_since_ck = 0; last_ck = now
                wrote_ck = f" ckpt={ws:.2f}s"
                # retention: keep only cur + prev (handled by checkpoint_atomic overwriting prev)
            eta_h = (len(intended) - done_cells) / max(1e-9, cum) / 3600
            log(logf, f"[unit {u['unit_id']+1}/{n_units}] cells={len(cixs)} cum={done_cells} "
                      f"inst={inst:.2f} cum={cum:.2f} c/s eta={eta_h:.2f}h rss={rss:.1f}GB{wrote_ck}")
            if (now - t_start) > args.watchdog_hours * 3600:
                checkpoint_atomic(acc, CK); log(logf, "[WATCHDOG] exceeded max hours; checkpointed and exiting"); sys.exit(4)
            if u["unit_id"] % 20 == 0 and free_gb(OUT) < args.disk_guard_gb:
                checkpoint_atomic(acc, CK); log(logf, f"[DISKGUARD] free<{args.disk_guard_gb}GB; checkpointed and exiting"); sys.exit(5)

    # ---- force final checkpoint ----
    ws = checkpoint_atomic(acc, CK); ck_write_secs.append(ws)
    log(logf, f"[final-checkpoint] {ws:.2f}s; consumed={len(acc.consumed())}/{n_units}")

    # ---- finalize (C1b assertions + aggregation) ----
    assert acc.consumed() == set(range(n_units)), "consumed != all units"
    finalize(OUT, EMB, acc, intended, gt, ntm, var_symbol, args, ck_write_secs, n_units, t_start, logf)
    log(logf, "[DONE] phase 3 finalize complete")


def is_nt_lookup(orig, intended, ntm):
    # small helper: is this orig_obs_idx an NT cell (from the intended/ntm alignment)
    pos = np.searchsorted(intended, orig)
    return bool(ntm[pos])


def decode_obs_var(f, col):
    vg = f["var"]; node = vg[col]
    if isinstance(node, h5py.Group):
        cats = node["categories"][:]; codes = node["codes"][:]
    else:
        codes = node[:]
        cats = vg["__categories"][col][:] if ("__categories" in vg and col in vg["__categories"]) else None
        if cats is None:
            return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in codes])
    cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
    return cats[codes]


def finalize(OUT, EMB, acc, intended, gt, ntm, var_symbol, args, ck_write_secs, n_units, t_start, logf):
    import glob
    # C1b finalize assertions
    unit_files = sorted(glob.glob(EMB + "/unit_*.npy"))
    assert len(unit_files) == n_units, (len(unit_files), n_units)
    all_ids = []; emb_list = []; constructs = []
    for uf in unit_files:
        arr = np.load(uf); ix = pd.read_parquet(uf.replace(".npy", ".parquet"))
        assert arr.shape[0] == len(ix)
        emb_list.append(arr.astype(np.float32)); all_ids.extend(ix.cell_id.astype(str).tolist())
        constructs.extend(ix.construct.astype(str).tolist())
    emb = np.concatenate(emb_list).astype(np.float32)
    all_ids = np.array(all_ids); constructs = np.array(constructs)
    assert len(all_ids) == len(intended) and len(np.unique(all_ids)) == len(intended), "dup/missing cell ids"
    # z: mu_NT (float32) + per-construct mean (float32) - mu_NT  (groupby indices)
    ntmask = constructs == "__NT_POOL__"
    assert int(ntmask.sum()) == 10691 or args.mode == "dry"
    groups = pd.DataFrame({"c": constructs}).groupby("c").indices     # dict construct -> row indices
    mu_nt_emb = emb[groups["__NT_POOL__"]].mean(axis=0)               # float32
    man = sorted(c for c in groups if c != "__NT_POOL__")
    zK = np.stack([emb[groups[c]].mean(axis=0) - mu_nt_emb for c in man]).astype(np.float32)
    # x_cell finalize (float64) -> map -> clip
    fin = acc.finalize(); xnat = fin["x_cell"]
    vgn = np.load(VGN, allow_pickle=True).astype(str); sel = np.load(SG); vcc_hvg = vgn[sel]
    xmap = []; nmap = None
    for c in man:
        mm, nmap = map_and_clip_to_vcc_hvg(xnat[c], var_symbol, vcc_hvg); xmap.append(mm)
    xK = np.stack(xmap).astype(np.float32)
    xnat_arr = np.stack([xnat[c] for c in man]).astype(np.float64)
    # Part F same-pass verify (bounded reread of 5 constructs)
    five = man[:5]
    with h5py.File(H5, "r") as fh:
        sp_max = 0.0
        for c in five:
            cells = np.sort(intended[(gt[intended] == c)])
            cc = fh["X"][cells, :].astype(np.float32)
            direct = cp10k_log1p_rows(cc).mean(axis=0) - fin["mu_nt"]
            sp_max = max(sp_max, float(np.abs(xnat[c] - direct).max()))
    # dtype asserts (L1)
    assert zK.dtype == np.float32 and xnat_arr.dtype == np.float64 and xK.dtype == np.float32
    # persist with Phase-2-matched dtypes
    np.save(OUT + "/z/z_k562_1903.npy", zK)                                   # float32
    np.save(OUT + "/x_cell/x_cell_k562_1903_mapped.npy", xK)                  # float32
    np.savez(OUT + "/x_cell/x_cell_k562_native.npz", ids=np.array(man), x=xnat_arr)   # float64
    pd.DataFrame({"construct": man, "row": range(len(man))}).to_parquet(OUT + "/z/k562_construct_index.parquet", index=False)
    counts = acc.per_construct_counts()
    cc_arr = np.array([counts[c] for c in man])
    manifest = {
        "n_units": n_units, "total_cells": int(len(intended)), "consumed_units": len(acc.consumed()),
        "z_shape": list(zK.shape), "x_cell_shape": list(xK.shape),
        "Z_AGGREGATION_DTYPE": "float32", "X_CELL_ACCUMULATION_DTYPE": "float64",
        "Z_PERSISTED_DTYPE": str(zK.dtype), "X_CELL_PERSISTED_DTYPE": str(xK.dtype),
        "X_CELL_NATIVE_PERSISTED_DTYPE": str(xnat_arr.dtype), "PERCELL_EMB_PERSISTED_DTYPE": str(emb.dtype),
        "mapped_hvg_count": int(nmap), "missing_hvg_count": int(len(vcc_hvg) - nmap),
        "nt_cell_count": int(ntmask.sum()), "same_pass_max_abs_diff": sp_max,
        "compensated_summation_max_delta": acc.compensated_max_delta(),
        "per_construct_count_min_median_max": [int(cc_arr.min()), int(np.median(cc_arr)), int(cc_arr.max())],
        "all_finite_z": bool(np.isfinite(zK).all()), "all_finite_x": bool(np.isfinite(xK).all()),
        "x_within_clip": bool((np.abs(xK) <= 10).all()),
        "z_all_zero_rows": int((np.abs(zK).sum(1) == 0).sum()), "x_all_zero_rows": int((np.abs(xK).sum(1) == 0).sum()),
        "ckpt_write_seconds_min_median_max": [float(np.min(ck_write_secs)), float(np.median(ck_write_secs)), float(np.max(ck_write_secs))],
        "elapsed_hours": (time.time() - t_start) / 3600,
    }
    json.dump(manifest, open(OUT + "/manifests/phase3_k562_manifest.json", "w"), indent=2, default=str)
    log(logf, "[finalize] " + json.dumps({k: manifest[k] for k in ["z_shape","x_cell_shape","Z_PERSISTED_DTYPE","X_CELL_PERSISTED_DTYPE","X_CELL_NATIVE_PERSISTED_DTYPE","mapped_hvg_count","same_pass_max_abs_diff","compensated_summation_max_delta"]}, default=str))


if __name__ == "__main__":
    main()
