"""Bounded unit tests for the x_cell accumulator (tiny synthetic arrays). Self-contained
runner (no pytest dependency): `python test_x_cell_accumulator.py` -> per-test PASS/FAIL
and a nonzero exit code on any failure."""
import os, sys, tempfile, shutil, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from x_cell_accumulator import XCellAccumulator, XCellError, cp10k_log1p_rows, map_and_clip_to_vcc_hvg

RNG = np.random.RandomState(0)

def _synth(n, g):
    """n cells x g genes of small non-negative integer counts (deterministic)."""
    return RNG.randint(0, 8, size=(n, g)).astype(np.float64)

def _direct(counts, cids, ntmask):
    rows = cp10k_log1p_rows(counts)
    mu_nt = rows[ntmask].mean(axis=0)
    out = {}
    for c in np.unique(np.asarray(cids)[~ntmask].astype(str)):
        sel = (np.asarray(cids).astype(str) == c) & (~ntmask)
        out[c] = rows[sel].mean(axis=0) - mu_nt
    return out, mu_nt

def _run_partition(counts, cids, ntmask, bounds, g, ckpt=None):
    acc = XCellAccumulator(g, checkpoint_dir=ckpt)
    for ci, (a, b) in enumerate(bounds):
        acc.update(counts[a:b], np.asarray(cids)[a:b], ntmask[a:b], chunk_id=ci)
    return acc

# --------------------------------------------------------------------------- #
def test_chunk_boundary_invariance():
    g = 6; n = 12
    counts = _synth(n, g)
    cids = np.array(["A","A","A","B","B","C","C","C","C","NT","NT","NT"], dtype=object)
    ntmask = np.array([c == "NT" for c in cids])
    accA = _run_partition(counts, cids, ntmask, [(0, n)], g)                 # one chunk
    accB = _run_partition(counts, cids, ntmask, [(0,3),(3,7),(7,10),(10,12)], g)  # splits B & C
    fa = accA.finalize()["x_cell"]; fb = accB.finalize()["x_cell"]
    m = max(float(np.abs(fa[k] - fb[k]).max()) for k in fa)
    assert m <= 1e-10, f"chunk-boundary max|Δ|={m}"

def test_direct_vs_streaming():
    g = 5; n = 20
    counts = _synth(n, g)
    cids = np.array((["A"]*6 + ["B"]*6 + ["NT"]*8), dtype=object)
    ntmask = np.array([c == "NT" for c in cids])
    acc = _run_partition(counts, cids, ntmask, [(0,7),(7,13),(13,20)], g)
    got = acc.finalize()["x_cell"]
    exp, _ = _direct(counts, cids, ntmask)
    m = max(float(np.abs(got[k] - exp[k]).max()) for k in exp)
    assert m <= 1e-10, f"direct-vs-streaming max|Δ|={m}"

def test_nt_accumulation():
    g = 4; n = 10
    counts = _synth(n, g)
    cids = np.array((["A"]*5 + ["NT"]*5), dtype=object)
    ntmask = np.array([c == "NT" for c in cids])
    acc = _run_partition(counts, cids, ntmask, [(0,4),(4,10)], g)
    mu_nt = acc.finalize()["mu_nt"]
    exp = cp10k_log1p_rows(counts[5:]).mean(axis=0)
    assert np.abs(mu_nt - exp).max() <= 1e-10
    assert acc.finalize()["nt_count"] == 5

def test_construct_counts():
    g = 3
    counts = _synth(9, g)
    cids = np.array((["A"]*2 + ["B"]*3 + ["NT"]*4), dtype=object)
    ntmask = np.array([c == "NT" for c in cids])
    acc = _run_partition(counts, cids, ntmask, [(0,5),(5,9)], g)
    cnt = acc.finalize()["counts"]
    assert cnt == {"A": 2, "B": 3}, cnt

def test_zero_total_cell_handling():
    g = 4
    counts = _synth(6, g)
    counts[0] = 0.0            # an all-zero-count cell (total==0 guard)
    cids = np.array((["A"]*3 + ["NT"]*3), dtype=object)
    ntmask = np.array([c == "NT" for c in cids])
    acc = _run_partition(counts, cids, ntmask, [(0,6)], g)
    x = acc.finalize()["x_cell"]["A"]
    assert np.isfinite(x).all()          # zero-count cell -> log1p(0)=0 vector, no NaN/Inf
    # cell 0 contributes zeros; mean over A = (0 + row1 + row2)/3
    rows = cp10k_log1p_rows(counts[:3])
    assert np.abs(acc.finalize()["mu_construct"]["A"] - rows.mean(axis=0)).max() <= 1e-12

def test_inconsistent_gene_dim_fails():
    acc = XCellAccumulator(5)
    try:
        acc.update(_synth(3, 6), ["A","A","NT"], np.array([0,0,1],bool), chunk_id=0)
    except XCellError:
        return
    raise AssertionError("expected XCellError on gene-dim mismatch")

def test_negative_counts_fails():
    acc = XCellAccumulator(4)
    c = _synth(3, 4); c[1,1] = -1
    try:
        acc.update(c, ["A","A","NT"], np.array([0,0,1],bool), chunk_id=0)
    except XCellError:
        return
    raise AssertionError("expected XCellError on negative counts")

def test_missing_nt_fails():
    g = 4
    acc = XCellAccumulator(g)
    acc.update(_synth(3, g), ["A","A","B"], np.array([0,0,0],bool), chunk_id=0)  # no NT
    try:
        acc.finalize()
    except XCellError:
        return
    raise AssertionError("expected XCellError when finalize() called with no NT cells")

def test_resume_from_checkpoint():
    g = 5; n = 25
    counts = _synth(n, g)
    cids = np.array((["A"]*8 + ["B"]*7 + ["NT"]*10), dtype=object)
    ntmask = np.array([c == "NT" for c in cids])
    bounds = [(0,5),(5,10),(10,15),(15,20),(20,25)]
    # uninterrupted
    full = _run_partition(counts, cids, ntmask, bounds, g).finalize()["x_cell"]
    # interrupted: chunks 0-2, then a NEW instance resumes and does 3-4
    d = tempfile.mkdtemp()
    try:
        a1 = XCellAccumulator(g, checkpoint_dir=d)
        for ci in range(3):
            a, b = bounds[ci]; a1.update(counts[a:b], cids[a:b], ntmask[a:b], chunk_id=ci)
        del a1                                   # simulate crash
        a2 = XCellAccumulator(g, checkpoint_dir=d)   # resumes from checkpoint
        # replay ALL chunks; consumed set makes 0-2 no-ops (structural anti-double-count)
        for ci in range(5):
            a, b = bounds[ci]; a2.update(counts[a:b], cids[a:b], ntmask[a:b], chunk_id=ci)
        res = a2.finalize()["x_cell"]
        m = max(float(np.abs(full[k] - res[k]).max()) for k in full)
        assert m == 0.0, f"resume != uninterrupted, max|Δ|={m}"
        assert a2.trace()["total_cells"] == n     # no double-count
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_duplicate_construct_id_fails():
    acc = XCellAccumulator(3)
    cids = np.array([5, "5", "NT"], dtype=object)   # int 5 and str "5" collide under str()
    ntmask = np.array([0,0,1], bool)
    try:
        acc.update(_synth(3,3), cids, ntmask, chunk_id=0)
    except XCellError:
        return
    raise AssertionError("expected XCellError on construct-id collision under str()")

def test_checkpoint_path_is_parameterized():
    g = 4
    d1 = tempfile.mkdtemp(); d2 = tempfile.mkdtemp()
    try:
        a1 = XCellAccumulator(g, checkpoint_dir=d1)
        a2 = XCellAccumulator(g, checkpoint_dir=d2)
        c = _synth(4, g); cids = np.array(["A","A","NT","NT"],dtype=object); nm=np.array([0,0,1,1],bool)
        a1.update(c, cids, nm, chunk_id=0)
        a2.update(c, cids, nm, chunk_id=0)
        assert os.path.exists(os.path.join(d1, "xcell_state.pkl"))
        assert os.path.exists(os.path.join(d2, "xcell_state.pkl"))
        assert a1.checkpoint_dir == d1 and a2.checkpoint_dir == d2
        # no hard-coded path leaked: the two dirs are distinct and both written
        assert d1 != d2
    finally:
        shutil.rmtree(d1, ignore_errors=True); shutil.rmtree(d2, ignore_errors=True)


TESTS = [test_chunk_boundary_invariance, test_direct_vs_streaming, test_nt_accumulation,
         test_construct_counts, test_zero_total_cell_handling, test_inconsistent_gene_dim_fails,
         test_negative_counts_fails, test_missing_nt_fails, test_resume_from_checkpoint,
         test_duplicate_construct_id_fails, test_checkpoint_path_is_parameterized]

if __name__ == "__main__":
    npass = 0; fails = []
    for t in TESTS:
        try:
            t(); print(f"PASS {t.__name__}"); npass += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}"); traceback.print_exc(); fails.append(t.__name__)
    print(f"\n{npass}/{len(TESTS)} passed")
    sys.exit(1 if fails else 0)
