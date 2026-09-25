"""Phase-0 component unit tests (a)-(g). CPU only; no model fit, no metric, no test-y statistic."""
import os, sys, inspect
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import hvg_guard as G
import fm_preprocess as PP
import controls as C
import fm_heads as H
import bootstrap_wiring as BW
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 3)))
sys.path.insert(0, _ROOT)
from src.training.magnitude_augment import _row_magnitude_features, fit_magnitude_stats


# ---- (a) HVG guard ---- #
def test_guard_raises_on_hvg_genes():
    for bad in [G.FORBIDDEN_PATH, "/x/y/hvg_genes.npy"]:
        try:
            G.assert_not_forbidden_path(bad); assert False
        except G.HVGGuardError:
            pass

def test_guard_raises_on_raw_hvg():
    try:
        G.guard_load_vcc_pseudobulk_args("raw", "hvg"); assert False
    except G.HVGGuardError:
        pass
    G.guard_load_vcc_pseudobulk_args("delta", "hvg")  # allowed

def test_guard_allows_authoritative():
    assert G.resolve_selected_genes_path().endswith("delta_hvg/selected_genes.npy")

def test_provenance_returns_int():
    v = G._provenance_only_hvg_count(np.array(["NOC2L", "ISG15"]), "delta_hvg")
    assert isinstance(v, int)


# ---- (b) m̃ raw L2 (M3) + train-only zscore ---- #
def test_row_magnitude_raw_l2_not_sqrt_d():
    X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)   # ||x||2 = sqrt(14)
    M = _row_magnitude_features(X)
    assert abs(float(M[0, 1]) - np.sqrt(14)) < 1e-5      # raw L2, NOT sqrt(14)/sqrt(3)

def test_train_zscore_on_train_is_unit():
    rng = np.random.default_rng(0)
    class S:  # shard shim
        def __init__(s, x): s.x = x
    X = rng.standard_normal((500, 2000)).astype(np.float32)
    stats = fit_magnitude_stats([S(X)])
    Mz = stats.transform(_row_magnitude_features(X))
    assert np.abs(Mz.mean(0)).max() < 1e-4 and np.abs(Mz.std(0) - 1).max() < 1e-3


# ---- (c) bridge assertions ---- #
def test_bridge_assert_logic():
    import z_label_bridge as BR
    good = {"matched": 11764, "z_orphans": 0, "dup_keys_z": 0, "dup_keys_labels": 0,
            "splits": {"training": 5740, "validation": 2052, "test": 3972},
            "z_all_finite": True, "label_alignment_max_abs_diff": 0.0}
    assert BR.assert_bridge(good)
    bad = dict(good); bad["z_orphans"] = 1
    assert not BR.assert_bridge(bad)


# ---- (d) scaler ---- #
def test_scaler_train_unit_val_not():
    rng = np.random.default_rng(1)
    tr = rng.standard_normal((400, 32)) * 3 + 5
    va = rng.standard_normal((200, 32)) * 7 - 2
    sc = PP.fit_block_scaler(tr)
    zt = PP.apply_scaler(sc, tr); zv = PP.apply_scaler(sc, va)
    assert np.abs(zt.mean(0)).max() < 1e-5 and np.abs(zt.std(0) - 1).max() < 1e-4
    assert np.abs(zv.mean(0)).max() > 1e-2      # frozen stats -> val not centered
    assert isinstance(PP.near_constant_count(sc), int)


# ---- (e) controls reproducibility + properties + leakage ---- #
def test_controls_reproducible_bitwise():
    assert np.array_equal(C.c1_features(100, "train"), C.c1_features(100, "train"))
    z = np.random.default_rng(3).standard_normal((100, 768)).astype(np.float32)
    assert np.array_equal(C.c5_features(z, "val"), C.c5_features(z, "val"))
    assert np.array_equal(C.c3_permutation(100, "test"), C.c3_permutation(100, "test"))

def test_c2_matrix_identical_and_shape():
    R1 = C.c2_projection_matrix(); R2 = C.c2_projection_matrix()
    assert np.array_equal(R1, R2) and R1.shape == (2000, 768)

def test_c3_is_permutation():
    p = C.c3_permutation(313, "RPE1")
    assert np.array_equal(np.sort(p), np.arange(313))

def test_controls_no_label_argument():
    # leakage guard: no control generator accepts a `y`/label argument
    for fn in [C.c1_features, C.c2_features, C.c3_features, C.c4_features, C.c5_features,
               C.c2_projection_matrix, C.c3_permutation, C.c4_permutation]:
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"y", "labels", "target", "endpoint"})


# ---- (f) bootstrap wiring runs on synthetic ---- #
def test_vcc_bootstrap_synthetic():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(80)
    out = BW.vcc_paired_delta_r2(y, [rng.standard_normal(80) for _ in range(3)],
                                 [rng.standard_normal(80) for _ in range(3)], B=200)
    assert np.isfinite([out["ci_lo"], out["ci_hi"], out["median"]]).all() and out["n"] <= 200

def test_external_bootstrap_synthetic():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((150, 3)); y = rng.standard_normal(150); cl = rng.integers(0, 30, 150)
    rho_b, contrast = BW.external_paired_delta_rho(P, y, cl, {"A": [0], "B": [1], "C": [2]}, B=200, seed=20260901)
    assert rho_b.shape == (200, 3)
    ci = contrast("A", "B")
    assert np.isfinite([ci["ci_low"], ci["ci_high"]]).all()


# ---- (g) heads: SPSM width 256, classical params, no fit ---- #
def test_classical_factory_params():
    _, ridge, scale = H.make_ridge(0); assert ridge.alpha == 1.0 and scale
    _, rf, _ = H.make_rf(0); assert rf.n_estimators == 200 and rf.max_depth == 20 and rf.min_samples_leaf == 2 and rf.n_jobs == 8
    _, hgb, _ = H.make_hgb(0)
    assert hgb.max_iter == 500 and hgb.max_depth == 6 and hgb.learning_rate == 0.05 and hgb.l2_regularization == 0.0
    assert hgb.early_stopping and hgb.validation_fraction == 0.1 and hgb.n_iter_no_change == 20

def test_spsm_width_256():
    assert H.spsm_rep_width(H.build_spsm(768)) == 256
    assert H.spsm_rep_width(H.build_spsm(772)) == 256
    try:
        H.build_spsm(2004); assert False   # only 768/772 allowed (input-width-only)
    except ValueError:
        pass
