"""(e) C1–C5 capacity / alignment control generators — Lock §13.

Feature-generation seeds 20270101–20270105 are SEPARATE from the head-training seeds
{0,1,2}. Leakage guard: every stochastic control is generated from array SHAPES ONLY —
never from `y`, never from validation/test statistics (Lock §13, §29). None of these
functions accept a label argument; C2's projection is one fixed matrix drawn once and
applied identically everywhere.
"""
import numpy as np

# split_index: train=0 / val=1 / test=2 / RPE1=3 / K562=4 (Lock §13)
SPLIT_INDEX = {"train": 0, "val": 1, "test": 2, "RPE1": 3, "K562": 4}
C1_SEED = 20270101
C2_SEED = 20270102
C3_SEED = 20270103
C4_SEED = 20270104
C5_SEED = 20270105
Z_DIM = 768
X_DIM = 2000


def _split_index(split):
    if split not in SPLIT_INDEX:
        raise ValueError(f"unknown split {split!r}; expected one of {list(SPLIT_INDEX)}")
    return SPLIT_INDEX[split]


# ---- C1 — 768-D i.i.d. Gaussian capacity floor (GATING) -------------------- #
def c1_features(n_rows, split, dim=Z_DIM):
    """Per-split i.i.d. N(0,1), shape (n_rows, dim), from ROW SHAPE ONLY.
    Deterministic per-split stream: default_rng(20270101 + split_index)."""
    rng = np.random.default_rng(C1_SEED + _split_index(split))
    return rng.standard_normal((int(n_rows), int(dim))).astype(np.float32)


# ---- C2 — one fixed Gaussian random projection x -> 768 (SECONDARY) --------- #
def c2_projection_matrix(x_dim=X_DIM, out_dim=Z_DIM):
    """The single fixed projection R in R^{2000x768}, R_ij ~ N(0, 1/768), drawn ONCE
    with seed 20270102. Applied identically to every split and both external screens."""
    rng = np.random.default_rng(C2_SEED)
    return (rng.standard_normal((int(x_dim), int(out_dim))) / np.sqrt(out_dim)).astype(np.float32)


def c2_features(x, R):
    """Project expression x (n, 2000) through the fixed matrix R (2000, 768)."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[1] != R.shape[0]:
        raise ValueError(f"x dim {x.shape[1]} != R rows {R.shape[0]}")
    return (x @ R).astype(np.float32)


# ---- C3 — row-shuffled z (ALIGNMENT_CONTROL) ------------------------------- #
def c3_permutation(n_rows, split):
    """Per-split permutation, seed 20270103 + split_index; the SAME permutation is used
    at train and eval for that split (destroys row identity, preserves z distribution)."""
    rng = np.random.default_rng(C3_SEED + _split_index(split))
    return rng.permutation(int(n_rows))


def c3_features(z, split):
    z = np.asarray(z)
    return z[c3_permutation(z.shape[0], split)]


# ---- C4 — [z ; row-shuffled m̃] (ALIGNMENT_CONTROL) ------------------------- #
def c4_permutation(n_rows, split):
    rng = np.random.default_rng(C4_SEED + _split_index(split))
    return rng.permutation(int(n_rows))


def c4_features(z, m_tilde, split):
    z = np.asarray(z, dtype=np.float32)
    m = np.asarray(m_tilde, dtype=np.float32)
    if z.shape[0] != m.shape[0]:
        raise ValueError("z and m̃ row counts differ")
    return np.concatenate([z, m[c4_permutation(m.shape[0], split)]], axis=1).astype(np.float32)


# ---- C5 — [z ; 4 i.i.d. Gaussian scalars] (DIMENSION_INCREMENT_CONTROL) ----- #
def c5_features(z, split):
    """[z ; 4 i.i.d. N(0,1)], per-row, seed 20270105 + split_index (from row shape only)."""
    z = np.asarray(z, dtype=np.float32)
    rng = np.random.default_rng(C5_SEED + _split_index(split))
    g = rng.standard_normal((z.shape[0], 4)).astype(np.float32)
    return np.concatenate([z, g], axis=1).astype(np.float32)
