"""(a) HVG guard — Lock §4.

Every ICLR code path that loads HVGs must go through this guard. It resolves the sole
authoritative artifact `cache/vcc/variants/delta_hvg/selected_genes.npy` and RAISES LOUDLY if
any resolved path basename is `hvg_genes.npy`, or if `load_vcc_pseudobulk` is invoked with the
non-authoritative `(feature_transform="raw", gene_selection="hvg")` combination (the default in
`run_vcc.py` that makes `hvg_genes.npy` reachable in principle — Preflight §F).

Preflight §F established no ICLR-planned path currently reaches `hvg_genes.npy`; this guard makes
the invariant enforced rather than incidental.
"""
import os as _o, sys as _s
_r = _o.path.dirname(_o.path.abspath(__file__))
while _r != _o.path.dirname(_r) and not _o.path.exists(_o.path.join(_r, "iga_paths.py")):
    _r = _o.path.dirname(_r)
if _r not in _s.path:
    _s.path.insert(0, _r)
import iga_paths
import os
import numpy as np

VCC_CACHE = iga_paths.IGA_NEURIPS + "/cache/vcc"
AUTHORITATIVE_SELECTED_GENES = VCC_CACHE + "/variants/delta_hvg/selected_genes.npy"
VAR_GENE_NAMES = VCC_CACHE + "/var_gene_names.npy"
FORBIDDEN_BASENAME = "hvg_genes.npy"
FORBIDDEN_PATH = VCC_CACHE + "/hvg_genes.npy"


class HVGGuardError(RuntimeError):
    """Raised when a non-authoritative HVG artifact / feature combination is reached."""


def assert_not_forbidden_path(path):
    """Raise loudly if `path`'s basename is the non-authoritative hvg_genes.npy."""
    if os.path.basename(str(path)) == FORBIDDEN_BASENAME:
        raise HVGGuardError(
            f"BLOCKED: attempt to resolve non-authoritative HVG artifact {path!r} "
            f"(basename {FORBIDDEN_BASENAME}). Lock §4: only {AUTHORITATIVE_SELECTED_GENES} is "
            f"authoritative (the two 2,000-HVG sets differ by 167 genes)."
        )
    return path


def guard_load_vcc_pseudobulk_args(feature_transform, gene_selection):
    """Raise loudly on the (raw, hvg) combination that resolves hvg_genes.npy (run_vcc.py default)."""
    if str(feature_transform) == "raw" and str(gene_selection) == "hvg":
        raise HVGGuardError(
            "BLOCKED: load_vcc_pseudobulk(feature_transform='raw', gene_selection='hvg') resolves "
            "cache/vcc/hvg_genes.npy (non-authoritative). Lock §4: ICLR paths must use "
            "feature_transform='delta', gene_selection='hvg' -> variants/delta_hvg/selected_genes.npy."
        )
    return feature_transform, gene_selection


def resolve_selected_genes_path():
    """Return the authoritative selected_genes.npy path, guarded."""
    return assert_not_forbidden_path(AUTHORITATIVE_SELECTED_GENES)


def load_selected_gene_indices():
    """Load the authoritative 2,000 HVG indices (int) into var_gene_names, via the guard."""
    return np.load(resolve_selected_genes_path())


def load_hvg_symbols():
    """Return the authoritative 2,000 HVG gene symbols (var_gene_names[selected_genes]), guarded."""
    sel = load_selected_gene_indices()
    vgn = np.load(VAR_GENE_NAMES, allow_pickle=True).astype(str)
    return vgn[sel]


# --------------------------------------------------------------------------- #
# PROVENANCE DIAGNOSTIC — the ONE permitted read of hvg_genes.npy.
# Isolated: not reachable from any feature path; returns an integer COUNT only;
# never returns, caches, or hands an array to anything. (Lock §4 diagnostic carve-out;
# Reconciliation §2.3(b).)
# --------------------------------------------------------------------------- #
def _provenance_only_hvg_count(screen_var_symbols, artifact):
    """PROVENANCE-ONLY. Return the integer count of HVG symbols (from `artifact`) that map into
    `screen_var_symbols`. `artifact` in {'delta_hvg', 'hvg_genes'}. Returns an int and nothing else.

    This is the sole location permitted to read hvg_genes.npy, to settle Reconciliation §2.3(b):
    which 2,000-HVG artifact the manuscript's "1,430 / 2,000 (71.5%)" figure was computed against.
    """
    vgn = np.load(VAR_GENE_NAMES, allow_pickle=True).astype(str)
    if artifact == "delta_hvg":
        idx = np.load(AUTHORITATIVE_SELECTED_GENES)
    elif artifact == "hvg_genes":
        idx = np.load(FORBIDDEN_PATH)  # permitted ONLY here, for provenance
    else:
        raise HVGGuardError(f"unknown artifact {artifact!r}")
    hvg_syms = set(vgn[idx].astype(str))
    screen = set(np.asarray(screen_var_symbols).astype(str).tolist())
    return int(len(hvg_syms & screen))  # integer count ONLY
