"""Single storage-root resolver for Project B (ICLR2027_2026-08-26).

All external storage (raw data, caches, models, derived outputs) is anchored under ONE
root, resolved in priority order:
  1. environment variable IGA_STORAGE_ROOT
  2. a file `storage_root.txt` sitting next to this module (repo root)
If neither is set, this FAILS LOUD. It never falls back to the old SSD, so a Project B
run can never silently read or write Project A's storage.
"""
import os


def _storage_root():
    r = os.environ.get("IGA_STORAGE_ROOT")
    if not r:
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage_root.txt")
        if os.path.exists(f):
            r = open(f).read().strip()
    if not r:
        raise RuntimeError(
            "IGA_STORAGE_ROOT is unset and no storage_root.txt was found next to iga_paths.py. "
            "Project B refuses to guess a storage path (no fallback to the old SSD). "
            "Set IGA_STORAGE_ROOT or create storage_root.txt with the absolute path to this "
            "project's *_DATA storage root."
        )
    r = os.path.abspath(r)
    if not os.path.isdir(r):
        raise RuntimeError("Configured IGA storage root does not exist: %s" % r)
    return r


STORAGE     = _storage_root()
IGA_NEURIPS = os.path.join(STORAGE, "IGA_NeurIPS")
GENEFORMER  = os.path.join(STORAGE, "IGA_ICLR2027_GENEFORMER")
MADA_DATA   = os.path.join(STORAGE, "MADA_data")
MADA        = os.path.join(STORAGE, "MADA")

__all__ = ["STORAGE", "IGA_NEURIPS", "GENEFORMER", "MADA_DATA", "MADA"]
