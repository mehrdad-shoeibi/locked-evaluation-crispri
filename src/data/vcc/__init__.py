"""Virtual Cell Challenge (VCC) — single-cell perturbation benchmark.

Entry points:

    from src.data.vcc import VccContextShard, VccSplit, load_vcc_pseudobulk

    train_shards = load_vcc_pseudobulk("/path/to/vcc", "train")

The loader caches pseudo-bulked shards under `$IGA_DATA_ROOT/cache/vcc/`
so all runs after the first reload in seconds.
"""

import os as _o, sys as _s
_r = _o.path.dirname(_o.path.abspath(__file__))
while _r != _o.path.dirname(_r) and not _o.path.exists(_o.path.join(_r, "iga_paths.py")):
    _r = _o.path.dirname(_r)
if _r not in _s.path:
    _s.path.insert(0, _r)
import iga_paths
from .loader import load_vcc_pseudobulk
from .schema import VccContextShard, VccDataset, VccSplit

__all__ = [
    "VccContextShard",
    "VccDataset",
    "VccSplit",
    "load_vcc_pseudobulk",
]
