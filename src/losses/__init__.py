"""Loss term primary backends (SPEC Section 6, locked).

Public API (torch-based primary):

    from src.losses import (
        prediction_mse,
        stability_mmd,
        relation_softspearman,
        separation_covpenalty,
        CompositeLoss,
    )

NumPy reference implementations (used for unit tests and as a framework-free
sanity layer) live in `numpy_ref.py`. The torch primary and NumPy reference
compute the same mathematical quantity and are cross-checked in tests.

If torch is not importable at module import time, the torch symbols become
lazy shims that raise a descriptive error when called; the NumPy references
remain available.
"""

from .numpy_ref import (
    mse_np,
    mmd_np,
    softspearman_loss_np,
    covpenalty_np,
)

try:  # torch is the production backend but not required to import this package
    import torch  # noqa: F401
    from .prediction import prediction_mse
    from .stability_mmd import stability_mmd
    from .relation_softspearman import relation_softspearman
    from .separation_covpenalty import separation_covpenalty
    from .composite import CompositeLoss

    _TORCH_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    _TORCH_AVAILABLE = False
    _IMPORT_ERROR = _e

    def _torch_unavailable(*a, **kw):
        raise ImportError(
            f"torch is required for this loss path, but was not importable: {_IMPORT_ERROR}"
        )

    prediction_mse = _torch_unavailable
    stability_mmd = _torch_unavailable
    relation_softspearman = _torch_unavailable
    separation_covpenalty = _torch_unavailable
    CompositeLoss = _torch_unavailable


__all__ = [
    "prediction_mse",
    "stability_mmd",
    "relation_softspearman",
    "separation_covpenalty",
    "CompositeLoss",
    "mse_np",
    "mmd_np",
    "softspearman_loss_np",
    "covpenalty_np",
]
