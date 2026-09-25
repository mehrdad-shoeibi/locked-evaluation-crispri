"""Backend-agnostic `TrainedModel` protocol.

Both the NumPy linear trainer (`numpy_trainer.TrainResult`) and the torch
trainer return different internal objects, but every downstream evaluator
in `run_synthetic` only needs three things:

    predict(x_np)       -> (n,)    pooled predictions on raw x
    encode(x_np)        -> (n, d)  representation z
    linear_predictor()  -> (d_x,)  effective linear map x -> y, in x-space

The `TrainedShim` wraps a torch `nn.Module` to expose these with numpy IO
so the evaluation code in `run_synthetic.run_one` is untouched across
backends. For the NumPy path, `numpy_trainer.TrainResult` already exposes
these methods (with a `.W_enc @ .w_head` composition for the linear
predictor); a compatibility adapter is provided here as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class TrainedModel(Protocol):
    model: str
    def predict(self, x: np.ndarray) -> np.ndarray: ...
    def encode(self, x: np.ndarray) -> np.ndarray: ...
    def linear_predictor(self) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# NumPy adapter (zero-cost — re-exports TrainResult with one new method)
# ---------------------------------------------------------------------------
def wrap_numpy(tr) -> "TrainedModel":
    """Attach a `linear_predictor` method to a numpy_trainer.TrainResult.

    We avoid modifying TrainResult itself to keep Phase-2/5 tests stable.
    """
    def linear_predictor() -> np.ndarray:
        return tr.W_enc @ tr.w_head
    tr.linear_predictor = linear_predictor   # type: ignore[attr-defined]
    return tr


# ---------------------------------------------------------------------------
# Torch adapter
# ---------------------------------------------------------------------------
if _HAS_TORCH:

    @dataclass
    class TorchTrained:
        """Backend-agnostic wrapper around a trained torch model.

        The wrapped `module` must return a `ModelOutput` (see `models.families`)
        whose `.y_pred` is (n,) and whose `.z` (or `.z_s` for DAFR) is (n, d).
        """
        module: "nn.Module"
        model: str
        device: str = "cpu"
        use_z_s_for_encoding: bool = False

        def _to_t(self, x: np.ndarray) -> "torch.Tensor":
            return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)

        @torch.no_grad()
        def predict(self, x: np.ndarray) -> np.ndarray:
            self.module.eval()
            out = self.module(self._to_t(x))
            return out.y_pred.detach().cpu().numpy().reshape(-1)

        @torch.no_grad()
        def encode(self, x: np.ndarray) -> np.ndarray:
            self.module.eval()
            out = self.module(self._to_t(x))
            z = out.z_s if (self.use_z_s_for_encoding and out.z_s is not None) else out.z
            return z.detach().cpu().numpy()

        @torch.no_grad()
        def linear_predictor(self) -> np.ndarray:
            """Effective linear weight mapping x -> y via a finite-difference probe.

            Uses `encode` and `predict` on a random Gaussian probe to estimate the
            best linear approximation to the (generally non-linear) learned map.
            Returned weight vector has shape (d_x,). Used by the
            stable_predictor_recovery metric, which is only qualitatively
            interpreted (|cos|), so a linear-probe estimate suffices.
            """
            # Probe x ~ N(0, I), regress y = <w, x>
            self.module.eval()
            # Infer d_x from first layer shape or by a zero-probe round trip
            d_x = _infer_input_dim(self.module)
            n_probe = max(2048, 4 * d_x)
            rng = np.random.default_rng(0)
            X = rng.standard_normal((n_probe, d_x)).astype(np.float32)
            y = self.predict(X)
            Xc = X - X.mean(axis=0, keepdims=True)
            yc = y - y.mean()
            A = Xc.T @ Xc + 1e-3 * np.eye(d_x)
            return np.linalg.solve(A, Xc.T @ yc)


    def _infer_input_dim(module: "nn.Module") -> int:
        """Find the input dim by walking to the first Linear layer with no
        prior Linear (i.e. the trunk's first layer)."""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                return m.in_features
        raise RuntimeError("Could not infer input dim from module — no Linear layer found.")

else:
    # Stub so imports don't fail without torch.
    class TorchTrained:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("TorchTrained requires torch; install torch to use it.")
