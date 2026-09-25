"""Torch backend bridge for `run_synthetic.run_one`.

This file is the thin glue layer between the backend-agnostic experiment
driver in `experiments.run_synthetic` and the production-path torch stack
(`models.families.build_model` + `training.trainer.train_torch`).

Responsibilities:
  * Construct the torch `nn.Module` for the requested family.
  * Build a `CompositeLoss` with the per-family enable flags locked by SPEC:
        DWP   : L_pred only
        SRM   : L_pred + lambda_stab * L_stab(z)
        SPSM  : L_pred + lambda_stab * L_stab(z) + lambda_rel * L_rel(y_pred)
        DAFR  : L_pred + lambda_stab * L_stab(z_s) + lambda_rel * L_rel(y_pred)
                + lambda_sep * L_sep(z_s, z_c)
  * Call `train_torch`.
  * Wrap the trained module in the backend-agnostic `TorchTrained` shim so
    the downstream evaluator in `run_one` can invoke `.predict/.encode/
    .linear_predictor()` without knowing which backend produced it.

The numerical values (λ defaults, epochs, lr, batch_size, device) come
from the caller's cfg dict. If a value is missing we fall back to the
TrainConfig dataclass defaults.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def train_family_torch(name: str, dataset, *, seed: int, cfg: Mapping[str, Any]):
    """Train a family via the torch path; return a `TorchTrained` shim.

    Parameters
    ----------
    name    : one of "DWP", "SRM", "SPSM", "DAFR"
    dataset : SyntheticDataset (train/val/test shards)
    seed    : int — forwarded to trainer for rng
    cfg     : dict-like. Recognized keys:
        epochs, batch_size, lr, weight_decay, grad_clip,
        lambda_stab, lambda_rel, lambda_sep,
        device, model_cfg (nested dict for encoder/head/factorization)

    Returns
    -------
    TorchTrained — exposes .predict(x), .encode(x), .linear_predictor().
    """
    if not _HAS_TORCH:
        raise ImportError(
            "train_family_torch requires PyTorch. Install with "
            "`bash scripts/setup_torch_env.sh` or use --backend numpy."
        )

    from ..losses.composite import CompositeLoss, LossWeights
    from ..losses.prediction import prediction_bce, prediction_mse
    from ..models.families import build_model
    from .shim import TorchTrained
    from .trainer import TrainConfig, train_torch

    # Per-family enable flags (SPEC §5, locked)
    stab_enabled = name in ("SRM", "SPSM", "DAFR")
    rel_enabled  = name in ("SPSM", "DAFR")
    sep_enabled  = name == "DAFR"

    # Default λ's chosen small so that λ_stab L_stab ≪ L_pred until stability
    # becomes load-bearing. These match the NumPy-sandbox defaults; the real
    # values may be swept in a follow-up phase.
    weights = LossWeights(
        lambda_stab=float(cfg.get("lambda_stab", 0.5)) if stab_enabled else 0.0,
        lambda_rel =float(cfg.get("lambda_rel",  0.5)) if rel_enabled  else 0.0,
        lambda_sep =float(cfg.get("lambda_sep",  0.1)) if sep_enabled  else 0.0,
    )

    train_cfg = TrainConfig(
        epochs      =int(cfg.get("epochs", 100)),
        batch_size  =int(cfg.get("batch_size", 256)),
        lr          =float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
        grad_clip   =float(cfg.get("grad_clip", 1.0)),
        lambda_stab =weights.lambda_stab,
        lambda_rel  =weights.lambda_rel,
        lambda_sep  =weights.lambda_sep,
        stab_enabled=stab_enabled,
        rel_enabled =rel_enabled,
        sep_enabled =sep_enabled,
        device      =str(cfg.get("device", "cpu")),
    )

    # Build module
    input_dim = dataset.train[0].x.shape[1]
    model_cfg = dict(cfg.get("model_cfg", {}))
    module = build_model(name, input_dim=input_dim, model_cfg=model_cfg)

    pred_loss_name = str(cfg.get("pred_loss", "mse"))
    if pred_loss_name == "bce":
        pred_loss_fn = prediction_bce
    elif pred_loss_name == "mse":
        pred_loss_fn = prediction_mse
    elif pred_loss_name in ("tail_rank", "tail_quintile"):
        # Phase 8C tail-weighted MSE. Cache the train y once.
        from .tail_losses import (
            make_rank_tail_weighted_mse,
            make_quintile_tail_weighted_mse,
        )
        train_y_concat = np.concatenate(
            [s.y for s in dataset.train if s.y.shape[0] > 0]
        )
        if pred_loss_name == "tail_rank":
            gamma = float(cfg.get("tail_gamma", 1.0))
            pred_loss_fn = make_rank_tail_weighted_mse(train_y_concat, gamma=gamma)
        else:  # tail_quintile
            q_weights = tuple(float(w) for w in cfg.get(
                "tail_quintile_weights", (2.0, 1.0, 1.0, 1.0, 2.0)
            ))
            pred_loss_fn = make_quintile_tail_weighted_mse(
                train_y_concat, q_weights=q_weights
            )
    elif pred_loss_name == "lds":
        # Phase 9A-Mini Label Distribution Smoothing weighted MSE.
        from .effect_balanced_losses import make_lds_weighted_mse
        train_y_concat = np.concatenate(
            [s.y for s in dataset.train if s.y.shape[0] > 0]
        )
        pred_loss_fn = make_lds_weighted_mse(
            train_y_concat,
            n_bins=int(cfg.get("lds_n_bins", 20)),
            sigma=float(cfg.get("lds_sigma", 2.0)),
            max_weight=float(cfg.get("lds_max_weight", 5.0)),
        )
    else:
        raise ValueError(
            f"Unknown pred_loss {pred_loss_name!r}; expected one of "
            "'mse', 'bce', 'tail_rank', 'tail_quintile', 'lds'."
        )
    composite = CompositeLoss(
        weights=weights,
        stab_enabled=stab_enabled,
        rel_enabled=rel_enabled,
        sep_enabled=sep_enabled,
        pred_loss_fn=pred_loss_fn,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    _ = train_torch(module, dataset, train_cfg, composite_loss=composite, seed=seed)

    return TorchTrained(
        module=module,
        model=name,
        device=train_cfg.device,
        use_z_s_for_encoding=(name == "DAFR"),
    )
