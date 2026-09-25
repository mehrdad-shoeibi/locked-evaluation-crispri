"""Metric modules, organized by the four groups in SPEC §7.

All metrics are pure-NumPy and operate on already-computed predictions or
representations. Per SPEC §11, metrics are computed offline from saved
`preds.parquet` / feature arrays so they can be added post-hoc without
retraining.

Public API:

    from src.metrics.predictive     import r2, pearson, spearman, ndcg_at_k
    from src.metrics.transfer       import gap, gap_rel, robustness_worst_ctx,
                                           spearman_gap
    from src.metrics.representation import mmd2_gaussian, linear_cka, rbf_cka,
                                           linear_probe_r2, context_separability
    from src.metrics.relation       import rank_consistency, pairwise_order_agreement,
                                           feature_importance_stability,
                                           stable_predictor_recovery
"""

from . import predictive, transfer, representation, relation  # re-export namespaces

__all__ = ["predictive", "transfer", "representation", "relation"]
