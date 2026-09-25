"""Head-family selection — NaN-safe (Phase-1 fragility fixed for Phase 4).

Phase 1 relied on Python `sorted()` ordering with NaN keys present (all NaN comparisons are False,
so ordering is input-order-dependent, not defined). It got the right seven selections empirically,
but that was not guaranteed by the code. This routine excludes any family whose validation score is
non-finite BEFORE any argmax or margin computation: a NaN family produces no result, is never coerced
to a number, is never ranked, and cannot be selected regardless of input order.
"""
import math


def select_family(val_means, families=None):
    """Return (selected, runner_up, margin, best_mean) using VCC validation R² only.
    val_means: dict family -> mean validation R² (may contain NaN/inf). Non-finite families are
    excluded up front. Raises if no family has a finite score."""
    if families is None:
        families = list(val_means)
    finite = [f for f in families if math.isfinite(val_means[f])]   # explicit NaN/inf filter
    if not finite:
        raise ValueError("no family produced a finite validation score")
    ranked = sorted(finite, key=lambda f: val_means[f], reverse=True)
    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    margin = (val_means[best] - val_means[runner]) if runner is not None else float("inf")
    return best, runner, float(margin), float(val_means[best])
