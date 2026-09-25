# D-TARGET-VAR — consumed target-structure diagnostic

Governed by lock amendment A12. **D-TARGET-VAR was proposed after the locked D-MECH result had
been observed and archived. Its specification and estimands were frozen before any D-TARGET-VAR
result was observed.** It is a design diagnostic, not a rescue analysis, and cannot alter the
consumed D-MECH interpretation. No D-MECH-NL decision is made here.

Lock commit `706b05737abe918c721c5e01a70c3679a53c62ad` · implementation commit `04620141ce7435f80834d6b6ad48378575d173a6`

Target `t = numpy.log1p(n_perturbed_cells), natural log, float64` · row key `(split, batch, target_gene)` · standardized: NO · splits pooled: NO

## Four cells

| field | TRAIN/target_gene | TRAIN/batch | VALIDATION/target_gene | VALIDATION/batch |
|---|---|---|---|---|
| `N_ROWS` | 5740 | 5740 | 2052 | 2052 |
| `N_GROUPS` | 140 | 48 | 49 | 48 |
| `GROUP_SIZE_MIN` | 1 | 114 | 1 | 39 |
| `GROUP_SIZE_MEDIAN` | 48.000000 | 119.000000 | 48.000000 | 43.000000 |
| `GROUP_SIZE_MAX` | 48 | 126 | 48 | 45 |
| `GROUP_SIZE_MEAN` | 41.000000 | 119.583333 | 41.877551 | 42.750000 |
| `SSB` | 1292.635105 | 10.717657 | 338.392502 | 5.020453 |
| `SSW` | 206.972225 | 1488.889673 | 75.286666 | 408.658715 |
| `TOTAL_SS` | 1499.607330 | 1499.607330 | 413.679168 | 413.679168 |
| `MSB` | 9.299533 | 0.228035 | 7.049844 | 0.106818 |
| `MSW` | 0.036959 | 0.261576 | 0.037587 | 0.203922 |
| `m0` | 40.967491 | 119.581274 | 41.790915 | 42.749036 |
| `VAR_BETWEEN` | 0.226096 | -0.000280 | 0.167794 | -0.002271 |
| `VAR_WITHIN` | 0.036959 | 0.261576 | 0.037587 | 0.203922 |
| `ICC` | 0.859500 | -0.001073 | 0.816989 | -0.011264 |
| `ETA_SQUARED` | 0.861982 | 0.007147 | 0.818007 | 0.012136 |
| `M_BAR_ARITHMETIC` | 41.000000 | 119.583333 | 41.877551 | 42.750000 |
| `M_BAR_SIZE_WEIGHTED` | 45.518815 | 119.680139 | 46.036062 | 42.795322 |
| `DESIGN_EFFECT_APPROX_ARITH` | 35.379986 | 0.872709 | 34.396507 | 0.529710 |
| `DESIGN_EFFECT_APPROX_WEIGHTED` | 39.263906 | 0.872605 | 37.793964 | 0.529199 |
| `ICC_DESIGN_EFFECT_APPROX_N_ARITH` | 162.238618 | 6577.224593 | 59.657221 | 3873.821135 |
| `ICC_DESIGN_EFFECT_APPROX_N_WEIGHTED` | 146.190244 | 6578.007848 | 54.294383 | 3877.558245 |
| `N_ROWS_TO_APPROX_N_RATIO_ARITH` | 35.379986 | 0.872709 | 34.396507 | 0.529710 |
| `N_ROWS_TO_APPROX_N_RATIO_WEIGHTED` | 39.263906 | 0.872605 | 37.793964 | 0.529199 |

## Batch caveat

- **TRAIN/batch** — N_BATCHES = 48, ICC = -0.001073. The batch ICC is a one-way estimate based on N_BATCHES = 48 groups; the between-batch component may be unstable when the number of groups is small.
- **VALIDATION/batch** — N_BATCHES = 48, ICC = -0.011264. The batch ICC is a one-way estimate based on N_BATCHES = 48 groups; the between-batch component may be unstable when the number of groups is small.

Both batch ICCs are negative. Their design effects fall below 1 and their approximate N
exceed N_ROWS: this is a mechanical consequence of the negative method-of-moments ICC estimate
and must not be interpreted literally as more independent observations than observed rows.

## Mandatory qualification

ICC_target_gene and ICC_batch are each separate one-way estimates. They are not components of a
joint decomposition, are mutually confounded to an unquantified degree, do not sum to a total, and
a larger value on one axis does not establish that variance belongs to that axis. Resolving that
attribution would require a design deliberately excluded from this lock.

Both design-effect approximations are conventional diagnostics. Neither is an exact count of
independent observations, neither is a power calculation, and neither is guaranteed to equal or
bracket the information available to a downstream prediction procedure. For positive ICC the
size-weighted variant is the more conservative of the two; for negative ICC that ordering reverses,
so neither is universally more conservative. ETA_SQUARED is a raw SS fraction and is not
interchangeable with ICC.

D-TARGET-VAR characterizes target clustering only. Whether this structure changes the design value
of a future nonlinear recoverability diagnostic is a separate post-diagnostic decision and is not
made here.
