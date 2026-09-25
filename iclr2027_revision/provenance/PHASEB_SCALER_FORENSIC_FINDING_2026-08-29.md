# Phase-B scaler provenance finding — 2026-08-29 UTC

Durable record of the historical scaler-provenance audit. This note changes no result, no gate
verdict, and no scientific decision. It is a provenance record only.

## Evidence

- Prior forensic run: `/home/mehrdad/iclr2027_phaseB_out/phaseB/hfm4_20260829T063027Z_8666e25de1a3`
  (external to this repository; not committed).
- Forensic manifest SHA-256: `9ba0200d422a7dba0c485caa3a9eb91c1bee8bb8228cac1c4b6aa469ef8fddfb`
- Forensic report SHA-256:   `8ef536d538ab147b11152a8b185fdb5c18224d69719f276b11163d478df9f9e0`

## Findings

1. The historical Phase-0 scaler exact-regeneration gate returned **MISMATCH** and remains MISMATCH.
2. The Phase-0 `z` and `C2` `StandardScaler` pickles were **written but never read** by Part-1
   execution code. The sole reference in `pipeline/` and `scripts/` is the write at
   `pipeline/run_phase0.py:133`.
3. Part 1 **re-fit** its operative `z`/`C1`/`C2` scaler state in process
   (`pipeline/run_phase1_select.py:24-26`) and applied it to every split
   (`pipeline/run_phase1_select.py:31,35`), per `lock:167-169`.
4. The exact operative Part-1 scaler state was **not persisted**.
5. No direct historical pre-scaler `z` or `C2` array, and no digest of any scaler-fit input, was
   found within the authorized Project-B scope. `phase0/phase0_results.json` records summary
   statistics only.
6. With bit-identical operands (verified by digest), changing only the BLAS thread count altered the
   float32 `x @ R` product: 755/768 column means differed, max relative difference 3.28e-06 —
   larger than the 1.33e-06 persisted-vs-reconstructed C2 mean gap.
7. Historical BLAS backend, BLAS version and thread count were **not preserved**.
8. Historical NumPy and SciPy versions were **not sufficiently preserved** (recorded under a
   `CURRENT_MACHINE` label, explicitly not a run record — `supplement/03_environment.md`).
9. The `z` discrepancy remains **unresolved / partially localized**: `mean_` reproduced exactly while
   variance-derived state did not, and no software stack tested reproduced the persisted state.
10. Three scikit-learn signals were observed: **1.7.2** in the persisted pickle metadata, **1.8.0**
    in a current-machine project environment record, **1.9.0** in the forensic gate environment.
    None of these establishes which stack the historical runtime actually used.

## Governing statements

> The preserved historical provenance is insufficient to establish bitwise regeneration of all
> preprocessing state used by the reported pipeline.

> The forensic finding does not show that Part 1 consumed a mismatched persisted scaler artifact;
> the persisted phase0 scaler pickles were not loaded by the Part-1 execution path.

## Classification

    HISTORICAL_PHASE0_EXACT_REGENERATION_GATE       = MISMATCH
    PART1_CONSUMED_PHASE0_SCALER_PICKLES            = NO
    PART1_OPERATIVE_SCALER_STATE_PERSISTED          = NO
    PART1_BITWISE_SOURCE_REGENERATION_DEMONSTRATED  = NO

The last line is deliberately *not* written as `PART1_REPRODUCIBLE_FROM_SOURCE = NO`: the audit did
not establish impossibility in principle, only that bitwise regeneration was not demonstrated from
the preserved record.
