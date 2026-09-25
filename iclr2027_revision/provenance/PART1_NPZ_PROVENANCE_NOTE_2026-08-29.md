# Part-1 NPZ Provenance Note — 2026-08-29

Post-hoc provenance record.
No scientific result was recomputed.
No outcome value was accessed.

## 1. Historical artifact finding

`iclr2027_revision/phase4/phase4_ext_predictions.npz` is a persisted historical prediction artifact
that existed locally but **was not bound to Git history** at the time of audit. It was excluded by
the repository-wide `*.npz` policy (`.gitignore:14`).

    path        iclr2027_revision/phase4/phase4_ext_predictions.npz
    bytes       346,604
    mtime       2026-08-13 02:15:09 -0400
    SHA-256     71977763748f670b8447a728b0a42c6f1bbfbe8fd6fb64e252e0277421ac8082

The digest above was computed **before** the Git tracking change made on 2026-08-29.

## 2. A7 operative dependency

A7 explicitly reused the already stored prediction artifact for its reported non-overlap
sensitivity — `experiment_specs/lock_amendment_A7.md:31` ("## Inputs (all already stored; nothing
refit, retrained, or regenerated)") and `:33-34` ("Stored per-construct, seed-averaged predictions:
`phase4/phase4_ext_predictions.npz` (`RPE1_P` 1932×7, `K562_P` 1903×7, arms = z, zm, C1, C2, C3, C4,
C5)"). That sensitivity is reported in the manuscript at `v7/main.tex:1100` ("On the retained
sets---$1{,}889$ constructs for RPE1 and $1{,}870$ for K562-essential---no verdict changes").

> An operative artifact consumed by the reported A7 sensitivity computation was not bound to the
> current Git history at the time of audit.

    A7_OPERATIVE_DEPENDENCY_ON_PERSISTED_NPZ = YES

## 3. Primary external result — a distinct and unresolved question

    PRIMARY_EXTERNAL_PERSISTED_NPZ_RELOAD_DEPENDENCY = NOT_DETERMINED

Persistence is documented (`experiment_specs/phase4_results.md:12`, `:95`). But the Phase-4 primary
scoring code is absent from the Project-B repository; no primary NPZ reload callsite is present;
and no in-memory primary scoring path is positively established. Persistence evidence therefore
cannot resolve what primary scoring consumed.

> The Phase-4 persistence event is attested by a contemporaneous governance record, but neither the
> persistence code nor the primary scoring consumption path is present in the searched Project-B
> repository scope. The record therefore establishes that the predictions were persisted before
> outcome access, but does not establish whether primary scoring subsequently reloaded the persisted
> NPZ or consumed the corresponding in-memory predictions.

A7's known persisted-artifact reuse is a **separate** execution context and is not evidence about
the primary scoring path.

### 3.1 Closing condition

> This verdict is resolvable only if the Phase-4 primary scoring code, or an equally direct
> execution record showing the scoring input path, is located. If that evidence is not recoverable
> from any authorized source, the question remains permanently NOT_DETERMINED; repeating the same
> Project-B repository audit cannot settle it.

    PRIMARY_RELOAD_VERDICT_CLOSING_CONDITION =
      LOCATE_PHASE4_PRIMARY_SCORING_CODE_OR_EQUIVALENT_DIRECT_EXECUTION_RECORD

## 4. Project-B Phase-4 code-availability gap

    PHASE4_CODE_FOUND_UNTRACKED             = NO
    PHASE4_EXT_PREDICTIONS_WRITE_PATH_FOUND = NO
    PRIMARY_SCORING_PATH_FOUND              = NO
    A7_RESCORE_CODE_PRESENT                 = NO
    PHASE4_CODE_PROVENANCE_GAP              = YES

> The executable Phase-4 producing code and A7 re-score driver were not found in the searched
> Project-B repository scope. This is a code-availability gap distinct from the prediction-artifact
> tracking gap.

> This finding does not establish that the code never existed or that it is absent outside Project B.

Six targeted searches over Project-B `*.py`/`*.sh` returned zero hits each: `savez.*phase4`,
`savez.*ext_predictions`, `dump.*phase4_primary`, `dump.*confound_diagnostic`,
`np.load.*ext_predictions`, and any `phase4_ext_predictions` reference. The only Project-B code
touching Phase-4 artifacts consists of four **tracked** figure/table readers of the Phase-4 JSON
outputs. `pipeline/bootstrap_wiring.py` provides the inference helper A7 cites
(`lock_amendment_A7.md:27`) and is tracked, but it is a library function, not the A7 driver.

## 5. Manuscript analysis-code release commitment

    CODE_RELEASE_COMMITMENT_FILE  = iclr2027_revision/v7/main.tex
    CODE_RELEASE_COMMITMENT_LINES = 776-779

Verbatim (`v7/main.tex:777-779`):

> "its amendment chain, the environment record, and the artifact digests accompany this submission
> as supplementary material; the analysis code will be released upon acceptance."

The manuscript states that the analysis code will be released upon acceptance
(`v7/main.tex:776-779`, quoted above). The executable Phase-4 producing code and A7 re-score driver
were not found in the searched Project-B scope. Whether that commitment can be met for Phase 4
depends on whether this code is recoverable from another authorized source. This note records that
dependency; it does not resolve it and proposes no manuscript change.

## 6. The 37/29 reconstruction script is a different object

`iclr2027_revision/provenance/reconstruct_nonoverlap_counts.py` is currently untracked and

    RECONSTRUCT_NONOVERLAP_COUNTS_RELATION_TO_A7 = RELATED_ONLY_CONCEPTUALLY

It reconstructs the 37/29 metadata-only counterfactual overlap counts (`:3`, `:76`
`EXPECT_COUNTERFACTUAL = {"RPE1": 37, "K562": 29}`), uses A7's 43/33 overlap counts only as
validation checks (`:75`), explicitly forbids outcome-column access (`:78-82` `FORBIDDEN_COLS`,
`:11-12`), computes no scientific metric, and contains no reference to A7's 1,889 / 1,870 retained
sets. It is not the A7 re-score driver. Its tracking state was deliberately left unchanged.

## 7. Related provenance record

Related provenance record: `PHASEB_SCALER_FORENSIC_FINDING_2026-08-29.md` documents the separate
historical Phase-B/Part-1 preprocessing-state provenance finding and its prospective Phase-B
numerical-baseline resolution.

## 8. Subsequent Git binding — what it does and does not establish

On 2026-08-29 the exact persisted historical prediction artifact identified in §1 was
**subsequently bound to Git history**.

**This establishes:**

- future repository checkouts can retrieve this exact persisted prediction artifact;
- its byte identity is now bound to repository Git history;
- the persisted prediction artifact explicitly reused by A7 is included in the repository.

**This does not establish, and is not claimed to establish:**

- any modification to a historical scientific result;
- any alteration of evaluation chronology;
- whether primary scoring reloaded this NPZ;
- recreation of the absent Project-B Phase-4 producer;
- recreation of the absent Project-B A7 re-score driver;
- end-to-end Phase-4 reproducibility;
- external timestamping;
- independent attestation.

The artifact is not retroactively preregistered, was not originally tracked, and was not
historically committed. Binding it now records availability, nothing more.

## 9. Existing A7 consistency record

`lock_amendment_A7.md:41` records that the full-set re-score reproduced `phase4_primary.json` to all
displayed digits. This is summarised here only as an **existing recorded consistency check**. It was
not recomputed, and no scientific array was inspected.

    A7_FULLSET_CONSISTENCY = DOCUMENTED_IN_EXISTING_RECORD

---

This note concerns artifact and code provenance availability. It does not alter or re-evaluate the
scientific validity of the reported Part-1 results.
