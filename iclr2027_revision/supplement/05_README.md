# Reproducibility supplement — contents and scope

This supplement collects the pre-result specification, its amendments, the environment of record, and
the pinned artifact digests for the locked foundation-model evaluation. It is anonymized: author names,
institutions, usernames, absolute paths, storage-device names, internal project names, hostnames and
email addresses have been removed or replaced with bracketed placeholders.

## Files

- **`01_experiment_lock.md`** — the pre-result specification ("lock"), verbatim except anonymization.
  It fixes, before any result existed: the frozen Geneformer-V2-104M representation and its construction;
  controls C1–C5; the four hypotheses H-FM1…H-FM4 with their gating controls, inference procedures and
  pass criteria; the bootstrap procedures and seeds; the reporting rules; and the reconciliation list.
- **`02_amendments/A1.md … A16.md`** — the amendment artifacts, verbatim except anonymization, in order.
  Every file in `02_amendments/` is a **derived supplementary copy**, not the contemporaneous original;
  each records its derived-copy creation date and the SHA-256 of its canonical source.
  `A1`–`A4` are the four **scientific** amendments, each recorded **before** the measurements it governs
  existed; each carries its own committed-date and committed-before-unblinding statement, and per their own
  text and the specification they add pre-registered secondaries and sensitivities and add, remove or reseed
  no control. `A6` and `A7` were committed **after** external unblinding: `A6` is an editorial wording change
  (four framing phrases), and `A7` records the execution of the pre-registered non-overlap sensitivity
  (specified in the lock, executed only after the primary was persisted). **There is no `A5.md`:** amendment
  A5 was a fileless editorial edit applied directly to the framing record and working manuscript and never
  produced a standalone artifact — its scope is stated in the shipped `02_amendments/A6.md` header and
  in the A5 section below.
  `A8` and `A9` are record-keeping and reporting reconciliations described below. `A10`, `A11` and `A12`
  are later still, all recorded **2026-08-29**, well after external unblinding: `A10` fixes the frozen
  Phase-B preprocessing state for post-hoc work, `A11` governs the linear depth-recovery probe and `A12`
  the target-structure diagnostic, both reported in the manuscript appendix "Sampling-depth diagnostics".
  Each of the three was locked before its own measurement existed, but **none belongs to the original
  pre-result specification**, and none changes a hypothesis, control, seed, endpoint or reported value.
- **`03_environment.md`** — the environment of record for the embedding run (`RUN_RECORD`) and the separate
  revision-analysis environment (`CURRENT_MACHINE`), each item labeled by source, including one recorded
  version conflict and the missing-environment-file gap.
- **`04_artifact_digests.md`** — full 64-character digests for the checkpoint, token dictionary,
  gene-median dictionary and Ensembl mapping dictionary, plus the repository commit; and, appended, the
  SHA-256 digests of the two archived harmonized external-endpoint (`y_primary`) CSVs (RPE1, K562-essential)
  that the primary external transfer analysis and the A7 sensitivity consume.

- **`06_governance/`** — later governance records that are **not** part of the pre-result lock.
  `kill_switch_27_construction_2026-08-29.md` records the author's construction, dated 2026-08-29, of the
  undefined term "validated embeddings" used by the lock's kill-switch clause. It is an interpretation
  stated before that clause's deadline; it is **not** an amendment, it does not modify the clause, and it
  does not claim to have been part of the original lock.

- **`part1_scientific.md`** — the governing scientific specification named by the lock header as the
  source of truth for hypotheses, controls, interpretation, multiplicity, budget and kill-switch.
  Included as a derived supplementary copy; its canonical SHA-256 is recorded in the copy's header.
- **`provenance/`** — sanitized supplementary derivatives of the frozen 37/29 counterfactual-exclusion
  reconstruction: the mini-spec, the result artifact and the human-readable report. Each derivative
  records its canonical source SHA-256 and its derivative creation date, and local path and machine
  identifiers have been redacted. The canonical artifacts, not these derivatives, are the frozen
  provenance record. The reconstruction script and its pre-run freeze record are **not** included here,
  because the canonical files embed machine-specific path context; they remain part of the project and
  code-release provenance record and are verifiable against the frozen script SHA-256 cited in
  `02_amendments/A9.md`.

## Amendments A5, A6, A7, A8 and A9

- **A8 (`02_amendments/A8.md`)** — authored and recorded 2026-08-27 (local project records). It was
  **not** under version control on that date; its version-control filing accompanies the current
  revision, and the authoritative filing date is the Git commit timestamp. It is record-keeping and
  changes no hypothesis, control, seed, endpoint, statistic, reported value or conclusion.
- **A9 (`02_amendments/A9.md`)** — a retrospective, post-unblinding governance and reporting
  reconciliation recorded with the current revision. It changes no hypothesis, endpoint, split,
  control, seed, metric, held-out access rule or scientific decision rule.
- **Version-control chronology.** A1–A4 first entered version control on 2026-08-25; A6–A7 on
  2026-08-26; A8 and A9 with the current revision. Authoring/recording dates and version-control
  dates are distinct and are reported separately in `A9.md`.

- **A5 is a fileless editorial amendment (no artifact).** It was an *editorial framing* edit applied
  directly to the manuscript framing record and the working manuscript during manuscript preparation —
  after all governed results existed. It rewords the Introduction "Findings" paragraph and a contributions
  preamble; it changes no hypothesis, control, seed, endpoint, statistic or reported value. It never
  produced a standalone amendment document, so there is no `A5.md`; its scope is recorded in the shipped
  `02_amendments/A6.md` header.
- **A6 is an editorial wording amendment (`02_amendments/A6.md`).** Committed 2026-08-26 after external
  unblinding, it replaces four occurrences of "genuine signal" / "genuinely" in the authoritative framing
  blocks with "measurable predictive information" (or the grammatical equivalent). It changes no hypothesis,
  control, seed, endpoint, statistic, reported value or conclusion.
- **A7 records the pre-registered non-overlap sensitivity (`02_amendments/A7.md`).** The VCC-train-overlap
  sensitivity was pre-registered in the lock (§25) and executed on 2026-08-26 — after the primary external
  result was persisted and after unblinding, both facts stated in the file. It re-scores already-stored
  predictions (no training, no refit, no test access) and reports the secondary alongside the primary;
  `CONCLUSION_CHANGES: NO`.

## What this supplement does NOT establish

- **No content digest binds a stored copy of the representation to the reported external correlations.**
  The pinned digests fix the *generating* artifacts (checkpoint, dictionaries, repository commit) and the
  full construction recipe, and the embedding step is recorded as deterministic — so `z` can be regenerated
  exactly. But no persisted manifest hashes a specific stored `z` array to the reported RPE1 / K562-essential
  correlations; that linkage is recorded in prose, not in a manifest. This supplement therefore supports
  *regeneration* of the representation, not *retrieval-with-proof* of the exact stored arrays used.
- **The exact all-in count of held-out (VCC test-split) accesses is not derivable from the durable record.**
  The record establishes exactly one governed scientific test evaluation and marks the test split untouched
  in every phase after the specification lock, but it separately (and deliberately non-merged) discloses a
  reproduction/diagnostic access episode without a standalone persisted log, so a single unqualified all-in
  access integer cannot be proven. The epistemic limits are set out in the shipped
  `02_amendments/A9.md` §4 ("What is NOT established" and "The defensible statement") and in the
  manuscript's held-out data access ledger.

These limits are shipped with the evidence rather than omitted.

## Two reporting items recorded with this revision

- **The matched-sensitivity result (A3 §2).** A3 fixed, before any of its own measurements existed and
  before the first documented external-outcome access in the surviving local record, a
  sensitivity that rebuilds the magnitude and representation
  blocks at matched per-construct cell counts. It was executed at both pre-specified counts
  (`n_match = 10` and `n_match = 27`), on all three screens, across all three pre-registered seeds. The
  cross-screen ordering changes with the matched count, which triggered A3's own conditional reporting
  obligation: the manuscript now discloses that the magnitude features, unlike the endpoint, are not
  size-matched across screens. Design, retention quantities and the bounded interpretation are in the
  manuscript appendix "Sampling-depth diagnostics". Note that A3's `n_match = 27` is a
  feature-side matched count derived as the maximum of the three screens' per-construct minima; it is a
  different intervention from the endpoint's fixed-size 27/788 subsampling, which derives its 27 from a
  VCC median. The two share a number, not a mechanism.

- **A gap in the amendment token taxonomy.** The specification's change-control clause enumerates three
  classification tokens but never defines them, and none fits an additive pre-registered sensitivity.
  `A2` and `A3` therefore carry the token `SENSITIVITY_ADDITION`, which is not among the three, and `A4`
  records the mismatch in place of a token. Each was recorded before the measurements it governs. This is
  a formal gap in the taxonomy; it does not bear on what was run, when it was run, or any reported value.
  It is disclosed in the manuscript appendix as well as here.

## Correction recorded in A9

An earlier version of `A9` recorded a sign-stability statistic for `ΔR²(z − C2)`. The frozen Phase-1
record persists that statistic only for `ΔR²(z − C1)`; none was ever recorded for `z − C2`. The
unsupported line was removed, and the same factual correction was applied to the manuscript. The
`z − C2` point estimate and its 95% interval are unchanged, as is every other value and decision in that
amendment. The correction is marked inline in `02_amendments/A9.md`. Any archive of this supplement
predating 2026-08-29 that still carries that line is superseded by this one.

## A13 — post-unblinding governance and reporting reconciliation

`02_amendments/A13.md` is a **post-unblinding governance and reporting reconciliation**, recorded
2026-08-29. It is not a prospective lock and not part of the original pre-result specification. It
changes no hypothesis, control, seed, endpoint, metric or primary result and computes no value; every
figure it cites is transcribed from an already-persisted artifact. It covers four items:

- **A3 RPE1 dual-population clarification.** A3 derived its matched count as `max(10, 25, 27)` from a
  minima table whose RPE1 minimum of 25 was measured on the checkpoint accumulator population, while the
  executed sensitivity ran on the persisted-embedding population, whose RPE1 minimum is 27. RPE1
  therefore retained 1,932 of 1,932 constructs and the anticipated 25–26-cell drop did not occur. The
  two RPE1 summaries, median 76 with IQR [50,118] and median 73 with IQR [48,113], describe the same
  1,932 constructs under two different cell populations and are not competing estimates of one.
- **Initial discharge of the A1 and A2 secondary reporting obligations.** Both secondaries were pre-specified,
  both were executed after the frozen primary was persisted, and both are now reported in the
  manuscript appendix. Neither replaces the frozen primary; the A2 linear head covers only
  magnitude-containing arms, re-evaluates no hypothesis, and leaves the representation-only results
  unchanged. A14 later completed A1's "in full" requirement by restoring the persisted $C_2$ and
  $C_4$ $x_{\mathrm{release}}$ rows.
- **Table 7 qualifier correction.** A row-spanning "train split" qualifier was correct for the VCC
  column and unsupported for the external column, which has no train split in that table. The qualifier
  was removed; both numbers are unchanged and no replacement qualifier was invented, because the exact
  population behind the external figure is not determinable from the governed record.
- **Chronology and release hygiene**, as described below.

## Two notes on how these records should be read

**Page counts and manuscript-state descriptions embedded in historical specification records refer to
the manuscript state at the time those records were written and are not descriptions of the final
release PDF.**

**Historical amendment records use "committed" in their contemporaneous local-record sense.** Git
history shows that A1–A4 first entered version control later; claims about their pre-result timing
therefore rely on the surviving contemporaneous local records and artifact chronology, not on a
pre-unblinding Git commit. The locked protocol required external outcomes to be held out until final
evaluation; separately, the surviving local record positively locates the first documented
external-outcome access at that point while not excluding undocumented earlier access. The two
statements are different in kind and are kept distinct in the manuscript.

## Notes on A5 and A6

- **A5** is a fileless editorial event recorded through the amendment chain; the contemporaneous scope
  description is preserved in the historical records.
- **A6** preserves an unfilled historical `FRAMING_HASH_AFTER` field from the contemporaneous record; it
  was **not** backfilled retrospectively, because a hash computed today would not be a historical hash.
  Final-release integrity is instead established by the release-level hashes and the build audit.
- **A6** records the framing language adopted at that revision stage. Subsequent manuscript edits
  preserve the same bounded scientific claim but do not necessarily retain those sentences verbatim.

## A14 — post-unblinding reporting completion

`02_amendments/A14.md` is a **post-unblinding reporting completion and release reconciliation**,
recorded 2026-08-29 after an independent audit of the previous release. Like A13 it is not a
prospective lock, changes no hypothesis, control, seed, endpoint, metric, estimator or frozen
primary result, and computes nothing. It closes four reporting defects: A1's
`x_release` comparator is now reported in full for every persisted arm it enters, with its feature
basis named and cross-referenced to the corresponding frozen-primary values; MAX-LINEAR is no
longer described as count-adjusted and is explicitly distinguished from the count-adjusted
partial-Spearman diagnostic, with its own persisted intervals labelled as belonging to it;
MAG+COUNT now carries the extrapolation limitation A4 recorded in advance and required to be stated
alongside the number; and the secondary section names its three amendment families instead of
mis-counting them. None of the six secondaries is promoted, and none replaces a frozen primary.

**Amendment inventory:** sixteen numbered amendments, fifteen as files. A5 remains fileless.

## Two further reading notes

**Chronology bound.** The locked protocol required external outcomes to be held out until final
evaluation. Separately, the surviving local record positively locates the *first documented*
external-outcome access at that point while not excluding undocumented earlier access. Current
manuscript and release prose use that bound; canonical historical records, A13 included, keep their
original wording and were not rewritten.

**A4 historical wording.** Historical mechanistic wording in A4 predates the final bounded
interpretation. The final manuscript treats the sampling-depth relationships descriptively and does
not infer that sampling depth causally explains the external transfer failure.

## A15 — post-unblinding wording reconciliation

`02_amendments/A15.md` is a **post-unblinding wording reconciliation**, recorded 2026-08-29 after
an independent audit of the previous release. It is not a prospective lock and not a scientific
amendment: it performs no computation and changes no dataset, endpoint, split, control, seed,
representation, model, head, metric, estimator, or numerical value. It aligns five wording sites to
the governing evidence bound — an appendix subsection title, one self-contradicting clause in the
chronology paragraph, this README's description of A3's timing, this README's description of what
A13 discharged, and one interpretive clause about the sampling-depth overlap that claimed more than
the evidence supports.

**Reading the amendment chain's three post-unblinding reporting records:** A13 is the reporting
reconciliation and the *initial* discharge of the A1 and A2 obligations; A14 completed A1's "in
full" requirement and reconciled the A4 secondary descriptions; A15 is wording only; and A16
closes two release defects, redirecting four references to a non-shipped internal evidence record
onto shipped artifacts and rescoping one chronology sentence. None of the four changes a frozen
primary result.
