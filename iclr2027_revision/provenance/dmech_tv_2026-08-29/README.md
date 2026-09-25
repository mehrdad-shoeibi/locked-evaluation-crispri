# D-MECH train/validation supporting diagnostic — consumed execution archive

**Archival record only.** The scientific result below was observed on 2026-08-29 and is final.
Nothing here re-executes, recomputes or reinterprets it.

```
DMECH_PRE_RESULT_AUTHORITY       = A11 repository lock
A11_SHA256                       = 4f3308eb8b9733b741b7eb4d1e2a2940ca29d050b315c61058d232fb9911a483
PLAN_SHA256                      = fb976c859bd72571c929ffc78e2b2c261d450bc3ac4a759bfd215161d44913cd
CLAUDE_PRE_RESULT_NOTE_AUTHORITY = NON_AUTHORITATIVE_WORKING_RECORD
```

The authoritative pre-result lock is the repository A11 record. Any Claude-workspace note is a
non-authoritative working record and may **not** be cited as proof that the design was
preregistered. No such note exists in this repository or on this filesystem.

## The consumed result

Executed under `STARTING_HEAD 64e44afe2c4d42edb8802a7c52818d3b8a6ad096`.

| alpha | validation R² |
|---|---|
| 0.001 | −0.14016045462719284 |
| 0.01 | −0.14008420483922368 |
| 0.1 | −0.13930009793397890 |
| 1.0 | −0.13292780213498245 |
| 10.0 | −0.10874826591187481 |
| 100.0 | −0.06905994852762753 |
| **1000.0** | **−0.014630115836128832** ← selected |

```
SELECTED_ALPHA         = 1000.0
SELECTED_VALIDATION_R2 = -0.014630115836128832
DMECH_RESULT_BRANCH    = NULL_OR_NONPOSITIVE
ALPHA_AT_GRID_BOUNDARY = YES        ALPHA_BOUNDARY_SIDE = MAX
TRAIN_ROWS = 5740   VALIDATION_ROWS = 2052   TARGET_SUPPORT_N_CELLS = [10, 121]
DETERMINISTIC_REPEATS = 3   REPEATS_IDENTICAL = YES   (in-process, fixed environment)
```

## Locked reading

> **"This probe did not linearly recover sampling-depth-associated information from the frozen
> Geneformer representation on VCC train/validation under the pre-specified estimator and grid"**
> — validation R² = −0.014630115836128832 — **"and the regularization optimum was not bracketed
> above within the pre-specified grid, so non-recovery is not established even for this estimator
> family."**

The MAX-boundary disclosure is mandatory here and is not discretionary.

**Reportable observations, and only these:** all seven pre-specified validation R² values were
negative; the values increased monotonically with alpha across the frozen tested grid; the selected
alpha was 1000.0; 1000.0 was the maximum frozen grid value; validation R² at the selected alpha was
−0.014630115836128832.

**The monotone pattern may not be read as** "the fits are pure noise", "the model is fitting noise",
"the optimum is above 1000", "R² would approach zero for larger alpha", or any claim about behaviour
outside the frozen grid.

**Not claimed, and not supported:** that Geneformer does not encode sampling depth; that `z` contains
no depth information; that sampling depth is absent from `z`; that a competing representation
explanation has been eliminated; that the external-transfer mechanism is known; that sampling depth
causally explains external failure; or that D-MECH supports or weakens `rho(n_cells, y)`.

## Boundaries

D-MECH does **not** negate H-FM4. H-FM4 and D-MECH probe different quantities and may not be combined
into an unregistered mechanistic inference.

The stored descriptive association `rho(n_cells, y) = +0.7045` was **not** tested by D-MECH, is not
recomputed here, and may not be strengthened or weakened by this result. No causal composition in
either direction.

## Integrity reconciliation (from the persisted run manifest)

```
13 guards PASS  (Guard A pipeline-refusal + Guard B direct-instrumentation positive controls)
RIDGE_FIT             expected 21 / observed 21 / MATCH      (7 alphas x 3 repeats, derived)
Z_GUARDED_LOADS       expected 96 / observed 96 / MATCH      (derived from the manifest line count)
TARGET_GUARDED_READS  expected  1 / observed  1 / MATCH
SCIENTIFIC_OPENS_OUTSIDE_BOUND_ROOT = 0
STANDARD_SCALER at project import 0/0 · before freeze 0/0 · synthetic guards 1/1 · REAL RUN 0/0
counter-object continuity verified; one pre-freeze reset; no post-freeze reset or reinstall
DMECH_REFIT_ON_COMBINED_TRAIN_VALIDATION = NO
```

## R² implementation provenance

```
R2_IMPL_PATH                     = iclr2027_revision/pipeline/phase1_harness.py
R2_IMPL_SHA256 (governed FILE)   = 68223292db4103a5c56c6a5b5278ccdad4737a6044a3fca282b19f0b865121f5
R2_IMPL_FUNCTION_SOURCE_SHA256   = 84c5df139b6d3a2ded8c8e0d372eadcead8389daf56f77c3cbf975abae99d808
R2_IMPL_BOUND_BY                 = VERBATIM_FUNCTION_SOURCE_FROM_SHA_VERIFIED_FILE_MODULE_NOT_IMPORTED
R2_IMPL_FUNCTION_SOURCE_HASH_STATUS =
    ESTABLISHED_DURING_PRE_RESULT_EXECUTION_IMPLEMENTATION_CLOSURE
R2_IMPL_FUNCTION_SOURCE_HASH_WAS_ORIGINAL_PREREG_AUTHORITY = NO
```

These two digests are **different things and must not be conflated**. The *file* digest is the
governed authority pinned in A11 §10.1. The *function-source* digest certifies the exact
`r2_pooled` source text that was extracted and executed; it was established during pre-result
execution-implementation closure and was **never** frozen in the original A11 design lock.

`phase1_harness` is not importable in the locked Phase-B environment (pandas, `iga_paths` and
`src.training` are absent from `phaseB_requirements_locked.txt`). The runner therefore extracted the
function source programmatically from the SHA-verified file — the same mechanism the executed H-FM4
runner used (`run_hfm4_tv.py:114-115`, "phase1_harness.py:92-95 verbatim").

## Timestamp discrepancy — recorded, not rewritten

```
RUN_ROOT_CREATION_TIMESTAMP      = 2026-08-29T21:23:39Z   (inode birth)
RUN_ROOT_BASENAME_TOKEN          = 20260829T212338Z
PRE_RUN_FREEZE_TIMESTAMP         = 20260829T212339Z       (also in results and run manifest)
TIMESTAMP_DISCREPANCY_SECONDS    = 1
```

`TIMESTAMP_DISCREPANCY_EXPLANATION` — these were **two separate clock captures, not one derivation**.
The launching shell captured a UTC timestamp to name the run directory; the runner then captured its
own UTC timestamp at process start, one second later, and that second value is what appears in the
freeze, the results and the run manifest. The run root was **not** renamed and no artifact was
rewritten to force agreement.

## Archive contents

| file | role |
|---|---|
| `../../pipeline/run_dmech_tv.py` | **A.** the executed runner, committed **verbatim** (not sanitized) |
| `dmech_results.json` / `.csv` | **J.** full alpha score table and consumed result (byte-identical to source) |
| `DMECH_PRE_RUN_FREEZE.sanitized.json` / `.md` | **C.** pre-run freeze summary |
| `dmech_guard_tests.sanitized.json` | **D.** guard and counter reconciliation |
| `dmech_input_access_manifest.sanitized.json` | **E.** scientific-input access summary |
| `dmech_environment.sanitized.json` | **F.** environment / reproducibility summary |
| `dmech_run_manifest.sanitized.json` | **B.** execution summary |
| `dmech_execution.sanitized.log` | chronological execution record |
| `dmech_external_artifact_manifest.json` | **G.** archival manifest: which bytes are original external execution bytes and which repository files are derived |
| `dmech_abort_ledger.json` | **H.** the two pre-result aborts |
| `dmech_sanitization_ledger.json` | **§11.** per-line sanitization ledger with invariance proofs |

`dmech_resolved_paths.json` is **external-only** and is deliberately **not** copied here; only its
digest `236b4f5b239e9135b972f885ba8b3ccdcf432805d2c0ec16547843cadd66cf2f` is recorded.

## Sanitization

Sanitization applies only to **derived** repository summaries — never to the executed runner. Only
absolute filesystem paths and the local username were replaced. Every changed line carries a
residual-equality proof: deleting each source string from the original and each replacement token
from the sanitized line leaves byte-identical text, so nothing outside the path/identity substitution
moved. `NO_NUMERIC_OR_HASH_LINE_ALTERED = YES` for all nine derived artifacts.

Original identifying strings are deliberately **absent** from the repository ledger; each is attested
by `SOURCE_LINE_SHA256`, and in full only by the external exact-diff artifact
`8ad41dfa30831be5701bcf08b78f5424aa4b62d19a382f6f6ad83401612cbe0c`, which is never committed.

## Release-review items

The executed runner necessarily embeds absolute paths — the governed Project-A exclusion list it
enforces. Those bytes are provenance evidence attested by `DMECH_RUNNER_SHA256`; rewriting them would
destroy the byte-level link to the freeze, so they are **recorded, not sanitized**:

- `run_dmech_tv.py:181-184` — the four governed Project-A roots in `A_ROOTS`
- `run_dmech_tv.py:237` — a Project-A path used by the boundary-rejection guard

These carry the local username and mount layout and must be reviewed before any public or
double-blind release. Sanitization at release time must preserve every content-identity hash and must
not retroactively alter the execution record. Broader Git author-history anonymization is a separate
matter and is not addressed here.
