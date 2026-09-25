<!--
POST-HOC ANONYMIZED SUPPLEMENTARY DERIVATIVE.
Derivative creation date: 2026-08-29
Canonical source: provenance/37_29_reconstruction_minispec.md
Canonical source SHA-256: ea20d6016818a8a7ac984cac91146f14f2aa82115ebd8289a54477fac8a43b3e
Local path and machine identifiers have been redacted.
The canonical project artifact, not this derivative, is the frozen provenance record.
Scientific counts, matching rule, gates, execution count, timestamps and the frozen
script SHA-256 are unaltered.
-->

# 37/29 Counterfactual Exclusion Reconstruction Mini-Spec

**Status: FROZEN FOR ONE-SHOT EXECUTION**

## 1. Purpose

Reconstruct the already-reported counterfactual metadata-only exclusion counts —
**RPE1: 37 constructs**, **K562-essential: 29 constructs** — without model execution,
retraining, prediction generation, metric recomputation, external outcome access, or GPU use.

**37 and 29 are expected historical values. They are NOT tuning targets and NOT acceptance
targets.** The scientific/set algorithm is frozen before execution. It was derived from the
historical rule recovered independently from executed R6 Task-B code, and validated against the
recorded 43/33 historical overlap counts, which it reproduced on first application without tuning.

## 2. RAW150 authority

`adata_Training.h5ad` `obs/target_gene`, excluding the exact historical non-targeting literal.
Governing definition: A7 §38-40 and lock §25 / §31 M1.

Path: `[STORAGE-ROOT]/[LEGACY-PROJECT-A]/data/vcc/train/adata_Training.h5ad`

## 3. RAW150 CSV equivalence

`pert_counts_Training.csv` (`target_gene` column only) was verified set-equal to the h5ad obs set:
RAW150_OBS = 150, RAW150_CSV = 150, intersection = 150, OBS_ONLY = [], CSV_ONLY = [], SET_EQUAL = YES.
The CSV is a corroborating cross-check; the h5ad remains the authority.

Path: `[STORAGE-ROOT]/[LEGACY-PROJECT-A]/data/vcc/train/pert_counts_Training.csv`
SHA-256: `633d202be221418bdbac16efd8cb169666de0ef6469a4e9bfdfb64666ea81a89`

## 4. Exact historical normalization rule

Recovered from executed R6 Task-B code (session record `78566c4b-1c24-4dbb-8b67-68366f6ab645.jsonl`,
entries `2026-08-26T22:14:01.775Z`, `22:14:25.261Z`, `22:15:50.309Z`):

- category strings taken as stored; bytes decoded only where necessary
  (`c.decode() if isinstance(c, bytes) else c`);
- membership by **exact Python string equality**;
- **no** whitespace stripping;
- **no** case normalization;
- **no** alias or synonym mapping;
- **no** Ensembl remapping (`target_ensg` is not used for membership);
- **no** punctuation normalization.

## 5. Exact NT rule

Exact, case-sensitive equality to the single literal `"non-targeting"`. No other token
(`NT`, `NTC`, `nan`, `non_targeting`) is treated as non-targeting. Removal is applied during set
construction, i.e. after per-row values are materialised and before any membership test.

## 6. Counting unit

**Constructs** — external endpoint rows. Gene counts are reported separately via `set()`, exactly
as the historical code did.

## 7. Duplicate policy

Constructs are counted individually and are **not** de-duplicated by gene. Where several constructs
map to one gene, each construct contributes to the construct count.

## 8. D10 and ELIGIBLE140

D10 = the ten target genes removed by the `min_cells_per_group = 10` eligibility floor, named in
`experiment_specs/implementation_preflight.md` Section G:

    ATP6V0C, BRD9, DNAJA3, EPHB4, FDPS, OXA1L, RNF20, SALL4, SLC39A6, TAF13

ELIGIBLE140 := RAW150 \ D10, with `|D10| = 10` and `|ELIGIBLE140| = 140` asserted.

`experiment_specs/implementation_preflight.md`
SHA-256: `468e5d97aca4ac306d5ae102aa510eda29f28a5d06d5ffbe1440a56341e4bc44`

## 9. Safe endpoint identifier projections

External endpoint CSVs are read with an identifier-only allowlist applied **at parse time**:
`usecols=['construct','target_symbol']`, `dtype=str`. A broad read followed by dropping columns is
prohibited. Column order is fixed to `['construct','target_symbol']`; source row order is preserved;
no sorting; no normalization; null values in either column are rejected.

Canonical projection digest definition:

    rows = [[construct_1, target_symbol_1], [construct_2, target_symbol_2], ...]
    blob = json.dumps(rows, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    digest = sha256(blob)

## 10. Identifier-projection SHA-256 values

    RPE1_ROWS                            = 1932
    RPE1_IDENTIFIER_PROJECTION_SHA256    = a245bf59071eb9526bc164c45f8deda04012297ca7df972d607ee047cb45b863

    K562_ROWS                            = 1903
    K562_IDENTIFIER_PROJECTION_SHA256    = 79436dd748851bd95edbc3bff7acf2e64c5fa89192c38d610790b05680f4bed3

## 11. Historical full-endpoint SHA-256 values — historical provenance only

    RPE1            353c04d3ac8e3380a3addf3abc1953d30799423f7cd467006adc36800e26a126
    K562-essential  051269eeeae93c75c84ca1fc050529a096ebd15e1990cb14bb24669c1b9672ff

**FULL ENDPOINT DIGEST RECOMPUTED DURING THIS RECONSTRUCTION: NO.**
These are recorded as historical provenance context only. The reconstruction never reads either
endpoint file in full and never recomputes these digests. Integrity of the consumed data is
established by the identifier-projection digests of Section 10 instead.

## 12. H5AD_METADATA_STRUCTURAL_FINGERPRINT

No whole-file cryptographic digest exists for `adata_Training.h5ad` (15,482,497,461 bytes); none is
recorded in `_SOURCE_MANIFEST_sha256.txt` or any supplement record, and none is computed here.

In its place the following structural fingerprint over `obs/target_gene` metadata is asserted:

    len(categories)            == 151
    number of distinct codes   == 151   (all categories used)
    count of "non-targeting"   == 1     (exactly one NT category)
    |RAW150|                   == 150

**This structural fingerprint is NOT equivalent to a whole-file cryptographic digest.** It
constrains only the obs metadata actually consumed.

## 13. Expected historical validation values

    RPE1            43 constructs / 42 genes
    K562-essential  33 constructs / 32 genes

## 14. Expected historical counterfactual values

    RPE1            37 constructs
    K562-essential  29 constructs

## 15. Statement on 37/29

37 and 29 are **expected historical values recorded in the manuscript and audit record**. They are
**not tuning targets** and **not acceptance criteria**. The algorithm is fixed before execution and
no branch is conditioned on producing them. The script derives the counts independently and then
compares.

## 16. Global historical validation gate

The script operates in three phases.

- **PHASE A** — load safe metadata and identifier projections.
- **PHASE B** — compute RAW150 historical overlaps for **both** screens and validate all four
  historical values (43c/42g and 33c/32g). Only when all four pass is
  `GLOBAL_HISTORICAL_VALIDATION = PASS` set.
- **PHASE C** — only after PHASE B passes may any ELIGIBLE140 membership mask or counterfactual
  count be computed. No RPE1 counterfactual calculation may occur before K562 historical validation
  has passed.

## 17. Stop-on-any-mismatch behavior

Any failed gate — pinned hash, structural fingerprint, negative categorical codes, CSV set
equivalence, D10 subset, cardinalities, schema, row counts, nulls, identifier-projection digests, or
any of the four historical values — terminates the run immediately with a nonzero exit and no
counterfactual output.

If the counterfactual counts differ from 37/29, the discrepancy is recorded in the provenance
artifact and the script exits nonzero. **No preprocessing, normalization, set-definition, NT,
duplicate, or counting-rule change is authorized in response.**

## 18-21. Safety invariants

- **No model execution.** The script imports no modeling or training library and invokes no project
  model, training, or prediction-pipeline module.
- **No external outcome column access.** Forbidden columns are never requested:
  `y_primary, y_mean, y_std, y_p5, y_p95, y_iqr, y_pooled, y_gemgroup, stored_ad, stored_ad_log1p,
  n_valid_draws, n_eligible_gemgroups, n_perturbed_pool, n_nt_pool, target_col, target_ensg`.
- **No external outcome value access.** No outcome value is read, stored, printed, or persisted.
- **No GPU.** `CUDA_VISIBLE_DEVICES` is forced empty before any import.
- Only `obs/target_gene/{categories,codes}` is read from the h5ad. `X`, `layers`, `raw`, `obsm`,
  `varm`, `obsp`, `varp` are never opened.

## 22. Persistence paths

    iclr2027_revision/provenance/37_29_reconstruction_minispec.md   (this document)
    iclr2027_revision/provenance/reconstruct_nonoverlap_counts.py   (frozen script)
    iclr2027_revision/provenance/37_29_PRE_RUN_FREEZE.json          (pre-run freeze record)
    iclr2027_revision/provenance/37_29_reconstruction.json          (result artifact)
    iclr2027_revision/provenance/37_29_reconstruction.md            (human-readable report)

## 23. One-shot execution rule

**EXECUTION AUTHORIZED: YES — ONE SHOT ONLY**

Authorized execution count: **1**. The mini-spec and script are hash-frozen in
`37_29_PRE_RUN_FREEZE.json` before invocation, and both hashes are re-asserted immediately before
execution. Neither file may be modified after the freeze.

Any nonzero reconstruction result terminates this execution attempt. No preprocessing,
normalization, set-definition, or counting-rule change is authorized in response.
