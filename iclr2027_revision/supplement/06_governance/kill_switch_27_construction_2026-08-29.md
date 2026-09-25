<!-- DERIVED SUPPLEMENTARY COPY. Not the contemporaneous original.
     Derived-copy creation date: 2026-08-29
     Canonical source: kill_switch_27_construction_2026-08-29.md
     Canonical source SHA-256: 05e7dd3c8f4dee09c53c6658726143ded12f9bb8c0bfc28a39ae981d1af5f803
     Anonymised: machine-local absolute paths replaced by bracketed tokens;
     author name and feature-pipeline name replaced by bracketed tokens.
     The canonical project artifact, not this copy, is the record of provenance. -->

# §27 Kill-switch — Author construction of "validated embeddings"

**Record date: 2026-08-29.** Created before the 2026-09-01 hard stop and before any further
Identification-Ladder external analysis.

**Repository state at creation:** branch `iclr2027-r2`, HEAD
`4703e8a48799f4235587a35d6be5e0aabcc7d52f`.

**Status of this document.** This is an *author construction record*. It is **not** an amendment. It
does **not** modify §27, §10.2, or any governing lock. It records, contemporaneously and before the
deadline, how the author construes an undefined term that the governing record uses but never defines.

---

## 1. The original clause — verbatim, four mirrored locations

The clause exists in four places. The wording below is reproduced exactly as written; nothing has been
altered or normalised.

### 1.1 Lock §27 — `experiment_specs/option_b_geneformer_experiment_lock.md:499-504`

> ## 27. Kill switch `[locked]`
>
> **Hard stop 2026-09-01.** If validated embeddings for VCC, RPE1, and K562-essential do not all exist by
> then, **cut Option B** and submit the strengthened paper without it. On cutting, do **not**: report a
> partial external screen; change the model after a failed extraction; relax eligibility; substitute
> K562-GWPS; swap FMs; drop the RPE1 43; or renegotiate the primary after partial results.

### 1.2 Lock §27, supplement copy — `supplement/01_experiment_lock.md:499-504`

Verified **byte-identical** to §1.1 by direct diff of both line ranges (empty diff). The two lock copies
differ elsewhere only by anonymisation (`[FEATURE-PIPELINE]` → `[FEATURE-PIPELINE]`); §27 contains no such token.

### 1.3 part1_scientific §10.2 — `experiment_specs/part1_scientific.md:345-353`

> ### 10.2 Kill switch
>
> **Hard stop: 1 September 2026.** If validated embeddings for VCC, RPE1, and K562-essential do not all
> exist by then, **cut Option B** and submit the strengthened paper without it. This is a pre-specified
> fallback and it is not renegotiable at the deadline.
>
> On cutting, do **not**: report a partial external screen; change the model after a failed extraction;
> relax construct eligibility; substitute K562-GWPS for K562-essential; swap in a different foundation
> model; or renegotiate the primary hypotheses after seeing partial results.

### 1.4 part1_scientific §10.2, supplement mirror — `supplement/part1_scientific.md:351-359`

Same text as §1.3, offset by six lines.

### 1.5 Precedence

Per the project governance statement recorded in `CLAUDE.md` and in the lock's own header:
`part1_scientific.md` is the **governing scientific specification**, and where the lock and that file
could be read as differing, **`part1_scientific.md` governs**. For the kill switch this ordering changes
nothing about the triggering condition — §27 and §10.2 state the same condition in the same terms — but
it does matter for the post-cut prohibition list, which differs between them (see §9 below).

---

## 2. The governance gap

Stated plainly, as facts about the record:

- **"validated embeddings" is used in the governing record.** It appears in the lock (`[locked]` §27, both
  copies) and in the governing scientific specification (§10.2, both copies).
- **The term is not operationally defined anywhere in the governing record.** An exhaustive search of
  `experiment_specs/`, `supplement/` and `provenance/` returns the term only in the four mirrored copies of
  this one clause. Every other occurrence of the word "validated" in the corpus is ordinary-English usage
  applied to a *different* object — SPSM hyperparameters (`option_b_geneformer_experiment_lock.md:156`), the
  Phase-3 runner's crash tests (`phase3_launch_and_monitoring.md:3`), the NT convention
  (`lock_amendment_A3.md:111`), the re-aggregation oracle
  (`matched_sensitivity_and_vcc_diagnostic.md:27`). None of these is attached to §27 and none defines the term.
- **No governing authority equates file existence alone with validation.** The lock attaches QC to each
  embedding-producing phase — §28 `[locked]` (lines 513-521): Phase 1 VCC "parity/QC", Phase 2 RPE1
  "QC only", Phase 3 K562-essential "QC only" — but enumerates no acceptance criteria and names no
  verdict field. §34 gives per-verdict falsifiers, not a validation standard. §35 lists a "VCC bridge/QC
  report", "RPE1 final `z`" and "K562-essential final `z`" as deliverables, not as criteria.
- **The read-only audit conducted on 2026-08-29 returned:** `VALIDATED_EMBEDDING_DEFINITION = NOT_DEFINED`,
  and consequently `KILL_SWITCH_CONDITION_MET = UNDETERMINED` with `AUTHOR_DECISION_REQUIRED = YES`.

**This record does not supply a retroactive definition and does not claim that the following construction
existed in the original lock.**

**This construction is being recorded before the 2026-09-01 deadline and before any further
Identification-Ladder external analysis.**

---

## 3. Author construction

```
AUTHOR_CONSTRUCTION_OF_VALIDATED =

"For purposes of applying §27, I construe 'validated embedding' to mean that
the required embedding artifact existed before the deadline and that the
contemporaneous phase-specific record documented successful QC/verification of
that embedding. Mere file existence alone is not sufficient. This construction
is stated on 2026-08-29 because the original governing record used the term
'validated' without defining it; it is not asserted to have been part of the
original lock."
```

```
AUTHOR_CONSTRUCTION_FAILURE_MODE =

"Under this construction, a required embedding would NOT be validated if any of
the following held: the artifact did not exist before the deadline; no
contemporaneous phase-specific record documented QC or verification of it; the
contemporaneous record documented a QC failure, an unresolved defect, or a
withheld verdict; or the record documented QC of a different artifact than the
one required by §27. Existence without a contemporaneous QC record is expressly
insufficient."
```

```
AUTHOR_CONSTRUCTION_ORIGIN = STATED_BY_AUTHOR_2026-08-29_NOT_PART_OF_ORIGINAL_LOCK
AUTHOR_CONSTRUCTION_SCOPE  = SECTION27_ONLY
```

The criterion is capable of returning **YES**, **NO**, or **UNDETERMINED** for each required embedding:
YES when existence and a passing contemporaneous QC record are both documented and no failure mode holds;
NO when any failure mode is affirmatively established; UNDETERMINED when the record is silent or
insufficient on a necessary element.

This construction is **not** generalised to any other provenance or scientific claim in this project. It
governs the reading of §27 and nothing else.

---

## 4. VCC — evidence and evaluation

### 4.1 Governing evidence

`experiment_specs/option_b_geneformer_experiment_lock.md:209-211`, §7 `[M]` — "VCC `z` — reuse, no
re-extraction":

> Locked/verified (ICLR-L3 §A, A2): raw 768-D; **11,764 / 11,764** manuscript rows join on
> `(split,batch,target_gene)`; per-row perturbed and NT **cell sets identical** on both sides (exact count
> agreement 11,764/11,764, no subsampling either side); all `z` finite; **no Geneformer re-extraction**.

Mirrored in the governing scientific specification at `experiment_specs/part1_scientific.md:424-425`:

> - **VCC bridge:** 11,764 / 11,764 rows join on `(split, batch, target_gene)`; per-row perturbed and NT
>   cell sets identical on both sides; all `z` finite; no re-extraction.

The four §7 facts required by this section are therefore all present in the governing record: the
11,764/11,764 manuscript-row join; exact perturbed/NT cell-set agreement; finite `z`; and no Geneformer
re-extraction.

### 4.2 Bridge re-verification evidence

`experiment_specs/phase1_vcc.md:191-195` records that the Phase-0 bridge carried the wrong label column and
was rebuilt and re-verified:

> - **L9 bridge corrected.** `phase0/bridge/vcc_z_label_bridge.parquet` carried `y_ad_distance_log1p`
>   (= `log1p(ad_stat_mean)` = the transcriptome-wide **breadth**, forbidden). Rebuilt with the correct
>   label **`shard.y = log1p(anderson_darling_counts)`**, re-verified: **11,764/11,764 matched, 0
>   duplicate keys, z all-finite, splits 5,740/2,052/3,972**, label column now
>   `y_shard_log1p_ADcounts`. `L9_BRIDGE_ARTIFACT_STATUS = corrected`.

`phase1_vcc.md:196-197` records the containment of that defect: `LABEL_BLAST_RADIUS` — "**only the bridge
embedded `y`** (now corrected)", with the `z`, C1/C2 scalers, projection matrix and magnitude blocks each
inspected and verified label-free.

Independent re-aggregation check, `experiment_specs/matched_sensitivity_and_vcc_diagnostic.md:33`: VCC
z re-aggregation oracle **EXACT (0.0)** over 350 constructs, my code path vs persisted `z`.

### 4.3 Recorded artifact path and already-recorded digest

| item | value | source |
|---|---|---|
| path | `iclr2027_revision/phase0/bridge/vcc_z_label_bridge.parquet` | `phase0_completion.md:98`; `hfm4_execution_note.md:93` |
| size | 64,209,931 B | `hfm4_execution_note.md:93` |
| SHA-256 | `396505130528572f561a31b8bfe20bd27f28a8ed65efc8398d796a7f6ce95416` | `hfm4_execution_note.md:93`; pinned again at `lock_amendment_A11_dmech_train_validation.md:606` |
| shape | 11,764 × 772 (768 `gf_z_emb_*` + 3 key columns + label) | `hfm4_execution_note.md:93` |

Digest **not recomputed in this task**; quoted from the pre-existing record. Upstream per-cell source
recorded as [FEATURE-PIPELINE] `full_feature_extraction_v2` (`matched_sensitivity_and_vcc_diagnostic.md:33`), reused
under §7 with no re-extraction.

Existence confirmed 2026-08-29 by **metadata only** (`stat`; no content read, no hashing): present,
64,209,931 B, mtime `2026-08-12 20:57:33 -0400` — consistent with the post-correction rebuild described at
`phase1_vcc.md:191-195`, and before the deadline.

### 4.4 Failure-mode evaluation

```
VCC_ARTIFACT_ABSENT_BEFORE_DEADLINE   = NO
VCC_NO_CONTEMPORANEOUS_QC_RECORD      = NO
VCC_QC_FAILURE_OR_UNRESOLVED_DEFECT   = NO
VCC_QC_OF_DIFFERENT_REQUIRED_ARTIFACT = NO
```

- **Absent before deadline — NO.** Artifact present, size matching the pinned record, mtime 2026-08-12,
  hash-pinned twice pre-deadline (`hfm4_execution_note.md:93`; `A11:606`).
- **No contemporaneous QC record — NO.** Two independent records exist: the governing §7 "Locked/verified"
  block (`:209-211`) covering the source `z`, and the Phase-1 record's re-verification of the rebuilt
  operative artifact (`phase1_vcc.md:193-195`).
- **QC failure / unresolved defect / withheld verdict — NO, with the defect stated explicitly.** A real
  defect *was* documented: the L9 bridge initially carried the forbidden breadth label. Under this
  construction the disqualifying condition is an **unresolved** defect. This one was resolved — the bridge
  was rebuilt, re-verified (11,764/11,764 matched, 0 duplicate keys, z all-finite), its status recorded as
  `corrected`, and its blast radius audited to a single artifact. The defect concerned the **label column,
  not `z`**; the re-verification explicitly reconfirms `z` all-finite on the rebuilt artifact. No verdict
  was withheld.
- **QC of a different required artifact — NO.** §27 requires a validated VCC embedding. §7's QC is of the
  VCC `z` on the 11,764 manuscript rows; the operative bridge carries those same 768 `gf_z_emb_*` columns
  keyed to the same rows, and was itself re-verified after rebuild.

```
VCC_SECTION27_VALIDATED = YES
```

Evidence: `option_b_geneformer_experiment_lock.md:209-211`; `part1_scientific.md:424-425`;
`phase1_vcc.md:191-197`; `matched_sensitivity_and_vcc_diagnostic.md:33`; `hfm4_execution_note.md:93`.

---

## 5. RPE1 — evidence and evaluation

### 5.1 Phase-2 QC record

`experiment_specs/phase2_rpe1_topup_and_xcell.md:100-106`, "Part J — QC (all label-free, all pass)":

> - `z_RPE1` shape **(1,932, 768)**, float32, all finite, 0 all-zero rows.
> - `x_cell_RPE1` shape **(1,932, 2,000)**, float32, all finite, **all within ±10**, 0 all-zero rows.
> - Construct id sets of `z_RPE1` and `x_cell_RPE1` are **identical to each other and to the manuscript's 1,932**
>   (both directions). None of [FEATURE-PIPELINE]'s 412 extra constructs appears.
> - Per-construct cell counts equal the **raw obs** counts for both arms

Recorded status block, `phase2_rpe1_topup_and_xcell.md`:

| field | value | line |
|---|---|---|
| `CONSTRUCT_SET_ASSERTIONS` | `PASS (1932 / 1889 / 43 / 412, set equality)` | 147 |
| `Z_REAGG_ORACLE` | `EXACT` | 158 |
| `Z_REAGG_ORACLE_MAX_ABS_DIFF_RAW` | `0.0 (float32 aggregation)` | 159 |
| `Z_REAGG_ORACLE_MAX_ABS_DIFF_POSTCAST` | `0.0 (1,450,752/1,450,752 bitwise-equal)` | 160 |
| `NT_EMB_ORACLE` | `EXACT` (max abs diff `0.0`) | 161-162 |
| `Z_RPE1_SHAPE` | `(1932, 768)` | 169 |
| `CONSTRUCT_SETS_MATCH` | `YES (z, x_cell, manuscript 1932 identical)` | 171 |
| `CELL_COUNTS_MATCH` | `YES (both paths == raw obs per construct)` | 172 |
| `ALL_FINITE` | `YES` | 173 |

Overall verdict, `phase2_rpe1_topup_and_xcell.md:138-140`:

> `PHASE3_READY = YES` — all conditions hold: Z_REAGG_ORACLE EXACT, NT_EMB_ORACLE EXACT, SAME_PASS_VERIFIED YES,
> NT_POOL_IDENTICAL YES, all Part J assertions pass, checkpoint written to SSD + loaded, [FEATURE-PIPELINE]_WRITES NONE, no
> label/model-fit, updated gate GO.

Corroborating independent check, `matched_sensitivity_and_vcc_diagnostic.md:32`: RPE1 z re-aggregation
oracle **EXACT (0.0)** for the 1,889 cached constructs, **1.9e-6** for the 43 top-up constructs (float32
accumulation).

### 5.2 Recorded artifact path and already-recorded digest

| item | value | source |
|---|---|---|
| path | `[LEGACY-EMBEDDING-ROOT]/rpe1_topup/z_rpe1_1932.npy` | `R2_REPORT.md:116` |
| size | 5,935,232 B | `R2_REPORT.md:116` |
| SHA-256 | `f115a777bd78f644219189b6589086978c2b57fd51b32d9ece39c5ba42fdd904` | `R2_REPORT.md:116` |
| shape | (1932, 768) f32 | `R2_REPORT.md:116` |

Digest **not recomputed in this task**. Existence confirmed 2026-08-29 by metadata only: present,
5,935,232 B, mtime `2026-08-10 23:43:01 -0400` — before the deadline. A second copy exists inside the
author-confirmed Project-B storage root at
`[PROJECT-B-DATA-ROOT]/[LEGACY-EMBEDDING-ROOT]/rpe1_topup/z_rpe1_1932.npy`
with identical size and mtime. **Neither copy was hashed in this task**, so byte-identity between the two
copies is not established here; it is recorded as an observation, not a claim.

### 5.3 Failure-mode evaluation

```
RPE1_ARTIFACT_ABSENT_BEFORE_DEADLINE   = NO
RPE1_NO_CONTEMPORANEOUS_QC_RECORD      = NO
RPE1_QC_FAILURE_OR_UNRESOLVED_DEFECT   = NO
RPE1_QC_OF_DIFFERENT_REQUIRED_ARTIFACT = NO
```

- **Absent before deadline — NO.** Present, mtime 2026-08-10, size matching the recorded 5,935,232 B.
- **No contemporaneous QC record — NO.** `phase2_rpe1_topup_and_xcell.md` is the Phase-2 session record and
  carries a complete Part-J QC block plus a machine-readable status block.
- **QC failure / unresolved defect / withheld verdict — NO.** Every Part-J assertion passes; both oracles
  are EXACT; an affirmative verdict token `PHASE3_READY = YES` is recorded. One deviation *is* recorded and
  is stated here for completeness: `phase2_rpe1_topup_and_xcell.md:10` records a **§33 ordering deviation,
  class `IMPLEMENTATION_CORRECTION`** — Phase 2 was run before Phase 1. That is a recorded and classified
  *ordering* deviation, not a QC failure and not a defect in the embedding; it does not bear on any Part-J
  assertion.
- **QC of a different required artifact — NO.** Part J QC is of `z_RPE1` at shape (1,932, 768) with
  construct-id sets identical to the manuscript's 1,932 — the object §27 and §23 require, including the 43.

```
RPE1_SECTION27_VALIDATED = YES
```

Evidence: `phase2_rpe1_topup_and_xcell.md:100-106`, `:138-140`, `:147`, `:158-162`, `:169-173`;
`matched_sensitivity_and_vcc_diagnostic.md:32`; `R2_REPORT.md:116`.

---

## 6. K562-essential — evidence and evaluation

K562-GWPS was **not** inspected, opened, considered, or substituted at any point in producing this record.

### 6.1 Phase-3 QC record

`experiment_specs/phase3_completion.md:1-8` establishes the record's method — independent recomputation,
with the runner's own self-report excluded as evidence:

> Independent verification of the Phase-3 K562-essential extraction. Every quantity below was
> **recomputed from the persisted files** and compared against its reference; the runner's own
> `[finalize]` line was treated as a self-report, not as evidence.

`phase3_completion.md:13-15`:

> **Verdict: ACCEPTED.** Every reference value reproduced; the re-aggregation oracle is **EXACT**
> (max|Δ| = 0.0); one recorded, benign deviation (the RPE1 *reference* data used in the Part-F
> construction check contains 2 non-finite native genes that map to none of the compared HVG columns).

`phase3_completion.md:74-75`:

> - `z_K562`: (1903, 768) float32; **all finite**; **0 all-zero rows**; **0 exactly-duplicated row
>   pairs** (a duplicate row would signal a construct-grouping bug — none).

Recorded status block, `phase3_completion.md`:

| field | value | line |
|---|---|---|
| `Z_SHAPE` | `(1903, 768) float32` | 220 |
| `DTYPES_MATCH_PHASE2` | `YES` | 222 |
| `ALL_FINITE` | `YES (z, x_cell mapped, x_cell native)` | 223 |
| `CONSTRUCT_SET_EQUALITY` | `PASS (1903 in / 1903 out; 0 symdiff both directions)` | 226 |
| `PARTITION_REDERIVED` | `PASS (236x1200 + 770; each cell once; union==283,970)` | 227 |
| `REAGG_ORACLE` | `EXACT` | 234 |
| `PHASE3_VERDICT` | `ACCEPTED` | 254 |
| `PHASE3_QC_COMPLETE` | (terminal token) | 258 |

Corroborating independent check, `matched_sensitivity_and_vcc_diagnostic.md:31`: K562-essential z
re-aggregation oracle **EXACT (0.0)** over 30 constructs.

### 6.2 Recorded artifact path and already-recorded digest

| item | value | source |
|---|---|---|
| path | `[LEGACY-EMBEDDING-ROOT]/k562_essential/z/z_k562_1903.npy` | `R2_REPORT.md:117` |
| size | 5,846,144 B | `R2_REPORT.md:117` |
| SHA-256 | `ce88160c8a525b3ea4608cfc8978ee0f71e24dde3d6fb0c5335f0bcf3f790a26` | `R2_REPORT.md:117` |
| shape | (1903, 768) f32 | `R2_REPORT.md:117` |

Digest **not recomputed in this task**. Existence confirmed 2026-08-29 by metadata only: present,
5,846,144 B, mtime `2026-08-11 11:47:44 -0400`. That mtime coincides with the Phase-3 finish timestamp
recorded at `phase3_completion.md:10-11` ("finished **2026-08-11T11:47:44**"). This is **corroboration of
contemporaneity only** — mtime is mutable metadata and is copy-dependent; it is **not** a hash binding and
is not offered as one. A second copy exists inside the Project-B storage root at
`[PROJECT-B-DATA-ROOT]/[LEGACY-EMBEDDING-ROOT]/k562_essential/z/z_k562_1903.npy`
with identical size and mtime; neither copy was hashed in this task.

### 6.3 Failure-mode evaluation

```
K562_ARTIFACT_ABSENT_BEFORE_DEADLINE   = NO
K562_NO_CONTEMPORANEOUS_QC_RECORD      = NO
K562_QC_FAILURE_OR_UNRESOLVED_DEFECT   = NO
K562_QC_OF_DIFFERENT_REQUIRED_ARTIFACT = NO
```

- **Absent before deadline — NO.** Present, mtime 2026-08-11, size matching the recorded 5,846,144 B.
- **No contemporaneous QC record — NO.** `phase3_completion.md` is a dedicated Phase-3 completion-QC
  record built by independent recomputation from the persisted files.
- **QC failure / unresolved defect / withheld verdict — NO, with the deviation stated explicitly.** The
  single recorded deviation (`:13-15`) is characterised in the record itself as **benign** and concerns the
  **RPE1 reference data** used in a construction cross-check — 2 non-finite native genes that map to **none**
  of the compared HVG columns — not the K562-essential `z`. `phase3_completion.md:170` records that
  "K562's normalized-bulk has 0 non-finite". An affirmative verdict is recorded (`ACCEPTED`,
  `PHASE3_QC_COMPLETE`); nothing was withheld.
- **QC of a different required artifact — NO.** The QC is of `z_K562` at (1903, 768) with construct-set
  equality against the 1,903 eligible constructs required by §24.

```
K562_ESSENTIAL_SECTION27_VALIDATED = YES
```

Evidence: `phase3_completion.md:1-8`, `:13-15`, `:74-75`, `:220-258`;
`matched_sensitivity_and_vcc_diagnostic.md:31`; `R2_REPORT.md:117`.

---

## 7. §27 determination, derived

Derived strictly from §§4-6 by the stated logic — all three YES ⇒ condition not met; any NO ⇒ condition
met; otherwise UNDETERMINED. The answer was not pre-committed.

```
SECTION27_REQUIRED_VALIDATED_EMBEDDINGS =

  VCC            = YES
  RPE1           = YES
  K562-essential = YES

KILL_SWITCH_CONDITION_MET = NO
OPTION_B_CUT              = NO
```

**Bounded reasoning.** All three required embeddings existed before the deadline, and for each one a
contemporaneous phase-specific record documents successful QC or verification with an affirmative verdict.
No failure mode under `AUTHOR_CONSTRUCTION_FAILURE_MODE` is established for any of the three. Three
anomalies appear in the record and are each stated above rather than omitted — the L9 bridge label defect
(documented and **corrected**; concerned the label column, not `z`), the §33 Phase-2/Phase-1 ordering
deviation (recorded and classified `IMPLEMENTATION_CORRECTION`), and the Part-F non-finite genes in the
RPE1 *reference* data (recorded as benign, and not in the K562-essential `z`). None is an unresolved
defect in a required embedding; none is a withheld verdict.

**This determination rests entirely on the construction stated in §3 of this document, which was stated by
the author on 2026-08-29 and is not asserted to have been part of the original lock.** Specifically, this
record does **not** state that §27 originally contained this definition; does **not** state that a
previously specified validation criterion was passed; does **not** state that all provenance gaps are
closed; and does **not** state that later-recorded hashes retroactively bind historical runs.

---

## 8. §27 vs §10.2 — text divergence

```
SECTION27_VS_SECTION10_2_DIVERGENCE =
The phrase "drop the RPE1 43" appears in §27 of the lock but is absent from
§10.2 of part1_scientific.md.
```

The lock's post-cut prohibition list contains seven items; §10.2's contains six. The missing item is the
RPE1-43 prohibition. The other six correspond, with §10.2 using slightly fuller wording ("relax construct
eligibility"; "substitute K562-GWPS for K562-essential"; "swap in a different foundation model";
"renegotiate the primary hypotheses after seeing partial results").

Independent standing constraint, `option_b_geneformer_experiment_lock.md:456`, §23 `[locked, measured]`,
verbatim:

> Primary includes all 1,932; **do not drop the 43** to save GPU.

```
COVERED_ELSEWHERE = YES — §23 independently prohibits dropping the RPE1 43.

STATUS =
The prohibition stands independently of §10.2's omission. This construction
record notes the divergence and does not amend, reconcile, or rewrite either
governing text.
```

No permission is inferred from the omission in §10.2. The §23 prohibition is unconditional and is not
scoped to the cut scenario, so it binds whether or not Option B is cut.

---

## 9. Open A17 — identity-binding gap

`R2_REPORT.md:120-122`, verbatim:

> **Stated plainly:** hashing these files today creates a **forward** provenance record only. It does **not**
> retroactively prove these specific arrays produced the reported correlations — nothing was recorded at run
> time binding them to the results. **This does not close A17.**

Substantive conclusions recorded here:

- The RPE1 and K562-essential external `z` hashes at `R2_REPORT.md:116-117` were recorded **after** the
  runs and provide **forward** provenance only.
- They do **not** retroactively prove that those exact byte-identical arrays generated the historical
  reported correlations.
- **A17 remains open.** This record does not resolve it and does not attempt to.
- **This §27 determination concerns existence plus recorded phase QC — not historical run-time hash
  binding.** The two questions are distinct, and a YES under §3's construction says nothing about A17.

```
FUTURE_RUNTIME_HASH_BINDING_REQUIREMENT =

Any prospective Identification-Ladder execution that uses these arrays must
record the exact input artifact hashes at execution/freeze time so that the new
analysis does not reproduce the same identity-binding gap.
```

This is a **forward governance requirement only**. It does not retroactively modify earlier runs, and it
does not by itself authorize any Ladder execution.

---

## 10. Deadline reading

§27 says "**by then**" without specifying an exact instant. The lock contains no time-of-day, no timezone,
and no boundary rule for the hard stop.

`iclr2027_revision/smoke/gate_manifest.json` contains the operational timestamp:

```
"kill_switch": "2026-09-01T00:00:00"
```

That timestamp is an **existing operationalization recorded in a session artifact** (the L6 smoke-test and
26B-gate manifest). It is **not** wording contained in §27 itself, and it is not part of any lock.

```
AUTHOR_DEADLINE_CONSTRUCTION        = 2026-09-01T00:00:00
AUTHOR_DEADLINE_CONSTRUCTION_ORIGIN = STATED_BY_AUTHOR_2026-08-29_USING_EXISTING_GATE_MANIFEST_OPERATIONALIZATION
```

**This construction record predates that timestamp.** It is dated 2026-08-29, three calendar days before
2026-09-01 and before the midnight instant adopted above. §27 is not rewritten.

---

## 11. Non-effects

This record does **NOT**:

- change scientific hypotheses;
- change eligibility;
- change the model;
- change the foundation model;
- substitute K562-GWPS;
- drop the RPE1 43;
- authorize a partial external screen;
- authorize new VCC-test scoring;
- authorize a new external-outcome look;
- authorize the Identification Ladder;
- change Phase-B preprocessing;
- alter historical results;
- alter historical locks;
- close A17;
- resolve unrelated VCC parquet bookkeeping discrepancies (specifically, the differing size/digest records
  at `implementation_preflight.md:115` — 132,134,861 B, sha `c531f392…` — versus `R2_REPORT.md:118` —
  468,781 B, sha `2e83f90e…` — for VCC row-feature parquets at two different paths; noted, unresolved,
  and outside this record's scope).

No governing lock file was modified in producing this record. §27 and §10.2 remain byte-identical in all
four locations.

---

## 12. Attestations for the session that produced this record

```
ORIGINAL_SECTION27_MODIFIED            = NO
VALIDATED_EMBEDDING_ORIGINALLY_DEFINED = NO
SCIENTIFIC_DESIGN_CHANGED              = NO
EXTERNAL_OUTCOME_VALUES_ACCESSED       = NO
VCC_TEST_OUTCOMES_ACCESSED             = NO
EXTERNAL_PREDICTIONS_ACCESSED          = NO
SCIENTIFIC_ARTIFACTS_HASHED            = NO
EMBEDDINGS_RECOMPUTED                  = NO
MODEL_EXECUTED                         = NO
IDENTIFICATION_LADDER_EXECUTED         = NO
GPU_USED                               = NO
MANUSCRIPT_CHANGED                     = NO
SUPPLEMENT_CHANGED                     = NO
PUSH_PERFORMED                         = NO
```

Artifact existence was established by filesystem **metadata only** (`stat`): no array was opened, no
content was read, and no digest was computed. All digests quoted in this record are transcribed from
pre-existing records at the cited file:line.
