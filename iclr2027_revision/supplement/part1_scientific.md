<!-- DERIVED SUPPLEMENTARY COPY. Not the contemporaneous original.
     Derived-copy creation date: 2026-08-29
     Canonical source: part1_scientific.md
     Canonical source SHA-256: 55f2f778563dedef1c487989d6d318a4b951867c1460eef35946cc148f1131df
     Anonymised: the feature-pipeline project name is rendered [FEATURE-PIPELINE],
     matching 01_experiment_lock.md. No clause, number or criterion differs.
     The canonical project artifact, not this copy, is the record of provenance. -->

# Option B — Scientific Specification (Part 1)

**Destination:** `iclr2027_revision/experiment_specs/part1_scientific.md`

**Status:** the engineering and provenance questions are closed by the ICLR-L2 / L3 / L3.5 audits.
This document fixes the scientific half — hypotheses, controls, decision rules, interpretation, and
reporting — which no audit covers, because it is not derivable from the repositories.

**How to use it:** this is the source of truth for *what the experiment is for*. The Experiment Lock
combines it with `prelock_final_decisions.md` and the four audit documents, which supply exact paths,
artifact hashes, code touchpoints, and extraction schedules. A lock written from the audits alone
would specify plumbing with no hypotheses attached.

**Nothing in this document may be changed once the first scientific result is computed.**

---

## 1. What is being tested

Whether the manuscript's magnitude–direction asymmetry — magnitude-only predictors transfer
positively across screens while expression-direction predictors do not — extends to a frozen
pretrained single-cell foundation-model representation.

Geneformer is the natural test case because its tokenizer retains only gene **rank order** after
per-cell count normalisation and gene-median scaling; it does not receive absolute per-cell
expression magnitude explicitly.

**Required wording discipline.** Never write "Geneformer is magnitude-blind." The licensed statement
is:

> Geneformer does not receive absolute per-cell expression magnitude explicitly; its tokenizer
> represents each cell through a rank-based encoding after per-cell count normalisation and gene-wise
> median scaling. A population-level embedding delta may nonetheless encode response-strength
> information through the detected-gene set and through rank displacement.

Whether it does is **H-FM4**, and the interpretation of every other result is conditional on it.

**Primary question, stated operationally:** does frozen Geneformer `z` carry predictive information
beyond a matched-capacity random control on VCC, and if so, does that information transfer to the two
external screens the way expression direction does, or the way magnitude does?

**Frozen representation only. No fine-tuning in the primary arm.** The question is what the
pretrained representation exposes; fine-tuning confounds that with head capacity. A short fine-tune
may appear as an appendix result only if time allows, and never as the primary.

---

## 2. Objects

Rows and labels are the manuscript's. Representation assets are [FEATURE-PIPELINE]'s, frozen. The bridge is new
code.

| object | definition | dim |
|---|---|---|
| `x` | manuscript expression features. **VCC:** log1p delta vs the per-batch NT mean. **External:** the distributed Replogle pseudobulk (`*_normalized_bulk_01.h5ad`), symbol-mapped into the VCC delta-HVG namespace, unmapped HVGs zero-imputed, clipped to ±10 | 2,000 |
| `m̃` | the four magnitude scalars of Eq. 1 — `mean\|x\|`, `‖x‖₂/√d`, `‖x‖∞`, `σ(x)` — z-scored on **training** statistics only, **recomputed over `cache/vcc/variants/delta_hvg/selected_genes.npy`** | 4 |
| `z` | frozen Geneformer CLS mean-embedding delta, `mean(E_pert) − mean(E_NT)`, **raw, never unit-normalised** in the primary | 768 |
| `x_cell` | **construction-matched comparator, external screens only**: `mean(log1p CP10K over all cells of the construct) − mean(over all global NT cells)`, then the manuscript's VCC-HVG mapping, zero-imputation, and clipping | 2,000 |

### 2.1 `z` construction — fixed

**VCC.** All cells with `(target_gene, batch)`, minimum 10 perturbed cells; NT reference =
**per-batch**. This matches the manuscript's `x` exactly: verified 11,764 / 11,764 rows joining on
`(split, batch, target_gene)` with per-row perturbed and NT cell counts agreeing on every row, no
subsampling on either side. Reuse the existing cached `z`; no re-extraction.

**External screens.** All cells of each manuscript-eligible construct — RPE1 **1,932**,
K562-essential **1,903** — minus **one global pooled NT mean embedding** over that screen's
non-targeting cells (RPE1 **11,485**, K562-essential **10,691**). No batch matching. **No 27/788
subsampling** — that rule is specific to the harmonized AD endpoint and does not apply to predictive
features. Deterministic means throughout.

No target-side fitting, calibration, or label-informed normalisation anywhere.

### 2.2 Two declarations that must appear in the paper

These must be stated as choices, not presented as inherited facts.

1. **The global pooled NT reference for `z` is a new preregistered design choice.** It is *supported
   by* the manuscript's own endpoint control policy — App. K pools controls globally as the primary
   policy, batch matching is computable for only 43 of 194 constructs, and the two agree at
   ρ = 0.91 (RPE1) / 0.97 (K562-essential) — and it *coincides with* [FEATURE-PIPELINE]'s precedent. It is **not**
   inherited from the manuscript's external predictive `x`, which carries no NT vector at all.

2. **On the external screens, `z` and `x` are not the same contrast.** On VCC they are: both are
   per-batch NT-subtracted deltas over identical cells. Externally, the manuscript's `x` is release
   pseudobulk carrying the release's own normalisation and no NT subtraction, while `z` is a
   perturbed-minus-NT delta we construct. `x_cell` exists precisely to close that gap, and the
   FM-versus-expression claim is made against `x_cell`, not against `x_release`, so that a
   construction mismatch cannot masquerade as a representation effect.

---

## 3. Evaluation

Taken from the manuscript's code unchanged. Nothing here is reimplemented.

- **In-distribution:** VCC held-out target genes. Pooled test R² with the **test-set mean**
  denominator, computed once over all 3,972 test rows. Spearman with average-tie ranks.
  (`src/metrics/predictive.py`.)
- **Transfer:** zero-shot from VCC to RPE1 and K562-essential, scored against the **harmonized
  target-gene endpoint** of Sec. 6.3 / App. K. **Spearman ρ is the criterion**; pooled R² is negative
  throughout by construction, may be reported descriptively, and is not a success criterion.
- **Uncertainty:** the manuscript's **paired cluster bootstrap resampling target genes**
  (`scripts/v4/harmonized_endpoint_v1/common.py`). Constructs sharing a target gene resample
  together. Replicate count, CI level, and seed are inherited from the manuscript lock and must be
  written explicitly into the Experiment Lock.
- **Verdict vocabulary,** as in Table 2: *resolved-positive*, *resolved-negative*, *unresolved*,
  according to whether the bootstrap interval excludes zero.
- **Seeds:** the frozen embedding is deterministic; variation arises only from head training. Report
  mean ± sample standard deviation, **ddof = 1**, consistent with the rest of the paper.

---

## 4. Head grid

The manuscript's own model families, so that the FM rows land on the same scale as the existing
tables: **Ridge**, **Random Forest**, **HistGradientBoosting**, and the **SPSM V4 backbone**. Each is
applied to `z` and to `[z; m̃]`.

**Hyperparameter grids are the manuscript's existing search spaces (App. F), enumerated verbatim in
the Lock. No widened search.** The identical search procedure is applied to every control.

**SPSM V4 adaptation — decided here, not at implementation time.** The backbone takes 2,004-d input
in the manuscript; `z` is 768-d and `[z; m̃]` is 772-d. **Only the input layer width is adapted.**
Depth, hidden widths, activations, LayerNorm placement, dropout, loss terms and their weights,
optimizer, learning rate, batch construction, and epoch count are unchanged from App. G. This
adaptation is a `NEW_PREREGISTERED_ICLR_CHOICE` and is stated in the paper. No other architectural
variant may be tried after results exist.

**Head selection is by validation R² only. No test metric may influence head selection.** The
selected configuration per feature set is frozen and recorded before any test number is computed.

---

## 5. Controls

The manuscript's signature control is that *four* i.i.d. Gaussian features do not reproduce the
magnitude gain. That says nothing about *768* of them. [FEATURE-PIPELINE]'s own pre-registered gate went INVALID
for exactly this reason — a 768-d Gaussian branch cleared its threshold over a 4-d baseline. **That
result is on a different endpoint and must not be cited in the paper**, but the design warning it
carries is endpoint-independent and is honoured here.

### 5.1 The gating floor

**C1 — 768 i.i.d. Gaussian features.** A true floor: it carries no information about anything.
Beating it answers *did the head learn anything from `z` at all, or is 768 dimensions by itself
enough to fit this target?* **C1 alone gates H-FM1.**

### 5.2 The compressed-expression benchmark — reported, not gating

**C2 — random projection of `x` → 768-d.** This is **not** a random control. By Johnson–Lindenstrauss,
a random 768-d projection of a 2,000-d vector preserves most of its geometry, so C2 is essentially
*`x` compressed* and should be nearly as predictive as `x` itself. It answers a different and much
harder question: *does FM pretraining buy anything over a naive linear compression of raw
expression?*

**C2 must never be used as the H-FM1 gate.** If `z` beat pure noise decisively but lost to compressed
`x`, gating on C2 would record H-FM1 as failed, render H-FM2 uninterpretable by its own conditionality
rule, and report the arm as "the FM failed" — when what actually happened is that it failed to beat a
strong expression baseline. That is a real and interesting finding, and it is reported as its own
result with its own interval; it is not the same finding, and it must not silently disable H-FM2.

### 5.3 Alignment controls

| control | question it answers |
|---|---|
| **C3** — row-shuffled `z` | does the `z` result require correct per-row identity, mirroring Sec. 5.2? |
| **C4** — `[z; row-shuffled m̃]` | does the `[z; m̃]` gain require correct per-row magnitude alignment? |
| **C5** — `[z; 4 i.i.d. Gaussian scalars]` | or is it simply four more dimensions? |

For every control the Lock fixes: seeds, generation procedure, whether generation is independent
within each split, train-time versus eval-time application for the shuffles, leakage prevention, and
an identical head-selection procedure to the corresponding primary arm. **No control may be
introduced after primary results are seen.**

---

## 6. Pre-registered hypotheses

### H-FM1 — in-distribution information

**Gate:** FM-only is *informative* if and only if the paired bootstrap interval on ΔR² versus **C1**
excludes zero on the positive side.

**Reported alongside, not gating:** ΔR² versus **C2**, with its interval, interpreted as the
compressed-expression benchmark of §5.2.

**Contextual reference points only**, not criteria: MAG-LINEAR ≈ **+0.208**, RF `[x; m̃]` ≈ **+0.370**,
MAG-SPSM ≈ **+0.225 ± 0.004**, SPSM ≈ **+0.082 ± 0.010**.

Success is explicitly **not** defined as "FM-only > MAG-LINEAR."

### H-FM2 — zero-shot transfer

**Pre-registered expectation:** FM-only behaves like the manuscript's expression-only group —
external Spearman **negative or unresolved in both** RPE1 and K562-essential.

**Conditional on H-FM1.** Interpretable only if H-FM1's C1 gate has established that `z` is
in-distribution-informative. A head that learned nothing yields ρ ≈ 0, which sits inside the
predicted band; "the FM representation does not transfer" and "our head never learned from `z`" must
not be confusable. If the C1 gate fails, near-zero external ρ **must not** be reported as a transfer
finding.

### H-FM3 — the magnitude increment

**Pre-registered expectation:** appending `m̃` to `z` improves transfer by a **resolved** margin in
**both** screens, mirroring SPSM → MAG-SPSM (Δρ = **+0.117** RPE1, **+0.086** K562-essential).

**Must survive C4 and C5.** If row-shuffled `m̃` or four i.i.d. Gaussian scalars reproduce the same
gain, no claim of per-row magnitude alignment may be made.

### H-FM4 — magnitude recoverability from `z`

**Probe form, decided here:** **four separate single-output Ridge probes**, one per magnitude scalar,
fit on the VCC **train** split. Rationale: the interpretation is per-component (is `mean|x|`
recoverable? is `‖x‖∞`?), separate probes allow per-component regularisation selection on VCC
validation, and a shared-penalty multi-output fit would conflate components. Regularisation is
selected on VCC validation only; VCC test is touched once.

**No absolute threshold.** Report each component against three references computed with the **same
probe family**:

- **Ceiling** — the same probe applied to `x`. Note `m̃` is a *nonlinear* function of `x`, so this
  ceiling will not be 1.0; that is the point of using a matched probe family.
- **Floor 1** — a random projection of `x` to 768-d.
- **Floor 2** — row-shuffled `z`.

The preregistered statement is *where `z` falls between the matched floors and the matched ceiling*,
per component. Report all four components; do not summarise to a single number and do not cherry-pick
the resolving component.

**Prior, stated in advance:** expect *partial* recoverability. [FEATURE-PIPELINE] treats `‖z‖` as a real signal and
deliberately unit-normalises `z` to isolate direction — direct evidence that `z` carries some
magnitude information. High recoverability is therefore neither a surprise nor a failure.

H-FM4 is mechanistic and supporting. It is not a new predictive target.

---

## 7. Interpretation table — fixed before any number exists

H-FM2 cannot be read independently of H-FM4. Every cell is a reportable outcome.

| | `m̃` **not** recoverable from `z` | `m̃` **is** recoverable from `z` |
|---|---|---|
| `z` transfers **poorly** | **A.** The asymmetry generalises: the frozen rank-based representation discards the transferable axis. *(the predicted cell)* Do not write "magnitude-blind." | **B.** Magnitude-related information is available but not exploited transferably. The claim narrows from *availability* to *exploitation*. |
| `z` transfers **well** | **C.** A genuine challenge to the headline — some axis other than magnitude transfers. Report as a limitation and refinement of contribution #1, not as a footnote. | **D.** The asymmetry is intact; rank displacement does retain amplitude information. The headline is **scoped**, not declared false. |

**E — H-FM1's C1 gate fails.** The chosen frozen representation and head did not establish usable
signal in-distribution. External near-zero results **cannot** be interpreted as biological transfer
failure. Report the negative in-distribution result honestly and stop there.

**F / G — the H-FM3 axis, read jointly with H-FM4.** If `[z; m̃]` improves over `z`, explicit
magnitude carries information beyond `z`. If it does not, `z` may already encode the useful magnitude
component, **or** magnitude adds nothing under this head. Do not decide between those two mechanisms
from H-FM3 alone; H-FM4 is what separates them.

**Falsification clause.** Report whichever cell the result occupies, including the cells that weaken
the paper. This is the condition on which Option B was chosen at all.

---

## 8. Multiplicity

The existing transfer table carries twelve prediction objects; the FM arm adds eight more plus five
controls. Under interval-based decision rules, some will resolve by chance.

**The primary inferential family is exactly four hypotheses: H-FM1, H-FM2, H-FM3, H-FM4.** Everything
else is labelled **SECONDARY** or **EXPLORATORY** in both the tables and the text. No secondary
comparison may be promoted to the central narrative after it is seen.

Within a hypothesis that spans two screens or four magnitude components, report **all** components
transparently. Use the paired bootstrap intervals and the pre-declared cross-screen consistency rule
("resolved in both screens"); do not select a p-value correction after seeing results.

---

## 9. Mandatory reporting

**Gene coverage asymmetry — declare in advance, either way.** Geneformer vocabulary coverage is
**99.42 %** (VCC, 17,975 / 18,080), **97.38 %** (RPE1, 8,520 / 8,749), **95.88 %** (K562-essential,
8,210 / 8,563). The expression arm's external mapping recovers only **1,430 / 2,000 = 71.5 %** of the
VCC HVGs. The FM arm therefore sees substantially more genes on the external screens than the
expression arm does.

- If `z` still fails to transfer **despite** near-complete coverage, the conclusion is *stronger* —
  the failure cannot be attributed to lost features.
- If `z` transfers well, ask honestly whether it is the representation or the coverage, and say so.

**Also mandatory:**

- per-row and per-construct cell counts, and the NT pool sizes used;
- the two declarations of §2.2, as design choices rather than inherited facts;
- which interpretation cell of §7 the result occupies, named explicitly;
- the C2 comparison and its interval, whatever it shows.

---

## 10. Ordering, the go/no-go gate, and the kill switch

**Phase 0 — preflight.** Reconcile `|G_tr|` (§12 M1); hard-pin `selected_genes.npy` with a loud
failure if `hvg_genes.npy` is reached; pin the Geneformer environment and record checkpoint,
tokenizer and dictionary hashes; isolate all output paths under `iclr2027_revision/`; implement the
bridge with unit tests; implement the `x_cell` same-pass accumulator; implement C1–C5; verify the
bootstrap import. **No scientific result is computed in this phase.**

**Smoke test**, on the identified path, with every write redirected out of [FEATURE-PIPELINE]. Measure cells/sec.

### 10.1 Post-smoke-test go/no-go gate

The Sep 1 kill switch fires too late to save the schedule. Immediately after the smoke test compute

```
projected_hours = (283,970 + 8,445) / measured_cells_per_second / 3600
```

and compare `2 × projected_hours` against the calendar remaining before Sep 1 (the factor of two
covers I/O contention, restarts, and validation):

- **GO** — fits with ≥ 5 days of slack for analysis. Proceed.
- **ESCALATE** — leaves < 5 days of slack. **Do not start the full extraction.** Report the measured
  rate and the projection; whether to continue, reduce scope, or cut is the author team's decision.
- **NO-GO** — exceeds the remaining calendar. Cut Option B now under the Sep 1 rule rather than at
  Sep 1.

Record the measured rate, the projection, and the verdict in the implementation manifest before any
full extraction begins.

**Phase 1 — VCC.** Key-join the existing `z` to the manuscript's target-gene labels. Parity checks.
Train/validation head selection, frozen. **VCC test once.** H-FM1 and H-FM4.

**Phase 2 — RPE1.** Embed exactly the missing **8,445** cells (the 43 constructs); combine with the
cached 1,889; build global-NT `z` and `x_cell`. QC only.

**Phase 3 — K562-essential.** Embed **283,970** cells (273,279 eligible perturbed + 10,691 NT, of
310,385 in the file). Build `z` and `x_cell`. QC only. `x_cell` accumulates in the **same streaming
pass** as tokenization; a second pass over ~19 GB of h5ad is not acceptable.

**Phase 4 — external evaluation,** only after all model choices are frozen: RPE1, K562-essential,
H-FM2, H-FM3, bootstrap intervals, the locked non-overlap sensitivity, and all controls.

No later phase may revise an earlier scientific choice.

### 10.2 Kill switch

**Hard stop: 1 September 2026.** If validated embeddings for VCC, RPE1, and K562-essential do not all
exist by then, **cut Option B** and submit the strengthened paper without it. This is a pre-specified
fallback and it is not renegotiable at the deadline.

On cutting, do **not**: report a partial external screen; change the model after a failed extraction;
relax construct eligibility; substitute K562-GWPS for K562-essential; swap in a different foundation
model; or renegotiate the primary hypotheses after seeing partial results.

---

## 11. Main-text budget

The submitted manuscript compiles to 29 pages with the **main body ending exactly on page 9**. The
[venue redacted] limit for initial review is **9 pages** (references and the ethics / reproducibility /
AI-use statements excluded). There is zero slack — and a **new summary
figure** is separately required conveying both headline results.

The FM arm's main-text footprint is therefore capped in advance:

- at most **one added table block** — FM rows appended to the existing transfer table, not a new
  standalone table;
- at most **one short paragraph** in Experiments and **one sentence** in Discussion;
- H-FM4, all controls C1–C5, `x_cell` versus `x_release`, and the VCC-train-overlap sensitivity live
  in the **appendix**, each with a one-line pointer from the main text.

Anything beyond this is appendix material by default. Any main-text expansion requires an explicit
cut stated in the same change note, weighed against the summary figure, which also needs main-text
space.

---

## 12. Manuscript items to fix separately — not part of this experiment

These surfaced during the audits. They belong on the revision list, not the Lock's execution path,
but M1 must be resolved before implementation begins.

**M1 — `|G_tr|` = 140 (manuscript §3) versus 150 unique target genes in the Training obs.** This is a
stated number in the problem setup, in the same sentence that gives 5,740 / 2,052 / 3,972, and any
reviewer with the public data can check it. Determine whether the manuscript's 140 reflects a
documented eligibility or minimum-cell filter the raw count omits — in which case say so — or whether
the number is wrong. Record the resolution. This belongs with the fixes, not the cleanups.

**M2 — VCC-training target overlap in the external evaluations.** RPE1 **43 / 1,932 = 2.23 %**;
K562-essential **33 / 1,903 = 1.73 %**. The manuscript does not address this, and Table 9's
"68 / 282 = 24.1 % overlap with VCC" is a **different quantity** that a reader could mistake for
answering it. At ~2 % the overlap cannot drive any result, so **keep all eligible constructs in the
primary** — but state the overlap explicitly and add the non-overlap sensitivity as a **secondary**
panel, using identical exclusion logic on both screens. It may not be used for model selection, head
selection, hyperparameters, or as a replacement for the primary conclusion. Report the primary first.

**M3 — retire or pin `hvg_genes.npy`.** No reported result used it — the deep-model provenance is
confirmed on `delta` + `selected_genes.npy` — but it remains reachable via `run_vcc.py`'s `raw`
default and differs from `selected_genes.npy` by **167 genes** (Jaccard 0.833). Pin
`variants/delta_hvg/selected_genes.npy` as the sole VCC HVG artifact before the code release.

**M4 — Eq. 1 versus implementation.** The manuscript defines `m₂ = ‖x‖₂/√d`;
`src/training/magnitude_augment.py` uses raw `‖x‖₂`. Because `1/√d` is a positive constant and `m̃` is
z-scored afterwards, the two are numerically identical post-standardisation — **nothing requires
recomputation**. But for a paper branded on pre-registration and about to release code, the formula
and the implementation should agree on paper.

**M5 — commit a Geneformer environment file.** It currently exists only as a conda environment on one
machine, against a gitignored clone at commit `ad8f66d` with `transformers` ~4.44–4.46. Reviewer PLav
already objected that release-on-acceptance is insufficient for this paper.

---

## 13. Locked facts carried in from the audits

Recorded here so this document is self-contained; the Lock cites the audits for provenance.

- **Deep-model feature provenance: CONFIRMED.** `phase9c_lite` hardcodes `feature_transform="delta"`,
  `gene_selection="hvg"` → `selected_genes.npy`, and the stored predictions reproduce the reported
  numbers exactly (MAG-SPSM per-seed 0.2224 / 0.2299 / 0.2238 → **0.2253 ± 0.0040**). Contribution #3
  is apples-to-apples. **Nothing requires recomputation.**
- **Geneformer:** Geneformer-V2-104M, frozen pretrained, `gc104M` vocabulary, 12 layers, 768 hidden,
  **CLS** token, `emb_layer = −1` (second-to-last), fp32, no fine-tuning.
- **VCC bridge:** 11,764 / 11,764 rows join on `(split, batch, target_gene)`; per-row perturbed and NT
  cell sets identical on both sides; all `z` finite; no re-extraction.
- **[FEATURE-PIPELINE] is out of scope except as a frozen representation source.** Its labels are a *different
  endpoint* — `log1p(mean over 18,080 per-gene AD)`, a transcriptome-wide breadth measure,
  rank-anticorrelated with the manuscript's target-gene endpoint at Spearman **−0.153**. Its magnitude
  features sit on a different HVG set (Jaccard 0.40). **No [FEATURE-PIPELINE] label, magnitude feature, supervised
  head, reported number, or scientific verdict may be used or cited anywhere in this paper** — including
  the INVALID gate verdict, whose design warning is honoured in §5 without citation.
