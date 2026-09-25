# Option B — Frozen-Geneformer Experiment Lock ([venue redacted])

**Status:** pre-result specification lock. Nothing here may change once the first scientific result is
computed. **Source of truth for hypotheses/controls/interpretation/multiplicity/budget/kill-switch:**
`iclr2027_revision/experiment_specs/part1_scientific.md` (sha256 `55f2f778…1131df`) — where this lock and
that file could be read as differing, **the file governs**. Engineering facts, paths, hashes, and
extraction schedules come from the five audit/prelock documents; every literal below was re-verified from
repository evidence in this session (citations inline).

**The one new ICLR result:** a frozen Geneformer evaluation of the manuscript's magnitude–direction
asymmetry. Specification only; no experiment is run in this session.

**Classification key:** `[N]` = INHERITED_FROM_[LEGACY-SUBMISSION] · `[M]` = REUSED_FROM_[FEATURE-PIPELINE]_REPRESENTATION_ONLY ·
`[I]` = NEW_PREREGISTERED_ICLR_CHOICE.

---

## 0. Governing architecture

- **[LEGACY-SUBMISSION] owns** `[N]`: the scientific task; row sets/splits; labels/endpoints; VCC expression `x`; the
  authoritative HVG artifact; the magnitude definition `m̃`; metrics; the bootstrap; existing baselines.
- **[FEATURE-PIPELINE] contributes ONLY** `[M]`: the frozen Geneformer checkpoint/tokenizer; existing frozen per-cell
  embeddings; the existing VCC raw embedding `z`; the reusable RPE1 frozen per-cell embeddings.
- **NEW ICLR CODE owns** `[I]`: the bridge; external `z` construction; `x_cell`; the FM heads; the
  capacity/alignment controls; the magnitude-recoverability probe; the new extraction manifests.

**Prohibited reuse (hard):** [FEATURE-PIPELINE] labels; [FEATURE-PIPELINE] MAG features; [FEATURE-PIPELINE] supervised heads; [FEATURE-PIPELINE] B0/G1/G2
predictions; [FEATURE-PIPELINE] supervised R² values; [FEATURE-PIPELINE] scientific verdicts (including the INVALID gate verdict —
its design warning is honoured in §13 without citation).

---

## 1. Scientific question

Does the manuscript's magnitude–direction asymmetry (magnitude-only predictors transfer positively across
screens; expression-direction predictors do not) extend to a **frozen pretrained single-cell FM
representation**?

**Wording discipline (mandatory).** Never write "Geneformer is magnitude-blind." The licensed statement:
> Geneformer does not receive absolute per-cell expression magnitude explicitly; its tokenizer represents
> each cell through a rank-based encoding after per-cell count normalisation and gene-wise median scaling.
> A population-level embedding delta may nonetheless encode response-strength information through the
> detected-gene set and through rank displacement.

The three questions with decision rules attached (the only ones posed): (a) does frozen `z` carry
predictive information beyond a matched-capacity random control on VCC; (b) if so, does FM-only transfer
like the expression-only group — Spearman negative or unresolved in **both** external screens; (c) does
appending `m̃` produce the resolved transfer increment seen for SPSM → MAG-SPSM. **No fine-tuning in the
primary arm.**

---

## 2. One-new-result scope `[locked]`

Option B is the single new result. **Excluded:** any third external screen; scGPT; UCE; any other FM; any
new endpoint; any benchmark-audit reframe; any import of [FEATURE-PIPELINE] supervised results. Every auxiliary result
below is labelled exactly `CONTROL`, `MECHANISM_DIAGNOSTIC`, or `SECONDARY_SENSITIVITY` and is never a
second headline.

---

## 3. Datasets / rows `[N]`

| dataset | rows | key |
|---|---|---|
| VCC | 11,764 total = 5,740 train / 2,052 val / 3,972 test | `(split, batch, target_gene)`, official manuscript split |
| RPE1 | **1,932** harmonized-eligible constructs | construct id `id_SYMBOL_promoter_ENSG` |
| K562-essential | **1,903** harmonized-eligible constructs | construct id |

Primary external analyses use **all** manuscript-eligible constructs. **Do not** inherit [FEATURE-PIPELINE]'s
VCC-training-overlap exclusion into the primary (see §25 for the secondary).

---

## 4. Authoritative HVG / feature provenance `[N]`

**Sole authoritative VCC 2,000-gene artifact:** `cache/vcc/variants/delta_hvg/selected_genes.npy`.

**VERIFIED execution/artifact evidence** (this session + ICLR-L3.5): the reported deep runs
(`scripts/phase9c_lite/run_3seed.py:37-43`, `run_magnitude.py:32-37`) **hardcode**
`load_vcc_pseudobulk(..., feature_transform="delta", gene_selection="hvg")` → via
`src/data/vcc/loader.py:97-106,622-629` → `variants/delta_hvg/selected_genes.npy`. Classical MAG-LINEAR /
RF `[x; m̃]` use the same delta path (`phase11_strong_baselines`/`phase10`). Stored MAG-SPSM predictions
reproduce **0.2224 / 0.2299 / 0.2238 → 0.2253 ± 0.0040** (matches Table 1).

`DEEP_MODEL_FEATURE_PROVENANCE = CONFIRMED_DELTA_SELECTED_GENES`. **No reported contribution requires
recomputation.**

**Guard `[I]`:** the implementation must **FAIL LOUDLY** if any new-ICLR code path resolves
`cache/vcc/hvg_genes.npy` (the `(raw,hvg)` default; 167 genes different, Jaccard 0.833). It is a live but
**non-authoritative** raw-default path; pin/retire before public release (§31 M4).

---

## 5. Locked Geneformer representation `[M]`

- **Model:** Geneformer-V2-104M, pretrained, **frozen, no fine-tuning**, `gc104M` vocabulary.
- **Architecture (verified `config.json`):** 12 transformer layers, hidden 768, 12 heads, vocab 20,275,
  max_position 4,096.
- **Representation:** per-cell, **CLS** token, `emb_layer = -1` (= **second-to-last** transformer layer),
  **fp32**.

**Verified artifact provenance:**

| artifact | path (under `[FEATURE-PIPELINE-REPO]/cache/foundation_models/`) | SHA-256 |
|---|---|---|
| Geneformer repo commit | `geneformer-repo` | git `ad8f66dfcda3ebbd148d916c01f31339c5b95a15` |
| V2-104M checkpoint | `geneformer-repo/Geneformer-V2-104M/model.safetensors` (417,571,156 B) | `fff5cba29ddd8792991fa77b4872246fbe548a178cebda3775cdc72b67780e7f` |
| token dictionary | `geneformer/geneformer/token_dictionary_gc104M.pkl` | `67c445f4385127adfc48dcc072320cd65d6822829bf27dd38070e6e787bc597f` |
| gene-median dictionary | `geneformer-repo/geneformer/gene_median_dictionary_gc104M.pkl` | `a51c53f6a771d64508dfaf61529df70e394c53bd20856926117ae5d641a24bf5` |
| Ensembl mapping dictionary | `geneformer-repo/geneformer/ensembl_mapping_dict_gc104M.pkl` | `0819bcbd869cfa14279449b037eb9ed1d09a91310e77bd1a19d927465030e95c` |
| gene-name→id dictionary | `geneformer-repo/geneformer/gene_name_id_dict_gc104M.pkl` | `fabfa0c2f49c598c59ae432a32c3499a5908c033756c663b5e0cddf58deea8e1` |

No alternative checkpoint / Geneformer generation / layer / pooling / raw-vs-unit-`z` may be chosen after
any result is observed. **Preflight verification (§37):** confirm the [FEATURE-PIPELINE] extraction that produced the
cached VCC/RPE1 `z` used exactly CLS + `emb_layer=-1` + fp32 + this checkpoint, so the new K562-essential
extraction is byte-identical in configuration.

---

## 6. Representation objects (from part1 §2) `[N]` def, `[M]` `z`, `[I]` `x_cell`

| object | definition | dim |
|---|---|---|
| `x` | manuscript expression. **VCC:** log1p delta vs per-batch NT over `selected_genes.npy`. **External:** distributed Replogle pseudobulk `*_normalized_bulk_01.h5ad` symbol-mapped into the VCC delta-HVG namespace, unmapped HVGs zero-imputed, clipped ±10 | 2,000 |
| `m̃` | four magnitude scalars `mean\|x\|`, `‖x‖₂/√d`, `‖x‖∞`, `σ(x)`, z-scored on **train** stats only, computed over `selected_genes.npy` | 4 |
| `z` | raw 768-D frozen Geneformer CLS mean-embedding delta `mean(E_pert) − mean(E_NT)`; **raw, never unit-normalised in the primary** | 768 |
| `x_cell` | construction-matched external comparator (§9) | 2,000 |

---

## 6B. FM representation-block preprocessing `[I]` `NEW_PREREGISTERED_ICLR_CHOICE`

`FM_BLOCK_SCALING = TRAIN_STANDARDIZE`.

**Decision timing — preregistered.** Source:
`iclr2027_revision/experiment_specs/pre_smoke_scale_architecture_check.md`.
Decided **PRE-SMOKE and PRE-TRAINING**, before any Option-B scientific result was
observed: no head had been fit, no VCC test number computed, no external evaluation
run. The measurements below use only cached representation arrays and no labels.

**Empirical motivation — not a scientific result.** The figures in the table are
descriptive statistics of the feature blocks, recorded to justify removing a
nuisance scale difference. They are not a finding of this paper and must not be
reported as one. (`pre_smoke_scale_architecture_check.md`, VCC train, 5,740 rows,
no labels used.)

| block | coord-std median | row L2 median |
|---|---|---|
| `z` (frozen Geneformer, raw) | 0.00275 | 0.0700 |
| `x` (delta-HVG) | 0.0734 | 3.12 |
| `C1` (i.i.d. `N(0,1)`, analytic) | 1 | 27.7128 (= √768) |

Raw `z` and `C1` would enter a scale-sensitive head **~400× apart** in row norm,
and `z` is ~45× smaller than the `x` on which the SPSM V4 hyperparameters (lr 1e-3,
25 epochs, no warmup, no input scaling) were validated. Since the H-FM1 gate is
`ΔR²(FM-only − C1)`, the gate could resolve on magnitude rather than information
content. This is not borderline.

**Locked convention.**

- Use `sklearn.preprocessing.StandardScaler` explicitly — not a hand-rolled
  mean/std division — so its zero-variance guard (`scale_ → 1` for near-constant
  coordinates) cannot amplify numerical noise into a unit-variance feature.
  Measured: `z` has **0** near-constant coordinates (min coord std 2.30e-4), so the
  guard does not bind here; it is retained as the correct general safeguard.
- Fit on **VCC train only**; apply the frozen statistics unchanged to validation,
  test, RPE1, and K562-essential. No scaler is ever fit on validation, test, or
  either external screen.
- `z` gets its own train-fitted scaler; `C1` its own; `C2` its own.
- `C3` (row-shuffle of `z`) uses the **same** scaler fitted on **unshuffled** `z`.
- In `[z; m̃]`: scaled `z` plus the manuscript's existing train-standardised
  magnitude block.
- **The same preprocessing applies to the H-FM4 probe inputs** — `z`, the ceiling
  `x`, the random-projection floor, and the row-shuffled-`z` floor — so that
  "where `z` falls between matched floors and the matched ceiling" is not a
  statement about scale. This extends the **preprocessing convention** of §18 to
  the probe inputs and **does not change the scientific definition of H-FM4**: the
  hypothesis, the four separate single-output Ridge probes, the validation-selected
  α grid, the ceiling and both floors, and the interpretation rule are all
  unchanged. Only the feature convention is now shared across them.
- No unit-L2 normalisation anywhere.
- Applied uniformly across all four head families so head selection never compares
  different feature conventions.

**Cross-family note (state in the paper).** Exactly one of the four families is
affected by this convention. Ridge already applies `StandardScaler` (App. F), so
pre-standardisation is a no-op for it; Random Forest and HistGradientBoosting are
invariant to monotone per-feature rescaling. Only SPSM V4 is scale-sensitive.
Because the Lock requires all four families to be reported, the preprocessing
asymmetry between the standardised FM block and the manuscript's raw expression
convention is auditable, and three of the four families are independent of it.

**Leakage.** The transform is an invertible, train-fitted affine map using no
labels and no test or external information.

**Supersession.** This replaces `implementation_preflight.md` D2.4 item 6
("FM SPSM input standardisation = NONE (raw)"), which preserved the legacy raw-`x`
convention. It is a **preprocessing** decision, declared in its own right and
separate from `SPSM_V4_ADAPTATION = INPUT_WIDTH_ONLY`, which changes only the input
layer width.

---

## 7. VCC `z` `[M]` — reuse, no re-extraction

`z_i = mean(E over perturbed cells of (batch, target_gene)) − mean(E over batch-local NT cells)`.

Locked/verified (ICLR-L3 §A, A2): raw 768-D; **11,764 / 11,764** manuscript rows join on
`(split,batch,target_gene)`; per-row perturbed and NT **cell sets identical** on both sides (exact count
agreement 11,764/11,764, no subsampling either side); all `z` finite; **no Geneformer re-extraction**.
Attach the **[LEGACY-SUBMISSION] target-gene-specific AD response-magnitude** `y`. **Never** attach or use [FEATURE-PIPELINE]'s
transcriptome-wide mean-over-genes breadth endpoint (rank-anticorrelated at Spearman −0.153).

---

## 8. External `z` `[I]`

For **both** RPE1 and K562-essential:
`z_g = mean(E over all cells of construct g) − mean(E over all global NT cells in that screen)`.

Locked: **all** cells of each eligible construct; **one global pooled NT** mean embedding per screen (RPE1
11,485 NT, K562-essential 10,691 NT); **no batch matching**; **no 27/788 subsampling** (that rule is
harmonized-AD-endpoint-only); deterministic means; no external fitting/calibration/label-informed
normalisation.

`GLOBAL_NT_FOR_Z = NEW_PREREGISTERED_ICLR_CHOICE`. Rationale (state in paper as a choice, not an inherited
fact): target-agnostic; deterministic; identical rule on both screens; consistent with the manuscript
**endpoint's** control policy (`protocol.json nt_policy_prereg_s5`: PRIMARY = "NT pooled across all
gem_groups"; App. J "globally pooled controls"; batch-matching computable for only 43/194 constructs;
agreement ρ = 0.91 RPE1 / 0.97 K562-essential); coincident with [FEATURE-PIPELINE] precedent. **It is NOT inherited from
`x_release`**, which carries no NT vector.

---

## 9. Construction parallelism / `x_cell` `[I]` mandatory

External `x_release` (release pseudobulk, no NT subtraction) and external `z` (pert−NT delta) are
**different contrasts**. `x_cell` closes the gap:
`x_cell_g = mean(log1p(CP10K) over all cells of g) − mean(log1p(CP10K) over all global NT cells)`, then
mapped into `selected_genes.npy` by the manuscript's gene-symbol mapping
(`scripts/phase11b_classical_transfer/run.py:80-113`): zero-impute unavailable VCC HVGs, preserve
authoritative feature order, `nan_to_num(nan=0, posinf=10, neginf=-10)`, clip ±10 (the verified manuscript
convention).

Locked: `x_release` = existing submitted comparator; `x_cell` = new construction-matched comparator;
**`x_cell` does not replace `x_release` or any submitted result.** The "FM representation vs expression"
claim is made against **`x_cell`**, so a construction mismatch cannot masquerade as a representation
effect. `x_cell` accumulates in the **same streaming raw-cell pass** as Geneformer tokenization
(`cp10k_log1p_mean` already computed from the raw counts read for tokenization; ICLR-L3.5 §B2.2) — a second
pass over the ~10.7 GB K562 h5ad is **not** acceptable.

---

## 10. Targets `[N]`

VCC: the manuscript target-gene-specific AD response-magnitude endpoint (existing). External: the existing
harmonized 27/788 endpoint and the locked construct lists (Sessions 6/11/13/14). **Do not regenerate
either. Prohibit [FEATURE-PIPELINE]'s mean-over-genes breadth endpoint.**

---

## 11. Head families `[N]` + input-width adaptation `[I]`

Candidate heads: **Ridge**, **Random Forest**, **HistGradientBoosting**, **SPSM V4 backbone**. Each applied
to `z` and `[z; m̃]`. **Selection = validation R² only; no test metric may influence head selection;** the
selected config per feature set is frozen and recorded before any test number.

**Exact settings (verified, verbatim — no widened search).**

App. F classical (`scripts/phase11_strong_baselines/run.py:124-143`; fixed single configs, not a grid):
- `Ridge(alpha=1.0, random_state=seed)` with `StandardScaler` (with_mean/with_std) fit on train.
- `RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=2, n_jobs=8, random_state=seed)`, no scaling.
- `HistGradientBoostingRegressor(max_iter=500, max_depth=6, learning_rate=0.05, l2_regularization=0.0, random_state=seed, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)`, no scaling.

App. G SPSM V4 (`src/models/encoders.py`, `src/training/torch_backend.py`, manuscript App. G; verified via
`phase9c_lite/run_3seed.py:89-90`):
- encoder = 3-layer MLP, hidden width **256**, LayerNorm, ReLU, **dropout p=0** → **256-dim** representation → linear head;
- loss `L_total = L_MSE + 0.5·L_stab + 0.5·L_rel` (MMD stability + soft-Spearman relation, τ=0.1);
- Adam, lr **1e-3**, weight_decay **1e-5**, batch **64**, **25 epochs**, model selection by validation R².
- **`SPSM_V4_ADAPTATION = INPUT_WIDTH_ONLY`** `[I]`: manuscript input 2,004 → **768** for `z`, **772** for
  `[z; m̃]`. Depth, hidden widths after the input layer, activations, LayerNorm placement, dropout, loss
  terms/weights, optimizer, lr, batch construction, and epochs are unchanged. State in the paper that only
  input dimensionality was adapted. No other architectural variant after results.
- **Preflight (§37):** confirm the reported SPSM/MAG-SPSM warmup setting (`--use-warmup --warmup-epochs 15`
  appears in the p6_2d shell runs but not in the `phase9c_lite` cfg template that produced the reported
  numbers) and hold it fixed identically for the FM heads.

Head-training seeds `[I]`: **{0, 1, 2}** (matches the manuscript 3-seed convention; the frozen embedding is
deterministic, so variation is head-only). Report **mean ± sample std (ddof=1)**.

---

## 12. Primary feature arms `[locked]`

| arm | input | scope | class |
|---|---|---|---|
| **P0** MAG-LINEAR | `m̃` (existing manuscript reference) | VCC + external | `[N]` |
| **P1** FM-only | raw `z` | VCC + external | `[I]` head over `[M]` z |
| **P2** FM+MAG | `[raw z ; m̃]` | VCC + external | `[I]` |
| **P3** `x_cell` | construction-matched expression | external only | `[I]` |
| **P4** `x_release` | submitted-paper expression | external only | `[N]` |

**Do not** use [FEATURE-PIPELINE] unit-normalised `z`, [FEATURE-PIPELINE]'s additive anchor, or any [FEATURE-PIPELINE] supervised head.

---

## 13. Capacity / alignment controls `[I]` — seeds fixed here

**C1 and C2 are different kinds of object and must not be treated as one class.**

- **C1 — 768-D i.i.d. Gaussian features.** Class **CAPACITY_FLOOR (GATING)**. Zero-information floor. **C1
  alone gates H-FM1.** Generation: per-row i.i.d. `N(0,1)`, **generated independently within each split**
  from row shape only (never from data values or labels), seed **20270101** (deterministic per-split
  stream: `default_rng(20270101 + split_index)`, split_index train=0/val=1/test=2/RPE1=3/K562=4).
- **C2 — random projection `x → 768-D`.** Class **COMPRESSED_EXPRESSION_BENCHMARK (SECONDARY, NON-GATING)**.
  By Johnson–Lindenstrauss ≈ `x` compressed. **Never the H-FM1 gate.** Generation: one fixed Gaussian
  projection matrix `R ∈ ℝ^{2000×768}`, `R_ij ~ N(0, 1/768)`, drawn **once** with seed **20270102** and
  applied identically to every split and both external screens (a projection must be the same linear map
  everywhere). Reported with its own interval as its own finding.
- **C3 — row-shuffled `z`.** ALIGNMENT_CONTROL. A permutation drawn **per split** with seed **20270103**
  (`default_rng(20270103 + split_index)`), applied to that split's `z` rows; the same permutation is used
  at train and eval for that split (destroys row identity while preserving the `z` distribution).
- **C4 — `[z ; row-shuffled m̃]`.** ALIGNMENT_CONTROL. Per-split permutation of `m̃` rows, seed **20270104**.
- **C5 — `[z ; 4 i.i.d. Gaussian scalars]`.** DIMENSION_INCREMENT_CONTROL. Per-row i.i.d. `N(0,1)`, per
  split, seed **20270105**.

**Feature-generation seeds (20270101–20270105) are separate from head-training seeds {0,1,2}.** No control
may be introduced after primary results; **no control seed may be changed because a result looks unusual.**
Every control uses the **identical head-selection procedure** as its corresponding primary arm. Leakage
guard: all stochastic controls are generated from array shapes only — never from `y`, never from
validation/test statistics.

---

## 14. Multiplicity `[N]`/`[I]`

**Primary inferential family = exactly four hypotheses: H-FM1, H-FM2, H-FM3, H-FM4.** Everything else is
labelled **SECONDARY** or **EXPLORATORY** in tables and text; no secondary comparison may be promoted after
it is seen. Within a hypothesis spanning two screens or four scalars, report **all** components; use the
paired bootstrap intervals and the pre-declared "resolved in both screens" rule; no post-hoc p-value
correction.

---

## 15. H-FM1 — in-distribution information

**Gate:** FM-only is INFORMATIVE iff the paired bootstrap interval on `ΔR² = R²(FM-only) − R²(C1)` **excludes
zero on the positive side**. **C1 alone is the gate.**

**Reported alongside, non-gating:** `ΔR² = R²(FM-only) − R²(C2)` with its interval — the compressed-expression
benchmark. *Rationale (state in lock):* if `z` beat pure noise but lost to compressed `x`, gating on C2
would falsely record H-FM1 failed and disable H-FM2 by conditionality; that is a real, separate finding,
not the same one.

**Contextual references only** (not criteria): MAG-LINEAR ≈ +0.208, RF `[x; m̃]` ≈ +0.370, MAG-SPSM ≈ +0.225
± 0.004, SPSM ≈ +0.082 ± 0.010. Success is **not** "FM-only > MAG-LINEAR."

**Bootstrap:** VCC in-distribution **row-level paired** bootstrap, **B = 10,000, RNG seed = 42, 95%
percentile (2.5/97.5)** (`scripts/bootstrap_ci/run_bootstrap.py:25-26,51-52`; `n=3,972`). `[N]`

---

## 16. H-FM2 — zero-shot transfer

Interpretable **only if** H-FM1's C1 gate passed. Pre-registered expectation: FM-only behaves like the
expression-only group — external Spearman **negative or unresolved in both** RPE1 and K562-essential. If the
C1 gate fails, near-zero external ρ **must not** be reported as a transfer finding; the correct statement is
"usable in-distribution information from `z` was not established under the locked head."

---

## 17. H-FM3 — magnitude increment

Compare **FM+MAG vs FM-only** on both screens; primary quantity = **paired Δ Spearman ρ**. Expectation: a
**resolved positive** increment in **both** screens, mirroring SPSM → MAG-SPSM (Δρ = +0.117 RPE1, +0.086
K562-essential). **Must survive C4 and C5** — if row-shuffled `m̃` or four Gaussian scalars reproduce the
gain, no per-row magnitude-alignment claim. No single-screen success claim.

**Bootstrap (external transfer, H-FM2/H-FM3):** manuscript **paired target-gene cluster** bootstrap
(`scripts/v4/harmonized_endpoint_v1/common.py cluster_bootstrap_corr`): constructs sharing a target
resample together; **B = 10,000**; **95% percentile** (2.5/97.5, `np.quantile` linear); per-screen seed
**RPE1 = 20260901**, **K562-essential = 20260902** (Sessions 13/14 locks; same draws as the existing
classical objects → paired). `[N]`

---

## 18. H-FM4 — magnitude recoverability `MECHANISM_DIAGNOSTIC` `[I]`

**Four separate single-output Ridge probes**, one per `m̃` scalar (not a shared multi-output fit). Fit on VCC
**train**; regularisation selected on VCC **validation**; VCC **test** touched once. References with the
**same probe family**, per component: **Ceiling** = probe `x → m̃_k` (nonlinear, so < 1.0 by design);
**Floor 1** = random-projection(`x`)→`m̃_k`; **Floor 2** = row-shuffled `z`→`m̃_k`. No absolute threshold;
report **all four** components by their location between matched floors and ceiling; do not summarise to one
number or cherry-pick. Prior: expect **partial** recoverability ([FEATURE-PIPELINE] unit-normalises `z` to isolate
direction — evidence `z` carries some magnitude information). Ridge α grid for the probes uses the same App. F
`Ridge(alpha=1.0)` fixed setting unless a validation-selected α grid is fixed at preflight (§37); if a grid
is used it must be enumerated at preflight and applied identically to ceiling and both floors.

---

## 19. Interpretation matrix — fixed before any number (part1 §7)

|  | `m̃` **not** recoverable | `m̃` **is** recoverable |
|---|---|---|
| `z` transfers **poorly** | **A** *(predicted)*: asymmetry generalises to the frozen representation. Do **not** write "magnitude-blind." | **B**: magnitude information available but not exploited transferably (availability → exploitation). |
| `z` transfers **well** | **C**: genuine challenge/refinement to contribution #1 — another transferable axis. Report as limitation, not a footnote. | **D**: rank displacement retains amplitude info; the headline is **scoped**, not false. |

**E** — C1 gate fails: representation/head established no usable in-distribution signal; external near-zero
**cannot** be read as biological transfer failure; report the negative ID result and stop.
**F/G** (H-FM3 axis): `[z; m̃]` improves → explicit magnitude adds beyond `z`; if not → `z` may already
encode useful magnitude **or** magnitude adds nothing under this head — **use H-FM4 to distinguish; do not
decide from H-FM3 alone.**

**Falsification clause:** report whichever cell the result occupies, including the ones that weaken the
paper. This is the condition on which Option B was chosen.

---

## 20. Metrics `[N]`

In-distribution primary: pooled test R² with **test-set-mean** denominator, computed once over all 3,972
rows; also Spearman (average-tie) and MAE where manuscript-standard (`src/metrics/predictive.py`,
sha256 `00b8952f…`). External primary: **Spearman ρ** vs the harmonized endpoint; external pooled R² is
descriptive only, never a success criterion. Metric code unchanged.

---

## 21. Bootstrap / uncertainty `[N]` — literal values

**Reuse, do not reimplement.**

| context | code | unit | B | CI | seed |
|---|---|---|---|---|---|
| VCC in-distribution (H-FM1 ΔR², H-FM4) | `scripts/bootstrap_ci/run_bootstrap.py` | row-level paired | **10,000** | **95% percentile (2.5/97.5)** | **42** |
| external transfer (H-FM2, H-FM3) | `scripts/v4/harmonized_endpoint_v1/common.py` | paired target-gene cluster | **10,000** | **95% percentile (2.5/97.5)** | **RPE1 20260901 · K562-essential 20260902** |

No new bootstrap implementation.

---

## 22. External zero-shot rule `[N]`

Train and select on **VCC only**. After head selection is frozen, evaluate zero-shot on RPE1 and
K562-essential. **No** external fitting / calibration / label-derived scaling / architecture selection /
hyperparameter selection / per-screen head choice. Report separately: `x_release`, `x_cell`, FM-only,
FM+MAG, and C1–C5 where applicable.

---

## 23. RPE1 extraction set `[locked, measured]`

Eligible 1,932; **1,889 already fully cached with all constituent cells** (ICLR-L3.5 C1a: cached==raw
cell count for all 1,889; 0 subset/excess); global NT **11,485 raw = 11,485 cached**; **43 missing
constructs = exactly the VCC-training-overlap set**. New GPU: **8,445** perturbed cells, **0** NT,
**8,445** total. Primary includes all 1,932; **do not drop the 43** to save GPU.

---

## 24. K562-essential extraction set `[locked, measured]`

Eligible 1,903; eligible perturbed cells **273,279**; global NT **10,691**; **total cells to embed =
283,970**; whole file **310,385**. Eligible-only extraction; do not embed ineligible constructs unless an
engineering requirement forces it, recorded as a deviation **before** viewing scientific results.

---

## 25. VCC-train-overlap sensitivity `SECONDARY_SENSITIVITY` `[locked]`

Primary = all eligible constructs. Secondary = exclude constructs whose target is in VCC training —
RPE1 **43/1,932 (42 targets, 2.23%)**, K562-essential **33/1,903 (32 targets, 1.73%)** — identical logic
both screens. `NONOVERLAP_SENSITIVITY = LOCKED_YES_SECONDARY`. It may **not** drive model/head/hyperparameter
selection or replace the primary conclusion. Report primary first.

---

## 26. Smoke test `[I]` (later session; not now)

Reuse `[FEATURE-PIPELINE]/scripts/phasec_rpe1_feature_smoke_geneformer_v1.py` extraction logic
(`TranscriptomeTokenizer(nproc=1, chunk_size=512, model_input_size=4096, model_version="V2")`, 20 constructs
+ deterministic NT subset, ≤ 30,000 cells) with **every write redirected under `iclr2027_revision/`** — no
smoke-test artifact may be written into [FEATURE-PIPELINE] (adapter = override `OUTDIR`, `WORKDIR`, `INPUT_DIR`; algorithm
unchanged). Measure: cell count; tokenization time; embedding time; total time; cells/sec; peak VRAM if
available; output shape; finite-value check; stable cell-index mapping; deterministic agreement on a small
repeat if feasible.

### 26B. Post-smoke GO / ESCALATE / NO-GO gate `[locked]`
`projected_hours = (283,970 + 8,445) / measured_cells_per_second / 3600`; compare **2 × projected_hours**
against the calendar to **2026-09-01**.
- **GO** — fits with **≥ 5 days** analysis slack → proceed.
- **ESCALATE** — fits but **< 5 days** slack → **do not start full extraction**; report measured rate +
  projection; the author team decides. Implementation must not silently reduce scope.
- **NO-GO** — exceeds remaining calendar → **cut Option B now** under the Sep-1 rule.
Record rate, projection, 2× hours, calendar slack, and verdict in the implementation manifest **before** any
full extraction.

---

## 27. Kill switch `[locked]`

**Hard stop 2026-09-01.** If validated embeddings for VCC, RPE1, and K562-essential do not all exist by
then, **cut Option B** and submit the strengthened paper without it. On cutting, do **not**: report a
partial external screen; change the model after a failed extraction; relax eligibility; substitute
K562-GWPS; swap FMs; drop the RPE1 43; or renegotiate the primary after partial results.

---

## 28. Execution order `[locked]`

**Phase 0 — preflight (no scientific result):** resolve §37 lookups; hard-pin `selected_genes.npy` with a
loud failure if `hvg_genes.npy` is reached; pin the Geneformer environment + record hashes; isolate all
outputs under `iclr2027_revision/`; implement the bridge with unit tests; implement the `x_cell` same-pass
accumulator; implement C1–C5; verify the bootstrap import; write the implementation manifest. →
**Smoke test + 26B gate.**
**Phase 1 — VCC:** key-join existing `z` to the manuscript labels; parity/QC; train/validation head
selection, frozen; **VCC test once**; H-FM1; H-FM4.
**Phase 2 — RPE1:** embed exactly the missing **8,445** cells; merge with cached 1,889; build global-NT `z`
and `x_cell`; QC only.
**Phase 3 — K562-essential:** embed **283,970** cells; build `z` and `x_cell` (`x_cell` in the same
streaming pass); QC only.
**Phase 4 — external evaluation** (only after all model choices frozen): RPE1, K562-essential, H-FM2, H-FM3,
paired bootstrap, locked non-overlap sensitivity, controls. **No later phase may revise an earlier
scientific choice.**

---

## 29. Leakage / post-hoc guards `[locked]`

Prohibited: external labels for head or architecture selection; external performance to choose controls;
fitting scalers on validation/test or on external labels; layer selection after results; raw-vs-unit `z`
switch after results; global-vs-batch NT switch after results; changing eligible rows after results;
silently dropping VCC-train overlap from the primary; tuning `x_cell` after external results; changing
control seeds after results; promoting a secondary to primary; widening App. F grids; changing App. G
architecture beyond the locked input-width adaptation.

---

## 30. Reporting / vocabulary coverage `[N]` mandatory

Geneformer vocab coverage (Ensembl `gene_id` → gc104M, verified ICLR-L3): VCC **17,975/18,080 = 99.42%**,
RPE1 **8,520/8,749 = 97.38%**, K562-essential **8,210/8,563 = 95.88%**. Expression external VCC-HVG mapping
**1,430/2,000 = 71.5%**. **State that the FM and expression arms have different gene coverage.** If `z` fails
despite ~96–97% coverage, the failure is **not** explained by gross vocabulary loss (stronger conclusion);
if `z` transfers well, ask honestly whether it is representation or coverage. Also report per-row/per-construct
cell counts, NT pool sizes, the two §8/§9 design declarations, the `x_release`/`z` construction mismatch and
`x_cell`'s role, and the named §19 interpretation cell.

### 30B. Main-text budget `[locked]`
Main body ends on page 9; ICLR limit 9 pages; **zero slack**; a new summary figure is separately requested.
FM footprint: **≤ one** added table block (FM rows appended to the existing transfer table), **≤ one**
Experiments paragraph, **≤ one** Discussion sentence; H-FM4, C1–C5, `x_cell` vs `x_release`, and the
overlap sensitivity go to the **appendix** with one-line main-text pointers. Any expansion requires an
explicit cut in the same change note, weighed against the figure.

---

## 31. Mandatory manuscript reconciliations `[N]` (revision list, not execution)

- **M1 (blocking preflight):** `|G_tr|` = 140 (manuscript §3) vs **150** unique target genes in the Training
  obs (verified this project). Determine whether 140 reflects a documented eligibility/min-cell filter or is
  an error; record the resolution before implementation. Not "cleanup."
- **M2:** VCC-training overlap RPE1 43/1,932 (2.23%), K562-essential 33/1,903 (1.73%); Table 9's 68/282 =
  24.1% is a **different** quantity. Keep all eligible in the primary; add the §25 secondary.
- **M3:** Eq. 1 `‖x‖₂/√d` vs code raw `‖x‖₂` (`src/training/magnitude_augment.py:35`) — train-only z-scoring
  makes them equivalent up to a positive constant; **nothing recomputes**; make notation consistent.
- **M4:** pin `selected_genes.npy` as the sole VCC HVG artifact; retire/guard `hvg_genes.npy` before release.
- **M5:** commit an ICLR Geneformer environment file (currently a conda env against gitignored clone
  `ad8f66d`, transformers ~4.44–4.46).
- **M6 — SPSM V4 encoder output dimension stated as 128; the executed network is
  256.** `main.tex:462` (Sec. 5) writes `f_θ : R^2004 → R^128`, and `main.tex:1007`
  (App. G) writes "128-dimensional representation" — but the encoder is a 3-layer
  MLP of width 256 whose output dimension is 256, and `main.tex:1004-1005` already
  states "hidden width 256". The manuscript is therefore internally inconsistent,
  which makes the correction unambiguous: both sentences should read 256. **No
  reported number is affected** — the 256-D network is what ran.
  **Do NOT change `main.tex:918`** (`d_in → 256 → 128 → 1`): that is the MAG-Anchor
  residual branch (App. D), a different network, and its 128 is correct.

---

## 32. Environment / reproducibility `[I]`

Create ICLR-only reproducibility artifacts recording: Python, torch, transformers, Geneformer commit
(`ad8f66d`), CUDA, scanpy, anndata, sklearn/scipy, and the §5 checkpoint/dictionary hashes. **Do not modify
[FEATURE-PIPELINE] environment files.**

---

## 33. Change control `[locked]`

After the first scientific result, every change needs a deviation note: (1) exact change; (2) reason;
(3) `BUG_FIX` / `IMPLEMENTATION_CORRECTION` / `SCIENTIFIC_REDESIGN`; (4) results already seen; (5) whether
those results could have motivated it. **Primary scientific redesign after seeing VCC test or external
outcomes is forbidden.**

---

## 34. Falsification conditions `[locked]`

State a falsifier for each major verdict; on discovery, STOP the affected phase and write a deviation report.
- **VCC z cell-set identity** → any per-row perturbed/NT count disagreement, or a hidden cell filter on
  either side. (Checked: 0/11,764.)
- **RPE1 cache completeness** → any covered construct whose cached count ≠ raw count. (Checked: 1889/1889.)
- **Checkpoint identity** → `model.safetensors` SHA ≠ `fff5cba2…`, or config ≠ 12L/768/vocab 20275.
- **selected_genes provenance** → any reported-run config resolving `hvg_genes.npy`.
- **External identifier mapping** → construct id ⟷ `obs_index` join < expected count on either screen.
- **`x_cell` construction parity** → `x_cell` NT pool ≠ the `z` global NT pool, or a different gene mapping.
- **Global NT membership** → an NT cell missing from the pooled reference, or a perturbed cell in it.
- **H-FM1 Gaussian-floor** → C1 features derived from anything other than shapes (leakage), or C1 not
  dimension-matched to `z`.

---

## 35. Planned output artifacts (list only; do not create) `[I]`

Implementation manifest; environment lock; hash manifest; VCC bridge/QC report; FM/control VCC metrics;
per-seed predictions; H-FM4 probe outputs; RPE1 top-up extraction manifest; RPE1 final `z`; RPE1 `x_cell`;
K562-essential extraction manifest; K562-essential final `z`; K562-essential `x_cell`; external zero-shot
predictions; paired bootstrap CI tables; non-overlap sensitivity; control/benchmark tables; final locked
outcome summary; manuscript-ready compact table block; appendix tables/figures. **All under an ICLR-only
path.** Never overwrite submitted-[LEGACY-SUBMISSION] or [FEATURE-PIPELINE] artifacts.

---

## 36. Environment-of-record note

The frozen embedding is deterministic; variation is head-only (seeds {0,1,2}). The Geneformer clone
(`ad8f66d`) is gitignored on one machine (§31 M5) — committing an environment file is a preflight blocker
for reproducibility, not for correctness of the numbers.

---

## 37. Mandatory preflight lookups

Values already resolved this session are embedded above and marked ✓; the remaining unresolved item is
labelled **LOOKUP**. All must be confirmed in Phase 0 before the first scientific computation.

1. **LOOKUP — `|G_tr|` 140 vs 150** (§31 M1). Genuinely unresolved; needs the manuscript's filter definition
   or an erratum. Blocking.
2. ✓ **Bootstrap literals** — VCC row-level B=10,000/seed 42/95%; external cluster B=10,000/95%/seeds
   20260901 (RPE1), 20260902 (K562-essential). Embedded (§21).
3. ✓ **Geneformer hashes** — checkpoint `fff5cba2…`, token `67c445f4…`, median `a51c53f6…`, ensembl
   `0819bcbd…`, name-id `fabfa0c2…`, repo `ad8f66d`. Embedded (§5). *Confirm at preflight that the cached
   VCC/RPE1 z were extracted with this exact checkpoint + CLS + emb_layer=-1 + fp32.*
4. ✓ **App. F grids** — Ridge/RF/HistGB fixed configs embedded verbatim (§11); no widened search.
5. ✓ **App. G SPSM V4** — 3×256 / LayerNorm / ReLU / dropout 0 / 128-rep / Adam 1e-3 / wd 1e-5 / batch 64 /
   25 epochs / λ=0.5 embedded (§11). *Confirm the warmup setting for the reported runs and hold it fixed.*
6. ✓ **Control seeds** — fixed at 20270101–20270105 (feature generation), head-training {0,1,2} (§13).
   *(H-FM4 probe α: confirm whether a validation-selected α grid replaces the fixed `alpha=1.0`; if so
   enumerate it and apply identically to ceiling/floors.)*

---

## FINAL LOCK STATUS

OPTION_B_SELECTED: YES
ONE_NEW_RESULT_RULE: LOCKED
SCIENTIFIC_SPECIFICATION: LOCKED
DEEP_MODEL_FEATURE_PROVENANCE: CONFIRMED_DELTA_SELECTED_GENES
GENEFORMER_MODEL: Geneformer-V2-104M frozen
VCC_Z: REUSE_EXISTING_RAW_Z
EXTERNAL_Z: ALL_CELLS_MINUS_GLOBAL_NT
EXTERNAL_CONSTRUCTION_MATCH_CONTROL: X_CELL
VCC_HVG_ARTIFACT: cache/vcc/variants/delta_hvg/selected_genes.npy
[FEATURE-PIPELINE]_LABELS_REUSED: NO
[FEATURE-PIPELINE]_MAG_REUSED: NO
[FEATURE-PIPELINE]_SUPERVISED_HARNESS_REUSED: NO
RPE1_NEW_GPU_CELLS: 8445
K562_ESSENTIAL_NEW_GPU_CELLS: 283970
EXTERNAL_PRIMARY_INCLUDES_VCC_TRAIN_OVERLAP: YES
NONOVERLAP_SENSITIVITY: LOCKED_YES
PRIMARY_HYPOTHESES: H-FM1,H-FM2,H-FM3,H-FM4
H_FM1_GATE: C1_ONLY
C2_ROLE: SECONDARY_COMPRESSED_EXPRESSION_BENCHMARK
CAPACITY_CONTROLS: LOCKED
MULTIPLICITY_STRUCTURE: LOCKED
MAIN_TEXT_BUDGET: LOCKED
POST_SMOKE_GO_NOGO_GATE: LOCKED
SEPTEMBER_1_KILL_SWITCH: LOCKED

MANDATORY_PREFLIGHT_LOOKUPS:
1. |G_tr| 140-vs-150 reconciliation (BLOCKING; the only unresolved specification item)
2. bootstrap replicate-count/seed/CI — RESOLVED and written literally (§21)
3. checkpoint/tokenizer/dictionary hashes — RESOLVED and written (§5); confirm cached-z extraction config match
4. App. F Ridge/RF/HistGB grids — RESOLVED, verbatim, no widened search (§11)
5. App. G SPSM V4 fixed settings — RESOLVED (§11); confirm reported-run warmup setting
6. fixed control/head RNG seeds — RESOLVED (§13); confirm H-FM4 probe α convention

READY_FOR_IMPLEMENTATION_PREFLIGHT: YES

NEXT SESSION:
IMPLEMENTATION PREFLIGHT ONLY. No full extraction, training, VCC test evaluation, or external evaluation
until implementation preflight verifies that this lock is executable exactly.
