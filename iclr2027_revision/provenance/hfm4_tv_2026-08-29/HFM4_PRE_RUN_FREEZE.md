# H-FM4 PRE-RUN FREEZE

**Written before the first Ridge fit on scientific data.**

- run_id: `hfm4_20260829T082844Z_b63c2a09324b`
- utc_freeze: 2026-08-29T08:28:47Z   (utc_start 2026-08-29T08:28:44Z)
- git: `b63c2a09324bd62e1e85c201344821e20c36de8a` on `iclr2027-r2` (dirty=False)
- output_root: `/home/mehrdad/iclr2027_phaseB_out/phaseB/hfm4_tv_20260829T082843Z_b63c2a0`
- scientific status: **TRAIN_VALIDATION_ONLY_SUPPORTING_DIAGNOSTIC**

## Pre-freeze declarations

```
HFM4_RESULT_OBSERVED_BEFORE_FREEZE       = NO
REAL_HFM4_RIDGE_FIT_COUNT_BEFORE_FREEZE = 0
VCC_TEST_ACCESSED_BEFORE_FREEZE         = NO
EXTERNAL_ACCESSED_BEFORE_FREEZE         = NO
```

## Governing documents (SHA-256)

- `iclr2027_revision/experiment_specs/option_b_geneformer_experiment_lock.md`  `6321daacafcaee01a0b97253d19968218104051a9a2e044482849a0ada0265e5`
- `iclr2027_revision/experiment_specs/part1_scientific.md`  `55f2f778563dedef1c487989d6d318a4b951867c1460eef35946cc148f1131df`
- `iclr2027_revision/experiment_specs/implementation_preflight.md`  `468e5d97aca4ac306d5ae102aa510eda29f28a5d06d5ffbe1440a56341e4bc44`
- `iclr2027_revision/experiment_specs/hfm4_execution_note.md`  `1b871c030a2489007282916386998dc7b73e350e15c63a0781776f1a51e89128`
- `iclr2027_revision/experiment_specs/lock_amendment_A10_phaseB_numerical_baseline.md`  `0d4fb9b270b04e59a9430da0f9bc1de9f802b6337c18d02ecf06529352b942a0`
- `iclr2027_revision/pipeline/phaseB_preprocessing.py`  `39063e4a0ec7e4af097597837dc9dc95c4839c9e21b6aa5e2ab08d9520dd07cb`
- `iclr2027_revision/pipeline/run_hfm4_tv.py`  `409004e573af08f4059a7b2ccd25498f7a5941ee063ee3ae79c88896366135dc`

## Environment lock (SHA-256)

- `iclr2027_revision/environment/phaseB/phaseB_requirements_locked.txt`  `6f1881604a6026792ef8f56568f58e37c164ea691b5937e2c0f1abdf8335a17f`
- `iclr2027_revision/environment/phaseB/phaseB_environment_README.md`  `d676cc4699e40ba6191d5130ae018263a10d8291bd4b9cb46938dc6a604bfb03`
- `iclr2027_revision/provenance/PHASEB_NUMERICAL_BASELINE_MANIFEST.json`  `da7da5adeb1b74a5b4c8b43874da4585e16f65f5e1d5c400c84037e664d0087c`

## Environment

```
python=3.13.12 numpy=2.4.4 scipy=1.17.1 sklearn=1.9.0
joblib=1.5.3 threadpoolctl=3.6.0 pyarrow=24.0.0
torch=None cuda=None gpu=None
thread_env={'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
threadpool_num_threads=[1, 1, 1] check=PASS
```

## Scope

```
authorized splits = ['VCC train', 'VCC validation']
forbidden splits  = ['VCC test', 'RPE1', 'K562', 'any external screen']
expected rows     = {'train': 5740, 'validation': 2052, 'vcc_test_reference_never_loaded': 3972}
canonical key     = (split, batch_name, sample_id) == bridge (split, batch, target_gene)
canonical order   = sorted(glob(shards/<split>/*.npz)) then stored sample_ids order (note §11 D3A)
probe inputs      = ['z', 'ceiling_x', 'C2', 'C3']
targets           = ['mean_abs_x', 'l2_norm_x', 'max_abs_x', 'std_x'] (object m_zscored)
alpha grid        = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
C2 matrix sha256  = e5bf7ce0f07cd68d73c41529b7f01441b1b5a6f9193d663bc28bc6f974d8f41a  (2000 -> 768, seed 20270102)
C3 seed rule      = default_rng(20270103 + split_index), controls.py:53-57; train=0, val=1
C3 seeds          = {'train': {'split_index': 0, 'seed': 20270103}, 'val': {'split_index': 1, 'seed': 20270104}}
```

## Scaler rule

```
A10 §4 LOAD ONLY — load, verify SHA, verify state, transform; never fit
STANDARD_SCALER_FIT_CALL_COUNT_EXPECTED = 0
z          ebf55308f3c6e62e43cd8b9aba084413549036c0f7e23a987ee6b8b390672e0e  n_features_in_=768 n_samples_seen_=5740
c2         2f6ae19992c7ec7422a3b4a8fe4fea21750a40fb862ecc15acc177d60593600d  n_features_in_=768 n_samples_seen_=5740
ceiling_x  827077e84b40844ccd5aa30fb78b8b0c2571fe98c96d4e1bd9b62bd914892766  n_features_in_=2000 n_samples_seen_=5740
REUSES the loaded z scaler (lock:171) — no C3 scaler exists
NONE (execution note §10 D3E)
```

## Result-handling rules

```
tie       : D1 — exact float64 equality; select the LARGEST tied alpha; no rounding/tolerance
non-finite: D2 — non-finite validation R2 is ineligible; all seven non-finite -> selected_alpha=null, PROBE_DEGENERATE, no fallback
CI        : D4 — NO H-FM4 confidence interval
bootstrap : D4 — NO H-FM4 bootstrap
p-value   : D4 — NO p-value
threshold : no absolute recoverability threshold (lock:393)
reporting : all 4 arms x 4 components = 16 cells reported; no composite; no cherry-pick
repeats   : REPEAT_1..3 are deterministic re-executions for determinism verification, NOT seeds; selection uses the single deterministic value (REPEAT_1)
```

## Synthetic fail-closed guard tests

| test | expected | observed | result |
|---|---|---|---|
| thread_policy_failure | PhaseBError: thread variable not 1 | `PhaseBError: thread variables not set to 1 before interpreter start: ['OMP_NUM_THREADS']` | **PASS** |
| wrong_scaler_sha | PhaseBError: scaler artifact SHA-256 mismatch | `PhaseBError: z scaler artifact SHA-256 mismatch: ebf55308f3c6e62e43cd8b9aba084413549036c0f7e23a987ee6b8b390672e0e` | **PASS** |
| split_test_request | PhaseBError before any test file is opened | `PhaseBError: FORBIDDEN SPLIT REQUESTED: 'test' — H-FM4 is train/validation only` | **PASS** |
| split_external_request | PhaseBError before any external file is opened | `PhaseBError: FORBIDDEN SPLIT REQUESTED: 'RPE1' — H-FM4 is train/validation only` | **PASS** |
| scaler_refit_forbidden_api | PhaseBError from phaseB_preprocessing.refit_forbidden | `PhaseBError: A10: Phase-B scientific runners must LOAD canonical scaler state, never refit.` | **PASS** |
| standardscaler_fit_instrumentation | PhaseBError from the patched StandardScaler.fit, and the counter increments | `PhaseBError: A10 §4: Phase-B scientific runners must LOAD canonical scaler state, never refit (StandardScaler.fit is forbidden).` | **PASS** |
| standardscaler_fit_transform_instrumentation | PhaseBError from the patched StandardScaler.fit_transform | `PhaseBError: A10 §4: Phase-B scientific runners must LOAD canonical scaler state, never refit (StandardScaler.fit_transform is forbidden).` | **PASS** |
| duplicate_keys | PhaseBError: duplicate shard keys | `PhaseBError: duplicate shard keys` | **PASS** |
| missing_keys | PhaseBError: keys missing from bridge | `PhaseBError: keys missing from bridge` | **PASS** |
| existing_output_dir | FileExistsError: output root already exists | `FileExistsError: output root already exists (no reuse, no overwrite): /home/mehrdad/iclr2027_phaseB_out/phaseB/hfm4_tv_20260829T082843Z_b63c2a0` | **PASS** |
| nonfinite_mixed | finite max 0.7 at alpha=1 selected; 4 non-finite ineligible | `selected_alpha=1.0 r2=0.7 n_finite=3 status=OK` | **PASS** |
| exact_tie_largest_alpha | alphas {1e-2, 1, 100} exactly tied at 0.42 -> select 100.0 | `selected_alpha=100.0 tie_applied=True tied_alphas=[0.01, 1.0, 100.0]` | **PASS** |
| all_nonfinite_probe_degenerate | selected_alpha=None, status=PROBE_DEGENERATE | `selected_alpha=None status=PROBE_DEGENERATE` | **PASS** |

StandardScaler instrumentation proof (counters after the deliberate synthetic calls): `{'fit': 1, 'fit_transform': 1}` — reset to 0 before the real run.

