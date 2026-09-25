# Environment of record

Two distinct environments are involved. They are labeled by source: `RUN_RECORD` (written by the
governed run) or `CURRENT_MACHINE` (read from the machine's current state during the revision, which
is not by itself evidence about the original run).

## A. Foundation-model embedding environment — `RUN_RECORD`

The frozen Geneformer embeddings (the representation `z`) were produced under this environment, as
recorded in the smoke-gate manifest and the extraction run logs persisted with the run.

| Item | Value | Source (label) |
|---|---|---|
| conda environment | `[CONDA-ENV]` | RUN_RECORD (gate manifest) |
| Python | 3.10.20 | RUN_RECORD (gate manifest) |
| PyTorch / CUDA build | 2.3.1+cu121 | RUN_RECORD (gate manifest; oracle/throughput result JSON) |
| transformers | 4.49.0 | RUN_RECORD (gate manifest; extraction run log) |

**CONFLICT (transformers version).** The pre-result specification estimated the extraction environment
as `transformers ~4.44–4.46` (a forward-looking note in the reconciliation list, item M5). The persisted
run manifest and the actual extraction run log record `transformers 4.49.0`. Both are reported; the
run-written value (4.49.0) is the environment that actually produced the embeddings. No resolution is
asserted beyond stating both.

## B. Revision analysis environment — `CURRENT_MACHINE`

The revision-round analysis (classical heads, bootstraps, tabulation) was run under a separate project
virtual environment. These versions were read from the machine during the revision and are recorded in
a project environment/artifact map; they are **not** the environment that produced the embeddings, and a
version read off the current machine is not evidence about the original run unless a run record says so.

| Package | Version | Source (label) |
|---|---|---|
| Python | 3.13.12 | CURRENT_MACHINE (project venv `pyvenv.cfg`) |
| PyTorch / CUDA build | 2.11.0+cu130 | CURRENT_MACHINE (env/artifact map) |
| numpy | 2.4.4 | CURRENT_MACHINE |
| pandas | 2.3.3 | CURRENT_MACHINE |
| scipy | 1.17.1 | CURRENT_MACHINE |
| scikit-learn | 1.8.0 | CURRENT_MACHINE |
| anndata | 0.12.11 | CURRENT_MACHINE |
| scanpy | 1.12.1 | CURRENT_MACHINE |
| h5py | 3.16.0 | CURRENT_MACHINE |

`transformers`, `scanpy`, `anndata`, `numpy`, `scipy`, `scikit-learn`, `pandas` are **not** independently
recorded for the embedding run (only Python, PyTorch and transformers are in the run manifest); the values
above for those packages therefore describe the current revision machine, not the embedding run.

## C. Hardware — `CURRENT_MACHINE`

| Item | Value | Source (label) |
|---|---|---|
| GPUs | 2× NVIDIA TITAN RTX, 24 GB each | CURRENT_MACHINE (env/artifact map) |
| GPU driver | 580.126.20 | CURRENT_MACHINE |
| CUDA (driver) | 13.0 | CURRENT_MACHINE |

The embedding run itself was a `cu121` build (§A); the run-time GPU model is not separately recorded in
the run manifest beyond the machine map, so the hardware line is labeled `CURRENT_MACHINE`.

## D. Pinned upstream artifacts — `RUN_RECORD`

Repository commit and artifact digests were pinned in the pre-result specification before any result was
produced (full 64-character digests in `04_artifact_digests.md`).

- Geneformer repository git commit: `ad8f66dfcda3ebbd148d916c01f31339c5b95a15` — RUN_RECORD (spec).

## E. Environment file — gap

**No portable environment file (conda `environment.yml` or pip `requirements.txt`) is committed** for
either environment. The specification's reconciliation list flagged this as an open item (M5: "commit an
ICLR Geneformer environment file"), and it was not closed. The only machine-side descriptor is the
project virtual-environment config, which is `CURRENT_MACHINE` state, not a run record. This is a
reproducibility gap and is stated as one.

## Q5 source classification

`Q5_ENV_SOURCE: MIXED` — the embedding-run Python/PyTorch/transformers are `RUN_RECORD` (gate manifest);
the analysis-package versions and the hardware are `CURRENT_MACHINE`; the upstream commit and artifact
digests are `RUN_RECORD` (spec). One `CONFLICT` (transformers: spec estimate 4.44–4.46 vs run record 4.49.0).
