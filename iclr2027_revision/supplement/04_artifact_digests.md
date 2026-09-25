# Artifact digests

Full 64-character digests and the repository commit, pinned in the pre-result specification before any
result was produced (`RUN_RECORD`). These identify the exact frozen foundation-model artifacts used to
produce the representation `z`. All are properties of the public Geneformer release and its dictionaries;
none is an internal or identifying value.

| Artifact | Filename | Size (bytes) | Digest |
|---|---|---|---|
| Repository commit | (Geneformer repository, git) | — | `ad8f66dfcda3ebbd148d916c01f31339c5b95a15` |
| V2-104M checkpoint | `Geneformer-V2-104M/model.safetensors` | 417,571,156 | `fff5cba29ddd8792991fa77b4872246fbe548a178cebda3775cdc72b67780e7f` |
| Token dictionary | `token_dictionary_gc104M.pkl` | — | `67c445f4385127adfc48dcc072320cd65d6822829bf27dd38070e6e787bc597f` |
| Gene-median dictionary | `gene_median_dictionary_gc104M.pkl` | — | `a51c53f6a771d64508dfaf61529df70e394c53bd20856926117ae5d641a24bf5` |
| Ensembl mapping dictionary | `ensembl_mapping_dict_gc104M.pkl` | — | `0819bcbd869cfa14279449b037eb9ed1d09a91310e77bd1a19d927465030e95c` |

Digest algorithm: SHA-256 (repository commit is the git commit hash).

Source: pre-result specification, artifact-pinning table (`RUN_RECORD`). The repository commit and the
checkpoint digest are additionally echoed in the persisted smoke-gate manifest (`RUN_RECORD`).

## Harmonized endpoint archive (added R8)

The per-construct harmonized external outcome (`y_primary`) was archived into `endpoint/` from the removable
evaluation volume so it is durably co-located with the results. Copy only; originals left in place. SHA-256
verified identical between source and archived copy.

| Screen | Archived file | Rows | Columns | SHA-256 |
|---|---|---|---|---|
| RPE1 | `endpoint/rpe1_endpoint_by_construct.csv` | 1932 | construct, target_symbol, target_ensg, target_col, n_perturbed_pool, n_nt_pool, y_primary, y_mean, y_std, y_p5, y_p95, y_iqr, n_valid_draws, y_pooled, y_gemgroup, n_eligible_gemgroups, stored_ad, stored_ad_log1p | `353c04d3ac8e3380a3addf3abc1953d30799423f7cd467006adc36800e26a126` |
| K562-essential | `endpoint/k562_essential_endpoint_by_construct.csv` | 1903 | (same 18 columns) | `051269eeeae93c75c84ca1fc050529a096ebd15e1990cb14bb24669c1b9672ff` |

These are the outcome vectors used by the non-overlap sensitivity (amendment A7) and the primary external
transfer analysis. The digests above were computed with `sha256sum` on the archived copies.
