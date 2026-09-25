# Release notes

This repository is a curated public release, not a mirror of the authors' working
archive. It contains what reproduces the results reported in the paper, and nothing
from two other efforts that share the same working tree: a separate unpublished
method, and the drafting record of a manuscript still under review.

## Provenance

| | |
|---|---|
| Working project | `ICLR2027_2026-08-26` on host `titans` |
| Copied from | `iga-neurips` @ `dfdff1e634de94ea…`, 2026-08-26 21:14 EDT |
| Release commit | `c6a0050a689aabdfafaf62b6e643093b0a05df13`, 2026-08-30 03:18:40 EDT |
| Working tree at that commit | clean (`git status --porcelain` empty) |

Every file here is byte-identical to that commit except the two listed below.

## The 8 modifications made for this release

**`src/models/families.py`** — removed build_iga_bundle (53 lines), 2 imports, 1 registry entry

- original sha256 `a7fccc4afbd0d81e560d2cda1791405a32d7884b80702826e8b40e0f5461355d`
- released sha256 `4708a733b4296d3e106fcbb70787ec2af2d596b5308b238ec693e65a339649e3`

**`src/training/trainer.py`** — synthetic-generator import made TYPE_CHECKING-only; two annotations quoted

- original sha256 `704c74632b43dabaaf65e98eecb290403c52e91b945245ecc6233964e327e6e4`
- released sha256 `695030c97c4c03f62e1d4742cb43296bb284fb58b260ee7cfcc2a72d1b183658`

**`src/losses/composite.py`** — removed the optional 'gated' stability backend: 2 constructor arguments, their validation block, their attribute assignments, the dispatch arm, and 'gated' from the allowed-backend set, plus the gate-diagnostic variable that only that arm ever wrote. The mmd arm is unchanged.

- original sha256 `0d488040ca99d04b813d436e7dfc8ea14cbb365959ea4ea323dfae97819c28fe`
- released sha256 `efbb6ee1cc7379a5e3ea0d8d18b644dc70e16e07971d900448accd96b2cb8728`

**`src/data/vcc/gene_selection.py`** — docstring only: references to the separate project and to datasets it uses were reworded. No code changed.

- original sha256 `893729177d4b4ef79e70868b314497e169fd1fd6a0f8a49064911b3e0a87b472`
- released sha256 `b6fae642dbde0973ef995e12ff6fabff3f9551b19dd267c89b657384f00d1927`

**`src/data/vcc/schema.py`** — docstring only: references to the separate project and to datasets it uses were reworded. No code changed.

- original sha256 `f9399ef67f1d5f6a1dc75729cee1792376cf69e13509ed9a94a7005a0e2e3d62`
- released sha256 `1b0370b3b6af3b7831ed0e49298e3df1b4ea5747d094874e542e0b672d697a9d`

**`src/losses/prediction.py`** — docstring only: references to the separate project and to datasets it uses were reworded. No code changed.

- original sha256 `bb81ace38ccc6ceaf9a40ce7a0c85262a77bc0bbab86263e048169601af34678`
- released sha256 `61e089f07d66b25ed0bbf957cba855148cd2f9c0f6d3e56f0bcdaf9a9a2973dd`

**`src/analysis/ (removed)`** — the whole module analysed a diagnostic written by the separate project's training loop. It is not in the pipeline's import closure and no paper result uses it.

**`pyproject.toml`** — project name and description rewritten; they named the separate project. The `wilds` extra and the `higher` dependency, used only by that project, were removed.

Every one removes or reworks material belonging to a separate unpublished project.
None touches a code path that produced a reported result: every reported run uses
`stab_backend='mmd'`, and the paper's arms are `z`, `zm`, the five controls `C1`-`C5`,
and the SPSM deep baseline.

Two directories were also dropped whole, `iclr2027_revision/audits/` and
`iclr2027_revision/experiment_specs/`, because 8 of their files describe that project.
Dropping them costs nothing: every numeric claim in the paper's results sections is
still recoverable from what ships here (verified, 56 of 56).

## What was deliberately left alone

**Absolute paths.** 29 files mention `/home/mehrdad` or `MehrdadSSD`. 26 of them are
historical records (pre-run freezes, amendments, environment captures, audits) and
rewriting a record would falsify it. The other 3 are the fail-loud path guards in
`pipeline/run_hfm4_tv.py`, `run_dmech_tv.py` and `run_d_target_var.py`, where the
paths appear in a *deny list* that refuses to read the predecessor project's storage.
Editing those would weaken the guard. Nothing here is secret: it is a Unix username
and a disk label.

## Data

No data is included. The paper uses public releases: the Virtual Cell Challenge
corpus and the Replogle RPE1 and K562-essential Perturb-seq datasets. Paths are
resolved by `iga_paths.py`, which reads `IGA_STORAGE_ROOT` or a `storage_root.txt`
placed beside it, and refuses to guess. Copy `storage_root.txt.example` to
`storage_root.txt` and point it at your own data root.

## Redactions

Two redactions were applied to this release. Both mask information; neither alters a
number, a decision rule, a date, or a result. They are present in the initial commit
rather than applied on top of an unredacted one, so the masked text was never published;
this section is the disclosure that stands in for a redaction diff. The venue name will
be restored by a later commit once the conference version's review concludes.

**1. Venue name.** The conference version of this work is under review at the time of
release. Three prose statements named that venue and now read `[venue redacted]`:

- `supplement/01_experiment_lock.md`, document title
- `supplement/02_amendments/A11.md`, line describing the storage tree
- `supplement/part1_scientific.md`, the page-limit note

Directory and branch names that contain the venue string (`ICLR2027_2026-08-26`,
`IGA_ICLR2027_GENEFORMER`, `iclr2027_revision`, `iclr2027-r2`) were **not** changed.
They are literal path bindings, and three of them appear inside the fail-loud deny lists
that refuse to read the predecessor project's storage. Renaming them would break the
guards and falsify how the pipeline actually resolved paths.

**2. Role attribution.** Ten passages attributed a decision to a specific member of the
author team. The attribution was generalised ("the advisor" to "the author team" or
"pre-specified"). **Every decision rule survives verbatim.** The 1 September hard stop,
the cut-Option-B fallback, the GO / ESCALATE / NO-GO branches, and the "exactly one new
result" constraint all read exactly as they did when they were fixed; only the sentence
naming who set them has changed. Nothing was deleted.

Files touched by the two redactions: 8. Per-file original and redacted SHA-256:

- `iclr2027_revision/supplement/01_experiment_lock.md` &nbsp; `aa2c6d2f56944f50` to `9f51b2701ce2d2e9`
- `iclr2027_revision/supplement/02_amendments/A1.md` &nbsp; `b49b7896f687f76a` to `7666773fe18084fa`
- `iclr2027_revision/supplement/02_amendments/A11.md` &nbsp; `697f1a0033220f75` to `dd5bee88ad4082d6`
- `iclr2027_revision/supplement/02_amendments/A2.md` &nbsp; `c51106a17ad01bf5` to `04ccfedc472328fd`
- `iclr2027_revision/supplement/02_amendments/A3.md` &nbsp; `c43aca2c90fd7441` to `f42183408f34054a`
- `iclr2027_revision/supplement/02_amendments/A4.md` &nbsp; `0192c50ca60b82ff` to `664978795a611fb6`
- `iclr2027_revision/supplement/06_governance/kill_switch_27_construction_2026-08-29.md` &nbsp; `59e9240d71c57447` to `b481fcb54839bd29`
- `iclr2027_revision/supplement/part1_scientific.md` &nbsp; `9ffb8cf1ede2bf51` to `24e1d7520a252b6e`
