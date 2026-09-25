# Locked evaluation of CRISPRi perturbation-effect prediction

Code, result artifacts and the pre-registered protocol for:

> **Can We Trust In-Distribution Success? Locked Evaluation Reveals Transfer Failure
> and Sampling-Depth Entanglement in CRISPRi Perturbation Prediction**
> Mehrdad Shoeibi, Niloofar Yousefi (University of Central Florida)
> NeurIPS 2026 Workshop on Trustworthy AI Evaluation (TAE), Sydney. Poster, non-archival.
> OpenReview: https://openreview.net/forum?id=DsVQi1WTsP

The contribution is an evaluation protocol, not a model. A frozen Geneformer-V2-104M
representation is scored on the Virtual Cell Challenge (VCC) and on two external
Replogle CRISPRi screens under a protocol that freezes heads and model selection
before any test metric is computed, withholds the external outcome labels until a
single final unblinding, and fixes analysis-governing choices before the evaluations
they govern, logging every later amendment with its timing and scope.

## What is here

| Path | Contents |
|---|---|
| `iclr2027_revision/pipeline/` | the locked analysis pipeline: control construction, head selection and freezing, the magnitude block, the zero-shot transfer scoring, the bootstrap wiring |
| `iclr2027_revision/phase0,1,4/` | the result artifacts the paper reports |
| `iclr2027_revision/endpoint/` | the harmonized target-gene endpoint rebuilt per external screen |
| `iclr2027_revision/supplement/` | the experiment lock, the numbered amendment chain A1–A16, the environment record, artifact digests, governance |
| `iclr2027_revision/provenance/` | pre-run freezes, run manifests, guard tests, input-access manifests |
| `iclr2027_revision/smoke/` | the gate manifest and the validation runs that had to pass before the real run |
| `iclr2027_revision/env/` | the pinned Geneformer environment, with checkpoint and dictionary SHA-256 digests |
| `src/` | the loaders, features, model families, losses and metrics the pipeline imports |

## Reproducing the reported numbers

The headline values can be read straight out of the stored artifacts without
re-running anything:

```bash
python - <<'PY'
import json
t = json.load(open('iclr2027_revision/phase1/test_table.json'))
p = json.load(open('iclr2027_revision/phase4/phase4_primary.json'))['screens']
g = lambda s, a: p[s]['arm_rho'][a]['rho']
print("in-distribution dR2(z - C1) :", round(t['z']['r2_mean'] - t['C1']['r2_mean'], 4))
print("RPE1 transfer rho_z         :", round(g('RPE1','z'), 4))
print("K562 transfer rho_z         :", round(g('K562','z'), 4))
print("magnitude increment d_rho   :", round(g('RPE1','zm')-g('RPE1','z'), 4),
                                        round(g('K562','zm')-g('K562','z'), 4))
PY
```

which prints `+0.1645`, `-0.1392`, `-0.2671`, `+0.0318 +0.1429`, matching the paper.

The sampling-depth result is in `iclr2027_revision/phase4/phase4_confound_diagnostic.json`
(`corr_ncells_y: 0.7045` on VCC, `partial_max|x|: 0.379`).

## Running the pipeline from data

Data is not included. The paper uses public releases: the Virtual Cell Challenge
corpus and the Replogle RPE1 and K562-essential Perturb-seq datasets.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp storage_root.txt.example storage_root.txt   # then edit it to point at your data root
```

`iga_paths.py` resolves every path from `IGA_STORAGE_ROOT` or from `storage_root.txt`,
and fails loudly rather than guessing.

## Scope and honesty notes

This is a curated release, not a mirror of the authors' working archive. Eight files
were modified and two directories dropped, to remove material belonging to a separate
unpublished project. Each one is listed in `RELEASE_NOTES.md` with the original and
released SHA-256 hashes. None touches a code path that produced a reported result, and
every numeric claim in the paper's results sections is still recoverable from what
ships here.

Two redactions were also applied, and are disclosed under *Redactions* in
`RELEASE_NOTES.md`: the name of the venue where the conference version is under review,
and the name of the team member to whom individual decisions were attributed. Neither
changes a number, a decision rule, or a date. Both are present from the initial commit
rather than applied on top of an unredacted one, and the venue name will be restored
once that review concludes.

Some execution paths behind the persisted external-evaluation artifacts were not
fully retained, and the classical-baseline tables carried over from an earlier
submission were not re-derived under the locked harness. The paper states both.
Nothing reported here depends on them.

## License

MIT, see `LICENSE`.
