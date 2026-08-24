# Contributing

Thanks for taking a look. This is a research repo, so the conventions below exist mostly to keep
results **reproducible and honest** rather than to enforce style.

## Environment: Docker only

Everything runs inside the `drug_response_env` image. Please don't `pip install` into your host
Python or run notebooks locally — the pinned versions in `requirements.txt` are the ones that
produced the published numbers, and results are version-sensitive.

```bash
docker build -t drug_response_env .
```

Convenience wrappers (identical commands, no absolute paths to paste):

| Task | macOS / Linux | Windows |
|---|---|---|
| Build image | `make build` | `.\tasks.ps1 build` |
| Run tests | `make test` | `.\tasks.ps1 test` |
| Lint | `make lint` | `.\tasks.ps1 lint` |
| Execute a notebook | `make notebook NB=03_baselines_comparison` | `.\tasks.ps1 notebook -Notebook 03_baselines_comparison` |
| Jupyter server | `make jupyter` | `.\tasks.ps1 jupyter` |

`make` is not installed on Windows by default, which is why `tasks.ps1` exists.

## Getting the data

`data/GDSC2_fitted_dose_response_27Oct23.csv` and `data/Model.csv` are tracked. The ~518 MB DepMap
expression matrix is **gitignored** and must be downloaded separately — see the Data section of
[`README.md`](README.md). The test suite does **not** need it.

## The two rules that must never be broken

Both are enforced in [`src/data.py`](src/data.py) and covered by tests. They fail *silently* if
broken, which is why they have dedicated regression tests:

1. **Filter expression to `IsDefaultEntryForModel == "Yes"`** — a *string*, not a boolean. Comparing
   against `True` matches nothing and returns an empty frame.
2. **Deduplicate on `["COSMIC_ID", "DRUG_NAME", "ModelID"]` only.** Adding a column that contains
   NaNs — `TCGA_DESC` especially — silently drops those rows, because `pandas.groupby` defaults to
   `dropna=True`.

**Never re-inline the merge in a notebook.** If the merge needs to change, change `src/data.py` and
re-run the affected notebooks. Notebooks 01, 02, 03 and 06 import it; 04 and 05 read only the arrays
notebook 03 saves to `output/`.

## Methodology

- **No leakage.** Scaling and *supervised* feature selection are fit inside CV folds / on training
  rows only. The unsupervised variance filter may be applied once, outside the loop.
- **Stratify by `OncotreeLineage`**, not `TCGA_DESC` (which has nulls). Collapse lineages with fewer
  than 5 cell lines to `"Other"`.
- **Always report against the lineage-only baseline.** Tissue is a confound — blood cancers are both
  distinct in expression and more sensitive — so a gene model that doesn't beat tissue alone hasn't
  shown anything.
- Report **pooled out-of-fold R²** over the shared folds in `output/cv_folds.npz`, so every model in
  the results table is comparable.

## Notebooks

Notebooks are hand-maintained `.ipynb` JSON and are committed **with their outputs**, deliberately —
it means results render on GitHub without anyone running Docker. `.gitattributes` marks them `-diff`
so the embedded base64 images don't produce unreadable diffs.

After editing a notebook, re-execute it in place so its stored outputs match its code:

```bash
make notebook NB=03_baselines_comparison
```

Notebooks 03 (~50 min) and 06 (~40 min) are slow; 01 and 02 take a couple of minutes.

## Tests and lint

```bash
make test    # 37 tests, runs without the 518 MB matrix
make lint    # ruff
```

CI runs both on every push and PR, plus a Docker build to prove the documented commands still work.

## Changing dependencies

`requirements.txt` is the human-readable list of direct dependencies; `requirements.lock` is the
full transitive freeze that Docker actually installs. Editing one without the other does nothing —
or worse, silently diverges. The sequence is:

```bash
# 1. edit requirements.txt
docker build -t drug_response_env .
docker run --rm drug_response_env pip freeze > requirements.lock
make notebook NB=03_baselines_comparison    # 3. confirm EXPERIMENTS.md still holds
```

Dependabot handles GitHub Actions updates automatically, but **pip version updates are switched
off** — see the comments in `.github/dependabot.yml`. Security advisories still open PRs. This is
deliberate: bumping scikit-learn, xgboost or torch invalidates the published results table until
notebook 03 is re-run, and Dependabot cannot regenerate the lock.

## Reporting results

Every modelling phase gets an appended entry in [`EXPERIMENTS.md`](EXPERIMENTS.md) — motivation, what
was tried, a results table, and the takeaway. Write it **as the phase happens**, not retroactively.

**Negative results belong in the log and in the README.** Several reasonable ideas here did not beat
a regularized linear model, and saying so plainly is more useful than quietly dropping them.
