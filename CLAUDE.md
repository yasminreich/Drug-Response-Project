# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Predict cancer cell line sensitivity (LN_IC50) to drugs from gene expression data using deep learning. The selected target drug is **Gemcitabine** — highest LN_IC50 variance (9.21) among drugs with ≥690 cell lines in the merged dataset.

## Environment

> **Always use Docker (`drug_response_env`) — never run notebooks or scripts locally with pip/python.**

**Build the image (once, or after changing requirements):**
```bash
docker build -t drug_response_env .
```

> **Daemon must be running.** If `docker` fails with a pipe/daemon error, start Docker Desktop first:
> `Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"`, then poll `docker info` until
> ready (~1 min). On Git-Bash, prefix docker commands with `MSYS_NO_PATHCONV=1` so `/app` args aren't
> rewritten to Windows paths. `requirements.txt` includes `xgboost` — rebuild the image after editing it.

**Execute a notebook non-interactively (e.g. to run and save outputs):**
```bash
docker run --rm \
    -v "C:/Users/yasmi/projects/Drug_Response_DL/Drug_Response_Project":/app \
    drug_response_env \
    jupyter nbconvert --to notebook --execute --inplace notebooks/<notebook>.ipynb
```

> **Windows note:** Use the absolute path for `-v` instead of `$(pwd)` to avoid Docker volume mount issues on Windows.

**Open Jupyter in the browser:**
```bash
docker build -t drug_response_env .
docker run -it --rm \
    -v $(pwd):/app \
    -p 8888:8888 \
    drug_response_env \
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
Then open the URL printed in the terminal (e.g. `http://127.0.0.1:8888/?token=...`) in your browser. The `-v $(pwd):/app` mount means any changes made inside the notebook are saved directly to your local files.

## Data

All data files live in `data/`. The large expression matrix is gitignored and must be obtained separately:

| File | Size | Status |
|---|---|---|
| `GDSC2_fitted_dose_response_27Oct23.csv` | 35 MB | tracked in git |
| `Model.csv` | 684 KB | tracked in git |
| `GDSC2_fitted_dose_response_27Oct23.xlsx` | 21 MB | **gitignored** — redundant duplicate of the `.csv`; nothing reads it |
| `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | 518 MB | **gitignored** — download from DepMap |

## Architecture & Data Pipeline

Three datasets are joined via two ID bridges:

```
GDSC2 (COSMIC_ID) ──→ Model.csv (COSMIC_ID / ModelID) ──→ Expression (ModelID)
```

**Merge result:** ~698 usable cell lines × 19,215 gene features after 3-way join.

**Key deduplication rule:** Group by `['COSMIC_ID', 'DRUG_NAME', 'ModelID']` only — never include `TCGA_DESC` or other metadata in the groupby keys, as NaN values in those columns cause rows to be silently dropped (pandas `dropna=True` default).

**Expression matrix:** Filter to `IsDefaultEntryForModel == 'Yes'` (string, not boolean) before merging — ensures one expression profile per cell line.

## Plot Output Convention

All notebooks must save figures to `output/` (gitignored) at `dpi=150` with `bbox_inches="tight"`. Set this up once in the setup cell:

```python
import os
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

Then before every `plt.show()`:

```python
plt.savefig(os.path.join(OUTPUT_DIR, "descriptive_name.png"), dpi=150, bbox_inches="tight")
plt.show()
```

Use lowercase snake_case filenames that describe the plot content (e.g. `gemcitabine_ln_ic50_distribution.png`).

## Code Structure

Notebooks run in order; each depends on the prior. Notebooks 04–06 reuse arrays saved by 03 to
`output/` (gitignored), so they do **not** reload the 518 MB expression matrix — except notebook 06,
which loads it once more to reach the unlabelled profiles its autoencoder needs.

- `notebooks/01_initial_EDA.ipynb` — Full EDA: loading, quality checks, 3-way merge, drug selection (Gemcitabine), LN_IC50 distribution.
- `notebooks/02_preprocessing.ipynb` — Rebuilds merge, builds feature matrix, lineage encoding, variance filter, Pearson + PCA *exploration* (stops before modelling).
- `notebooks/03_baselines_comparison.ipynb` — Leak-free comparison harness (5-fold stratified CV). Sections 0–7: lineage-only baseline, all-genes Ridge, top-N MI, ElasticNetCV, PCA contrast; per-lineage + solid-tumour-only confound analysis; persists `X`/`y`/folds/genes to `output/`. **Section 8 (Phase 1):** nested CV for unbiased hyperparameter selection (K, ElasticNet l1_ratio/alpha). **Section 9 (Phase 2):** RandomForest + XGBoost on the top-1000 MI representation, same folds.
- `notebooks/04_mlp.ipynb` — PyTorch MLP (BatchNorm + Dropout) on the best representation, same folds, vs Ridge.
- `notebooks/05_interpretability.ipynb` — SHAP on the holdout split + biology check (DCK/SLC29A1 ranks vs Pearson).
- `notebooks/06_autoencoder.ipynb` — **Phase 3:** denoising autoencoder over all 1,699 expression
  profiles (one AE per fold, leak-free) → Ridge on the 64-d embedding, plus a transductive contrast;
  multi-task MLP across the 12 best-covered drugs vs a single-task control. Same folds as 03.
- `src/data.py` — **Single source of truth for the 3-way merge**, dedup, variance filter, lineage
  collapsing and the stratified fold/holdout builders. Notebooks **01, 02, 03 and 06** import from
  here rather than each keeping their own copy; 04 and 05 touch no raw data, reading only the arrays
  03 saves to `output/`. Use `join_gdsc_to_models` / `dedup_rows` for all-drug work (notebook 01's
  drug selection) and `build_merged` / `dedup_drug_rows` for a single target drug. Replaces the old
  `preprocess_data.py`, which skipped both mandatory rules below and would have produced a
  different, wrong dataset.
- `tests/` — pytest suite covering the dedup rule, the `IsDefaultEntryForModel` filter, the
  no-leakage contract, and split determinism/stratification. Runs without the 518 MB matrix
  (synthetic expression fixture joined to the real tracked `Model.csv`/GDSC2 CSV).
- `.github/workflows/ci.yml` — CI: runs the tests and verifies the Docker image builds.
- `README.md` — Human-facing project overview (goal, data, pipeline, results, what-didn't-work).
- `EXPERIMENTS.md` — Running log of each modelling attempt; **append a new entry per phase** (do not write it retroactively).

> **Notebooks import shared logic from `src/data.py`.** Because they execute with
> `cwd=notebooks/`, they prepend the repo root to `sys.path` before importing. Never
> re-inline the merge in a notebook — change `src/data.py` and re-run.

> **Notebooks are hand-maintained `.ipynb` JSON** — there are no builder scripts. Edit them directly
> (NotebookEdit tool), then execute in Docker with `nbconvert --execute --inplace`.

## Methodology Rules (must follow)

- **No leakage:** scaling and feature selection are fit **inside CV folds / on train only**, via
  `sklearn.pipeline.Pipeline` or manual per-fold fitting. The Pearson/PCA on all 694 rows in
  notebook 02 is *exploration only* — it must never choose final features.
- **Stratify by `OncotreeLineage`** (not `TCGA_DESC`, which has nulls); collapse lineages with
  < 5 cell lines to `"Other"`. Tissue is a confound (blood cancers are distinct + more sensitive),
  so always compare against the **lineage-only baseline**.

## Results So Far (Gemcitabine, 5-fold stratified CV, pooled OOF R²)

| Model | R² |
|---|---|
| Lineage-only (confound baseline) | 0.20 |
| **ElasticNet (nested, l1 tuned)** (best) | **0.46** |
| ElasticNetCV (l1=0.5) | 0.45 |
| XGBoost (top-1000 MI) | 0.38 |
| All genes / Top-1000 MI (Ridge) | 0.36 |
| Top-K MI → Ridge (nested) | 0.35 |
| PCA-50 (Ridge) | 0.33 |
| RandomForest (top-1000 MI) | 0.32 |
| Autoencoder-64 → Ridge (Phase 3) | 0.30 |
| MLP (top-1000 genes) | 0.27 |
| Multi-task MLP, 12 drugs (Phase 3) | 0.29 |

Genes roughly **double** the tissue-only signal. Solid-tumours-only R² ≈ 0.21 (positive ⇒ not just
the blood-cancer shortcut). SHAP top gene is **SLFN11** (known DNA-damage/Gemcitabine sensitivity
biomarker); **DCK** ranks ~279/1000; SLC29A1 is weak in this cohort.

**Nothing beats regularized linear at p≫n, n≈694** (expected): the MLP (0.27), RandomForest (0.32),
XGBoost (0.38), the autoencoder embedding (0.30) and the multi-task MLP (0.29) all trail
ElasticNet (0.46). XGBoost edges out Ridge on the same features but not
ElasticNet's L1/L2 selection over all genes. **Phase 1** confirmed the earlier R² numbers were only
mildly optimistic — nested CV shifts them < 0.02, and ElasticNet prefers a near-Lasso `l1_ratio`
(0.9–1.0), not the originally hardcoded 0.5. See `EXPERIMENTS.md` for the full per-phase log.

## Extension Plan (4 phases, one at a time — confirm results with the user before starting the next)

- ~~**Phase 1** — nested CV for unbiased hyperparameter selection.~~ ✅ done (notebook 03, Section 8)
- ~~**Phase 2** — RandomForest + XGBoost in the harness.~~ ✅ done (notebook 03, Section 9)
- ~~**Phase 3** — autoencoder on the full expression matrix + multi-task MLP.~~ ✅ done (notebook 06).
  Neither beat ElasticNet: AE-64 → Ridge 0.296 (below even PCA-50's 0.334), multi-task 0.290 vs a
  0.294 single-task control. The transductive AE gained only +0.005 over the leak-free per-fold
  protocol, so strict per-fold refitting costs essentially nothing.
- ~~**Phase 4** — finalize documentation and engineering.~~ ✅ done. `src/data.py` is imported by
  every notebook that touches raw data (01/02/03/06); 37 tests; CI runs lint + tests + Docker build;
  `requirements.lock` pins the full transitive tree (pinning only direct deps silently broke clean
  builds); MIT license, `CITATION.cff`, `CONTRIBUTING.md`, `Makefile`/`tasks.ps1`.

**All four phases are complete.** If work resumes, the highest-value directions are external
validation on an independent panel, running the harness across many drugs (`src/data.py` already
takes a `target_drug` argument), or modelling within tissue — solid-tumours-only R² ≈ 0.21 is the
part that's actually unsolved. More model capacity is *not* promising: three phases of it all lost
to ElasticNet.

After each phase: append an `EXPERIMENTS.md` entry, update `README.md`, commit + push, and pause for
user confirmation.

## Git Workflow

After every meaningful change: stage, commit with a descriptive message, and push to `origin main`.
```bash
git add <files>
git commit -m "concise description of what and why"
git push
```
The expression matrix (`OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`) must never be committed — it is gitignored.
