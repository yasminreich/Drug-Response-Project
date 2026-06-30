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
| `GDSC2_fitted_dose_response_27Oct23.xlsx` | 21 MB | tracked in git |
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

Notebooks run in order; each depends on the prior. Notebooks 03–05 reuse arrays saved by 03 to
`output/` (gitignored), so they do **not** reload the 518 MB expression matrix.

- `notebooks/innitial_EDA.ipynb` — Full EDA: loading, quality checks, 3-way merge, drug selection (Gemcitabine), LN_IC50 distribution.
- `notebooks/02_preprocessing.ipynb` — Rebuilds merge, builds feature matrix, lineage encoding, variance filter, Pearson + PCA *exploration* (stops before modelling).
- `notebooks/03_baselines_comparison.ipynb` — Leak-free comparison harness (5-fold stratified CV): lineage-only baseline, all-genes Ridge, top-N MI, ElasticNetCV, PCA contrast; per-lineage + solid-tumour-only confound analysis. Persists `X`/`y`/folds/genes to `output/`.
- `notebooks/04_mlp.ipynb` — PyTorch MLP (BatchNorm + Dropout) on the best representation, same folds, vs Ridge.
- `notebooks/05_interpretability.ipynb` — SHAP on the holdout split + biology check (DCK/SLC29A1 ranks vs Pearson).
- `src/preprocess_data.py` — Early standalone merge script (superseded by the notebooks; kept for reference).

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
| **ElasticNetCV** (best) | **0.45** |
| All genes / Top-1000 MI (Ridge) | 0.36 |
| PCA-50 (Ridge) | 0.33 |
| MLP (top-1000 genes) | 0.27 |

Genes roughly **double** the tissue-only signal. Solid-tumours-only R² ≈ 0.21 (positive ⇒ not just
the blood-cancer shortcut). The MLP does **not** beat regularized linear (expected at p≫n, n≈694).
SHAP top gene is **SLFN11** (known DNA-damage/Gemcitabine sensitivity biomarker); **DCK** ranks
~279/1000; SLC29A1 is weak in this cohort.

## Next Steps (Planned)

1. **Tune / extend** — hyperparameter search for ElasticNet & MLP; try gene-set (pathway) features.
2. **Confound modelling** — explicitly residualize lineage, or per-lineage models, to isolate mechanism.
3. **Robustness** — nested CV for an unbiased estimate; external validation if another cohort is available.

## Git Workflow

After every meaningful change: stage, commit with a descriptive message, and push to `origin main`.
```bash
git add <files>
git commit -m "concise description of what and why"
git push
```
The expression matrix (`OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`) must never be committed — it is gitignored.
