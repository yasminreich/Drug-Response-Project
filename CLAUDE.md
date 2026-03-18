# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Predict cancer cell line sensitivity (LN_IC50) to drugs from gene expression data using deep learning. The selected target drug is **Gemcitabine** — highest LN_IC50 variance (9.21) among drugs with ≥690 cell lines in the merged dataset.

## Environment

**Run locally with Jupyter:**
```bash
pip install -r requirements.txt
jupyter notebook
```

**Run in Docker:**
```bash
docker build -t drug_response_env .
docker run -p 8888:8888 drug_response_env
```
Then open the URL printed in the terminal (e.g. `http://127.0.0.1:8888/?token=...`) in your browser. Notebooks will open with the `drug_response_env` kernel.

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

## Code Structure

- `notebooks/innitial_EDA.ipynb` — Full EDA: data loading, quality checks, 3-way merge, drug selection (Gemcitabine), and LN_IC50 distribution analysis. Run all cells top-to-bottom; cells depend on prior state.
- `src/preprocess_data.py` — Early-stage preprocessing script. Loads and merges all three datasets. **Note:** does not yet apply variance filtering, standardisation, or train/val/test splitting — those are next steps.

## Next Steps (Planned)

1. **Feature selection** — remove near-zero-variance genes, apply variance threshold or mutual information filter against Gemcitabine LN_IC50
2. **Dataset construction** — build `X` (694 × ~19k genes) and `y` (LN_IC50), stratified train/val/test split by tissue type (`TCGA_DESC`)
3. **Modelling** — Ridge regression baseline → MLP with dropout + batch norm → SHAP interpretability (DCK and SLC29A1 expected as top features for Gemcitabine)

## Git Workflow

After every meaningful change: stage, commit with a descriptive message, and push to `origin main`.
```bash
git add <files>
git commit -m "concise description of what and why"
git push
```
The expression matrix (`OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`) must never be committed — it is gitignored.
