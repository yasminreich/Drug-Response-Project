# Predicting Cancer Drug Response from Gene Expression

Predicting cancer cell-line **sensitivity to Gemcitabine** (`LN_IC50`) from gene-expression
profiles, and asking a sharper question underneath it: **how much of the signal is just tissue
type, and how much do genes genuinely add on top?**

> `LN_IC50` = log half-maximal inhibitory concentration. **Lower = more sensitive** to the drug.

## Why Gemcitabine

Across the merged dataset, Gemcitabine has the **highest `LN_IC50` variance (9.21)** among drugs
tested on ≥ 690 cell lines — the most signal to model, and a drug with known expression-based
sensitivity biomarkers (e.g. *SLFN11*, *DCK*) to sanity-check against.

## Data

Three public datasets, joined via two ID bridges:

```
GDSC2 (COSMIC_ID) ──→ Model.csv (COSMIC_ID / ModelID) ──→ DepMap Expression (ModelID)
```

| File | Contents | Notes |
|---|---|---|
| `GDSC2_fitted_dose_response_27Oct23.csv` | Drug response (`LN_IC50`) per cell line | tracked in git |
| `Model.csv` | Bridges COSMIC_ID ↔ DepMap ModelID + lineage | tracked in git |
| `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | ~19k gene TPM expression matrix | **518 MB, gitignored** — download from [DepMap](https://depmap.org/portal/) |

After the 3-way join and a variance filter: **694 cell lines × 16,820 genes** — a classic
**p ≫ n** problem (far more features than samples), which drives most of the modelling choices below.

## Method & pipeline

Everything is **leak-free**: scaling and feature selection are fit **inside each CV fold only**.
Evaluation is 5-fold cross-validation, **stratified by tissue lineage** (`OncotreeLineage`), reported
as **pooled out-of-fold R²** (one honest estimate over all 694 cell lines).

The **tissue confound** is central: blood cancers (myeloid/lymphoid) form a distinct, more-sensitive
cluster, so tissue predicts *both* expression and response. Every gene model is therefore compared
against a **lineage-only baseline**, and re-checked on **solid tumours only** (blood cancers removed).

Notebooks run in order; each depends on the previous. Notebooks 03–05 reuse arrays saved to
`output/` so they never reload the 518 MB matrix.

| Notebook | What it does |
|---|---|
| `notebooks/01_initial_EDA.ipynb` | Loading, quality checks, 3-way merge, drug selection, `LN_IC50` distribution |
| `notebooks/02_preprocessing.ipynb` | Rebuild merge, feature matrix, lineage encoding, variance filter, Pearson/PCA *exploration* |
| `notebooks/03_baselines_comparison.ipynb` | Leak-free 5-fold comparison harness; per-lineage + solid-tumour-only confound checks; **nested CV** (Section 8) |
| `notebooks/04_mlp.ipynb` | PyTorch MLP (BatchNorm + Dropout) on the best representation, same folds, vs Ridge |
| `notebooks/05_interpretability.ipynb` | SHAP on a holdout split + biology check (known Gemcitabine genes) |

## Results

Pooled out-of-fold R² (5-fold stratified CV, n = 694). Higher is better.

| Model | Pooled OOF R² |
|---|---|
| **ElasticNet (l1_ratio tuned, nested CV)** — best | **0.46** |
| ElasticNetCV (l1_ratio = 0.5) | 0.45 |
| XGBoost (top-1000 MI) | 0.38 |
| All genes (Ridge) | 0.36 |
| Top-K MI → Ridge (nested CV) | 0.35 |
| PCA-50 → Ridge | 0.33 |
| RandomForest (top-1000 MI) | 0.32 |
| MLP (top-1000 genes) | 0.27 |
| Lineage-only (confound baseline) | 0.20 |

**Headline findings:**
- **Genes roughly double the tissue-only signal** (0.20 → ~0.46). Expression carries real
  predictive information beyond tissue type.
- **The signal is not just the blood-cancer shortcut** — solid-tumours-only R² stays positive
  (≈ 0.21).
- **Biology checks out.** Top SHAP gene is **SLFN11**, a well-known DNA-damage / Gemcitabine
  sensitivity biomarker; **DCK** (the canonical Gemcitabine-activating enzyme) surfaces further down.

## What we tried, and what didn't work

Being honest about negative results — several reasonable ideas did **not** beat regularized linear
models, which is expected at p ≫ n with n ≈ 694:

- **A neural network (MLP) underperformed** (R² 0.27) vs. regularized linear (~0.46). With ~19k
  features and ~694 samples, there isn't enough data for a deep model to win — regularized linear is
  the right tool here.
- **Tree ensembles didn't beat linear either.** XGBoost (0.38) was respectable — it edged out Ridge
  on the same features — but couldn't match ElasticNet's L1/L2 selection across all genes;
  RandomForest (0.32) landed near the tissue-only baseline. At p ≫ n, signal is spread thinly across
  many weak genes (a near-additive structure linear models exploit), while greedy tree splits chase a
  few features and overfit. Same lesson as the MLP: extra flexibility doesn't pay at n ≈ 694.
- **PCA-50 → Ridge (0.33) lost to keeping the genes** (all-genes Ridge 0.36; ElasticNet 0.46).
  Compressing to 50 components discards useful signal *and* the gene-level interpretability.
- **The number of selected genes (top-K MI) barely matters.** Nested CV picked K inconsistently
  across folds (500–2000) with near-identical scores — the selection is not the lever.
- **Nested CV barely moved the numbers** (< 0.02 R² in both directions) — the earlier estimates were
  only mildly optimistic. Its value was methodological rigor, not a better score. ElasticNet prefers
  a near-**Lasso** `l1_ratio` (0.9–1.0), not the originally hardcoded 0.5.

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the full running log of each attempt, result, and takeaway.

## Reproducing

All work runs inside Docker (`drug_response_env`) — see [`CLAUDE.md`](CLAUDE.md) for the exact build
and execute commands. In short:

```bash
docker build -t drug_response_env .
docker run --rm -v "<absolute-repo-path>":/app drug_response_env \
    jupyter nbconvert --to notebook --execute --inplace notebooks/<notebook>.ipynb
```

Plots and intermediate arrays are written to `output/` (gitignored).

## Roadmap

- ~~**Phase 1** — nested CV for unbiased hyperparameter selection.~~ ✅ done
- ~~**Phase 2** — non-linear tree models (RandomForest, XGBoost) in the same harness.~~ ✅ done
- **Phase 3** — a deeper neural approach: unsupervised autoencoder on the full expression matrix,
  then a regressor on the learned embedding; optionally multi-task across several drugs.
- **Phase 4** — finalize documentation.
