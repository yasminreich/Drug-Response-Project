# Experiments Log

A running log of models/attempts as the project evolves — what was tried, the result, and the
takeaway. Newest phase at the bottom. Metrics are **pooled out-of-fold R²** on Gemcitabine
`LN_IC50`, 5-fold stratified CV (stratified on `OncotreeLineage`, seed 42, n=694), unless noted.

## Baseline (pre-Phase-1, notebook 03 Sections 0–7)

| Model | Pooled OOF R² |
|---|---|
| Lineage-only (confound baseline) | 0.200 |
| ElasticNetCV (l1_ratio=0.5 fixed) | 0.448 |
| All genes (Ridge) | 0.362 |
| Top-1000 MI (Ridge) | 0.356 |
| Top-500 MI (Ridge) | 0.356 |
| Top-2000 MI (Ridge) | 0.344 |
| PCA-50 (Ridge) | 0.334 |

Genes roughly double the tissue-only signal; solid-tumours-only R² stays positive (not just the
blood-cancer shortcut). Established in the initial modelling pipeline.

---

## Phase 1 — Nested CV (unbiased hyperparameter selection)

**Date:** 2026-07-23 · **Where:** `notebooks/03_baselines_comparison.ipynb`, Section 8

**Motivation.** The baseline table reports `Top-500/1000/2000 MI` as three separate models and uses a
fixed `ElasticNet l1_ratio=0.5`. Reading the *best* Top-K off that table tunes K on the same
out-of-fold data used to score it → optimistically biased. (Ridge `alpha` via `RidgeCV`/GCV and
`ElasticNetCV`'s internal alpha search were already train-only, so not leaky.)

**What was tried.** Nested CV: an inner 5-fold CV on each outer fold's *training rows only* picks the
hyperparameters; the winning config is refit on the full outer-train and scored once on the untouched
outer-test.
- **Top-K MI → Ridge (nested):** inner CV selects `K* ∈ {500, 1000, 2000}` per fold.
- **ElasticNetCV (l1 tuned, nested):** `l1_ratio ∈ {.1,.5,.7,.9,.95,.99,1}` **and** `alpha` selected
  by inner CV on outer-train only.

**Result — biased vs unbiased (pooled OOF R²):**

| Selection | Biased | Unbiased (nested) | Δ |
|---|---|---|---|
| Top-K MI → Ridge | 0.356 (best K post-hoc) | **0.346** | −0.010 |
| ElasticNet | 0.448 (l1=0.5 fixed) | **0.460** (l1 tuned) | +0.012 |

Per-fold choices (`output/nested_hp_choices.csv`): `K*` = 500, 2000, 500, 500, 1000 (unstable — K
barely matters); ElasticNet `l1_ratio*` = 0.9, 0.95, 1.0, 0.9, 1.0 (consistently near-Lasso, i.e.
the data prefers sparse selection over the fixed 0.5).

**Takeaway.** Honest nested tuning moves the numbers by **< 0.02 R² in both directions**, so the
earlier ranking holds: the Top-K MI estimate was only mildly optimistic (−0.010 once de-biased), and
tuning ElasticNet's `l1_ratio` toward Lasso gives a small, real gain (+0.012). **ElasticNet remains
the best model at ~0.46**, comfortably above the lineage-only 0.20 baseline. The main value here is
methodological: the reported R² is now an unbiased estimate, not a max-over-configs.

**Artifacts:** `output/model_comparison_nested.csv`, `output/nested_hp_choices.csv`,
`output/nested_cv_comparison.png`.

---

## Phase 2 — Non-linear tree models (RandomForest + XGBoost)

**Date:** 2026-07-23 · **Where:** `notebooks/03_baselines_comparison.ipynb`, Section 9
(added `xgboost` to `requirements.txt`; Docker image rebuilt)

**Motivation.** Do non-linear models capture drug-response structure the linear models miss? Add
`RandomForestRegressor` and `XGBRegressor` to the harness, same folds, same representation as
`Top-1000 MI (Ridge)` (per-fold top-1000 MI genes from Section 3 — leak-free; trees are
scale-invariant so raw values used). Regularized defaults, **not tuned** (RF: 500 trees,
`max_features="sqrt"`, `min_samples_leaf=2`; XGB: 600 rounds, `lr=0.03`, `max_depth=3`,
`subsample=0.8`, `colsample_bytree=0.5`).

**Result (pooled OOF R²):**

| Model | Pooled OOF R² |
|---|---|
| ElasticNet (nested, l1 tuned) — best linear | 0.460 |
| **XGBoost** (top-1000 MI) | **0.384** |
| All genes (Ridge) | 0.362 |
| Top-1000 MI (Ridge) | 0.356 |
| **RandomForest** (top-1000 MI) | **0.316** |
| Lineage-only baseline | 0.200 |

**Takeaway.** **Neither tree beats the best linear model** (best tree 0.384 vs ElasticNet 0.460;
Δ −0.076). Nuance: XGBoost (0.384) is respectable — it edges out *Ridge* on the same top-1000
features (0.356) and even all-genes Ridge (0.362) — but it can't match ElasticNet's L1/L2 selection
across all ~16.8k genes. RandomForest (0.316) lands near the bottom, barely above the tissue-only
baseline, and XGBoost's per-fold R² is the most unstable in the table (std 0.11 vs ~0.05 for linear).

**Why, in plain terms (p ≫ n).** With ~16.8k mostly-noisy genes and only 694 samples, the gene→IC50
signal is spread thinly across many weakly-predictive genes — a smooth, near-additive structure that
penalized linear models exploit efficiently. Greedy, axis-aligned tree splits instead commit hard to
a few features and overfit sparse data; boosting recovers some of that (hence XGBoost > RF > single
Ridge subset), but not enough to overtake regularized linear. Consistent with the MLP result: at this
sample size, added model flexibility doesn't pay.

**Artifacts:** `output/model_comparison_full.csv`, `output/model_comparison_trees.png`.
