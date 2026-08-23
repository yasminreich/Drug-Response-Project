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

---

## Phase 3 — Deep representations: autoencoder embeddings & multi-task learning

**Date:** 2026-08-23 · **Where:** `notebooks/06_autoencoder.ipynb`

**Motivation.** Every model so far only saw the **694 labelled** cell lines. The DepMap expression
matrix holds **1,699** default profiles — **2.45x more** than we have Gemcitabine labels. Two ways to
buy signal that don't require new labels:
1. **Unsupervised pretraining** — learn a representation from all 1,699 profiles, regress on the
   embedding.
2. **Multi-task learning** — predict several drugs at once and let a shared trunk pool the signal.

**What was tried.** Same 5 folds as notebook 03 throughout, so numbers are directly comparable.

- **Denoising autoencoder** over all 16,820 variance-filtered genes
  (16,820 -> 512 -> 128 -> **64** -> 128 -> 512 -> 16,820), input dropout 0.1, Adam (lr 2e-3,
  wd 1e-5), batch 256, <=100 epochs with early stopping on a 10% *reconstruction* holdout (never on
  `y`). Then `RidgeCV` on the 64-d embedding.
  **Leakage protocol:** the 1,005 unlabelled profiles are in no test fold, so they are always safe;
  labelled cell lines are restricted to the fold's training rows, so **one autoencoder is trained
  per fold** (1,560 rows each).
- **Transductive variant** — a single autoencoder over all 1,699 profiles, including test-fold
  *expression* (never their `y`). Reported separately to price that weaker assumption.
- **Multi-task MLP** — shared trunk (1000 -> 256 -> 64, BatchNorm + Dropout 0.3) with one linear head
  per drug over the **12 best-covered GDSC2 drugs**, masked MSE, on the per-fold top-1000 MI genes
  (selected on training rows only). A **single-task control** runs the identical code path with the
  Gemcitabine head alone, so the comparison isolates multi-tasking rather than architecture.

**Result (pooled OOF R²):**

| Model | Pooled OOF R² |
|---|---|
| ElasticNet (nested, l1 tuned) — best overall | 0.460 |
| PCA-50 -> Ridge (linear compression, for contrast) | 0.334 |
| **AE-64 -> Ridge (transductive)** | **0.301** |
| **AE-64 -> Ridge (leak-free, per-fold AE)** | **0.296** |
| **Single-task MLP (control)** | **0.294** |
| **Multi-task MLP (12 drugs)** | **0.290** |
| Lineage-only baseline | 0.200 |

Per-fold AE R²: 0.287 / 0.307 / 0.260 / 0.244 / 0.373 (mean 0.294 +/- 0.045); autoencoders stopped
at 65-88 epochs with validation reconstruction MSE ~0.49-0.50.

**Takeaway — neither idea worked, and the *way* they failed is informative.**

1. **Unsupervised pretraining did not pay.** AE-64 (0.296) lands far below ElasticNet (0.460) and,
   more tellingly, **below PCA-50 (0.334)** — a *linear* 50-component compression beats a *non-linear*
   64-dimensional one on the same genes. Compression itself is the problem, not its flexibility:
   squeezing ~16.8k genes into tens of dimensions optimises for **reconstruction**, which is
   dominated by the high-variance tissue/lineage axes, not for the thin drug-response signal
   ElasticNet finds by selecting individual genes. The extra 1,005 unlabelled profiles do not fix
   this, because the bottleneck was never sample size for *representation* learning — it was that the
   objective is the wrong one.
2. **Transduction is worth almost nothing here: +0.005 R².** Letting the autoencoder see test-fold
   expression barely moves the result, which means the conservative per-fold protocol costs us
   essentially no performance. Cheap rigor — worth stating, because papers often take the
   transductive shortcut and imply it matters.
3. **Multi-task learning did not help: −0.004 vs. its own single-task control.** Note the 12 selected
   drugs were tested on **all 694** cell lines (a 100%-dense target matrix), so this was the
   *favourable* case — no missing-data sparsity to overcome — and sharing a trunk still gave nothing.
   Per-fold it was also less stable than the control (fold 3: 0.154 vs 0.233). With n = 694 the trunk
   has no shortage of *tasks*; it has a shortage of *samples*, and multi-tasking doesn't create any.
4. **A consistency check that passed:** the single-task MLP control (0.294) closely reproduces
   notebook 04's independently-implemented MLP (0.27) on the same folds and features.

**The Phase 1-3 pattern is now unambiguous.** Nested CV (Phase 1), tree ensembles (Phase 2), and deep
representation learning (Phase 3) have each failed to beat a regularized linear model. At p >> n with
n = 694 and signal spread thinly across many weakly-predictive genes, **ElasticNet's L1/L2 selection
over all ~16.8k genes remains the right tool**, and every added layer of flexibility has cost
accuracy. That is the finding, not a failure to find one.

**Artifacts:** `output/model_comparison_phase3.csv`, `output/phase3_results.csv`,
`output/phase3_ae_folds.csv`, `output/phase3_comparison.png`, `output/phase3_training_curves.png`.
