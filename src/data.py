"""Shared data loading, merging and splitting for the Gemcitabine pipeline.

This module is the single source of truth for the 3-way join. Notebooks 02 and 03
previously each carried their own copy of this code; any drift between them would
have meant notebook 02's exploratory plots describing a different matrix than the
one notebook 03 actually models, with nothing to catch it.

The two rules that must never be broken (see CLAUDE.md):

1. Filter the expression matrix to ``IsDefaultEntryForModel == "Yes"`` (a *string*,
   not a boolean) so each cell line contributes exactly one profile.
2. Group by ``["COSMIC_ID", "DRUG_NAME", "ModelID"]`` and nothing else. Including a
   column with NaNs (``TCGA_DESC`` in particular) silently drops rows, because
   ``pandas.groupby`` defaults to ``dropna=True``.

Both are enforced here and covered by ``tests/``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# Repo root, resolved from this file so callers work regardless of cwd
# (notebooks execute with cwd=notebooks/, tests with cwd=repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

GDSC_FILENAME = "GDSC2_fitted_dose_response_27Oct23.csv"
MODEL_FILENAME = "Model.csv"
EXPR_FILENAME = "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"

# Non-gene columns in the DepMap expression export.
EXPR_META_COLS = [
    "ModelID",
    "SequencingID",
    "IsDefaultEntryForModel",
    "ModelConditionID",
    "IsDefaultEntryForMC",
]

# Grouping keys for deduplication. Do NOT add metadata columns here.
DEDUP_KEYS = ["COSMIC_ID", "DRUG_NAME", "ModelID"]

VARIANCE_THRESHOLD = 0.1
MIN_LINEAGE_COUNT = 5
N_SPLITS = 5
RANDOM_SEED = 42
TARGET_DRUG = "Gemcitabine"
BLOOD_LINEAGES = ["Lymphoid", "Myeloid"]


def _resolve(data_dir):
    return Path(data_dir) if data_dir is not None else DATA_DIR


def load_gdsc2(data_dir=None):
    """Load the raw GDSC2 fitted dose-response table."""
    return pd.read_csv(_resolve(data_dir) / GDSC_FILENAME)


def load_model_bridge(data_dir=None):
    """Load Model.csv reduced to the COSMIC_ID <-> ModelID bridge plus lineage.

    Rows without a COSMIC_ID cannot be joined to GDSC2 and are dropped.
    """
    model = pd.read_csv(_resolve(data_dir) / MODEL_FILENAME, low_memory=False)
    model_clean = (
        model[["ModelID", "CellLineName", "COSMICID", "OncotreeLineage"]]
        .rename(columns={"COSMICID": "COSMIC_ID"})
        .dropna(subset=["COSMIC_ID"])
        .copy()
    )
    model_clean["COSMIC_ID"] = model_clean["COSMIC_ID"].astype(int)
    return model_clean


def filter_default_entries(expr_raw):
    """Keep one canonical expression profile per cell line and drop meta columns.

    ``IsDefaultEntryForModel`` holds the strings "Yes"/"No", not booleans — comparing
    against ``True`` would silently match nothing and return an empty frame.
    """
    if expr_raw.columns[0].startswith("Unnamed"):
        expr_raw = expr_raw.drop(columns=[expr_raw.columns[0]])

    expr = expr_raw[expr_raw["IsDefaultEntryForModel"] == "Yes"].copy()
    gene_cols = [c for c in expr.columns if c not in EXPR_META_COLS]
    return expr[["ModelID"] + gene_cols].copy(), gene_cols


def load_expression(data_dir=None):
    """Load the DepMap expression matrix (~518 MB) and keep default entries only.

    Returns ``(expr_final, gene_cols)``.
    """
    expr_raw = pd.read_csv(_resolve(data_dir) / EXPR_FILENAME)
    return filter_default_entries(expr_raw)


def dedup_drug_rows(gdsc_model, target_drug=TARGET_DRUG):
    """Filter to one drug and collapse replicate measurements to one row per cell line.

    Replicates are averaged. Grouping uses only ``DEDUP_KEYS``; ``TCGA_DESC`` is
    carried through as an aggregate rather than a key precisely because its NaNs
    would otherwise drop rows.
    """
    subset = gdsc_model[gdsc_model["DRUG_NAME"] == target_drug].copy()

    agg = dict(
        CellLineName=("CellLineName", "first"),
        OncotreeLineage=("OncotreeLineage", "first"),
        TCGA_DESC=("TCGA_DESC", "first"),
        LN_IC50=("LN_IC50", "mean"),
    )
    if "AUC" in subset.columns:
        agg["AUC"] = ("AUC", "mean")

    return subset.groupby(DEDUP_KEYS, as_index=False).agg(**agg)


def build_merged(gdsc, model_clean, expr_final, target_drug=TARGET_DRUG):
    """Run the two-hop join: GDSC2 -> Model bridge -> expression matrix."""
    gdsc_model = gdsc.merge(model_clean, on="COSMIC_ID", how="inner")

    # Keep only rows whose cell line has an expression profile.
    expr_ids = set(expr_final["ModelID"].unique())
    gdsc_model = gdsc_model[gdsc_model["ModelID"].isin(expr_ids)].copy()

    target_df = dedup_drug_rows(gdsc_model, target_drug=target_drug)
    return target_df.merge(expr_final, on="ModelID", how="inner")


def split_merged(merged, gene_cols):
    """Split the merged frame into ``(X_raw, y, gene_cols_present, meta)``.

    No variance filter is applied. Notebook 02 uses this because it explores the
    unfiltered matrix before filtering; notebook 03 goes via ``build_drug_matrix``.
    """
    gene_cols_present = [c for c in gene_cols if c in merged.columns]

    X_raw = merged[gene_cols_present].values.astype(np.float32)
    y = merged["LN_IC50"].values.astype(np.float32)
    meta = merged[
        ["ModelID", "COSMIC_ID", "CellLineName", "OncotreeLineage", "TCGA_DESC"]
    ].copy()

    if np.isnan(X_raw).any():
        raise ValueError("NaNs found in the gene expression matrix")
    if np.isnan(y).any():
        raise ValueError("NaNs found in LN_IC50")

    return X_raw, y, gene_cols_present, meta


def build_drug_matrix(merged, gene_cols, variance_threshold=VARIANCE_THRESHOLD):
    """Split the merged frame into ``(X, y, gene_names, meta)``.

    The variance filter is unsupervised (it never looks at ``y``), so applying it
    once outside the CV loop does not leak. Supervised selection must stay inside
    the folds.
    """
    X_raw, y, gene_cols_present, meta = split_merged(merged, gene_cols)

    vt = VarianceThreshold(variance_threshold).fit(X_raw)
    keep_mask = vt.get_support()
    gene_names = [g for g, keep in zip(gene_cols_present, keep_mask) if keep]
    X = X_raw[:, keep_mask].astype(np.float32)

    return X, y, gene_names, meta


def collapse_rare_lineages(labels, min_count=MIN_LINEAGE_COUNT):
    """Collapse lineages with fewer than ``min_count`` cell lines into "Other".

    Stratified CV cannot split a class with fewer members than there are folds, and
    tiny lineages carry no learnable signal anyway.
    """
    s = pd.Series(labels).fillna("Unknown")
    counts = s.value_counts()
    rare = counts[counts < min_count].index
    return s.where(~s.isin(rare), "Other").values


def add_lineage_columns(meta, blood_lineages=BLOOD_LINEAGES):
    """Attach collapsed/encoded lineage labels and the blood-cancer flag to ``meta``.

    Returns ``(meta, lineage_encoded, label_encoder)``. ``meta`` is modified in place.
    """
    lineage_label = collapse_rare_lineages(meta["OncotreeLineage"].values)
    le = LabelEncoder()
    lineage_encoded = le.fit_transform(lineage_label)

    meta["lineage_label"] = lineage_label
    meta["lineage_encoded"] = lineage_encoded
    meta["is_blood"] = np.isin(lineage_label, blood_lineages)

    return meta, lineage_encoded, le


def make_folds(X, lineage_encoded, n_splits=N_SPLITS, random_state=RANDOM_SEED):
    """Build lineage-stratified CV folds. Deterministic under a fixed seed."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(X, lineage_encoded))


def make_holdout(y, lineage_encoded, test_size=0.2, random_state=RANDOM_SEED):
    """Build the stratified 80/20 holdout split used by notebooks 04 and 05."""
    return train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        stratify=lineage_encoded,
        random_state=random_state,
    )


def load_gemcitabine_dataset(data_dir=None, target_drug=TARGET_DRUG,
                             variance_threshold=VARIANCE_THRESHOLD):
    """End-to-end convenience loader: raw CSVs -> ``(X, y, gene_names, meta)``.

    Requires the ~518 MB expression matrix, so tests exercise the individual steps
    against fixtures instead of calling this.
    """
    gdsc = load_gdsc2(data_dir)
    model_clean = load_model_bridge(data_dir)
    expr_final, gene_cols = load_expression(data_dir)
    merged = build_merged(gdsc, model_clean, expr_final, target_drug=target_drug)
    return build_drug_matrix(merged, gene_cols, variance_threshold=variance_threshold)
