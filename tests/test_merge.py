"""Tests for the 3-way merge and the two rules CLAUDE.md calls mandatory."""

import numpy as np
import pandas as pd
import pytest

from src import data as sdata


class TestDefaultEntryFilter:
    """Rule 1: IsDefaultEntryForModel == "Yes" (string, not boolean)."""

    def test_yields_one_profile_per_cell_line(self, expr_final):
        final, _ = expr_final
        assert final["ModelID"].duplicated().sum() == 0
        assert len(final) == 30  # 30 models, duplicate "No" rows removed

    def test_drops_meta_columns(self, expr_final):
        final, gene_cols = expr_final
        for col in sdata.EXPR_META_COLS:
            if col != "ModelID":
                assert col not in final.columns
        assert len(gene_cols) == 50

    def test_keeps_default_rows_not_the_duplicates(self, synthetic_expression):
        expr_raw, genes = synthetic_expression
        final, _ = sdata.filter_default_entries(expr_raw)
        # "No" rows were offset by +100; if the filter inverted, values would exceed it.
        assert final[genes].values.max() < 100.0

    def test_boolean_comparison_would_match_nothing(self, synthetic_expression):
        """Guards the specific footgun: the column holds strings, not booleans."""
        expr_raw, _ = synthetic_expression
        assert len(expr_raw[expr_raw["IsDefaultEntryForModel"] == True]) == 0  # noqa: E712
        assert len(expr_raw[expr_raw["IsDefaultEntryForModel"] == "Yes"]) == 30

    def test_drops_unnamed_index_column(self, synthetic_expression):
        expr_raw, _ = synthetic_expression
        with_index = expr_raw.copy()
        with_index.insert(0, "Unnamed: 0", range(len(with_index)))
        final, gene_cols = sdata.filter_default_entries(with_index)
        assert "Unnamed: 0" not in final.columns
        assert len(gene_cols) == 50


class TestDedup:
    """Rule 2: group by COSMIC_ID/DRUG_NAME/ModelID only."""

    def test_nan_tcga_desc_does_not_drop_rows(self, gdsc_gemcitabine):
        """The regression CLAUDE.md warns about.

        Grouping on a column containing NaNs silently drops those rows, because
        pandas groupby defaults to dropna=True. Here every TCGA_DESC is NaN, so a
        buggy implementation returns an empty frame.
        """
        subset = gdsc_gemcitabine.copy()
        subset["TCGA_DESC"] = np.nan

        result = sdata.dedup_drug_rows(subset)

        expected = subset[sdata.DEDUP_KEYS].drop_duplicates().shape[0]
        assert len(result) == expected
        assert len(result) > 0

        # Demonstrate the bug this guards against.
        buggy = subset.groupby(
            sdata.DEDUP_KEYS + ["TCGA_DESC"], as_index=False
        ).agg(LN_IC50=("LN_IC50", "mean"))
        assert len(buggy) == 0

    def test_one_row_per_cell_line(self, gdsc_gemcitabine):
        result = sdata.dedup_drug_rows(gdsc_gemcitabine)
        assert result["ModelID"].duplicated().sum() == 0

    def test_replicates_are_averaged(self):
        df = pd.DataFrame(
            {
                "COSMIC_ID": [1, 1, 2],
                "DRUG_NAME": ["Gemcitabine"] * 3,
                "ModelID": ["ACH-1", "ACH-1", "ACH-2"],
                "CellLineName": ["A", "A", "B"],
                "OncotreeLineage": ["Lung", "Lung", "Skin"],
                "TCGA_DESC": [np.nan, np.nan, "SKCM"],
                "LN_IC50": [1.0, 3.0, 5.0],
            }
        )
        result = sdata.dedup_drug_rows(df).sort_values("ModelID").reset_index(drop=True)
        assert len(result) == 2
        assert result.loc[0, "LN_IC50"] == 2.0  # mean of 1.0 and 3.0
        assert result.loc[1, "LN_IC50"] == 5.0

    def test_filters_to_target_drug_only(self, gdsc_gemcitabine):
        result = sdata.dedup_drug_rows(gdsc_gemcitabine)
        assert set(result["DRUG_NAME"].unique()) == {sdata.TARGET_DRUG}


class TestJoinGdscToModels:
    """The all-drug join shared by notebook 01 and build_merged."""

    def test_without_expression_keeps_all_bridged_rows(self, gdsc, model_bridge):
        joined = sdata.join_gdsc_to_models(gdsc, model_bridge)

        assert len(joined) > 0
        assert "ModelID" in joined.columns
        assert joined["COSMIC_ID"].isin(model_bridge["COSMIC_ID"]).all()
        # All drugs, not just the target one.
        assert joined["DRUG_NAME"].nunique() > 1

    def test_expression_restricts_to_profiled_cell_lines(self, gdsc, model_bridge, expr_final):
        final, _ = expr_final

        unrestricted = sdata.join_gdsc_to_models(gdsc, model_bridge)
        restricted = sdata.join_gdsc_to_models(gdsc, model_bridge, final)

        assert len(restricted) < len(unrestricted)
        assert restricted["ModelID"].isin(final["ModelID"]).all()
        # The synthetic matrix holds 30 models, so at most 30 cell lines survive.
        assert restricted["ModelID"].nunique() <= 30

    def test_build_merged_matches_the_helper(self, gdsc, model_bridge, expr_final):
        """build_merged must go through the same join, not its own copy."""
        final, gene_cols = expr_final

        via_helper = sdata.join_gdsc_to_models(gdsc, model_bridge, final)
        expected_ids = set(
            sdata.dedup_drug_rows(via_helper)["ModelID"]
        )
        merged = sdata.build_merged(gdsc, model_bridge, final)

        assert set(merged["ModelID"]) == expected_ids


class TestDedupRows:
    def test_all_drug_dedup_keeps_every_drug(self, gdsc_gemcitabine, gdsc, model_bridge):
        joined = sdata.join_gdsc_to_models(gdsc, model_bridge)
        result = sdata.dedup_rows(joined)

        assert result["DRUG_NAME"].nunique() == joined["DRUG_NAME"].nunique()
        # One row per (COSMIC_ID, DRUG_NAME, ModelID).
        assert result.duplicated(subset=sdata.DEDUP_KEYS).sum() == 0

    def test_averages_auc_when_present(self):
        df = pd.DataFrame(
            {
                "COSMIC_ID": [1, 1],
                "DRUG_NAME": ["D", "D"],
                "ModelID": ["ACH-1", "ACH-1"],
                "CellLineName": ["A", "A"],
                "OncotreeLineage": ["Lung", "Lung"],
                "TCGA_DESC": [np.nan, np.nan],
                "LN_IC50": [1.0, 3.0],
                "AUC": [0.2, 0.4],
            }
        )
        out = sdata.dedup_rows(df)
        assert len(out) == 1
        assert out.loc[0, "AUC"] == pytest.approx(0.3)


class TestBuildMerged:
    def test_two_hop_join_against_synthetic_expression(
        self, gdsc, model_bridge, expr_final
    ):
        final, gene_cols = expr_final
        merged = sdata.build_merged(gdsc, model_bridge, final)

        # Only cell lines present in the (30-model) synthetic matrix survive.
        assert len(merged) <= 30
        assert merged["ModelID"].isin(final["ModelID"]).all()
        assert merged["ModelID"].duplicated().sum() == 0
        assert "LN_IC50" in merged.columns

    def test_build_drug_matrix_shapes_align(self, gdsc, model_bridge, expr_final):
        final, gene_cols = expr_final
        merged = sdata.build_merged(gdsc, model_bridge, final)
        X, y, gene_names, meta = sdata.build_drug_matrix(
            merged, gene_cols, variance_threshold=0.0
        )

        assert X.shape[0] == len(y) == len(meta) == len(merged)
        assert X.shape[1] == len(gene_names)
        assert X.dtype == np.float32
        assert not np.isnan(X).any()

    def test_variance_filter_drops_constant_genes(self, gdsc, model_bridge, expr_final):
        final, gene_cols = expr_final
        final = final.copy()
        final["CONSTANT (999)"] = 1.0
        gene_cols = gene_cols + ["CONSTANT (999)"]

        merged = sdata.build_merged(gdsc, model_bridge, final)
        _, _, gene_names, _ = sdata.build_drug_matrix(
            merged, gene_cols, variance_threshold=0.0
        )
        assert "CONSTANT (999)" not in gene_names
