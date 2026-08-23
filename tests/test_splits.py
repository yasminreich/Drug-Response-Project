"""Tests for lineage collapsing and the stratified CV / holdout splits."""

import numpy as np
import pandas as pd

from src import data as sdata


class TestCollapseRareLineages:
    def test_rare_lineages_become_other(self):
        labels = ["Lung"] * 10 + ["Skin"] * 6 + ["Eye"] * 2 + ["Bone"] * 1
        out = sdata.collapse_rare_lineages(labels, min_count=5)

        assert set(out) == {"Lung", "Skin", "Other"}
        assert (out == "Other").sum() == 3  # Eye (2) + Bone (1)

    def test_common_lineages_untouched(self):
        labels = ["Lung"] * 10 + ["Skin"] * 6
        out = sdata.collapse_rare_lineages(labels, min_count=5)
        assert set(out) == {"Lung", "Skin"}

    def test_nan_becomes_unknown_then_possibly_other(self):
        labels = ["Lung"] * 10 + [np.nan] * 2
        out = sdata.collapse_rare_lineages(labels, min_count=5)
        assert "Other" in set(out)  # 2 Unknowns are rare -> Other
        assert not pd.isna(pd.Series(out)).any()

    def test_boundary_is_inclusive(self):
        """A lineage with exactly min_count members is kept, not collapsed."""
        labels = ["Lung"] * 10 + ["Skin"] * 5
        out = sdata.collapse_rare_lineages(labels, min_count=5)
        assert "Skin" in set(out)
        assert "Other" not in set(out)


class TestLineageColumns:
    def _meta(self):
        return pd.DataFrame(
            {
                "OncotreeLineage": ["Lung"] * 6
                + ["Myeloid"] * 6
                + ["Lymphoid"] * 6
                + ["Eye"] * 2
            }
        )

    def test_adds_expected_columns(self):
        meta, encoded, le = sdata.add_lineage_columns(self._meta())
        for col in ("lineage_label", "lineage_encoded", "is_blood"):
            assert col in meta.columns
        assert len(encoded) == len(meta)
        assert len(le.classes_) == len(set(meta["lineage_label"]))

    def test_blood_flag_marks_myeloid_and_lymphoid(self):
        meta, _, _ = sdata.add_lineage_columns(self._meta())
        assert meta["is_blood"].sum() == 12
        assert meta.loc[meta["OncotreeLineage"] == "Lung", "is_blood"].sum() == 0

    def test_rare_lineage_collapsed_before_encoding(self):
        meta, _, _ = sdata.add_lineage_columns(self._meta())
        assert "Eye" not in set(meta["lineage_label"])
        assert "Other" in set(meta["lineage_label"])


class TestFolds:
    def _data(self, n=200):
        rng = np.random.RandomState(0)
        X = rng.rand(n, 10).astype(np.float32)
        lineage = np.array([0, 1, 2, 3] * (n // 4))
        return X, lineage

    def test_deterministic_under_fixed_seed(self):
        X, lineage = self._data()
        a = sdata.make_folds(X, lineage, random_state=42)
        b = sdata.make_folds(X, lineage, random_state=42)

        assert len(a) == sdata.N_SPLITS
        for (tr_a, te_a), (tr_b, te_b) in zip(a, b):
            np.testing.assert_array_equal(tr_a, tr_b)
            np.testing.assert_array_equal(te_a, te_b)

    def test_different_seed_gives_different_folds(self):
        X, lineage = self._data()
        a = sdata.make_folds(X, lineage, random_state=42)
        c = sdata.make_folds(X, lineage, random_state=7)
        assert not np.array_equal(a[0][1], c[0][1])

    def test_test_folds_partition_the_data(self):
        X, lineage = self._data()
        folds = sdata.make_folds(X, lineage)

        all_test = np.concatenate([te for _, te in folds])
        np.testing.assert_array_equal(np.sort(all_test), np.arange(len(X)))

    def test_train_and_test_never_overlap(self):
        X, lineage = self._data()
        for tr, te in sdata.make_folds(X, lineage):
            assert len(np.intersect1d(tr, te)) == 0

    def test_folds_are_stratified_by_lineage(self):
        X, lineage = self._data()
        overall = np.bincount(lineage) / len(lineage)

        for _, te in sdata.make_folds(X, lineage):
            fold_dist = np.bincount(lineage[te], minlength=len(overall)) / len(te)
            np.testing.assert_allclose(fold_dist, overall, atol=0.05)


class TestHoldout:
    def test_split_sizes_and_disjointness(self):
        rng = np.random.RandomState(0)
        y = rng.rand(200)
        lineage = np.array([0, 1, 2, 3] * 50)

        train_idx, test_idx = sdata.make_holdout(y, lineage)

        assert len(test_idx) == 40  # 20% of 200
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        np.testing.assert_array_equal(
            np.sort(np.concatenate([train_idx, test_idx])), np.arange(200)
        )

    def test_deterministic(self):
        rng = np.random.RandomState(0)
        y = rng.rand(200)
        lineage = np.array([0, 1, 2, 3] * 50)

        a = sdata.make_holdout(y, lineage, random_state=42)
        b = sdata.make_holdout(y, lineage, random_state=42)
        np.testing.assert_array_equal(a[1], b[1])
