"""Guards for the project's central methodological claim: no leakage.

Scaling and *supervised* feature selection must be fit on training rows only. These
tests encode that contract and demonstrate, on synthetic data with no real signal,
how badly the leaky alternative misleads — which is exactly why the harness in
notebook 03 fits everything per fold.
"""

import numpy as np
import pytest
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import data as sdata


@pytest.fixture
def pn_data():
    """p >> n with pure noise: the true out-of-fold R^2 must be <= ~0."""
    rng = np.random.RandomState(0)
    X = rng.randn(120, 2000).astype(np.float32)
    y = rng.randn(120).astype(np.float32)
    lineage = np.array([0, 1, 2, 3] * 30)
    return X, y, lineage


def _pipeline(k=50):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_regression, k=k)),
            ("model", Ridge(alpha=1.0)),
        ]
    )


class TestScalerFitOnTrainOnly:
    def test_scaler_uses_train_statistics_not_full_data(self, pn_data):
        X, y, lineage = pn_data
        train_idx, test_idx = sdata.make_folds(X, lineage)[0]

        pipe = _pipeline().fit(X[train_idx], y[train_idx])
        scaler = pipe.named_steps["scale"]

        np.testing.assert_allclose(
            scaler.mean_, X[train_idx].mean(axis=0), rtol=1e-5, atol=1e-5
        )
        # And it must NOT equal the full-data mean.
        assert not np.allclose(scaler.mean_, X.mean(axis=0), rtol=1e-3, atol=1e-3)

    def test_transform_of_test_rows_is_not_recentred(self, pn_data):
        """Test rows keep a non-zero mean after transform; only train is centred."""
        X, y, lineage = pn_data
        train_idx, test_idx = sdata.make_folds(X, lineage)[0]

        scaler = StandardScaler().fit(X[train_idx])
        assert abs(scaler.transform(X[train_idx]).mean()) < 1e-6
        assert abs(scaler.transform(X[test_idx]).mean()) > 1e-6


class TestSelectionFitOnTrainOnly:
    def test_selected_features_differ_across_folds(self, pn_data):
        """Per-fold supervised selection must depend on the fold's training rows.

        Identical selections across every fold would mean selection saw all the data.
        """
        X, y, lineage = pn_data
        selections = []
        for train_idx, _ in sdata.make_folds(X, lineage):
            pipe = _pipeline().fit(X[train_idx], y[train_idx])
            selections.append(set(np.where(pipe.named_steps["select"].get_support())[0]))

        assert len({frozenset(s) for s in selections}) > 1
        # Folds share training rows, but on noise the overlap should stay small.
        assert len(selections[0] & selections[1]) < len(selections[0])


class TestLeakageInflatesScores:
    def test_leak_free_cv_reports_no_signal_on_pure_noise(self, pn_data):
        X, y, lineage = pn_data

        oof = np.zeros_like(y)
        for train_idx, test_idx in sdata.make_folds(X, lineage):
            pipe = _pipeline().fit(X[train_idx], y[train_idx])
            oof[test_idx] = pipe.predict(X[test_idx])

        # There is no signal, so honest pooled OOF R^2 must not look predictive.
        assert r2_score(y, oof) < 0.1

    def test_selecting_before_cv_fabricates_signal(self, pn_data):
        """The failure mode this project's methodology exists to avoid."""
        X, y, lineage = pn_data

        # Leaky: choose features using ALL rows, including future test rows.
        leaked = SelectKBest(f_regression, k=50).fit(X, y)
        X_leaked = leaked.transform(X)

        oof = np.zeros_like(y)
        for train_idx, test_idx in sdata.make_folds(X, lineage):
            model = Ridge(alpha=1.0).fit(X_leaked[train_idx], y[train_idx])
            oof[test_idx] = model.predict(X_leaked[test_idx])

        # Pure noise, yet leakage manufactures an apparently strong result.
        assert r2_score(y, oof) > 0.3


class TestVarianceFilterIsUnsupervised:
    def test_variance_filter_ignores_y(self, pn_data):
        """Justifies applying the variance filter once, outside the CV loop."""
        X, y, lineage = pn_data
        rng = np.random.RandomState(1)

        import pandas as pd

        gene_cols = [f"G{i}" for i in range(X.shape[1])]
        merged = pd.DataFrame(X, columns=gene_cols)
        merged["LN_IC50"] = y
        for col in ["ModelID", "COSMIC_ID", "CellLineName", "OncotreeLineage", "TCGA_DESC"]:
            merged[col] = "x"

        _, _, genes_a, _ = sdata.build_drug_matrix(merged, gene_cols)

        shuffled = merged.copy()
        shuffled["LN_IC50"] = rng.permutation(y)
        _, _, genes_b, _ = sdata.build_drug_matrix(shuffled, gene_cols)

        # Shuffling y must not change which genes survive.
        assert genes_a == genes_b
