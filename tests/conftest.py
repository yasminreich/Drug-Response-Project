"""Shared fixtures.

The 518 MB DepMap expression matrix is gitignored, so CI cannot rely on it. Tests
use a small synthetic expression matrix combined with the *real* tracked Model.csv
and GDSC2 CSV, so the join logic is exercised against genuine COSMIC_ID/ModelID
values rather than invented ones.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import data as sdata  # noqa: E402

DATA_DIR = REPO_ROOT / "data"


@pytest.fixture(scope="session")
def model_bridge():
    """The real COSMIC_ID <-> ModelID bridge from the tracked Model.csv."""
    return sdata.load_model_bridge(DATA_DIR)


@pytest.fixture(scope="session")
def gdsc():
    """The real GDSC2 dose-response table (tracked, ~36 MB)."""
    return sdata.load_gdsc2(DATA_DIR)


@pytest.fixture(scope="session")
def gdsc_gemcitabine(gdsc, model_bridge):
    """GDSC2 joined to the bridge, filtered to Gemcitabine. Real IDs, still raw."""
    joined = gdsc.merge(model_bridge, on="COSMIC_ID", how="inner")
    return joined[joined["DRUG_NAME"] == sdata.TARGET_DRUG].copy()


@pytest.fixture(scope="session")
def synthetic_expression(model_bridge):
    """A small expression matrix shaped like the DepMap export.

    Uses 30 real ModelIDs so it joins against the real bridge. Every model gets a
    duplicate non-default row, so tests can prove the IsDefaultEntryForModel filter
    is what collapses them to one profile each.
    """
    rng = np.random.RandomState(0)
    model_ids = model_bridge["ModelID"].drop_duplicates().head(30).tolist()
    genes = [f"GENE{i} ({1000 + i})" for i in range(50)]

    rows = []
    for flag in ("Yes", "No"):
        for mid in model_ids:
            row = {
                "ModelID": mid,
                "SequencingID": f"{mid}-{flag}",
                "IsDefaultEntryForModel": flag,
                "ModelConditionID": f"MC-{mid}",
                "IsDefaultEntryForMC": flag,
            }
            # "No" rows carry deliberately different values, so if the filter ever
            # breaks the resulting matrix is numerically distinguishable.
            offset = 0.0 if flag == "Yes" else 100.0
            row.update({g: rng.rand() * 5 + offset for g in genes})
            rows.append(row)

    return pd.DataFrame(rows), genes


@pytest.fixture
def expr_final(synthetic_expression):
    """The synthetic matrix after the default-entry filter."""
    expr_raw, _ = synthetic_expression
    final, gene_cols = sdata.filter_default_entries(expr_raw)
    return final, gene_cols
