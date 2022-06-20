from unittest.mock import patch
from dataclasses import make_dataclass
from itertools import product

import pytest
import numpy as np
import pandas as pd
import awswrangler as wr
import pandera as pa

from mbl.name_space import Columns
from mbl.schema import RandomHeisenbergEDSchema, RandomHeisenbergFoldingTSDRGSchema
from mbl.analysis.level_statistic import LevelStatistic, AverageOrder


@pytest.fixture(scope="function")
def expected_table(request) -> pd.DataFrame:
    expected_table = {
        "ed": {
            0: [Columns.level_id, "bigint", False, ""],
            1: [Columns.en, "double", False, ""],
            2: [Columns.total_sz, "double", False, ""],
            3: [Columns.edge_entropy, "double", False, ""],
            4: [Columns.bipartite_entropy, "double", False, ""],
            5: [Columns.trial_id, "bigint", False, ""],
            6: [Columns.seed, "bigint", False, ""],
            7: [Columns.system_size, "bigint", True, ""],
            8: [Columns.disorder, "double", True, ""],
            9: [Columns.penalty, "double", True, ""],
            10: [Columns.s_target, "bigint", True, ""],
            11: [Columns.offset, "double", True, ""],
        },
        "tsdrg": {
            0: [Columns.level_id, "bigint", False, ""],
            1: [Columns.en, "double", False, ""],
            2: [Columns.variance, "double", False, ""],
            3: [Columns.total_sz, "double", False, ""],
            4: [Columns.edge_entropy, "double", False, ""],
            5: [Columns.trial_id, "string", False, ""],
            6: [Columns.seed, "bigint", False, ""],
            7: [Columns.offset, "double", False, ""],
            8: [Columns.max_en, "double", False, ""],
            9: [Columns.min_en, "double", False, ""],
            10: [Columns.system_size, "bigint", True, ""],
            11: [Columns.disorder, "double", True, ""],
            12: [Columns.truncation_dim, "bigint", True, ""],
            13: [Columns.relative_offset, "double", True, ""],
            14: [Columns.penalty, "double", True, ""],
            15: [Columns.s_target, "bigint", True, ""],
            16: [Columns.method, "string", True, ""],
        },
    }[request.param]
    return pd.DataFrame.from_dict(
        expected_table,
        orient="index",
        columns=["Column Name", "Type", "Partition", "Comment"],
    )


@pytest.mark.parametrize(
    "table_name, expected_table",
    [("ed_table", "ed"), ("tsdrg_table", "tsdrg")],
    indirect=["expected_table"],
)
def test_athena_table(table_name: str, expected_table: pd.DataFrame):
    table_df = wr.catalog.table(
        database=LevelStatistic.Metadata.database,
        table=getattr(LevelStatistic.Metadata, table_name),
    )
    pd.testing.assert_frame_equal(table_df, expected_table)  # order sensitive


@pytest.mark.parametrize(
    "schema, n, h, chi, seed",
    [
        (RandomHeisenbergEDSchema, 10, 0.5, None, 2020),
        (RandomHeisenbergFoldingTSDRGSchema, 10, 0.5, 16, 2020),
    ],
)
def test_athena_query(schema: pa.SchemaModel, n: int, h: float, chi: int, seed: int):
    df = LevelStatistic.athena_query(n=n, h=h, chi=chi, seed=seed)
    schema.validate(df, lazy=True)


@pytest.fixture(scope="function")
def mock_gap(request) -> np.ndarray:
    size = 10**6
    gap = {
        "poisson": pd.Series(np.sort(np.random.standard_normal(size))).diff(),
        "wigner_dyson": pd.Series(),
    }[request.param]
    return gap.to_numpy()


@pytest.mark.parametrize(
    "mock_gap, expected_r_value",
    [
        ("poisson", 2 * np.log(2) - 1),
        # ("wigner_dyson", 4 - 2 * np.sqrt(3))
    ],
    indirect=["mock_gap"],
)
def test_gap_ratio(mock_gap: np.ndarray, expected_r_value: float):
    assert np.isnan(mock_gap[0])
    r_value = LevelStatistic.gap_ratio(mock_gap)
    assert np.isnan(r_value[0])
    assert np.isnan(r_value[-1])
    avg_r_value = pd.Series(r_value).mean()
    np.testing.assert_allclose(avg_r_value, expected_r_value, atol=1e-3)


@pytest.fixture(scope="function", params=[(3, 5), (10, 16)])
def mock_raw_df(request) -> pd.DataFrame:
    n_trials, n_levels = request.param
    Row = make_dataclass("Row", [(Columns.en, float), (Columns.seed, int)])
    return pd.DataFrame(
        [
            Row(en, seed)
            for seeds in np.random.randint(2000, 2500, n_trials)
            for en, seed in product(
                np.sort(np.random.standard_normal(size=n_levels)), [seeds]
            )
        ]
    )


def mock_drop_duplicates(*args, **kwargs):
    pass


def test_extract_gap(mock_raw_df, monkeypatch):
    monkeypatch.setattr(pd.DataFrame, "drop_duplicates", mock_drop_duplicates)
    df = LevelStatistic.extract_gap(mock_raw_df)
    assert (df[Columns.energy_gap][~df[Columns.energy_gap].isnull()]).gt(0).all()
    assert (df[Columns.gap_ratio][~df[Columns.gap_ratio].isnull()]).between(0, 1).all()
    assert (df.groupby(Columns.seed)[Columns.energy_gap].nth(0).isnull()).all()
    assert (df.groupby(Columns.seed)[Columns.gap_ratio].nth(0).isnull()).all()
    assert (df.groupby(Columns.seed)[Columns.gap_ratio].nth(-1).isnull()).all()


def test_averaged_gap_ratio(mock_raw_df, monkeypatch):
    monkeypatch.setattr(pd.DataFrame, "drop_duplicates", mock_drop_duplicates)
    df = LevelStatistic.extract_gap(mock_raw_df)
    r1 = LevelStatistic.averaged_gap_ratio(df, AverageOrder.LEVEL_FIRST)
    r2 = LevelStatistic.averaged_gap_ratio(df, AverageOrder.DISORDER_FIRST)
    np.testing.assert_allclose(r1, r2, atol=1e-12)


def test_fetch_gap_ratio(mock_raw_df, monkeypatch):
    monkeypatch.setattr(pd.DataFrame, "drop_duplicates", mock_drop_duplicates)
    with patch.object(LevelStatistic, "athena_query", return_value=mock_raw_df):
        r1 = LevelStatistic.fetch_gap_ratio(
            n=8, h=10.0, chi=32, total_sz=0, order=AverageOrder.LEVEL_FIRST
        )
        r2 = LevelStatistic.fetch_gap_ratio(
            n=8, h=10.0, chi=32, total_sz=0, order=AverageOrder.DISORDER_FIRST
        )
        np.testing.assert_allclose(r1, r2, atol=1e-12)
