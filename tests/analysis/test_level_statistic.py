# fmt: off
# flake8: noqa
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
    return {
        "ed": pd.DataFrame(
            {
                "Column Name": [
                    Columns.level_id,
                    Columns.en,
                    Columns.total_sz,
                    Columns.edge_entropy,
                    Columns.bipartite_entropy,
                    Columns.trial_id,
                    Columns.seed,
                    Columns.system_size,
                    Columns.disorder,
                    Columns.penalty,
                    Columns.s_target,
                    Columns.offset,
                ],
                "Type": [
                    "bigint",
                    "double",
                    "double",
                    "double",
                    "double",
                    "bigint",
                    "bigint",
                    "bigint",
                    "double",
                    "double",
                    "bigint",
                    "double",
                ],
                "Partition": [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                "Comment": "",
            }
        ),
        "tsdrg": pd.DataFrame(
            {
                "Column Name": [
                    Columns.level_id,
                    Columns.en,
                    Columns.variance,
                    Columns.total_sz,
                    Columns.edge_entropy,
                    Columns.trial_id,
                    Columns.seed,
                    Columns.offset,
                    Columns.max_en,
                    Columns.min_en,
                    Columns.system_size,
                    Columns.disorder,
                    Columns.truncation_dim,
                    Columns.relative_offset,
                    Columns.penalty,
                    Columns.s_target,
                    Columns.method,
                ],
                "Type": [
                    "bigint",
                    "double",
                    "double",
                    "double",
                    "double",
                    "string",
                    "bigint",
                    "double",
                    "double",
                    "double",
                    "bigint",
                    "double",
                    "bigint",
                    "double",
                    "double",
                    "bigint",
                    "string",
                ],
                "Partition": [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                "Comment": "",
            }
        ),
    }[request.param]


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
    pd.testing.assert_frame_equal(table_df, expected_table)


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



def test_gap_ratio(level_statistic):
    np.testing.assert_allclose(
        level_statistic.gap_ratio(
            np.array(
                [
                    np.nan,
                    0.00726675,
                    0.00550455,
                    0.01220888,
                    0.00061159,
                    0.00013423,
                    0.00775276,
                    0.00357282,
                    0.04445829,
                    0.00015914,
                ]
            )
        ),
        np.array(
            [
                np.nan,
                0.75749753,
                0.45086416,
                0.05009375,
                0.21948545,
                0.01731444,
                0.46084533,
                0.08036352,
                0.00357958,
                np.nan,
            ]
        ),
        atol=1e-5,
    )
    np.testing.assert_array_equal(level_statistic.gap_ratio(np.array([0.0478])), np.nan)
    np.testing.assert_array_equal(
        level_statistic.gap_ratio(np.array([-0.0478, 0.0478])),
        np.array([-1, np.nan]),
    )


def test_extract_gap(level_statistic):
    df = level_statistic.athena_query(8, 10.0, total_sz=0)
    print(df)
    # df2 = self.agent.extract_gap(20, 0.5, chi=64, total_sz=0, seed=1948)
    # print(df2)
    # df[Columns.level_id] = df.groupby([Columns.seed]).cumcount()


def test_averaged_gap_ratio(level_statistic):
    df = level_statistic.extract_gap(n=10, h=1.0, total_sz=0)
    r = level_statistic.averaged_gap_ratio(df, AverageOrder.LEVEL_FIRST)
    print(r)
    r2 = level_statistic.averaged_gap_ratio(df, AverageOrder.DISORDER_FIRST)
    print(r2)
