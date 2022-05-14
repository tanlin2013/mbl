import pytest
import numpy as np

from mbl.analysis.level_statistic import (
    LevelStatistic,
    AverageOrder
)


class TestLevelStatistic:

    @pytest.fixture(scope='class')
    def level_statistic(self):
        # raw_df = pd.read_csv(f'{Path(__file__).parent}/random_heisenberg_config.csv')
        return LevelStatistic()

    def test_athena_query(self, level_statistic):
        # df = wr.catalog.table(database="random_heisenberg", table="tsdrg")
        df = level_statistic.athena_query(18, 0.5, chi=16, total_sz=0)
        print(df)

    def test_gap_ratio(self, level_statistic):
        np.testing.assert_allclose(
            level_statistic.gap_ratio(
                np.array([np.nan, 0.00726675, 0.00550455, 0.01220888, 0.00061159, 0.00013423,
                          0.00775276, 0.00357282, 0.04445829, 0.00015914])
            ),
            np.array([np.nan, 0.75749753, 0.45086416, 0.05009375, 0.21948545, 0.01731444,
                      0.46084533, 0.08036352, 0.00357958, np.nan]),
            atol=1e-5
        )
        np.testing.assert_array_equal(
            level_statistic.gap_ratio(np.array([0.0478])),
            np.nan
        )
        np.testing.assert_array_equal(
            level_statistic.gap_ratio(np.array([-0.0478, 0.0478])),
            np.array([-1, np.nan])
        )

    def test_extract_gap(self, level_statistic):
        df = level_statistic.athena_query(8, 10.0, total_sz=0)
        print(df)
        # df2 = self.agent.extract_gap(20, 0.5, chi=64, total_sz=0, seed=1948)
        # print(df2)
        # df[Columns.level_id] = df.groupby([Columns.seed]).cumcount()

    def test_averaged_gap_ratio(self, level_statistic):
        df = level_statistic.extract_gap(n=10, h=1.0, total_sz=0)
        r = level_statistic.averaged_gap_ratio(df, AverageOrder.LEVEL_FIRST)
        print(r)
        r2 = level_statistic.averaged_gap_ratio(df, AverageOrder.DISORDER_FIRST)
        print(r2)
