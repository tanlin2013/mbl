from dataclasses import dataclass
from typing import Sequence, Dict, Tuple

import ray
import pandas as pd
import awswrangler as wr

from mbl.name_space import Columns
from mbl.distributed import Distributed
from mbl.analysis.level_statistic import LevelStatistic, AverageOrder


class ETL:
    @dataclass(frozen=True)
    class Metadata:
        database: str = "random_heisenberg"
        ed_table: str = "ed"
        tsdrg_table: str = "folding_tsdrg"
        s3_path: str = "s3://many-body-localization/gap_ratio"
        gap_ratio_table: str = "gap_ratio"

    @classmethod
    def create_gap_ratio_table(cls, params: Sequence[Dict]) -> pd.DataFrame:
        """
        CREATE TABLE AS SELECT (CTAS) approach to dump results from
        :func:`~LevelStatistic.fetch_gap_ratio` into another table.
        This method will run in parallel with ray.

        Args:
            params: List of dictionary that contains kwargs to
                :func:`~LevelStatistic.fetch_gap_ratio`.

        Returns:

        Notes:
            Performance may be bounded by the bandwidth of internet.
            Please adjust the number of cpus in ray.init() accordingly.
        """

        @ray.remote(num_cpus=1)
        def wrapper(kwargs: Dict) -> Tuple[Dict, Dict]:
            _df = LevelStatistic.athena_query(**kwargs)
            _df = LevelStatistic.extract_gap(_df)
            kwargs[Columns.algorithm] = (
                cls.Metadata.tsdrg_table if "chi" in kwargs else cls.Metadata.ed_table
            )
            avg_r1 = kwargs.copy()
            avg_r2 = kwargs.copy()
            for container, order in zip(
                [avg_r1, avg_r2],
                [AverageOrder.LEVEL_FIRST, AverageOrder.DISORDER_FIRST],
            ):
                container[Columns.avg_order] = order.name
                container[Columns.gap_ratio] = LevelStatistic.averaged_gap_ratio(
                    _df, order=order
                )
            return avg_r1, avg_r2

        data = Distributed.map_on_ray(wrapper, params)
        data = list(sum(data, ()))  # flatten list of tuples
        df = pd.DataFrame(data).rename(
            columns={
                "n": Columns.system_size,
                "h": Columns.disorder,
                "chi": Columns.truncation_dim,
                "total_sz": Columns.total_sz,
            },
            errors="ignore",  # ignore non-existing key
        )
        wr.s3.to_parquet(
            df=df,
            path=cls.Metadata.s3_path,
            index=False,
            dataset=True,
            mode="append",
            schema_evolution=True,
            partition_cols=[
                Columns.algorithm,
                Columns.system_size,
                Columns.disorder,
                Columns.avg_order,
            ],
            database=cls.Metadata.database,
            table=cls.Metadata.gap_ratio_table,
        )
        return df
