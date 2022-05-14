import ray
import awswrangler as wr
from awswrangler.exceptions import QueryFailed
from botocore.exceptions import ClientError
from time import sleep
from mbl.name_space import Columns
from mbl.analysis.level_statistic import (
    LevelStatistic
)


def retry(func, *args, **kwargs):
    for attempt in range(3):
        try:
            func(*args, **kwargs)
            break
        except ClientError:
            sleep(3)
        except QueryFailed:
            pass


@ray.remote(num_cpus=1)
def main(kwargs):
    df = LevelStatistic.athena_query(**kwargs)
    retry(
        wr.s3.to_parquet,
        df=df,
        path="s3://many-body-localization/dataframe/tsdrg",
        index=False,
        dataset=True,
        mode="append",
        partition_cols=[
            Columns.system_size,
            Columns.disorder,
            Columns.truncation_dim,
            Columns.penalty,
            Columns.s_target,
            Columns.offset
        ],
        database="random_heisenberg",
        table="tsdrg2"
    )


if __name__ == "__main__":

    params = [
        {
            'n': n,
            'h': h,
            'chi': chi,
            'penalty': 0,
            's_target': 0,
            'offset': 0
        }
        for n in [8, 10, 12, 14, 16, 18, 20]
        for chi in [2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6]
        for h in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0]
    ]

    # Distributed.map_on_ray(main, params)

    query_final_state = wr.athena.repair_table(
        table='tsdrg',
        database='random_heisenberg'
    )
    print(query_final_state)

    print(
        wr.catalog.get_partitions(
            database='random_heisenberg',
            table='tsdrg',
        )
    )

    # print(wr.catalog.get_tables(database='random_heisenberg'))

    # wr.catalog.add_parquet_partitions(
    #     database='random_heisenberg',
    #     table='tsdrg',
    #     partitions_values={
    #         's3://many-body-localization/dataframe/tsdrg/system_size=8/': ['8']
    #     }
    # )
