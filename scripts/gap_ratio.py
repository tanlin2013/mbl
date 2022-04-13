import ray
import pandas as pd
import awswrangler as wr
from botocore.exceptions import ClientError
from time import sleep
from mbl.name_space import Columns
from mbl.distributed import Distributed
from mbl.level_statistic import LevelStatistic


def retry(func, *args, **kwargs):
    for attempt in range(3):
        try:
            func(*args, **kwargs)
            break
        except ClientError:
            sleep(3)


@ray.remote(num_cpus=1)
def fetch_gap_ratio(kwargs):
    df = LevelStatistic().extract_gap(**kwargs)
    df = pd.DataFrame(
        {
            Columns.system_size: [kwargs.get('n')],
            Columns.disorder: [kwargs.get('h')],
            Columns.truncation_dim: [kwargs.get('chi')],
            Columns.total_sz: [kwargs.get('total_sz')],
            Columns.gap_ratio: [LevelStatistic.averaged_gap_ratio(df)]
        }
    )
    retry(
        wr.s3.to_parquet,
        df=df,
        path="s3://many-body-localization/gap_ratio",
        index=False,
        dataset=True,
        mode="append",
        database="random_heisenberg",
        table="gap_ratio"
    )


if __name__ == "__main__":

    params = [
        {
            'n': n,
            'h': h,
            'chi': chi,
            'total_sz': total_sz
        }
        for n in [8, 10, 12, 14, 16, 18, 20]
        for chi in [2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6]
        for h in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0]
        for total_sz in [None, 0, 1]
    ]

    ray.init(num_cpus=16)
    results = Distributed.map_on_ray(fetch_gap_ratio, params)
