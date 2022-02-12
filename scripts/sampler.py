# import ray
import numpy as np
import pandas as pd
import awswrangler as wr
from botocore.exceptions import ClientError
from pathlib import Path
from time import sleep
# from dask import config
# from dask.distributed import LocalCluster
from mbl.model import RandomHeisenbergED, RandomHeisenbergTSDRG
from mbl.distributed import Distributed


def retry(func, *args, **kwargs):
    for attempt in range(3):
        try:
            func(*args, **kwargs)
            break
        except ClientError:
            sleep(3)


def main1(kwargs) -> pd.DataFrame:
    agent = RandomHeisenbergED(**kwargs)
    df = agent.df
    retry(
        wr.s3.to_parquet,
        df=df,
        path="s3://many-body-localization/dataframe/ed",
        dataset=True,
        mode="append",
        database="random_heisenberg",
        table="ed"
    )
    return df


def main2(kwargs) -> pd.DataFrame:
    # print(kwargs)
    agent = RandomHeisenbergTSDRG(**kwargs)
    df = agent.df
    # retry(
    #     wr.s3.to_parquet,
    #     df=df,
    #     path="s3://many-body-localization/dataframe/tsdrg",
    #     index=False,
    #     dataset=True,
    #     mode="append",
    #     database="random_heisenberg",
    #     table="tsdrg"
    # )
    # path = Path(f"{Path(__file__).parents[1]}/data/tree")
    # path.mkdir(parents=True, exist_ok=True)
    # filename = "-".join([f"{k}_{v}" for k, v in kwargs.items()])
    # agent.save_tree(f"{path}/{filename}")
    del agent
    return df


def resource_aware_func(**kwargs):
    n = kwargs.get('n')
    chi = kwargs.get('chi')
    return {
        'num_cpus': 1,
        'memory': (n - 1 - np.log2(chi)) * (chi ** 4) * 8
    }


def scopion():
    from dask_jobqueue import SLURMCluster
    return SLURMCluster(
        cores=32,
        memory="10G",
        processes=30,
        queue='scopion1',
        walltime="00:30:00",
        header_skip=["--mem"],
        scheduler_options={"host": "192.168.1.254"},
        # host='192.168.1.254',
        # extra=['--no-dashboard'],
        env_extra=['module load singularity', ],  # ensure singularity is loaded
        python=f"singularity run mbl.sif python",  # use python in container
    )


if __name__ == "__main__":

    penalty = 0.0
    s_target = 0
    offset = 0.0
    n_conf = 500

    params = [
        {
            'n': n,
            'h': h,
            'chi': chi,
            'trial_id': trial_id,
            'seed': seed,
            'penalty': penalty,
            's_target': s_target,
            'offset': offset
        }
        for n in [8, 10, 12, 14, 16, 18][::-1]
        for chi in [2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7][::-1]
        for h in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0]
        for trial_id, seed in enumerate(range(1900, 1900 + n_conf))
    ]

    # if "random_heisenberg" not in wr.catalog.databases().values:
    #     wr.catalog.create_database("random_heisenberg")

    # config.set(
    #     {
    #         "distributed.comm.timeouts.tcp": "50s",
    #         "distributed.workers.memory.target": 1,
    #         "distributed.workers.memory.spill": False,
    #         "distributed.workers.memory.pause": 10,
    #         "distributed.workers.memory.terminate": False
    #     }
    # )

    # cluster = scopion()
    # print(cluster.job_script())
    # cluster = LocalCluster(
    #     n_workers=32,
    #     threads_per_worker=1,
    #     memory_limit="700MiB",
    #     memory_target_fraction=1,
    #     memory_pause_fraction=2
    # )
    # cluster.adapt(
    #     minimum=0,
    #     maximum=32,
    #     maximum_memory="28GiB",
    #     memory_ratio=10,
    #     n=10,  # number of workers to close by once
    #     target_duration="30s",
    #     wait_count=1  # scale down more gently
    # )
    results = Distributed.map_on_ray(main2, params, resource_aware_func)
    # print(wr.catalog.table(database="random_heisenberg", table="tsdrg"))
    # merged_df = pd.concat(results)
    # merged_df.to_parquet(f'~/data/random_heisenberg_tsdrg.parquet', index=False)
    # merged_df.to_parquet(f'{Path(__file__).parents[1]}/data/random_heisenberg_tsdrg.parquet', index=False)
