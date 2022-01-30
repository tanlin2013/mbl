import ray
import pandas as pd
import awswrangler as wr
from pathlib import Path
from dask_jobqueue import SLURMCluster
from mbl.model import RandomHeisenbergED, RandomHeisenbergTSDRG
from mbl.distributed import Distributed


def main1(kwargs) -> pd.DataFrame:
    agent = RandomHeisenbergED(**kwargs)
    df = agent.df
    wr.s3.to_parquet(
        df=df,
        path="s3://many-body-localization/data/",
        dataset=True,
        mode="overwrite",
        database="random_heisenberg",
        table="ed"
    )
    return agent.df


@ray.remote(memory=2 * 1024 ** 3)
def main2(kwargs) -> pd.DataFrame:
    agent = RandomHeisenbergTSDRG(**kwargs)
    # agent.save_tree("")
    df = agent.df
    wr.s3.to_parquet(
        df=df,
        path="s3://many-body-localization/data/",
        dataset=True,
        database="random_heisenberg",
        table="tsdrg"
    )
    return df


def scopion():
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
        for n in [8, 10, 12, 14, 16]
        for h in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0]
        for chi in [2**4, 2**5, 2**6, 2**7]
        for trial_id, seed in enumerate(range(1900, 1900 + n_conf))
    ]

    # if "random_heisenberg" not in wr.catalog.databases().values:
    #     wr.catalog.create_database("random_heisenberg")

    # cluster = scopion()
    # print(cluster.job_script())
    # cluster.adapt(
    #     minimum=4,
    #     maximum=32,
    #     target_duration="1200",  # measured in CPU time per worker -> 120 seconds at 10 cores / worker
    #     wait_count=4  # scale down more gently
    # )
    results = Distributed.map_on_ray(main2, params)
    # print(wr.catalog.table(database="random_heisenberg", table="tsdrg"))
    merged_df = pd.concat(results)
    # merged_df.to_parquet(f'~/data/random_heisenberg_tsdrg.parquet', index=False)
    merged_df.to_parquet(f'{Path(__file__).parents[1]}/data/random_heisenberg_tsdrg.parquet', index=False)
