import os
import logging
import pandas as pd
from pathlib import Path
from dask_jobqueue import SLURMCluster
from mbl.distributed import Distribute


def main1(kwargs) -> pd.DataFrame:
    from mbl.model import RandomHeisenbergED
    agent = RandomHeisenbergED(**kwargs)
    return agent.df


def main2(kwargs) -> pd.DataFrame:
    # import awswrangler as wr
    from mbl.model import RandomHeisenbergTSDRG

    agent = RandomHeisenbergTSDRG(**kwargs)
    # agent.save_tree("")
    df = agent.df
    # wr.s3.to_parquet(
    #     df=df,
    #     path="",
    #     dataset=True,
    #     database="awswrangler_test",
    #     table="noaa"
    # )
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
        python=f"singularity run {os.environ['SINGULARITY_CONTAINER']} python",  # use python in container
    )


if __name__ == "__main__":

    penalty = 0.0
    s_target = 0
    offset = 0.0
    n_conf = 10

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
        for n in [8, 10]
        for h in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0]
        for chi in [2**4, 2**6, 2**8]
        for trial_id, seed in enumerate(range(1900, 1900 + n_conf))
    ]

    cluster = scopion()
    logging.info(cluster.job_script())
    cluster.adapt(
        minimum=4,
        maximum=48,
        target_duration="1200",  # measured in CPU time per worker -> 120 seconds at 10 cores / worker
        wait_count=4  # scale down more gently
    )
    results = Distribute.map_on_dask(main2, params, cluster)
    # wr.catalog.table(database="awswrangler_test", table="noaa")
    merged_df = pd.concat(results)
    merged_df.to_parquet(f'~/data/random_heisenberg_tsdrg.parquet', index=False)
    # merged_df.to_parquet(f'{Path(__file__).parents[1]}/data/random_heisenberg_tsdrg.parquet', index=False)
