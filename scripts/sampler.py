import os
import click
from pathlib import Path

import yaml
import ray
from ray import tune

from mbl.name_space import Columns
from mbl.workflow.grid_search import (
    run,
    RandomHeisenbergTSDRGGridSearch,
    RandomHeisenbergFoldingTSDRGGridSearch,
)
from mbl.analysis.energy_bounds import EnergyBounds


def scopion():
    from dask_jobqueue import SLURMCluster

    return SLURMCluster(
        cores=32,
        memory="10G",
        processes=30,
        queue="scopion1",
        walltime="00:30:00",
        header_skip=["--mem"],
        scheduler_options={"host": "192.168.1.254"},
        # host='192.168.1.254',
        # extra=['--no-dashboard'],
        env_extra=["module load singularity"],  # ensure singularity is loaded
        python="singularity run mbl.sif python",  # use python in container
    )


def config_parser():
    with open(Path(__file__).parent / "run_config.yml", "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    config = {
        "n": data[Columns.system_size],
        "h": data[Columns.disorder],
        "chi": data[Columns.truncation_dim],
        "seed": list(range(*data[Columns.seed].values())),
        "penalty": data[Columns.penalty],
        "s_target": data[Columns.s_target],
        "offset": data[Columns.offset],
        "overall_const": data[Columns.overall_const],
    }
    return {k: tune.grid_search(v) for k, v in config.items()}


@click.command()
@click.option("-U", "--tracking_uri", default="http://localhost:8787", type=str)
@click.option("-E", "--mlflow_s3_endpoint", default="http://localhost:9000", type=str)
@click.option(
    "-N", "--experiment_name", default="random_heisenberg-tsdrg-energy_bounds", type=str
)
@click.option(
    "--num_cpus", default=16, type=int, help="Number of total available cpus."
)
@click.option("--cpu", default=1, type=int, help="Number of cpu per task.")
@click.option("--memory", default=1, type=float, help="Memory size per task in GB.")
@click.option(
    "--verbose",
    default=3,
    type=int,
    help="0, 1, 2, or 3. Verbosity mode. "
         "0 = silent, "
         "1 = only status updates, "
         "2 = status and brief trial results, "
         "3 = status and detailed trial results",
)
def main(
    tracking_uri: str,
    mlflow_s3_endpoint: str,
    experiment_name: str,
    num_cpus: int,
    cpu: int,
    memory: float,
    verbose: int,
):
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_s3_endpoint
    ray.init(num_cpus=num_cpus, object_store_memory=memory * 1024 ** 3)
    run(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        experiment=RandomHeisenbergTSDRGGridSearch.experiment,
        configs=config_parser(),
        resources_per_trial={"cpu": cpu},
        verbose=verbose,
    )


if __name__ == "__main__":
    main()

    # if "random_heisenberg" not in wr.catalog.databases().values:
    #     wr.catalog.create_database("random_heisenberg")
    # print(wr.catalog.table(database="random_heisenberg", table="tsdrg"))
