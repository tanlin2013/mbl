import os
import click
from pathlib import Path

import yaml
import ray
from ray import tune

from mbl.name_space import Columns
from mbl.pipeline.grid_search import (
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


def config_parser(local_path: str, artifact_path: str):
    with open(Path(__file__).parent / "run_config.yml", "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return {
        "n": tune.grid_search(data[Columns.system_size]),
        "h": tune.grid_search(data[Columns.disorder]),
        "chi": tune.grid_search(data[Columns.truncation_dim]),
        "seed": tune.grid_search(
            list(
                range(
                    data[Columns.seed]["start"],
                    data[Columns.seed]["end"],
                    data[Columns.seed]["step"],
                )
            )
        ),
        "penalty": data[Columns.penalty],
        "s_target": data[Columns.s_target],
        "offset": data[Columns.offset],
        "overall_const": data[Columns.overall_const],
        "local_path": local_path,
        "artifact_path": artifact_path,
    }


@click.command()
@click.option("-U", "--tracking_uri", type=str)
@click.option(
    "-P", "--artifact_path", default="/random_heisenberg/tsdrg/tree", type=str
)
@click.option(
    "-N", "--experiment_name", default="random_heisenberg-tsdrg-energy_bounds", type=str
)
@click.option("--save_artifact", default=True, type=bool)
@click.option(
    "--num_cpus", default=16, type=int, help="Number of total available cpus."
)
@click.option("--cpu", default=1, type=int, help="Number of cpu per task.")
@click.option("--memory", default=1, type=float, help="Memory size per task in GB.")
def main(
    tracking_uri: str,
    artifact_path: str,
    experiment_name: str,
    save_artifact: bool,
    num_cpus: int,
    cpu: int,
    memory: float,
):
    ray.init(num_cpus=num_cpus)
    run(
        experiment=RandomHeisenbergTSDRGGridSearch,
        config=config_parser(local_path=os.getcwd(), artifact_path=artifact_path),
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        save_artifact=save_artifact,
        resources_per_trial={"cpu": cpu, "memory": memory * 1024 ** 3},
    )


if __name__ == "__main__":
    main()

    # if "random_heisenberg" not in wr.catalog.databases().values:
    #     wr.catalog.create_database("random_heisenberg")
    # print(wr.catalog.table(database="random_heisenberg", table="tsdrg"))
