import click
import subprocess
from pathlib import Path
from typing import Union

import yaml
import ray
from ray import tune

from mbl.name_space import Columns
from mbl.workflow.grid_search import (
    run,
    RandomHeisenbergTSDRGGridSearch,
    RandomHeisenbergFoldingTSDRGGridSearch,
)


def config_parser(workflow: str):
    with open(Path(__file__).parent / "run_config.yml", "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    config = {
        "n": data[Columns.system_size],
        "h": data[Columns.disorder],
        "chi": data[Columns.truncation_dim],
        "seed": list(range(*data[Columns.seed].values())),
        "penalty": data[Columns.penalty],
        "s_target": data[Columns.s_target],
    }
    if workflow == RandomHeisenbergTSDRGGridSearch.__name__:
        config.update(
            {
                "offset": data[Columns.offset],
                "overall_const": data[Columns.overall_const],
                "method": data[Columns.method],
            }
        )
    elif workflow == RandomHeisenbergFoldingTSDRGGridSearch.__name__:
        config.update(
            {
                "relative_offset": data[Columns.relative_offset],
                "method": data[Columns.method],
            }
        )
    return {k: tune.grid_search(v) for k, v in config.items()}


@click.command()
@click.option("-U", "--tracking_uri", default="http://mlflow:5000", type=str)
@click.option("-W", "--workflow", default="RandomHeisenbergTSDRGGridSearch", type=str)
@click.option(
    "-N",
    "--experiment_name",
    default="random_heisenberg-tsdrg",
    type=str,
)
@click.option(
    "--num_cpus", default=34, type=int, help="Number of total available cpus."
)
@click.option("--cpu", default=1, type=int, help="Number of cpu per task.")
@click.option("--memory", default=4, type=float, help="Memory size per task in GB.")
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
@click.option(
    "--resume",
    default=False,
    type=click.Choice(
        ["ERRORED_ONLY", "LOCAL", "REMOTE", "PROMPT", "AUTO", True, False],
        case_sensitive=True,
    ),
)
def main(
    tracking_uri: str,
    workflow: str,
    experiment_name: str,
    num_cpus: int,
    cpu: int,
    memory: float,
    verbose: int,
    resume: Union[bool, str],
):
    tags = {
        k: subprocess.check_output(v.split()).decode("ascii").strip()
        for k, v in {
            "docker.image.id": "hostname",
            "git.commit": "git rev-parse --short HEAD",
        }.items()
    }
    ray.init(
        num_cpus=num_cpus,
        object_store_memory=memory * 1024**3,
    )
    experiment = globals()[workflow].experiment
    run(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        experiment=experiment,
        configs=config_parser(workflow),
        resources_per_trial={"cpu": cpu},
        verbose=verbose,
        tags=tags,
        resume=resume,
    )


if __name__ == "__main__":
    main()
