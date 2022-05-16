import abc
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Union, List

import validators
import awswrangler as wr
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback, mlflow_mixin
from mlflow import log_params, log_metric, log_artifact

from mbl.name_space import Columns
from mbl.experiment.random_heisenberg import (
    RandomHeisenbergTSDRG,
    RandomHeisenbergFoldingTSDRG,
)


def run(
    experiment,
    config: Dict[str, Dict[str, List]],
    tracking_uri: str,
    experiment_name: str,
    tags: Dict[str, str] = None,
    save_artifact: bool = True,
    num_samples: int = 1,
    *args,
    **kwargs,
):
    assert not isinstance(
        validators.url(tracking_uri), validators.ValidationFailure
    )
    tune.run(
        experiment,
        config=config,
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri=tracking_uri,
                experiment_name=experiment_name,
                tags=tags,
                save_artifact=save_artifact,
            )
        ],
        num_samples=num_samples,
        *args,
        **kwargs,
    )


@mlflow_mixin
class GridSearch(abc.ABC, tune.Trainable):
    @abc.abstractmethod
    def setup(self, config: Dict[str, Union[int, float, str]]):
        return NotImplemented


@mlflow_mixin
class RandomHeisenbergTSDRGGridSearch(GridSearch):
    @dataclass
    class Metadata:
        s3_path: str = "s3://many-body-localization/random_heisenberg/tsdrg"
        database: str = "random_heisenberg"
        table: str = "tsdrg"

    def setup(self, config: Dict[str, Union[int, float, str]]):
        log_params(config)
        experiment = RandomHeisenbergTSDRG(**config)
        filename = Path(config["local_path"]) / (
            "-".join([f"{k}_{v}" for k, v in config.items()]) + ".p"
        )
        experiment.save_tree(str(filename))
        log_artifact(str(filename), config["artifact_path"])
        df = experiment.compute_df()
        log_metric(key="mean_variance", value=df[Columns.variance].mean())

        wr.s3.to_parquet(
            df=df,
            path=self.Metadata.s3_path,
            index=False,
            dataset=True,
            mode="append",
            schema_evolution=True,
            partition_cols=[
                Columns.system_size,
                Columns.disorder,
                Columns.truncation_dim,
                Columns.overall_const,
                Columns.penalty,
                Columns.s_target,
            ],
            database=self.Metadata.database,
            table=self.Metadata.table,
        )


@mlflow_mixin
class RandomHeisenbergFoldingTSDRGGridSearch(GridSearch):
    @dataclass
    class Metadata:
        s3_path: str = "s3://many-body-localization/random_heisenberg/folding_tsdrg"
        database: str = "random_heisenberg"
        table: str = "folding_tsdrg"

    def setup(self, config: Dict[str, Union[int, float, str]]):
        log_params(config)
        experiment = RandomHeisenbergFoldingTSDRG(**config)
        filename = Path(config["local_path"]) / (
            "-".join([f"{k}_{v}" for k, v in config.items()]) + ".p"
        )
        experiment.save_tree(str(filename))
        log_artifact(str(filename), config["artifact_path"])
        df = experiment.compute_df()
        log_metric(key="mean_variance", value=df[Columns.variance].mean())

        wr.s3.to_parquet(
            df=df,
            path=self.Metadata.s3_path,
            index=False,
            dataset=True,
            mode="append",
            schema_evolution=True,
            partition_cols=[
                Columns.system_size,
                Columns.disorder,
                Columns.truncation_dim,
                Columns.relative_offset,
                Columns.penalty,
                Columns.s_target,
            ],
            database=self.Metadata.database,
            table=self.Metadata.table,
        )
