import abc
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Union, List

import awswrangler as wr
from mlflow import log_params, log_metric, log_artifact
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback, mlflow_mixin

from mbl.name_space import Columns
from mbl.experiment.random_heisenberg import (
    RandomHeisenbergTSDRG,
    RandomHeisenbergFoldingTSDRG,
)


class GridSearch(abc.ABC):
    def __init__(self, tracking_uri: str):
        self._tracking_uri = tracking_uri

    @abc.abstractmethod
    def experiment(self):
        return NotImplemented

    def run(
        self,
        config: Dict[str, Dict[str, List]],
        experiment_name: str,
        tags: Dict[str, str] = None,
        save_artifact: bool = True,
        *args,
        **kwargs,
    ):
        tune.run(
            self.experiment,
            config=config,
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=self._tracking_uri,
                    experiment_name=experiment_name,
                    tags=tags,
                    save_artifact=save_artifact,
                )
            ],
            num_samples=1,
            *args,
            **kwargs,
        )


class RandomHeisenbergTSDRGGridSearch(GridSearch):
    @dataclass
    class Metadata:
        s3_path: str = "s3://many-body-localization/random_heisenberg/tsdrg"
        database: str = "random_heisenberg"
        table: str = "tsdrg"

    def __init__(self, tracking_uri: str, local_path: str, artifact_path: str = None):
        super().__init__(tracking_uri)
        self._local_path = local_path
        self._artifact_path = artifact_path

    @mlflow_mixin
    def experiment(self, **params: Dict[str, Union[int, float]]):
        log_params(params)
        experiment = RandomHeisenbergTSDRG(**params)
        filename = Path(self._local_path) / (
            "-".join([f"{k}_{v}" for k, v in params.items()]) + ".p"
        )
        experiment.save_tree(str(filename))
        log_artifact(str(filename), self._artifact_path)
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


class RandomHeisenbergFoldingTSDRGGridSearch(GridSearch):
    @dataclass
    class Metadata:
        s3_path: str = "s3://many-body-localization/random_heisenberg/folding_tsdrg"
        database: str = "random_heisenberg"
        table: str = "folding_tsdrg"

    def __init__(self, tracking_uri: str, local_path: str, artifact_path: str = None):
        super().__init__(tracking_uri)
        self._local_path = local_path
        self._artifact_path = artifact_path

    @mlflow_mixin
    def experiment(self, **params: Dict[str, Union[int, float]]):
        log_params(params)
        experiment = RandomHeisenbergFoldingTSDRG(**params)
        filename = Path(self._local_path) / (
            "-".join([f"{k}_{v}" for k, v in params.items()]) + ".p"
        )
        experiment.save_tree(str(filename))
        log_artifact(str(filename), self._artifact_path)
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
