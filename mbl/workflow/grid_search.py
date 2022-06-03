import abc
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Union, Callable

import boto3
import mlflow
import pandas as pd
from ray import tune
from ray.tune.integration.mlflow import mlflow_mixin
import awswrangler as wr

from mbl.name_space import Columns
from mbl.experiment.random_heisenberg import (
    RandomHeisenbergTSDRG,
    RandomHeisenbergFoldingTSDRG,
)
from mbl.analysis.energy_bounds import EnergyBounds


def run(
    tracking_uri: str,
    experiment_name: str,
    experiment: Callable,
    configs: Dict,
    resources_per_trial: Dict = None,
    token: str = None,
    **kwargs,
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow_config = {
        "mlflow": {
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
            "token": token,
        }
    }
    configs.update(mlflow_config)

    tune.run(
        tune.with_parameters(experiment),
        config=configs,
        num_samples=1,
        resources_per_trial=resources_per_trial,
        **kwargs,
    )


class GridSearch(abc.ABC):
    @staticmethod
    @mlflow_mixin
    @abc.abstractmethod
    def experiment(config: Dict[str, Union[int, float, str]]):
        return NotImplemented


class RandomHeisenbergTSDRGGridSearch(GridSearch):
    @dataclass
    class AthenaMetadata:
        s3_path: str = "s3://many-body-localization/random_heisenberg/tsdrg"
        database: str = "random_heisenberg"
        table: str = "tsdrg"

    @staticmethod
    @mlflow_mixin
    def experiment(config: Dict[str, Union[int, float, str]]):
        boto3.setup_default_session(profile_name="minio")
        config.pop("mlflow")
        mlflow.log_params(config)
        experiment = RandomHeisenbergTSDRG(**config)
        filename = Path("-".join([f"{k}_{v}" for k, v in config.items()]) + ".p")
        experiment.save_tree(str(filename))
        mlflow.log_artifact(str(filename))
        df = experiment.compute_df()
        mlflow.log_metric(key="mean_variance", value=df[Columns.variance].mean())
        mlflow.log_dict(df.to_dict(), "df.json")
        RandomHeisenbergTSDRGGridSearch.to_s3_parquet(df)

    @classmethod
    def to_s3_parquet(cls, df: pd.DataFrame):
        boto3.setup_default_session(profile_name="default")
        wr.s3.to_parquet(
            df=df,
            path=cls.AthenaMetadata.s3_path,
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
            database=cls.AthenaMetadata.database,
            table=cls.AthenaMetadata.table,
        )


class RandomHeisenbergFoldingTSDRGGridSearch(GridSearch):
    @dataclass
    class AthenaMetadata:
        s3_path: str = "s3://many-body-localization/random_heisenberg/folding_tsdrg"
        database: str = "random_heisenberg"
        table: str = "folding_tsdrg"

    @staticmethod
    @mlflow_mixin
    def experiment(config: Dict[str, Union[int, float, str]]):
        config.pop("mlflow")
        config.update(EnergyBounds.retrieve(**config))  # TODO: this can be a bottleneck
        boto3.setup_default_session(profile_name="minio")
        mlflow.log_params(config)
        experiment = RandomHeisenbergFoldingTSDRG(**config)
        filename = Path("-".join([f"{k}_{v}" for k, v in config.items()]) + ".p")
        experiment.save_tree(str(filename))
        mlflow.log_artifact(str(filename))
        df = experiment.compute_df()
        mlflow.log_metric(key="mean_variance", value=df[Columns.variance].mean())
        mlflow.log_dict(df.to_dict(), "df.json")
        RandomHeisenbergFoldingTSDRGGridSearch.to_s3_parquet(df)

    @classmethod
    def to_s3_parquet(cls, df: pd.DataFrame):
        boto3.setup_default_session(profile_name="default")
        wr.s3.to_parquet(
            df=df,
            path=cls.AthenaMetadata.s3_path,
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
            database=cls.AthenaMetadata.database,
            table=cls.AthenaMetadata.table,
        )
