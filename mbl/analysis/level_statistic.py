from enum import Enum
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import awswrangler as wr

from mbl.name_space import Columns


class AverageOrder(Enum):
    LEVEL_FIRST = 1
    DISORDER_FIRST = 2


class LevelStatistic:
    @dataclass
    class Metadata:
        database: str = "random_heisenberg"
        ed_table: str = "ed2"
        tsdrg_table: str = "folding_tsdrg"

    def __init__(self, raw_df: pd.DataFrame = None):
        self._raw_df = raw_df

    @property
    def raw_df(self) -> pd.DataFrame:
        return self._raw_df

    @raw_df.setter
    def raw_df(self, raw_df: pd.DataFrame):
        self._raw_df = raw_df

    @classmethod
    def query_elements(
        cls,
        n: int,
        h: float,
        penalty: float = 0.0,
        s_target: int = 0,
        seed: int = None,
        chi: int = None,
        total_sz: int = None,
        tol: float = 1e-12,
    ) -> List[str]:
        query = [
            f"({Columns.system_size} = {n})",
            f"({Columns.disorder} = {h})",
            f"({Columns.penalty} = {penalty})",
            f"({Columns.s_target} = {s_target})",
        ]
        if chi is not None:
            query.append(f"({Columns.truncation_dim} = {chi})")
        if seed is not None:
            query.append(f"({Columns.seed} = {seed})")
        if total_sz is not None:
            query.append(f"(ABS({Columns.total_sz} - {total_sz}) < {tol})")
        return query

    def local_query(
        self,
        n: int,
        h: float,
        penalty: float = 0.0,
        s_target: int = 0,
        seed: int = None,
        chi: int = None,
        total_sz: int = None,
        tol: float = 1e-12,
    ) -> pd.DataFrame:
        query = self.query_elements(
            n, h, penalty, s_target, seed, chi, total_sz, tol
        )
        return self.raw_df.query(
            " & ".join(query).replace("=", "==").replace("ABS", "abs")
        )

    @classmethod
    def athena_query(
        cls,
        n: int,
        h: float,
        penalty: float = 0.0,
        s_target: int = 0,
        seed: int = None,
        chi: int = None,
        total_sz: int = None,
        tol: float = 1e-12,
        **kwargs,
    ) -> pd.DataFrame:
        query = cls.query_elements(
            n, h, penalty, s_target, seed, chi, total_sz, tol
        )
        table = cls.Metadata.ed_table if chi is None else cls.Metadata.tsdrg_table
        return wr.athena.read_sql_query(
            f"SELECT * FROM {table} WHERE {' AND '.join(query)}",
            database=cls.Metadata.database,
            **kwargs,
        )

    def extract_gap(
        self,
        n: int,
        h: float,
        penalty: float = 0.0,
        s_target: int = 0,
        seed: int = None,
        chi: int = None,
        total_sz: int = None,
        tol: float = 1e-12,
        **kwargs,
    ) -> pd.DataFrame:
        df = (
            self.local_query(n, h, penalty, s_target, seed, chi, total_sz, tol)
            if self.raw_df is not None
            else self.athena_query(
                n, h, penalty, s_target, seed, chi, total_sz, tol
            )
        )
        df.drop_duplicates(
            subset=[
                Columns.system_size,
                Columns.disorder,
                Columns.penalty,
                Columns.s_target,
                Columns.seed,
                Columns.level_id,
            ],
            keep="first",
            inplace=True,
        )
        df[Columns.energy_gap] = df.groupby([Columns.seed])[Columns.en].diff()
        df[Columns.gap_ratio] = df.groupby([Columns.seed])[
            Columns.energy_gap
        ].transform(lambda x: LevelStatistic.gap_ratio(x.to_numpy()))
        return df.reset_index(drop=True)

    @staticmethod
    def gap_ratio(x: np.ndarray) -> np.ndarray:
        r = np.minimum(x[:-1] / x[1:], x[1:] / x[:-1])
        return np.append(r, np.nan)

    @staticmethod
    def level_average(df: pd.DataFrame) -> pd.Series:
        return df.groupby([Columns.seed])[Columns.gap_ratio].mean()

    @staticmethod
    def disorder_average(df: pd.DataFrame) -> pd.Series:
        return df.groupby([Columns.level_id])[Columns.gap_ratio].mean()

    @staticmethod
    def averaged_gap_ratio(
        df: pd.DataFrame, order: AverageOrder = AverageOrder.LEVEL_FIRST
    ) -> float:
        return (
            LevelStatistic.level_average(df).mean()
            if order is AverageOrder.LEVEL_FIRST
            else LevelStatistic.disorder_average(df).mean()
        )
