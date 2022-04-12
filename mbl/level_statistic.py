import numpy as np
import pandas as pd
import awswrangler as wr
from enum import Enum
from mbl.name_space import Columns
from typing import List


class AverageOrder(Enum):
    LevelFirst = 1
    DisorderFirst = 2


class LevelStatistic:

    def __init__(self, raw_df: pd.DataFrame = None):
        self._raw_df = raw_df

    @property
    def raw_df(self):
        return self._raw_df

    @raw_df.setter
    def raw_df(self, raw_df: pd.DataFrame):
        self._raw_df = raw_df

    @staticmethod
    def query_elements(n: int, h: float, penalty: float = 0.0, s_target: int = 0, seed: int = None,
                       chi: int = None, total_sz: int = None, tol: float = 1e-12) -> List[str]:
        query = [
            f'({Columns.system_size} = {n})',
            f'({Columns.disorder} = {h})',
            f'({Columns.penalty} = {penalty})',
            f'({Columns.s_target} = {s_target})'
        ]
        if chi is not None:
            query.append(f'({Columns.truncation_dim} = {chi})')
        if seed is not None:
            query.append(f'({Columns.seed} = {seed})')
        if total_sz is not None:
            query.append(f'(ABS({Columns.total_sz} - {total_sz}) < {tol})')
        return query

    def local_query(self, n: int, h: float, penalty: float = 0.0, s_target: int = 0, seed: int = None,
                    chi: int = None, total_sz: int = None, tol: float = 1e-12) -> pd.DataFrame:
        query = LevelStatistic.query_elements(n, h, penalty, s_target, seed, chi, total_sz, tol)
        return self.raw_df.query(
            ' & '.join(query).replace('=', '==').replace('ABS', 'abs')
        )

    @staticmethod
    def athena_query(n: int, h: float, penalty: float = 0.0, s_target: int = 0, seed: int = None,
                     chi: int = None, total_sz: int = None, tol: float = 1e-12) -> pd.DataFrame:
        query = LevelStatistic.query_elements(**locals())
        table = 'ed' if chi is None else 'tsdrg'
        return wr.athena.read_sql_query(
            f"SELECT * FROM {table} WHERE {' AND '.join(query)}",
            database="random_heisenberg"
        )

    def extract_gap(self, n: int, h: float, penalty: float = 0.0, s_target: int = 0, seed: int = None,
                    chi: int = None, total_sz: int = None, tol: float = 1e-12) -> pd.DataFrame:
        df = self.local_query(n, h, penalty, s_target, seed, chi, total_sz, tol) if self.raw_df is not None \
            else LevelStatistic.athena_query(n, h, penalty, s_target, seed, chi, total_sz, tol)
        df.drop_duplicates(
            subset=[
                Columns.system_size,
                Columns.disorder,
                Columns.penalty,
                Columns.s_target,
                Columns.seed,
                Columns.level_id
            ],
            keep='first', inplace=True
        )
        df[Columns.energy_gap] = df.groupby([Columns.seed])[Columns.en].diff()
        df[Columns.gap_ratio] = df.groupby([Columns.seed])[Columns.energy_gap]\
            .transform(lambda x: LevelStatistic.gap_ratio(x.to_numpy()))
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
    def averaged_gap_ratio(df: pd.DataFrame,
                           order: AverageOrder = AverageOrder.LevelFirst) -> float:
        return LevelStatistic.level_average(df).mean() if order is AverageOrder.LevelFirst \
            else LevelStatistic.disorder_average(df).mean()
