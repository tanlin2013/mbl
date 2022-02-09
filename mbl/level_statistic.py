import numpy as np
import pandas as pd
import awswrangler as wr
from enum import Enum


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

    def local_query(self, n: int, h: float, penalty: float = 0.0, s_target: int = 0,
                    chi: int = None, total_sz: int = None, tol: float = 1e-12) -> pd.DataFrame:
        return self.raw_df.query(
            f'(SystemSize == {n}) & '
            f'(Disorder == {h}) & '
            f'(Penalty == {penalty}) & '
            f'(STarget == {s_target}) & '
            f'(TruncationDim == {chi}) & '
            f'(abs(TotalSz - {total_sz}) < {tol})'
        )

    @staticmethod
    def athena_query(n: int, h: float, penalty: float = 0.0, s_target: int = 0,
                     chi: int = None, total_sz: int = None, tol: float = 1e-12) -> pd.DataFrame:
        return wr.athena.read_sql_query(
            f'(systemsize = {n}) AND '
            f'(disorder = {h}) AND '
            f'(penalty = {penalty}) AND '
            f'(starget = {s_target}) AND '
            f'(truncationdim = {chi}) AND '
            f'(ABS(totalsz - {total_sz}) < {tol})',
            database="random_heisenberg",
            categories=[
                "levelid", "en", "variance", "totalsz",
                "edgeentropy", "truncationdim", "systemsize",
                "disorder", "trailid", "seed", "penalty",
                "starget", "offset"
            ]
        )

    def extract_gap(self, n: int, h: float, penalty: float = 0.0, s_target: int = 0,
                    chi: int = None, total_sz: int = None, tol: float = 1e-12) -> pd.DataFrame:
        df = self.local_query(**locals()) if self.raw_df is not None \
            else self.athena_query(**locals())
        df['EnergyGap'] = df.groupby(['TrialID'])['En'].apply(lambda x: x.diff())
        df['GapRatio'] = self.gap_ratio(df['EnergyGap'].to_numpy())
        return df.reset_index(drop=True)

    @staticmethod
    def gap_ratio(x: np.ndarray) -> np.ndarray:
        r = np.minimum(x[:-1] / x[1:], x[1:] / x[:-1])
        return np.append(r, np.nan)

    @staticmethod
    def level_average(df: pd.DataFrame) -> pd.Series:
        return df.groupby(['TrialID'])['GapRatio'].mean()

    @staticmethod
    def disorder_average(df: pd.DataFrame) -> pd.Series:
        return df.groupby(['LevelID'])['GapRatio'].mean()

    @staticmethod
    def averaged_gap_ratio(df: pd.DataFrame,
                           order: AverageOrder = AverageOrder.LevelFirst) -> float:
        return LevelStatistic.level_average(df).mean() if order is AverageOrder.LevelFirst \
            else LevelStatistic.disorder_average(df).mean()
