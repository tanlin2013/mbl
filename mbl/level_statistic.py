import numpy as np
import pandas as pd
from enum import Enum
from typing import Union


class AverageOrder(Enum):
    LevelFirst = 1
    DisorderFirst = 2


class LevelStatistic:

    def __init__(self, raw_df: pd.DataFrame):
        self._raw_df = raw_df

    @property
    def raw_df(self):
        return self._raw_df

    @staticmethod
    def gap_ratio(x: np.ndarray) -> np.ndarray:
        r = np.minimum(x[:-1] / x[1:], x[1:] / x[:-1])
        return np.append(r, np.nan)

    def extract_gap(self, N: int, h: float, penalty: float = 0.0, s_target: int = 0,
                    total_sz: Union[int, None] = None, tol: float = 1e-12) -> pd.DataFrame:
        df = self.raw_df.query(
            f'(SystemSize == {N}) & '
            f'(Disorder == {h}) & '
            f'(Penalty == {penalty}) & '
            f'(STarget == {s_target})'
        )
        if total_sz is not None:
            df = df.query(f'abs(TotalSz - {total_sz}) < {tol}')
        df['EnergyGap'] = df.groupby(['TrialID'])['En'].apply(lambda x: x.diff())
        df['GapRatio'] = self.gap_ratio(df['EnergyGap'].to_numpy())
        return df.reset_index(drop=True)

    def level_average(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby(['TrialID'])['GapRatio'].mean()

    def disorder_average(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby(['LevelID'])['GapRatio'].mean()

    def averaged_gap_ratio(self, df: pd.DataFrame,
                           order: AverageOrder = AverageOrder.LevelFirst) -> float:
        return self.level_average(df).mean() if order is AverageOrder.LevelFirst \
            else self.disorder_average(df).mean()
