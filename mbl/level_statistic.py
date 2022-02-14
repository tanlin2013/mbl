import numpy as np
import pandas as pd
import awswrangler as wr
from enum import Enum
from mbl.model import Columns


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
        query = f'({Columns.system_size} == {n}) & ' \
            f'({Columns.disorder} == {h}) & ' \
            f'({Columns.penalty} == {penalty}) & ' \
            f'({Columns.s_target} == {s_target})'
        if total_sz is not None:
            query += f' & (abs({Columns.total_sz} - {total_sz}) < {tol})'
        if chi is not None:
            query += f' & ({Columns.truncation_dim} == {chi})'
        return self.raw_df.query(query)

    @staticmethod
    def athena_query(n: int, h: float, penalty: float = 0.0, s_target: int = 0,
                     chi: int = None, total_sz: int = None, tol: float = 1e-12) -> pd.DataFrame:
        table = 'ed' if chi is None else 'tsdrg'
        query = f'SELECT * FROM {table} WHERE' \
                f'({Columns.system_size} = {n}) AND ' \
                f'({Columns.disorder} = {h}) AND ' \
                f'({Columns.penalty} = {penalty}) AND ' \
                f'({Columns.s_target} = {s_target})'
        if total_sz is not None:
            query += f' AND (ABS({Columns.total_sz} - {total_sz}) < {tol})'
        if table == 'tsdrg':
            query += f' AND ({Columns.truncation_dim} = {chi})'
        return wr.athena.read_sql_query(query, database="random_heisenberg")

    def extract_gap(self, n: int, h: float, penalty: float = 0.0, s_target: int = 0,
                    chi: int = None, total_sz: int = None, tol: float = 1e-12) -> pd.DataFrame:
        df = self.local_query(**locals()) if self.raw_df is not None \
            else LevelStatistic.athena_query(n, h, penalty, s_target, chi, total_sz, tol)
        df[Columns.energy_gap] = df.groupby([Columns.trial_id])[Columns.en].apply(lambda x: x.diff())
        df[Columns.gap_ratio] = LevelStatistic.gap_ratio(df[Columns.energy_gap].to_numpy())
        return df.reset_index(drop=True)

    @staticmethod
    def gap_ratio(x: np.ndarray) -> np.ndarray:
        r = np.minimum(x[:-1] / x[1:], x[1:] / x[:-1])
        return np.append(r, np.nan)

    @staticmethod
    def level_average(df: pd.DataFrame) -> pd.Series:
        return df.groupby([Columns.trial_id])[Columns.gap_ratio].mean()

    @staticmethod
    def disorder_average(df: pd.DataFrame) -> pd.Series:
        return df.groupby([Columns.trial_id])[Columns.gap_ratio].mean()

    @staticmethod
    def averaged_gap_ratio(df: pd.DataFrame,
                           order: AverageOrder = AverageOrder.LevelFirst) -> float:
        return LevelStatistic.level_average(df).mean() if order is AverageOrder.LevelFirst \
            else LevelStatistic.disorder_average(df).mean()
