from enum import Enum
from functools import wraps
from dataclasses import dataclass
from typing import List, Union, Callable

import numpy as np
import pandas as pd
import awswrangler as wr
import modin.pandas as mpd

from mbl.name_space import Columns


class AverageOrder(Enum):
    LEVEL_FIRST = 1
    DISORDER_FIRST = 2


class LevelStatistic:
    @dataclass
    class Metadata:
        database: str = "random_heisenberg"
        ed_table: str = "ed"
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
        relative_offset: float = None,
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
        if relative_offset is not None:
            query.append(f"({Columns.relative_offset} = {relative_offset})")
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
        relative_offset: float = None,
        total_sz: int = None,
        tol: float = 1e-12,
    ) -> pd.DataFrame:
        query = self.query_elements(
            n, h, penalty, s_target, seed, chi, relative_offset, total_sz, tol
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
        relative_offset: float = None,
        total_sz: int = None,
        tol: float = 1e-12,
        **kwargs,
    ) -> pd.DataFrame:
        query = cls.query_elements(
            n, h, penalty, s_target, seed, chi, relative_offset, total_sz, tol
        )
        table = cls.Metadata.ed_table if chi is None else cls.Metadata.tsdrg_table
        return wr.athena.read_sql_query(
            f"SELECT * FROM {table} WHERE {' AND '.join(query)}",
            database=cls.Metadata.database,
            **kwargs,
        )

    def check_modin_df(func: Callable):
        @wraps(func)
        def wrapper(
            cls, df: Union[pd.DataFrame, mpd.DataFrame], *args, **kwargs
        ) -> mpd.DataFrame:
            if not isinstance(df, mpd.DataFrame):
                assert isinstance(df, pd.DataFrame)
                df = mpd.DataFrame(df)
            return func(cls, df, *args, **kwargs)

        return wrapper

    @classmethod
    def _get_subset(cls, columns: mpd.Index) -> List[str]:
        subset = [
            Columns.system_size,
            Columns.disorder,
            Columns.penalty,
            Columns.s_target,
            Columns.seed,
            Columns.level_id,
        ]
        if Columns.truncation_dim in columns:
            subset.append(Columns.truncation_dim)
        if Columns.relative_offset in columns:
            subset.append(Columns.relative_offset)
        return subset

    @classmethod
    @check_modin_df
    def extract_gap(cls, df: mpd.DataFrame) -> mpd.DataFrame:
        df.drop_duplicates(
            subset=cls._get_subset(df.columns),
            keep="first",
            inplace=True,
        )
        df[Columns.energy_gap] = df.groupby(Columns.seed)[Columns.en].diff()
        # TODO: check why there are negative gpas?
        df[Columns.energy_gap][df[Columns.energy_gap] < 0] = np.nan
        df[Columns.gap_ratio] = df.groupby(Columns.seed)[Columns.energy_gap].transform(
            lambda x: cls.gap_ratio(x.to_numpy())
        )
        df[Columns.energy_gap] = df.groupby([Columns.seed])[Columns.en].diff()
        df[Columns.gap_ratio] = df.groupby([Columns.seed])[
            Columns.energy_gap
        ].transform(lambda x: cls.gap_ratio(x.to_numpy()))
        return df.reset_index(drop=True)

    @staticmethod
    def gap_ratio(gap: np.ndarray) -> np.ndarray:
        assert np.isnan(gap[0])
        assert (gap[~np.isnan(gap)] > 0).all()
        next_gap = np.roll(gap, -1)
        return np.minimum(gap / next_gap, next_gap / gap)

    @classmethod
    @check_modin_df
    def level_average(cls, df: mpd.DataFrame) -> mpd.Series:
        return df.groupby(Columns.seed)[Columns.gap_ratio].mean()

    @classmethod
    @check_modin_df
    def disorder_average(cls, df: mpd.DataFrame) -> mpd.Series:
        return df.groupby(Columns.level_id)[Columns.gap_ratio].mean()

    @classmethod
    @check_modin_df
    def averaged_gap_ratio(
        cls, df: mpd.DataFrame, order: AverageOrder = AverageOrder.LEVEL_FIRST
    ) -> float:
        return {
            AverageOrder.LEVEL_FIRST: cls.level_average(df).mean(),
            AverageOrder.DISORDER_FIRST: cls.disorder_average(df).mean(),
        }[order]

    @classmethod
    def fetch_gap_ratio(
        cls,
        n: int,
        h: float,
        chi: int = None,
        total_sz: int = None,
        order: AverageOrder = AverageOrder.LEVEL_FIRST,
    ) -> float:
        df = cls.athena_query(n=n, h=h, chi=chi, total_sz=total_sz)
        return cls.averaged_gap_ratio(cls.extract_gap(df), order=order)
