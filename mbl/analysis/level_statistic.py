from enum import Enum
from functools import wraps
from dataclasses import dataclass
from typing import List, Union, Callable

import numpy as np
import pandas as pd
import awswrangler as wr
import modin.pandas as mpd

from mbl import logger
from mbl.name_space import Columns


class AverageOrder(Enum):
    LEVEL_FIRST = 1
    DISORDER_FIRST = 2


class LevelStatistic:
    @dataclass(frozen=True)
    class Metadata:
        database: str = "random_heisenberg"
        ed_table: str = "ed"
        tsdrg_table: str = "folding_tsdrg"

    def __init__(self, raw_df: pd.DataFrame = None):
        r"""
        Analyze the level statistic on :math:`K`-rounds of experiment (disorder trials),
        with each having :math:`N` levels,

        .. math::

            \langle r \rangle = \mathbb{E}_{k, n}\, r_n^{(k)},

        where :math:`r` is the gap ratio parameter defined in :func:`~gap_ratio`.

        Note that the order of taking average does matter when
        constraining in certain charge sector only,
        instead of the full spectrum.

        Args:
            raw_df: The raw data. If provided, local queries will be executed.
                Default None to extract data from AWS Athena.

        Examples:
            >>> agent = LevelStatistic()
            >>> df = agent.fetch_gap_ratio()
        """
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

    def df_cleaning(func: Callable):
        @wraps(func)
        def wrapper(cls, *args, **kwargs) -> pd.DataFrame:
            df = func(cls, *args, **kwargs)
            return (
                df.drop_duplicates(
                    subset=cls._get_subset(df.columns),
                    keep="first",
                )
                .sort_values([Columns.seed, Columns.en], ascending=True)
                .reset_index(drop=True)
            )

        return wrapper

    @df_cleaning
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
    @df_cleaning
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
        # TODO: maybe it will be better to select only the essential columns
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
            # if not isinstance(df, mpd.DataFrame):
            #     assert isinstance(df, pd.DataFrame)
            #     df = mpd.DataFrame(df)
            return func(cls, df, *args, **kwargs)

        return wrapper

    @classmethod
    @check_modin_df
    def extract_gap(cls, df: mpd.DataFrame) -> mpd.DataFrame:
        """
        Feed in the DataFrame obtained through either
        :func:`~LevelStatistic.local_query` or :func:`~LevelStatistic.athena_query`,
        then compute the energy gap and the gap ratio parameter.

        Args:
            df:

        Returns:

        """
        df.drop_duplicates(
            subset=cls._get_subset(df.columns),
            keep="first",
            inplace=True,
        )
        df[Columns.energy_gap] = df.groupby(Columns.seed)[Columns.en].diff()
        if (df[Columns.energy_gap] < 0).any():
            logger.critical("Encounter negative energy gap, set value to NaN.")
            df[Columns.energy_gap][df[Columns.energy_gap] < 0] = np.nan
        df[Columns.gap_ratio] = df.groupby(Columns.seed)[Columns.energy_gap].transform(
            lambda x: cls.gap_ratio(x.to_numpy())
        )
        df[Columns.level_id] = (
            df.groupby(Columns.seed)[Columns.en]
            .rank(method="first", ascending=True)
            .astype("int64")
        )
        return df.reset_index(drop=True)

    @staticmethod
    def gap_ratio(gap: np.ndarray) -> np.ndarray:
        r"""
        For k-th disorder trial, the gap ratio is defined as

        .. math::

            r_n^{(k)} = \min\left(
                \frac{\delta_n}{\delta_{n+1}}, \frac{\delta_{n+1}}{\delta_n}
            \right),

        where :math:`\delta_n = E_{n+1} - E_n` is the n-th energy gap.

        Args:
            gap:

        Returns:

        """
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
        relative_offset: float = None,
        order: AverageOrder = AverageOrder.LEVEL_FIRST,
    ) -> float:
        df = cls.athena_query(
            n=n, h=h, chi=chi, total_sz=total_sz, relative_offset=relative_offset
        )
        return cls.averaged_gap_ratio(cls.extract_gap(df), order=order)
