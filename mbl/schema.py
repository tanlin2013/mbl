import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import Series

from mbl.name_space import Columns


class RandomHeisenbergEDSchema(pa.SchemaModel):

    level_id: Series[int] = pa.Field(ge=0)
    en: Series[float]
    total_sz: Series[float]
    edge_entropy: Series[float] = pa.Field(nullable=True)
    bipartite_entropy: Series[float] = pa.Field(nullable=True)
    system_size: Series[int] = pa.Field(gt=1)
    disorder: Series[float]
    trial_id: Series[str] = pa.Field(
        nullable=True, coerce=True
    )  # allow to coerce to int for legacy reason
    seed: Series[int]
    penalty: Series[float]
    s_target: Series[int]
    offset: Series[float]

    @pa.check(Columns.total_sz)
    def close_to_integer(cls, series: Series[float]) -> Series[bool]:
        return Series(np.isclose(series, np.rint(series), atol=1e-12))

    @pa.check(r"entropy$", regex=True)
    def bound_in(cls, series: Series[float]) -> Series[bool]:
        return (-1e-12 < series) & (series < np.log(2) + 1e-12)


class RandomHeisenbergTSDRGSchema(pa.SchemaModel):

    level_id: Series[int] = pa.Field(ge=0)
    en: Series[float]
    variance: Series[float]
    total_sz: Series[float]
    edge_entropy: Series[float] = pa.Field(nullable=True)
    truncation_dim: Series[int]
    system_size: Series[int] = pa.Field(gt=1)
    disorder: Series[float]
    trial_id: Series[str] = pa.Field(nullable=True, coerce=True)
    seed: Series[int]
    penalty: Series[float]
    s_target: Series[int]
    offset: Series[float]
    overall_const: Series[float]
    method: Series[str] = pa.Field(isin=["min", "max"], coerce=True)

    @pa.check(Columns.total_sz)
    def close_to_integer(cls, series: Series[float]) -> Series[bool]:
        return Series(np.isclose(series, np.rint(series), atol=1e-12))

    @pa.check(Columns.edge_entropy)
    def bound_in(cls, series: Series[float]) -> Series[bool]:
        return (-1e-12 < series) & (series < np.log(2) + 1e-12)


class RandomHeisenbergFoldingTSDRGSchema(pa.SchemaModel):
    level_id: Series[int] = pa.Field(ge=0)
    en: Series[float]
    variance: Series[float]
    total_sz: Series[float]
    edge_entropy: Series[float] = pa.Field(nullable=True)
    truncation_dim: Series[int]
    system_size: Series[int] = pa.Field(gt=1)
    disorder: Series[float]
    trial_id: Series[str] = pa.Field(nullable=True, coerce=True)
    seed: Series[int]
    penalty: Series[float]
    s_target: Series[int]
    offset: Series[float]
    max_en: Series[float] = pa.Field(nullable=True)
    min_en: Series[float] = pa.Field(nullable=True)
    relative_offset: Series[float] = pa.Field(ge=0, le=1)
    method: Series[str] = pa.Field(isin=["min", "max"], coerce=True)

    @pa.check(Columns.total_sz)
    def close_to_integer(cls, series: Series[float]) -> Series[bool]:
        return Series(np.isclose(series, np.rint(series), atol=1e-12))

    @pa.check(Columns.edge_entropy)
    def bound_in(cls, series: Series[float]) -> Series[bool]:
        return (-1e-12 < series) & (series < np.log(2) + 1e-12)

    @pa.dataframe_check
    def energy_bounds(cls, df: pd.DataFrame) -> Series[bool]:
        if np.isnan(df[Columns.max_en]).all() and np.isnan(df[Columns.min_en]).all():
            return np.isnan(df[Columns.max_en]) & np.isnan(df[Columns.min_en])
        return Series(df[Columns.max_en] > df[Columns.min_en])

    @pa.dataframe_check
    def offset_within(self, df: pd.DataFrame) -> Series[bool]:
        return Series(df[Columns.min_en] < df[Columns.offset]) & Series(
            df[Columns.offset] < df[Columns.max_en]
        )
