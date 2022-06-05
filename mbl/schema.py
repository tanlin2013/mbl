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
    disorder: Series[float] = pa.Field(coerce=True)
    trial_id: Series[str] = pa.Field(nullable=True)
    seed: Series[int]
    penalty: Series[float] = pa.Field(coerce=True)
    s_target: Series[int]
    offset: Series[float] = pa.Field(ge=0, le=1, coerce=True)

    @pa.check(Columns.total_sz)
    def close_to_integer(cls, series: Series[float]) -> Series[bool]:
        return Series(np.isclose(series, np.rint(series), atol=1e-12))

    @pa.check(Columns.edge_entropy)
    def bound_in(cls, series: Series[float]) -> Series[bool]:
        return (-1e-12 < series) & (series < np.log(2) + 1e-12)

    @pa.check(Columns.bipartite_entropy)
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
    disorder: Series[float] = pa.Field(coerce=True)
    trial_id: Series[str] = pa.Field(nullable=True)
    seed: Series[int]
    penalty: Series[float] = pa.Field(coerce=True)
    s_target: Series[int]
    offset: Series[float] = pa.Field(coerce=True)
    overall_const: Series[float] = pa.Field(coerce=True)

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
    disorder: Series[float] = pa.Field(coerce=True)
    trial_id: Series[str] = pa.Field(nullable=True)
    seed: Series[int]
    penalty: Series[float] = pa.Field(coerce=True)
    s_target: Series[int]
    offset: Series[float] = pa.Field(coerce=True)
    max_en: Series[float] = pa.Field(coerce=True)
    min_en: Series[float] = pa.Field(coerce=True)
    relative_offset: Series[float] = pa.Field(ge=0, le=1, coerce=True)

    @pa.check(Columns.total_sz)
    def close_to_integer(cls, series: Series[float]) -> Series[bool]:
        return Series(np.isclose(series, np.rint(series), atol=1e-12))

    @pa.check(Columns.edge_entropy)
    def bound_in(cls, series: Series[float]) -> Series[bool]:
        return (-1e-12 < series) & (series < np.log(2) + 1e-12)

    @pa.dataframe_check
    def energy_bounds(cls, df: pd.DataFrame) -> Series[bool]:
        return df[Columns.max_en] > df[Columns.min_en]
