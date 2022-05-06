import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import Series

from mbl.name_space import Columns


class EDSchema(pa.SchemaModel):

    level_id: Series[int] = pa.Field(ge=0)
    en: Series[float]
    total_sz: Series[float]
    edge_entropy: Series[float] = pa.Field(ge=0, le=np.log(2), nullable=True)
    bipartite_entropy: Series[float] = pa.Field(ge=0, le=np.log(2), nullable=True)
    system_size: Series[int] = pa.Field(gt=1)
    disorder: Series[float] = pa.Field(coerce=True)
    trial_id: Series[pd.Int64Dtype] = pa.Field(nullable=True, coerce=True)
    seed: Series[int]
    penalty: Series[float] = pa.Field(coerce=True)
    s_target: Series[int]
    offset: Series[float] = pa.Field(ge=0, le=1, coerce=True)

    @pa.check(Columns.total_sz)
    def close_to_integer(cls, series: Series[float]) -> Series[bool]:
        return Series(np.isclose(series, np.rint(series), atol=1e-12))


class TSDRGSchema(pa.SchemaModel):

    level_id: Series[int] = pa.Field(ge=0)
    en: Series[float]
    variance: Series[float]
    total_sz: Series[float]
    edge_entropy: Series[float] = pa.Field(ge=0, le=np.log(2), nullable=True)
    truncation_dim: Series[int]
    system_size: Series[int] = pa.Field(gt=1)
    disorder: Series[float] = pa.Field(coerce=True)
    trial_id: Series[pd.Int64Dtype] = pa.Field(nullable=True, coerce=True)
    seed: Series[int]
    penalty: Series[float] = pa.Field(coerce=True)
    s_target: Series[int]
    offset: Series[float] = pa.Field(ge=0, le=1, coerce=True)

    @pa.check(Columns.total_sz)
    def close_to_integer(cls, series: Series[float]) -> Series[bool]:
        return Series(np.isclose(series, np.rint(series), atol=1e-12))
