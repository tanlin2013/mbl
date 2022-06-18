import orjson  # noqa: F401
import numpy as np
import pandas as pd
import awswrangler as wr
import streamlit as st

from mbl.name_space import Columns
from mbl.analysis.level_statistic import LevelStatistic, AverageOrder
from mbl.analysis.energy_bounds import EnergyBounds


@st.cache(persist=True)
def load_data(
    n: int,
    h: float,
    chi: int = None,
    relative_offset: float = None,
    total_sz: int = None,
) -> pd.DataFrame:
    df = LevelStatistic.athena_query(
        n, h, chi=chi, relative_offset=relative_offset, total_sz=total_sz
    )
    df = LevelStatistic.extract_gap(df)
    if chi is not None:
        df["error"] = np.sqrt(df[Columns.variance].to_numpy() / chi)
    return df._to_pandas()


@st.cache(persist=True)
def fetch_gap_ratio(
    n: int,
    h: float,
    chi: int = None,
    total_sz: int = None,
    order: AverageOrder = AverageOrder.LEVEL_FIRST,
) -> float:
    return LevelStatistic.fetch_gap_ratio(**locals())


@st.cache(persist=True)
def fetch_energy_bounds(method: str):
    groups = [
        Columns.system_size,
        Columns.disorder,
        Columns.seed,
        Columns.overall_const,
        Columns.truncation_dim,
    ]
    assert method in ["min", "max"]
    return wr.athena.read_sql_query(
        f"""
        SELECT *
        FROM {EnergyBounds.Metadata.table}
        WHERE {Columns.en} IN (
            SELECT {method.upper()}({Columns.en})
            FROM {EnergyBounds.Metadata.table}
            GROUP BY {', '.join(groups)}
        )
        """,
        database=EnergyBounds.Metadata.database,
    ).sort_values(
        by=[
            Columns.system_size,
            Columns.disorder,
            Columns.seed,
            Columns.overall_const,
            Columns.truncation_dim,
        ]
    )
