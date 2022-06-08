import orjson
import numpy as np
import pandas as pd
import awswrangler as wr
import streamlit as st

from mbl.name_space import Columns
from mbl.analysis.level_statistic import LevelStatistic
from mbl.analysis.energy_bounds import EnergyBounds


@st.cache(persist=True)
def load_data(
    n: int,
    h: float,
    chi: int = None,
    relative_offset: float = None,
    total_sz: int = None,
) -> pd.DataFrame:
    df = LevelStatistic().extract_gap(
        n, h, chi=chi, relative_offset=relative_offset, total_sz=total_sz
    )
    if chi is not None:
        df["error"] = np.sqrt(df[Columns.variance].to_numpy() / chi)
    return df


@st.cache(persist=True)
def fetch_gap_ratio(n: int, h: float, chi: int = None, total_sz: int = None) -> float:
    return LevelStatistic().averaged_gap_ratio(load_data(**locals()))


@st.cache(persist=True)
def fetch_energy_bounds():
    items = [
        Columns.system_size,
        Columns.disorder,
        Columns.truncation_dim,
        Columns.seed,
        f"MIN({Columns.en}) AS min_en",
        Columns.overall_const,
        Columns.total_sz,
    ]
    groups = [
        Columns.system_size,
        Columns.disorder,
        Columns.truncation_dim,
        Columns.seed,
        Columns.overall_const,
    ]
    return wr.athena.read_sql_query(
        f"""
        SELECT {', '.join(items)}
        FROM {EnergyBounds.Metadata.table}
        GROUP BY {', '.join(groups)}
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
