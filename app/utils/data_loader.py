from typing import Dict

import numpy as np
import pandas as pd
import awswrangler as wr
import ray
import streamlit as st
import modin.pandas as mpd

from mbl.name_space import Columns
from mbl.distributed import Distributed
from mbl.analysis.level_statistic import LevelStatistic
from mbl.analysis.energy_bounds import EnergyBounds


@st.cache(persist=True, allow_output_mutation=True)
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
        df["error"] = np.sqrt(df[Columns.variance].to_numpy())
    if isinstance(df, mpd.DataFrame):
        return df._to_pandas()
    return df


@st.cache(persist=True)
def fetch_gap_ratio(params: Dict) -> pd.DataFrame:
    @ray.remote
    def wrapper(kwargs):
        return LevelStatistic.fetch_gap_ratio(**kwargs)

    rs = Distributed.map_on_ray(wrapper, params)
    for k, param in enumerate(params):
        param[Columns.gap_ratio] = rs[k]
    return pd.DataFrame(params)


@st.cache(persist=True, allow_output_mutation=True)
def fetch_energy_bounds(method: str, overall_const: float):
    groups = [
        Columns.system_size,
        Columns.disorder,
        Columns.seed,
        Columns.overall_const,
        Columns.truncation_dim,
    ]
    assert method in ["min", "max"]
    table = {-1: "tsdrg_min", 1: "tsdrg"}[int(overall_const)]
    if method == "max" and overall_const == -1:
        return None
    return wr.athena.read_sql_query(
        f"""
        SELECT *
        FROM {table}
        WHERE {Columns.en} IN (
            SELECT {method.upper()}({Columns.en})
            FROM {table}
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
