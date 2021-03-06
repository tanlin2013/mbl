from typing import Tuple

import pandas as pd
import streamlit as st
import orjson  # noqa: F401 (See https://plotly.com/python/renderers/#performance)
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from mbl.name_space import Columns


@st.cache(persist=True)
def density_histogram_2d(df: pd.DataFrame, x: str, y: str, title: str):
    fig = ff.create_2d_density(df[x], df[y], title=title, point_size=3)
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    return fig


@st.cache(persist=True)
def scatter_with_error_bar(df: pd.DataFrame, x: str, y: str, title: str):
    error_x = "error" if x == Columns.en else None
    error_y = "error" if y == Columns.en else None
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=Columns.seed,
        error_x=error_x,
        error_y=error_y,
        marginal_x="histogram",
        marginal_y="histogram",
        title=title,
    )
    return fig


@st.cache(persist=True, allow_output_mutation=True)
def energy_bounds_scaling(
    df: pd.DataFrame,
    y: str,
    seeds: Tuple[int, int],
    title: str,
    xaxis_title: str,
    yaxis_title: str,
) -> go.Figure:
    fig = go.Figure()
    for seed in range(*seeds):
        res = df.query(f"{Columns.seed} == {seed}")
        fig.add_trace(
            go.Scatter(
                x=res[Columns.truncation_dim],
                y=res[y],
                name=f"{Columns.seed} = {seed}",
                mode="lines+markers",
                line={"dash": "dash"},
                marker={"size": 10},
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    return fig
