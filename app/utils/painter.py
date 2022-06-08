import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from mbl.name_space import Columns
from .data_loader import fetch_gap_ratio


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
        error_x=error_x,
        error_y=error_y,
        marginal_x="histogram",
        marginal_y="histogram",
        title=title,
    )
    return fig


# @st.cache(persist=True)
# def scaling_line(xs, zs, x_label: str, z_label: str, title: str, xaxis_title: str):
#     fig = go.Figure()
#     for z in zs:
#         r = [fetch_gap_ratio(n, h, chi=chi, total_sz=total_sz) for x in xs]
#         fig.add_trace(
#             go.Scatter(
#                 x=xs,
#                 y=r,
#                 name=f"{z_label} = {z}",
#                 mode="lines+markers",
#                 line={"dash": "dash"},
#                 marker={"size": 10},
#             )
#         )
#     fig.update_layout(
#         title=title, xaxis_title=xaxis_title, yaxis_title="Averaged gap ratio r"
#     )
#     return fig


# @st.cache(persist=True)
# def energy_bounds_scaling(chis):
#     fig = go.Figure()
#     for chi in chis:
#         go.Scatter(
#
#         )
#     return fig
