# flake8: noqa
import io

import streamlit as st

from mbl.name_space import Columns
from utils.data_loader import *
from utils.painter import *


# Title
st.markdown("# 📷 tSDRG estimation for energy bounds")
st.sidebar.markdown("# 📷 tSDRG estimation for energy bounds")
st.markdown(
    "The purpose of this part is to find the energy bounds of spectrum, "
    "i.e. the highest excited energy and the ground state energy, "
    "so that we can shift into the interior part of spectrum accordingly. "
    "Later, we will take the spectral transformation"
)
st.latex(r"H \rightarrow (H - \sigma)^2")
st.markdown(
    r"To determine the proper \sigma in relative scale,"
    r" we therefore look for the energy bounds first."
)

# Sidebar choices
n = st.sidebar.radio("System size n", (8, 10, 12, 14, 16, 18, 20))
h = st.sidebar.radio("Disorder strength h", (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0))
overall_const = st.sidebar.radio("Overall const", (1, -1))
method = st.sidebar.radio("Method", ("min", "max"))
seeds = st.sidebar.slider("The range of seeds to draw", 2000, 2200, (2000, 2020))

# Load the data
st.header("1. Data Table")

st.caption("**Note**: Please wait, it may take a while to load the data.")
df = fetch_energy_bounds(method)
with st.expander("Click to view the table"):
    st.dataframe(df)
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

# Data visualization
st.header("2. Visualization")

query = [
    f"{Columns.system_size} == {n}",
    f"{Columns.disorder} == {h}",
    f"{Columns.overall_const} == {overall_const}",
]
mini_df = df.query(" & ".join(query))
mini_df["en"] = mini_df["en"] * mini_df[Columns.overall_const]
st.plotly_chart(
    energy_bounds_scaling(
        mini_df,
        y=Columns.en,
        seeds=seeds,
        title="Fig. Convergence of energy up to finite bond dimensions",
        xaxis_title="Bond dim chi",
        yaxis_title="Energy",
    )
)

st.plotly_chart(
    energy_bounds_scaling(
        mini_df,
        y=Columns.total_sz,
        seeds=seeds,
        title="Fig. Total Sz sector tSDRG converged to on different bond dimensions",
        xaxis_title="Bond dim chi",
        yaxis_title="Total Sz",
    )
)

# mini_df[Columns.variance][mini_df[Columns.variance] < 0] = 1e-24
fig = energy_bounds_scaling(
    mini_df,
    y=Columns.variance,
    seeds=seeds,
    title="Fig. Energy variance up to finite bond dimensions",
    xaxis_title="Bond dim chi",
    yaxis_title="Variance",
)
fig.update_yaxes(type="log", range=[-8, 0])
st.plotly_chart(fig)
st.caption(
    "**Note**: Nearly perpendicular line may appear due to the presence of "
    "small or even negative variance in log scale."
)
