import io

import numpy as np
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
overall_const = st.sidebar.radio("Overall const", (-1, 1))
num_seeds = st.sidebar.radio("Number of seeds to draw", (6, 8, 10, 20, 30, 40, 50))
st.sidebar.caption("**Note**: seeds will be randomly picked up")

# Load the data
st.header("1. Data Table")

st.caption("**Note**: Please wait, it may take a while to load the data.")
df = fetch_energy_bounds()
with st.expander("Click to view the table"):
    st.dataframe(df)
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

# Data visualization
st.header("2. Visualization")

chis = [2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6]
query = [
    f"{Columns.system_size} == {n}",
    f"{Columns.disorder} == {h}",
    f"{Columns.overall_const} == {overall_const}",
]
res = df.query(" & ".join(query))
res["en"] = res["min_en"] * res[Columns.overall_const]
fig = go.Figure()
for seed in np.random.choice(np.arange(2000, 2500), num_seeds):
    fig.add_trace(
        go.Scatter(
            x=chis,
            y=res.query(f"{Columns.seed} == {seed}")["en"],
            name=f"{Columns.seed} = {seed}",
            mode="lines+markers",
            line={"dash": "dash"},
            marker={"size": 10},
        )
    )
fig.update_layout(
    title="Fig. Convergence of energy up to finite bond dimension",
    xaxis_title="Bond dim chi",
    yaxis_title="Energy",
)
st.plotly_chart(fig)
