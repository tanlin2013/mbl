import io
from itertools import count

import streamlit as st
import pandas as pd
from mbl.name_space import Columns
from mbl.analysis.level_statistic import LevelStatistic, AverageOrder

from utils.data_loader import *
from utils.painter import *


# Title
st.markdown("# 💡 Results from Exact Diagonalization")
st.sidebar.markdown("# 💡 Exact Diagonalization")

# Sidebar choices
n = st.sidebar.radio("System size n", (8, 10, 12))
h = st.sidebar.radio("Disorder strength h", (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0))
total_sz = st.sidebar.radio("Restrict Total Sz in", (None, 0, 1))
n_conf = st.sidebar.radio(
    "Number of disorder trials for drawing plots",
    (8, 10, 20, 30, None),
)
st.sidebar.caption(
    "**Warning**: The larger, the slower. `None` means using all samples."
)

# Load the data
st.header("1. Data Table")

st.caption("**Note**: Please wait, it may take a while to load the data.")
df = load_data(n, h, total_sz=total_sz)
with st.expander("Click to view the table"):
    st.dataframe(df)
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

# Level statistics
st.header("2. Gap ratio parameter (r-value)")

st.markdown(
    "The order of taking average is somewhat important. "
    "Here we consider both, with their **relative difference** labeled below."
)
st.markdown("#")
r1 = LevelStatistic.averaged_gap_ratio(df, AverageOrder.LEVEL_FIRST)
r2 = LevelStatistic.averaged_gap_ratio(df, AverageOrder.DISORDER_FIRST)
rel_diff = (r1 - r2) / max(r1, r2) * 100
col1, col2 = st.columns(2)
with col1:
    st.metric(
        label="(Level-then-disorder) averaged < r >",
        value=f"{r1:.12f}",
        delta=f"{rel_diff:.6f} %"
    )
with col2:
    st.metric(
        label="(Disorder-then-level) averaged < r >",
        value=f"{r2:.12f}",
        delta=f"{-1 * rel_diff:.6f} %"
    )
st.markdown("#")
st.caption("**Note**: Theoretical values are")
st.caption("* Ergodic phase: < r > ~ `0.53589(8)`.")
st.caption("* Localized phase: < r > ~ `0.38629(4)`.")

# Data visualization
st.header("3. Visualization")

st.caption("**Hint**: Change parameters on the sidebar.")

if n_conf is not None:
    grouped = df.groupby([Columns.seed])
    mini_df = pd.concat([grouped.get_group(g) for g in list(grouped.groups)[:n_conf]])
else:
    mini_df = df.copy()

plot_pairs = [
    (Columns.en, Columns.level_id),
    (Columns.gap_ratio, Columns.en),
    (Columns.en, Columns.edge_entropy),
    (Columns.gap_ratio, Columns.edge_entropy),
    (Columns.gap_ratio, Columns.bipartite_entropy),
    (Columns.en, Columns.energy_gap),
    (Columns.en, Columns.total_sz),
    (Columns.gap_ratio, Columns.total_sz),
]

for k, v in zip(count(start=1), plot_pairs):
    if not (Columns.total_sz in v and total_sz is not None):
        try:
            with st.container():
                st.plotly_chart(density_histogram_2d(mini_df, v[0], v[1], f"Fig. {k}:"))
        except ValueError:
            pass
        except KeyError:
            pass

# Plot scaling relations
st.subheader("Scaling")

if st.button("Click to start this time-consuming part"):
    ns = [8, 10, 12]
    hs = [0.5, 3.0, 4.0, 6.0, 10.0]

    fig = go.Figure()
    for h in hs:
        r = [fetch_gap_ratio(n, h, total_sz=total_sz) for n in ns]
        fig.add_trace(
            go.Scatter(
                x=ns,
                y=r,
                name=f"h = {h}",
                mode="lines+markers",
                line={"dash": "dash"},
                marker={"size": 10},
            )
        )
    fig.update_layout(
        title="Fig. Finite size scaling of averaged gap ratio",
        xaxis_title="System size n",
        yaxis_title="Averaged gap ratio r",
    )
    st.write(fig)

    fig = go.Figure()
    for n in ns:
        r = [fetch_gap_ratio(n, h, total_sz=total_sz) for h in hs]
        fig.add_trace(
            go.Scatter(
                x=hs,
                y=r,
                name=f"n = {n}",
                mode="lines+markers",
                line={"dash": "dash"},
                marker={"size": 10},
            )
        )
    fig.update_layout(
        title="Fig. Finite size scaling of averaged gap ratio",
        xaxis_title="Disorder strength h",
        yaxis_title="Averaged gap ratio r",
    )
    st.write(fig)
