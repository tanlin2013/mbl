# flake8: noqa
import io
from itertools import count

import streamlit as st
import pandas as pd
from mbl.name_space import Columns
from mbl.analysis.level_statistic import LevelStatistic, AverageOrder
from mbl.workflow.etl import ETL

from utils.data_loader import *
from utils.painter import *


# Title
st.markdown("# ⏳ Looking up energy window with tSDRG and spectral folding")
st.sidebar.markdown("# ⏳ Looking up energy window with tSDRG and spectral folding")

# Sidebar choices
n = st.sidebar.radio("System size n", (8, 10, 12, 14, 16, 18, 20))
h = st.sidebar.radio("Disorder strength h", (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0))
chi = st.sidebar.radio("Truncation dim chi", (2**3, 2**4, 2**5, 2**6))
relative_offset = st.sidebar.radio("Relative offset", (0.1, 0.5, 0.9))
total_sz = st.sidebar.radio("Restrict Total Sz in", (0, 1, None))
n_conf = st.sidebar.radio(
    "Number of disorder trials for drawling plots",
    (50, 100, 200, None),
)
st.sidebar.caption(
    "**Warning**: The larger, the slower. `None` means using all samples."
)

# Load the data
st.header("1. Data Table")
st.caption("**Note**: Please wait, it may take a while to load the data.")
df = load_data(n, h, chi=chi, total_sz=total_sz, relative_offset=relative_offset)
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
        delta=f"{rel_diff:.6f} %",
    )
with col2:
    st.metric(
        label="(Disorder-then-level) averaged < r >",
        value=f"{r2:.12f}",
        delta=f"{-1 * rel_diff:.6f} %",
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
    (Columns.en, Columns.energy_gap),
    (Columns.gap_ratio, Columns.en),
    (Columns.en, Columns.edge_entropy),
    (Columns.gap_ratio, Columns.edge_entropy),
    (Columns.gap_ratio, Columns.bipartite_entropy),
    (Columns.en, Columns.total_sz),
    (Columns.gap_ratio, Columns.total_sz),
]

if n_conf is not None:
    grouped = df.groupby([Columns.seed])
    mini_df = pd.concat([grouped.get_group(g) for g in list(grouped.groups)[:n_conf]])
else:
    mini_df = df.copy()

for k, v in zip(count(start=1), plot_pairs):
    if not (Columns.total_sz in v and total_sz is not None):
        try:
            with st.container():
                st.plotly_chart(
                    scatter_with_error_bar(mini_df, v[0], v[1], f"Fig. {k}:")
                )
        except ValueError:
            pass
        except KeyError:
            pass

st.markdown("#### 3.b. Scaling")

r_df = fetch_gap_ratio(algorithm=ETL.Metadata.tsdrg_table)
with st.expander("Click to view the table"):
    st.dataframe(r_df)
    buffer = io.StringIO()
    r_df.info(buf=buffer)
    st.text(buffer.getvalue())

order = st.radio(
    "Average order", (AverageOrder.LEVEL_FIRST.name, AverageOrder.DISORDER_FIRST.name)
)
st.caption(
    "**Hint**: apart from the average order, "
    "change also the parameters that are not part of the (x-y-z) axes on the sidebar."
)

scaling_plots = [
    {
        "query": [
            f"({Columns.system_size} == {n})",
            f"({Columns.total_sz} == {total_sz})",
            f"({Columns.avg_order} == '{order}')",
            f"({Columns.relative_offset} == {relative_offset})",
        ],
        "x": Columns.disorder,
        "y": Columns.gap_ratio,
        "z": Columns.truncation_dim,
    },
    {
        "query": [
            f"({Columns.disorder} == {h})",
            f"({Columns.total_sz} == {total_sz})",
            f"({Columns.avg_order} == '{order}')",
            f"({Columns.relative_offset} == {relative_offset})",
        ],
        "x": Columns.system_size,
        "y": Columns.gap_ratio,
        "z": Columns.truncation_dim,
    },
    {
        "query": [
            f"({Columns.truncation_dim} == {chi})",
            f"({Columns.total_sz} == {total_sz})",
            f"({Columns.avg_order} == '{order}')",
            f"({Columns.relative_offset} == {relative_offset})",
        ],
        "x": Columns.system_size,
        "y": Columns.gap_ratio,
        "z": Columns.disorder,
    },
]

for plot in scaling_plots:
    fig = px.line(
        r_df.query(" & ".join(plot["query"])).sort_values(by=[plot["z"], plot["x"]]),
        x=plot["x"],
        y=plot["y"],
        color=plot["z"],
        symbol=plot["z"],
    )
    st.plotly_chart(fig)
