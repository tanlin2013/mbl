import orjson  # https://plotly.com/python/renderers/#performance
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
from itertools import count
from mbl.level_statistic import (
    Columns,
    LevelStatistic,
    AverageOrder
)

pio.renderers.default = 'iframe'
# run_on = 'prod' if os.getenv('USER') == 'appuser' else 'local'


@st.cache(persist=True)
def load_data(n: int, h: float, chi: int, total_sz: int, n_conf: int, filename: str = None):
    raw_df = pd.read_parquet(filename) if filename is not None else None
    agent = LevelStatistic(raw_df)
    df = agent.extract_gap(n, h, chi=chi, total_sz=total_sz).query(f'{Columns.trial_id} < {n_conf}')
    if chi is not None:
        df['error'] = np.sqrt(df[Columns.variance].to_numpy() / chi)
    return df


@st.cache(persist=True)
def density_histogram_2d(df: pd.DataFrame, x: str, y: str, title: str):
    fig = ff.create_2d_density(
        df[x], df[y],
        title=title,
        point_size=3
    )
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y
    )
    return fig


@st.cache(persist=True)
def scatter_with_error_bar(df: pd.DataFrame, x: str, y: str, title: str):
    error_x = "error" if x == Columns.en else None
    error_y = "error" if y == Columns.en else None
    fig = px.scatter(
        df,
        x=x,
        y=y,
        # color=Columns.seed,
        error_x=error_x,
        error_y=error_y,
        marginal_x='histogram',
        marginal_y='histogram',
        title=title
    )
    return fig


if __name__ == "__main__":

    table = st.sidebar.radio('Database table', ('ed', 'tsdrg'))
    options_n = (8, 10, 12) if table == 'ed' else (8, 10, 12, 14, 16, 18, 20)
    n = st.sidebar.radio('System size N', options_n)
    h = st.sidebar.radio('Disorder strength h', (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0))
    chi = st.sidebar.radio('Truncation dim', (2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6)) if table == 'tsdrg' else None
    total_sz = st.sidebar.radio("Restrict Total Sz in", (None, 0, 1))
    options_n_conf = (8, 10) if table == 'ed' else (10, 20, 30, 40, 50)
    n_conf = st.sidebar.radio(
        "Number of disorder trials "
        "(can't be too large, just because we can't draw too many dots in the plot)",
        options_n_conf
    )

    st.markdown('# Level statistics of Heisenberg model with random transversed field')
    st.write("Please wait, for every first click on the sidebar, it may take a while to load the data.")

    df = load_data(n, h, chi=chi, total_sz=total_sz, n_conf=n_conf)

    st.markdown('### 1. Gap ratio parameter (r-value)')
    r1 = LevelStatistic.averaged_gap_ratio(df, AverageOrder.LevelFirst)
    st.write(f'(Level-then-disorder) averaged `r = {r1}`')
    r2 = LevelStatistic.averaged_gap_ratio(df, AverageOrder.DisorderFirst)
    st.write(f'(Disorder-then-level) averaged `r = {r2}`')
    st.write(f'Relative difference is `{abs(r1 - r2)/max(r1, r2) * 100} %`')
    st.write('**Note**: Theoretical value is ~0.5307 for the delocalized phase and ~0.3863 for the localized phase.')

    st.markdown('### 2. Data Table')
    st.dataframe(df)

    plot_pairs = [
        (Columns.en, Columns.level_id),
        (Columns.gap_ratio, Columns.en),
        (Columns.en, Columns.edge_entropy),
        (Columns.gap_ratio, Columns.edge_entropy),
        (Columns.en, Columns.energy_gap),
        (Columns.en, Columns.total_sz),
        (Columns.gap_ratio, Columns.total_sz)
    ]

    st.markdown('### 3. Plots')
    for k, v in zip(count(start=1), plot_pairs):
        if not (Columns.total_sz in v and total_sz is not None):
            try:
                if table == 'ed':
                    st.write(
                        density_histogram_2d(df, v[0], v[1], f'Fig. {k}:')
                    )
                elif table == 'tsdrg':
                    st.write(
                        scatter_with_error_bar(df, v[0], v[1], f'Fig. {k}:')
                    )
            except KeyError:
                pass
