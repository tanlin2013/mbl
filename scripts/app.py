import orjson  # https://plotly.com/python/renderers/#performance
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
import plotly.graph_objects as go
from itertools import count
from mbl.name_space import Columns
from mbl.level_statistic import (
    LevelStatistic,
    AverageOrder
)

pio.renderers.default = 'iframe'
# run_on = 'prod' if os.getenv('USER') == 'appuser' else 'local'


@st.cache(persist=True)
def load_data(n: int, h: float, chi: int, total_sz: int, filename: str = None):
    raw_df = pd.read_parquet(filename) if filename is not None else None
    agent = LevelStatistic(raw_df)
    df = agent.extract_gap(n, h, chi=chi, total_sz=total_sz)
    if chi is not None:
        df['error'] = np.sqrt(df[Columns.variance].to_numpy() / chi)
    return df


@st.cache(persist=True)
def fetch_gap_ratio(n: int, h: float, chi: int = None, total_sz: int = None) -> float:
    return LevelStatistic().averaged_gap_ratio(
        load_data(**locals())
    )


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
        error_x=error_x,
        error_y=error_y,
        marginal_x='histogram',
        marginal_y='histogram',
        title=title
    )
    return fig


if __name__ == "__main__":

    st.set_page_config(layout="wide")

    table = st.sidebar.radio("Database table", ('ed', 'tsdrg'))
    options_n = (8, 10, 12) if table == 'ed' else (8, 10, 12, 14, 16, 18, 20)
    n = st.sidebar.radio("System size n", options_n)
    options_h = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0)
    h = st.sidebar.radio("Disorder strength h", options_h)
    options_chi = (2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6)
    chi = st.sidebar.radio("Truncation dim chi", options_chi) if table == 'tsdrg' else None
    total_sz = st.sidebar.radio("Restrict Total Sz in", (None, 0, 1))
    options_n_conf = (8, 10, 20, 30, 40, 50, None)
    n_conf = st.sidebar.radio(
        "Number of disorder trials for making plots (The larger, the slower. None means using all samples)",
        options_n_conf
    )

    st.markdown('# Level statistics of Heisenberg model with random transversed field')
    st.write('Please wait, it may take a while to load the data.')

    df = load_data(n, h, chi=chi, total_sz=total_sz)

    st.markdown('### 1. Data Table')
    st.dataframe(df)

    st.markdown('### 2. Gap ratio parameter (r-value)')
    r1 = LevelStatistic.averaged_gap_ratio(df, AverageOrder.LevelFirst)
    st.write(f'(Level-then-disorder) averaged `r = {r1}`')
    r2 = LevelStatistic.averaged_gap_ratio(df, AverageOrder.DisorderFirst)
    st.write(f'(Disorder-then-level) averaged `r = {r2}`')
    st.write(f'Relative difference is `{abs(r1 - r2)/max(r1, r2) * 100} %`')
    st.write('**Note**: Theoretical value is')
    st.write('* Ergodic phase: ~`0.5307`.')
    st.write('* Localized phase: ~`0.3863`.')

    st.markdown('### 3. Visualization')
    st.markdown('#### 3.a. Interactive')
    st.write('**Hint**: Change parameters on the sidebar.')
    plot_pairs = [
        (Columns.en, Columns.level_id),
        (Columns.gap_ratio, Columns.en),
        (Columns.en, Columns.edge_entropy),
        (Columns.gap_ratio, Columns.edge_entropy),
        (Columns.gap_ratio, Columns.bipartite_entropy),
        (Columns.en, Columns.energy_gap),
        (Columns.en, Columns.total_sz),
        (Columns.gap_ratio, Columns.total_sz)
    ]

    if n_conf is not None:
        grouped = df.groupby([Columns.seed])
        mini_df = pd.concat([grouped.get_group(g) for g in list(grouped.groups)[:n_conf]])
    else:
        mini_df = df.copy()

    for k, v in zip(count(start=1), plot_pairs):
        if not (Columns.total_sz in v and total_sz is not None):
            try:
                if table == 'ed':
                    st.write(
                        density_histogram_2d(mini_df, v[0], v[1], f'Fig. {k}:')
                    )
                elif table == 'tsdrg':
                    st.write(
                        scatter_with_error_bar(mini_df, v[0], v[1], f'Fig. {k}:')
                    )
            except ValueError:
                pass
            except KeyError:
                pass

    st.markdown('#### 3.b. Scaling')

    hs = [0.5, 3.0, 4.0, 6.0, 10.0]
    fig = go.Figure()
    for h in hs:
        r = [fetch_gap_ratio(n, h, chi=chi, total_sz=total_sz) for n in options_n]
        fig.add_trace(go.Scatter(
            x=options_n, y=r,
            name=f'h = {h}', mode='lines+markers', line={'dash': 'dash'}, marker={'size': 10}
        ))
    fig.update_layout(title='Fig. Finite size scaling of averaged gap ratio',
                      xaxis_title='System size n',
                      yaxis_title='Averaged gap ratio r')
    st.write(fig)

    fig = go.Figure()
    for n in options_n:
        r = [fetch_gap_ratio(n, h, chi=chi, total_sz=total_sz) for h in hs]
        fig.add_trace(go.Scatter(
            x=hs, y=r,
            name=f'n = {n}', mode='lines+markers', line={'dash': 'dash'}, marker={'size': 10}
        ))
    fig.update_layout(title='Fig. Finite size scaling of averaged gap ratio',
                      xaxis_title='Disorder strength h',
                      yaxis_title='Averaged gap ratio r')
    st.write(fig)

    if table == 'tsdrg':
        fig = go.Figure()
        for chi in options_chi:
            r = [fetch_gap_ratio(n, h, chi=chi, total_sz=total_sz) for n in options_n]
            fig.add_trace(go.Scatter(
                x=options_n, y=r,
                name=f'chi = {chi}', mode='lines+markers', line={'dash': 'dash'}, marker={'size': 10}
            ))
        fig.update_layout(title='Fig. Finite size scaling of averaged gap ratio',
                          xaxis_title='System size n',
                          yaxis_title='Averaged gap ratio r')
        st.write(fig)

        fig = go.Figure()
        for chi in options_chi:
            r = [fetch_gap_ratio(n, h, chi=chi, total_sz=total_sz) for h in hs]
            fig.add_trace(go.Scatter(
                x=hs, y=r,
                name=f'chi = {chi}', mode='lines+markers', line={'dash': 'dash'}, marker={'size': 10}
            ))
        fig.update_layout(title='Fig. Finite size scaling of averaged gap ratio',
                          xaxis_title='Disorder strength h',
                          yaxis_title='Averaged gap ratio r')
        st.write(fig)
