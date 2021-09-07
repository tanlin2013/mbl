import orjson
import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
import plotly.io as pio
from itertools import count
from pathlib import Path
from mbl.level_statistic import (
    LevelStatistic,
    AverageOrder
)


pio.renderers.default = 'iframe'


@st.cache(persist=True)
def load_data():
    raw_df = pd.read_parquet('s3://many-body-localization-random-heisenberg/config/random_heisenberg_config.parquet')
    # raw_df = pd.read_parquet(f'{Path(__file__).parents[1]}/data/random_heisenberg_config.parquet')
    return LevelStatistic(raw_df)


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


if __name__ == "__main__":

    agent = load_data()

    N = st.sidebar.radio('System size N', (8, 10))
    h = st.sidebar.radio('Disorder strength h', (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0))
    total_sz = st.sidebar.radio("Restrict TotalSz in", (None, 0, 1))
    n_conf = st.sidebar.radio(
        "Number of disorder trials "
        "(can't be too large due to limited memory for making plots)",
        (4, 6, 8, 10)
    )
    df = agent.extract_gap(N, h, total_sz=total_sz).query(f'TrialID < {n_conf}')

    st.markdown('# Level statistics of Heisenberg model with random transversed field')
    st.markdown('##')
    r1 = agent.averaged_gap_ratio(df, AverageOrder.LevelFirst)
    st.write(f'The (level-then-disorder) averaged gap-ratio parameter r = {r1}')
    r2 = agent.averaged_gap_ratio(df, AverageOrder.DisorderFirst)
    st.write(f'The (disorder-then-level) averaged gap-ratio parameter r = {r2}')
    st.write(f'The relative difference is {abs(r1 - r2)/max(r1, r2) * 100} %')

    st.markdown('#')
    st.dataframe(df)

    plot_pairs = [
        ('En', 'LevelID'),
        ('GapRatio', 'En'),
        ('En', 'EntanglementEntropy'),
        ('GapRatio', 'EntanglementEntropy'),
        ('En', 'EnergyGap'),
        ('En', 'TotalSz'),
        ('GapRatio', 'TotalSz')
    ]

    st.markdown('#')
    for k, v in zip(count(start=1), plot_pairs):
        if not ('TotalSz' in v and total_sz is not None):
            st.write(
                density_histogram_2d(df, v[0], v[1], f'Fig. {k}: {v[0]}(xaxis) - {v[1]}(yaxis)')
            )
