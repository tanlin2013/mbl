import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
from pathlib import Path
from mbl.level_statistic import (
    LevelStatistic,
    AverageOrder
)


@st.cache(persist=True)
def load_data():
    raw_df = pd.read_csv(f'{Path(__file__).parent}/random_heisenberg_config.csv')
    return LevelStatistic(raw_df)


if __name__ == "__main__":

    agent = load_data()

    N = 10
    h = st.sidebar.radio('Disorder strength h', (1.0, 4.0))
    total_sz = st.sidebar.radio("Restrict TotalSz in", (None, 0, 1))
    n_conf = st.sidebar.radio("Number of disorder trials "
                              "(can't be too large due to limited memory for making plots)", (2, 4, 6, 8))
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

    st.markdown('#')
    fig = ff.create_2d_density(
        df['En'], df['LevelID'],
        title='Fig 1. En(x-axis) - Level(y-axis)',
        point_size=3
    )
    st.write(fig)

    fig = ff.create_2d_density(
        df['GapRatio'], df['En'],
        title='Fig 2. GapRatio(x-axis) - En(y-axis)',
        point_size=3
    )
    st.write(fig)

    fig = ff.create_2d_density(
        df['En'], df['TotalSz'],
        title='Fig 3. En(x-axis) - TotalSz(y-axis)',
        point_size=3
    )
    st.write(fig)

    fig = ff.create_2d_density(
        df['GapRatio'], df['TotalSz'],
        title='Fig 4. GapRatio(x-axis) - TotalSz(y-axis)',
        point_size=3
    )
    st.write(fig)
