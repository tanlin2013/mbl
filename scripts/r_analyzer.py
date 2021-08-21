import pandas as pd
from mbl.level_statistic import (
    LevelStatistic,
    AverageOrder
)


if __name__ == "__main__":

    N = 10
    h = 1.0
    total_sz = 0

    raw_df = pd.read_csv('/Users/tandaolin/Desktop/random_heisenberg_config.csv')
    agent = LevelStatistic(raw_df)
    df = agent.extract_gap(N, h, total_sz=total_sz)
    print(df.head(10))

    r = agent.averaged_gap_ratio(df, AverageOrder.LevelFirst)
    print(r)
    r2 = agent.averaged_gap_ratio(df, AverageOrder.DisorderFirst)
    print(r2)
