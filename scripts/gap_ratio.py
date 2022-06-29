from itertools import product

import ray
import pandas as pd

from mbl.workflow.etl import ETL


if __name__ == "__main__":

    params = [
        {
            "n": n,
            "h": h,
            "chi": chi,
            "total_sz": total_sz,
            "relative_offset": relative_offset,
        }
        for n, chi, h, total_sz, relative_offset in product(
            [8, 10, 12, 14, 16, 18, 20],
            [2**3, 2**4, 2**5, 2**6],
            [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0],
            [None, 0, 1],
            [0.1, 0.5, 0.9],
        )
    ]

    ray.init(num_cpus=34)
    df = ETL.create_gap_ratio_table(params)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    print(df.head(5))
    print(df.info())
