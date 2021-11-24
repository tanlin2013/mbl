import ray
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from mbl import model
from mbl.distributed_computing import distribute


flag = 'spectral_folded_random_heisenberg'
camel_case_flag = ''.join(word.title() for word in flag.split('_'))


@ray.remote
def main(N: int, h: float, penalty: float, s_target: int, trial_id: int) -> pd.DataFrame:
    return getattr(model, f'{camel_case_flag}ED')(N, h, penalty, s_target, trial_id).df


if __name__ == "__main__":

    penalty = 0.0
    s_target = 0
    n_conf = 100

    params = [
        (N, h, penalty, s_target, trial_id)
        for N in [8, 10]
        for h in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0]
        for trial_id in range(n_conf)
    ]

    ray.init()
    jobs = [main.remote(*i) for i in params]
    for _ in tqdm(distribute(jobs), total=len(params)):
        pass

    merged_df = pd.concat(ray.get(jobs))
    ray.shutdown()

    merged_df.to_parquet(f'{Path(__file__).parents[1]}/data/{flag}_config.parquet', index=False)
