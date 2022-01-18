import ray
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from mbl import model
from mbl.distributed_computing import distribute


flag = 'random_heisenberg'
camel_case_flag = ''.join(word.title() for word in flag.split('_'))


@ray.remote
def main(N: int, h: float, penalty: float, s_target: int, trial_id: int) -> pd.DataFrame:
    return getattr(model, f'{camel_case_flag}ED')(N, h, penalty, s_target, trial_id).df


@ray.remote(memory=10 * 1024 ** 3)
def main2(N: int, h: float, chi: int, penalty: float, s_target: int, trial_id: int, seed: int) -> pd.DataFrame:
    return model.RandomHeisenbergTSDRG(N, h, chi, penalty, s_target, trial_id, seed).df


if __name__ == "__main__":

    penalty = 0.0
    s_target = 0
    n_conf = 100

    params = [
        (N, h, chi, penalty, s_target, trial_id, seed)
        for N in [8, 10]
        for h in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0]
        for chi in [2**4, 2**6, 2**8, 2**10]
        for trial_id, seed in enumerate(range(1900, 1900+n_conf))
    ]

    ray.init()
    jobs = [main2.remote(*i) for i in params]
    for _ in tqdm(distribute(jobs), total=len(params)):
        pass

    merged_df = pd.concat(ray.get(jobs))
    ray.shutdown()

    merged_df.to_parquet(f'{Path(__file__).parents[1]}/data/{flag}_tsdrg_config.parquet', index=False)
