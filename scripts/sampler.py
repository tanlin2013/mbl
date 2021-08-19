import pandas as pd
import ray
from tqdm import tqdm
from mbl.model import RandomHeisenberg
from mbl.distributed_computing import distribute


@ray.remote
def main(N: int, h: float, penalty: float, s_target: int, trial_id: int) -> pd.DataFrame:
    return RandomHeisenberg(N, h, penalty, s_target, trial_id).df


if __name__ == "__main__":

    N = 10
    penalty = 0.0
    s_target = 0
    n_conf = 500

    params = [
        (N, h, penalty, s_target, trial_id)
        for h in [1.0, 4.0]
        for trial_id in range(n_conf)
    ]

    ray.init()
    jobs = [main.remote(*i) for i in params]
    for _ in tqdm(distribute(jobs), total=len(params)):
        pass

    merged_df = pd.concat(ray.get(jobs))
    ray.shutdown()

    merged_df.to_csv('/Users/tandaolin/Desktop/random_heisenberg_config.csv', index=False)
