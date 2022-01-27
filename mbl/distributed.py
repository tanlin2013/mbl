import ray
from tqdm import tqdm
from dask.distributed import Client, progress
from typing import Callable, Sequence, List


class Distribute:

    @staticmethod
    def map_on_ray(func: Callable, params: Sequence) -> List:
        """

        Args:
            func:
            params:

        Returns:

        """
        def assignee(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])

        @ray.remote
        def wrapped_func(*args, **kwargs):
            return func(*args, **kwargs)

        ray.init()
        try:
            jobs = [func.remote(i) for i in params]
        except AttributeError:
            jobs = [wrapped_func.remote(i) for i in params]
        for _ in tqdm(assignee(jobs), total=len(params)):
            pass
        results = ray.get(jobs)
        ray.shutdown()
        return results

    @staticmethod
    def map_on_dask(func: Callable, params: Sequence, cluster=None) -> List:
        """

        Args:
            func:
            params:
            cluster:

        Returns:

        """
        client = Client() if cluster is None else Client(cluster)
        futures = client.map(func, params)
        progress(futures)
        return client.gather(futures)
