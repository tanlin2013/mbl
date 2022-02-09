import ray
from tqdm import tqdm
from ray.remote_function import RemoteFunction
from dask.distributed import Client, progress
from typing import Callable, Sequence, List


class Distributed:

    @staticmethod
    def map_on_ray(func: Callable, params: Sequence, mem_aware_func: Callable = None) -> List:
        """

        Args:
            func:
            params:
            mem_aware_func:

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
        if isinstance(func, RemoteFunction):
            jobs = [func.remote(i) for i in params] if mem_aware_func is None \
                else [func.options(memory=mem_aware_func(**i)).remote(i) for i in params]
        else:
            jobs = [wrapped_func.remote(i) for i in params] if mem_aware_func is None \
                else [wrapped_func.options(memory=mem_aware_func(**i)).remote(i) for i in params]
        for _ in tqdm(assignee(jobs), total=len(params)):
            pass
        results = ray.get(jobs)
        ray.shutdown()
        return results

    @staticmethod
    def map_on_dask(func: Callable, params: Sequence, cluster=None, **kwargs) -> List:
        """

        Args:
            func:
            params:
            cluster:

        Returns:

        """
        client = Client() if cluster is None else Client(cluster)
        futures = client.map(func, params, **kwargs)
        progress(futures)
        return client.gather(futures)
