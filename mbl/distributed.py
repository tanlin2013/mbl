import ray
from tqdm import tqdm
from itertools import islice
from ray.remote_function import RemoteFunction
from dask.distributed import Client, progress
from typing import Callable, Sequence, List


class Distributed:

    @staticmethod
    def map_on_ray(func: Callable, params: Sequence,
                   resource_aware_func: Callable = None, chunk_size: int = 32) -> List:
        """

        Args:
            func:
            params:
            resource_aware_func:
            chunk_size:

        Returns:

        """
        def chunk(lst):
            lst = iter(lst)
            return iter(lambda: tuple(islice(lst, chunk_size)), ())

        def assignee(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])

        if not ray.is_initialized:
            ray.init()
        func = ray.remote(func) if not isinstance(func, RemoteFunction) else func
        jobs = [func.remote(i) for i in params] if resource_aware_func is None \
            else [func.options(resource_aware_func(**i)).remote(i) for i in params]
        results = []
        for chunked_job in tqdm(chunk(jobs), desc='chunk', total=chunk_size):
            for _ in tqdm(assignee(list(chunked_job)), desc='subtask', position=1, total=len(chunked_job)):
                pass
            results += ray.get(chunked_job)
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
