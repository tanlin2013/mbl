import ray
import numpy as np
from tqdm import tqdm, trange
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
        def chunk(obj_ids):
            obj_ids = iter(obj_ids)
            return iter(lambda: list(islice(obj_ids, chunk_size)), [])

        def assignee(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])

        if not ray.is_initialized:
            ray.init()
        func = ray.remote(func) if not isinstance(func, RemoteFunction) else func
        results = []
        for chunk_params in tqdm(chunk(params), desc='All chunks', total=int(np.ceil(len(params) / chunk_size))):
            jobs = [func.remote(i) for i in chunk_params] if resource_aware_func is None \
                else [func.options(resource_aware_func(**i)).remote(i) for i in chunk_params]
            for done_job in tqdm(assignee(jobs), desc='Each chunk', position=1, total=len(jobs)):
                results += [done_job]
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
