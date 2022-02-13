import ray
from tqdm import tqdm
from ray.remote_function import RemoteFunction
from dask.distributed import Client, progress
from typing import Callable, Sequence, List


class Distributed:

    @staticmethod
    def map_on_ray(func: Callable, params: Sequence) -> List:
        """

        Args:
            func:
            params:

        Returns:

        """
        def watch(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])

        if not ray.is_initialized:
            ray.init()
        func = ray.remote(func) if not isinstance(func, RemoteFunction) else func
        jobs = [func.remote(i) for i in params]
        results = []
        for done_job in tqdm(watch(jobs), desc='Completed jobs', total=len(jobs)):
            results.append(done_job)
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
