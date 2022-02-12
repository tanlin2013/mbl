import ray
import psutil
import numpy as np
from tqdm import tqdm
from itertools import islice
from ray.remote_function import RemoteFunction
from dask.distributed import Client, progress
from typing import Callable, Sequence, List


class Distributed:

    @staticmethod
    def get_workload():
        cpu_load_1, _, _ = psutil.getloadavg()
        mem = psutil.virtual_memory()
        return cpu_load_1 / psutil.cpu_count() / (mem.available * 1e-9)

    @staticmethod
    def map_on_ray(func: Callable, params: Sequence, chunk_size: int = 7) -> List:
        """

        Args:
            func:
            params:
            chunk_size:

        Returns:

        """
        def chunk(inputs):
            inputs = iter(inputs)
            return iter(lambda: list(islice(inputs, chunk_size)), [])

        def watch(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])

        if not ray.is_initialized:
            ray.init()
        func = ray.remote(func) if not isinstance(func, RemoteFunction) else func
        results = []
        for chunk_params in tqdm(chunk(params), desc='All chunks', total=int(np.ceil(len(params) / chunk_size))):
            jobs = [func.remote(i) for i in chunk_params]
            for done_job in tqdm(watch(jobs), desc='Current chunk', position=1, total=len(jobs)):
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
