from typing import Callable, Sequence, List

import ray
from tqdm import tqdm
from ray.remote_function import RemoteFunction

# from dask.distributed import Client, progress
# from dask_jobqueue import SLURMCluster


class Distributed:
    @staticmethod
    def map_on_ray(func: Callable, params: Sequence) -> List:
        """

        Args:
            func:
            params:

        Returns:

        Warnings:
            The results are not order-preserving as the order in input `params`.
        """

        def watch(obj_ids: List[ray.ObjectRef]):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])

        if not ray.is_initialized:
            ray.init()
        func = ray.remote(func) if not isinstance(func, RemoteFunction) else func
        jobs = [func.remote(i) for i in params]
        results = []
        for done_job in tqdm(watch(jobs), desc="Completed jobs", total=len(jobs)):
            results.append(done_job)
        ray.shutdown()
        return results


#     @staticmethod
#     def map_on_dask(func: Callable, params: Sequence, cluster=None) -> List:
#         """
#
#         Args:
#             func:
#             params:
#             cluster:
#
#         Returns:
#
#         """
#         client = Client() if cluster is None else Client(cluster)
#         futures = client.map(func, params)
#         progress(futures)
#         return client.gather(futures)
#
#
# def scopion(config: Dict = None) -> SLURMCluster:
#     config = (
#         {
#             "cores": 32,
#             "memory": "10G",
#             "processes": 30,
#             "queue": "scopion1",
#             "walltime": "00:30:00",
#             "header_skip": ["--mem"],
#             "scheduler_options": {"host": "192.168.1.254"},
#             # host: '192.168.1.254',
#             # extra: ['--no-dashboard'],
#             "env_extra": ["module load singularity"],  # ensure singularity is loaded
#             "python": "singularity run mbl.sif python",  # use python in container
#         }
#         if config is None
#         else config
#     )
#     return SLURMCluster(config)
