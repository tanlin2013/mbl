from typing import List, Optional

from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType, FileInfo

from mbl.name_space import Columns


class RunSearcher(MlflowClient):
    def __init__(self, tracking_uri: str, **kwargs):
        super().__init__(tracking_uri=tracking_uri, **kwargs)

    @staticmethod
    def filter(
        n: int,
        h: float,
        penalty: float = 0.0,
        s_target: int = 0,
        seed: int = None,
        chi: int = None,
        offset: float = None,
        overall_const: float = None,
        relative_offset: float = None,
        method: str = None,
    ):
        """

        Args:
            n:
            h:
            penalty:
            s_target:
            seed:
            chi:
            offset:
            overall_const:
            relative_offset:
            method:

        Returns:

        Examples:
        Query string takes the form,

            query = (
                "params.model = 'CNN' "
                "and params.layers = '10' "
                "and metrics.`prediction accuracy` >= 0.945"
            )
        """
        query = [
            f"params.{'n'} = '{n}'",
            f"params.{'h'} = '{h}'",
            f"params.{Columns.penalty} = '{penalty}'",
            f"params.{Columns.s_target} = '{s_target}'",
            f"params.{'chi'} = '{chi}'",
        ]
        if seed is not None:
            query.append(f"params.{Columns.seed} = '{seed}'")
        if offset is not None:
            query.append(f"params.{Columns.offset} = '{offset}'")
        if relative_offset is not None:
            query.append(f"params.{Columns.relative_offset} = '{relative_offset}'")
        if overall_const is not None:
            query.append(f"params.{Columns.overall_const} = '{overall_const}'")
        if method is not None:
            query.append(f"params.{Columns.method} = '{method}'")
        return " and ".join(query)

    def query(self, experiment_id: int, filter_string: str = None, **kwargs):
        filter_string = "" if filter_string is None else filter_string
        return self.search_runs(
            experiment_ids=[str(experiment_id)],
            filter_string=filter_string,
            run_view_type=ViewType.ACTIVE_ONLY,
            **kwargs,
        )

    def list_artifacts(
        self,
        experiment_id: int,
        filter_string: str,
        path: Optional[str] = None,
        **kwargs,
    ) -> List[FileInfo]:
        (run,) = self.query(
            experiment_id=experiment_id,
            filter_string=filter_string,
            **kwargs,
        )
        path = "" if path is None else path
        return super().list_artifacts(run.info.run_id, path)

    def download_artifacts(
        self,
        experiment_id: int,
        filter_string: str,
        path: Optional[str] = None,
        dst_path: Optional[str] = None,
        **kwargs,
    ):
        (run,) = self.query(
            experiment_id=experiment_id,
            filter_string=filter_string,
            **kwargs,
        )
        path = "" if path is None else path
        super().download_artifacts(run.info.run_id, path, dst_path)
