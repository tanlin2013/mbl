import os
import tempfile
from pathlib import Path

import pytest

from mbl.analysis.run_searcher import RunSearcher


@pytest.fixture(scope="function")
def client():
    os.environ["AWS_ACCESS_KEY_ID"] = "tanlin2013"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "Sv5TYupU"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://do-doublets.phys.nthu.edu.tw:8786/"
    os.environ["MLFLOW_TRACKING_USERNAME"] = "tanlin2013"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "$p@t1@lly"
    return RunSearcher(tracking_uri="https://do-doublets.phys.nthu.edu.tw:8080/")


def test_query(client):
    filter_string = client.filter(n=8, h=0.5, chi=64, seed=2024, method="min")
    runs = client.query(experiment_id=4, filter_string=filter_string)
    assert len(runs) == 3  # for 3 `relative_offset`s
    # _ = [client.delete_run(run.info.run_id) for run in res]


@pytest.mark.parametrize(
    "n, h, chi, seed, method, relative_offset", [(8, 0.5, 64, 2024, "min", 0.1)]
)
def test_list_artifacts(client, n, h, chi, seed, method, relative_offset):
    filter_string = client.filter(
        n=n, h=h, chi=chi, seed=seed, method=method, relative_offset=relative_offset
    )
    artifacts = client.list_artifacts(experiment_id=4, filter_string=filter_string)
    assert len(artifacts) == 2
    assert artifacts[0].path == "df.json"
    assert Path(artifacts[1].path).suffix == ".p"


@pytest.mark.parametrize(
    "n, h, chi, seed, method, relative_offset", [(8, 0.5, 64, 2024, "min", 0.1)]
)
def test_down_artifacts(client, n, h, chi, seed, method, relative_offset):
    filter_string = client.filter(
        n=n, h=h, chi=chi, seed=seed, method=method, relative_offset=relative_offset
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        client.download_artifacts(
            experiment_id=4, filter_string=filter_string, dst_path=tmp_dir
        )
        assert len(os.listdir(tmp_dir)) == 2
        assert "df.json" in os.listdir(tmp_dir)
