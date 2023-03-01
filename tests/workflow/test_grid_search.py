# To mock mlflow tracking server, see
# (https://github.com/mlflow/mlflow/blob/master/tests/tracking/test_rest_tracking.py)
import sys
import os
import logging
import tempfile
from unittest import mock

import pytest
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.file_utils import path_to_local_file_uri
from tests.workflow.mlflow.integration_test_utils import _init_server

# Root directory for all stores (backend or artifact stores) created during this suite
SUITE_ROOT_DIR = tempfile.mkdtemp("test_rest_tracking")
# Root directory for all artifact stores created during this suite
SUITE_ARTIFACT_ROOT_DIR = tempfile.mkdtemp(suffix="artifacts", dir=SUITE_ROOT_DIR)

_logger = logging.getLogger(__name__)


def _get_sqlite_uri():
    path = path_to_local_file_uri(os.path.join(SUITE_ROOT_DIR, "test-database.bd"))
    path = path[len("file://") :]

    # NB: It looks like windows and posix have different requirements on number of
    # slashes for whatever reason. Windows needs uri like 'sqlite:///C:/path/to/my/file'
    # whereas posix expects sqlite://///path/to/my/file
    prefix = "sqlite://" if sys.platform == "win32" else "sqlite:////"
    return prefix + path


# Backend store URIs to test against
BACKEND_URIS = [
    _get_sqlite_uri(),  # SqlAlchemy
    path_to_local_file_uri(
        os.path.join(SUITE_ROOT_DIR, "file_store_root")
    ),  # FileStore
]

# Map of backend URI to tuple (server URL, Process). We populate this map by
# constructing a server per backend URI
BACKEND_URI_TO_SERVER_URL_AND_PROC = {
    uri: _init_server(backend_uri=uri, root_artifact_uri=SUITE_ARTIFACT_ROOT_DIR)
    for uri in BACKEND_URIS
}


@pytest.fixture()
def tracking_server_uri(backend_store_uri):
    url, _ = BACKEND_URI_TO_SERVER_URL_AND_PROC[backend_store_uri]
    return url


@pytest.fixture()
def mlflow_client(tracking_server_uri):
    """Provides an MLflow Tracking API client pointed at the local tracking server."""
    mlflow.set_tracking_uri(tracking_server_uri)
    yield mock.Mock(wraps=MlflowClient(tracking_server_uri))
    mlflow.set_tracking_uri(None)


class TestRandomHeisenbergTSDRGGridSearch:
    def test_experiment(self, mlflow_client):
        pass
