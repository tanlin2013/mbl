from importlib import metadata

import awswrangler as wr
from botocore.config import Config


__version__ = metadata.version(__name__)


wr.config.botocore_config = Config(
    retries={
        "max_attempts": 3,
        "mode": "standard"
    },
    connect_timeout=30,
    max_pool_connections=20
)
