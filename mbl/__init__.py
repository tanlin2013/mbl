import logging
from importlib import metadata

import awswrangler as wr
from botocore.config import Config

# -- Version ----------------------
__version__ = metadata.version(__name__)

# -- Define logger and the associated formatter and handler -------------
formatter = logging.Formatter(
    "%(asctime)s [%(filename)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)

logger = logging.getLogger("mbl")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# -- AWS Data Wrangler configuration
wr.config.botocore_config = Config(
    retries={"max_attempts": 3, "mode": "standard"},
    connect_timeout=30,
    max_pool_connections=20,
)
