#!/bin/bash
set -euxo pipefail

poetry run cruft check
poetry run safety check -i 39462 -i 40291
poetry run bandit -c pyproject.toml -r mbl/
poetry run isort --check --diff mbl/ tests/
poetry run black --check mbl/ tests/
poetry run flake8 mbl/ tests/
poetry run mypy \
           --install-types \
           --non-interactive \
           mbl/
#  https://mypy.readthedocs.io/en/stable/running_mypy.html#library-stubs-not-installed
