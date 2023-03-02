#!/bin/bash
set -euxo pipefail

poetry run isort mbl/ tests/
poetry run black mbl/ tests/
