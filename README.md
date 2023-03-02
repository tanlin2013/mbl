# mbl: Many-body localization

_________________

[![PyPI version](https://badge.fury.io/py/mbl.svg)](http://badge.fury.io/py/mbl)
[![Downloads](https://pepy.tech/badge/mbl)](https://pepy.tech/project/mbl)
[![codecov](https://codecov.io/gh/tanlin2013/mbl/branch/main/graph/badge.svg)](https://codecov.io/gh/tanlin2013/mbl)
[![Join the chat at https://gitter.im/tanlin2013/mbl](https://badges.gitter.im/tanlin2013/mbl.svg)](https://gitter.im/tanlin2013/mbl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://timothycrosley.github.io/isort/)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://pypi.python.org/pypi/mbl/)
[![Docker build](https://github.com/tanlin2013/mbl/actions/workflows/build.yml/badge.svg)](https://github.com/tanlin2013/mbl/actions/workflows/build.yml)
[![Test Status](https://github.com/tanlin2013/mbl/actions/workflows/test.yml/badge.svg)](https://github.com/tanlin2013/mbl/actions/workflows/test.yml)
[![Lint Status](https://github.com/tanlin2013/mbl/actions/workflows/lint.yml/badge.svg)](https://github.com/tanlin2013/mbl/actions/workflows/lint.yml)
[![job-deploy](https://github.com/tanlin2013/mbl/actions/workflows/job-deploy.yml/badge.svg)](https://github.com/tanlin2013/mbl/actions/workflows/job-deploy.yml)
[![service-deploy](https://github.com/tanlin2013/mbl/actions/workflows/service-deploy.yml/badge.svg)](https://github.com/tanlin2013/mbl/actions/workflows/service-deploy.yml)
[![cdk-deploy](https://github.com/tanlin2013/mbl/actions/workflows/aws-cdk.yml/badge.svg)](https://github.com/tanlin2013/mbl/actions/workflows/aws-cdk.yml)
_________________

[Documentation](https://tanlin2013.github.io/mbl/) |
[Dashboard](https://streamlit-mbl.herokuapp.com/)
_________________

MBL is a research program aims at studying the delocalization transitions with numerical methods,
currently with exact diagonalization (ED) and tree tensor strong-disorder RG (tSDRG).
And the support of matrix product state (MPS) is under planning.

Models
------
* Random-field Heisenberg model

Prerequisite
------------
* An AWS account
* MLflow tracking server

For the purpose of better data wrangling and MLOps experience,
this repo highly relies on [aws data wrangler](https://aws-data-wrangler.readthedocs.io/en/stable/)
and [mlflow tracking](https://mlflow.org/docs/latest/tracking.html).
For backend algorithms, one may refer to [tnpy](https://tanlin2013.github.io/tnpy/).


Installation
------------
* using Docker:
  ```
  docker run --rm -it -v $(PWD)/data:/home/data tanlin2013/mbl
  ```
* using pip:
  ```
  pip install git+https://github.com/tanlin2013/mbl@main
  ```

Getting started
---------------
Run a sampler (with parallel runs supported by [ray](https://ray.io/)).

```
python scripts/sampler.py
```

License
-------
© Tan Tao-Lin, 2021. Licensed under
a [MIT](https://github.com/tanlin2013/mbl/master/LICENSE)
license.

Reference
-------
[D. A. Abanin, E. Altman, I. Bloch and M. Serbyn, Colloquium: Many-body localization, thermalization, and entanglement. Rev. Mod. Phys. 91, 021001 (2019).](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.91.021001)
