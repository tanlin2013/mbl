FROM python:3.9
MAINTAINER "TaoLin" <tanlin2013@gmail.com>

ARG WORKDIR=/home/mbl
ENV PYTHONPATH="${PYTHONPATH}:$WORKDIR" \
    PATH="/root/.local/bin:$PATH"
WORKDIR $WORKDIR
COPY . $WORKDIR

# Ray dashboard
EXPOSE 8265

# Install fortran, blas, lapack
RUN apt update && \
    apt-get install -y --no-install-recommends \
      gfortran libblas-dev liblapack-dev graphviz
RUN apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*

# Install required python packages
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry config virtualenvs.create false --local && \
    poetry install --without dev --extras "mlops distributed"

ENTRYPOINT /bin/bash
