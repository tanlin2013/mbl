FROM tanlin2013/tnpy:latest
MAINTAINER "TaoLin" <tanlin2013@gmail.com>

ARG WORKDIR=/home/mbl
ENV PYTHONPATH "${PYTHONPATH}:$WORKDIR"
WORKDIR $WORKDIR

# Install required python packages
COPY . $WORKDIR
RUN pip install --upgrade pip
RUN pip install -r requirements/simulation.txt

# Install mbl
RUN python setup.py install

ENTRYPOINT /bin/bash