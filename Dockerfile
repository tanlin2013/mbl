FROM tanlin2013/tnpy:latest
MAINTAINER "TaoLin" <tanlin2013@gmail.com>

ARG WORKDIR=/home/mbl
ENV PYTHONPATH "${PYTHONPATH}:$WORKDIR"
WORKDIR $WORKDIR
EXPOSE 8080

# Install required python packages
COPY . $WORKDIR
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install tnpy==0.0.9 && \
    python setup.py install && \
    rm requirements.txt setup.py && \
    rm -rf mbl

ENTRYPOINT /bin/bash