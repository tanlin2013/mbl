FROM python:3.9

ARG WORKDIR=/home
WORKDIR $WORKDIR
COPY requirements.txt $WORKDIR

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    mkdir /mlflow/

EXPOSE 5000

CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --default-artifact-root sftp://${SFTPUSER}:${SFTPPASSWORD}@${SFTPHOST}/${ARTIFACTPATH} \
    --backend-store-uri mysql+pymysql://${USERNAME}:${PASSWORD}@${HOST}/${DATABASE}
