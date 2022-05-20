FROM python:3.9

ARG WORKDIR=/home
WORKDIR $WORKDIR
COPY requirements.txt $WORKDIR

RUN mkdir ~/.ssh/ && \
    echo "${SSH_PRIVATE_KEY}" > ~/.ssh/id_rsa && \
    chmod 600 ~/.ssh/id_rsa && \
    ssh-keyscan -t rsa ${SFTPHOST} >> ~/.ssh/known_hosts

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    mkdir /mlflow/

EXPOSE 5000

CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --default-artifact-root sftp://${SFTPUSER}@${SFTPHOST}/${ARTIFACTPATH} \
    --backend-store-uri mysql+pymysql://${USERNAME}:${PASSWORD}@${HOST}/${DATABASE}
