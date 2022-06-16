# MLflow On-Premise Deployment using Docker Compose
Easily deploy an MLflow tracking server with 1 command.

MinIO S3 is used as the artifact store and MySQL server is used as the backend store.

## How to run

1. Clone (download) this repository

    ```bash
    git clone https://github.com/sachua/mlflow-docker-compose.git
    ```

2. `cd` into the `mlflow-docker-compose` directory

3. Build and run the containers with `docker-compose`

    ```bash
    docker-compose up -d --build
    ```

4. Access MLflow UI with http://localhost:5000

5. Access MinIO UI with http://localhost:9000

## Containerization

The MLflow tracking server is composed of 4 docker containers:

* MLflow server
* MinIO object storage server
* MySQL database server

## Example

1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. Install MLflow with extra dependencies, including scikit-learn

    ```bash
    pip install mlflow[extras]
    ```

3. Set environmental variables

    ```bash
    export MLFLOW_TRACKING_URI=http://localhost:5000
    export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
    ```
4. Set MinIO credentials

    ```bash
    cat <<EOF > ~/.aws/credentials
    [default]
    aws_access_key_id=minio
    aws_secret_access_key=minio123
    EOF
    ```

5. Train a sample MLflow model

    ```bash
    mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=0.42
    ```

    * Note: To fix ModuleNotFoundError: No module named 'boto3'

        ```bash
        #Switch to the conda env
        conda env list
        conda activate mlflow-3eee9bd7a0713cf80a17bc0a4d659bc9c549efac #replace with your own generated mlflow-environment
        pip install boto3
        ```

 6. Serve the model (replace with your model's actual path)
    ```bash
    mlflow models serve -m S3://mlflow/0/98bdf6ec158145908af39f86156c347f/artifacts/model -p 1234
    ```

 7. You can check the input with this command
    ```bash
    curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://127.0.0.1:1234/invocations
    ```

## Customization

### 1. Nginx Frontend

#### a. SSL certification (https)

To enable https with SSL certification,
it's required to mount public key and private key (in `.cert` or `.pem`) into
the container path `/etc/nginx/cert`.
One should put them in the path `/opt/docker/nginx/cert` of the based system.

Or alternatively, if using `let's encript`,
one could mount them to `/etc/letsencrypt/live/<domain_name>` in container.

#### b. Configuration files
Configuration files are placed in `/mlflow-tracker/minio/nginx` of this repo.

#### c. Set up htpasswd for basic auth
We’ll create the first user as follows (replace `first_username with username of your choice):
```
sudo htpasswd -c /etc/nginx/.htpasswd first_username
```
You will be asked to supply and confirm a password for the user.
Leave out the -c argument for any additional users you wish to add so you don’t overwrite the file:
```
sudo htpasswd /etc/nginx/.htpasswd another_user
```
The directory `/etc/nginx` is on host, and it should be mounted into the container at runtime.
See [here](https://www.digitalocean.com/community/tutorials/how-to-set-up-password-authentication-with-apache-on-ubuntu-18-04-quickstart)
for more information.

### 2. Environment file
One should replace the default environment file `/mlflow-tracker/minio/.env`
in this repo, with your own one.
Preferably in this path `/opt/docker/mlflow/.env` of the based system.
Or, one can overwrite it with
```
docker compose --env-file <path/to/your/.env> up
```

## References
https://github.com/sachua/mlflow-docker-compose
